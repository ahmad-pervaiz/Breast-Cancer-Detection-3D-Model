"""
Box-prompted SAM, fine-tuned decoder-only.

Wraps segment_anything's Sam model (image_encoder + prompt_encoder +
mask_decoder) so it exposes the same forward contract as src/models/unet.py's
UNet - raw logits, shape (B, out_channels, H, W) at the dataset's native
resolution - so DiceBCELoss and MetricAccumulator work unchanged regardless
of which model produced the logits.

The image encoder (~89.7M params) and, by default, the prompt encoder
(~6.2K params) are frozen: only the mask decoder (~4M params) is trained.
See configs/sam_finetune.yaml for why - the whole point of using SAM here is
its pretrained visual features, which 5-7 patients can't teach from scratch
(see improving_model.md's overfitting/shortcut-learning findings on the
from-scratch U-Net).
"""
from pathlib import Path

import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class SAMFineTune(nn.Module):
    def __init__(self, sam: nn.Module, freeze_image_encoder: bool = True,
                 freeze_prompt_encoder: bool = True):
        super().__init__()
        self.sam = sam
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        if freeze_image_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad = False

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        preprocessed = self.sam.preprocess(images)
        if self.freeze_image_encoder:
            with torch.no_grad():
                return self.sam.image_encoder(preprocessed)
        return self.sam.image_encoder(preprocessed)

    def _encode_prompt(self, boxes: torch.Tensor):
        if self.freeze_prompt_encoder:
            with torch.no_grad():
                return self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
        return self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)

    def forward(self, images: torch.Tensor, boxes: torch.Tensor, original_size: tuple) -> torch.Tensor:
        """images: (B, 3, 1024, 1024) raw [0, 255]. boxes: (B, 4) xyxy, already in
        the 1024-space (see src/data/sam_dataset.py). original_size: (H, W) to
        upscale the decoder's low-res output back to - the dataset's native
        image_size, matching the mask/target resolution.

        segment_anything's mask_decoder is built for ONE image + N prompts (its
        `predict_masks` does `repeat_interleave(image_embeddings, tokens.shape[0])`,
        i.e. it repeats a single image's embedding to match multiple prompts for
        THAT image - see Sam.forward's own per-image loop in the upstream
        library). For N different images with one box each (our training batch),
        that repeat_interleave silently produces the wrong shape instead of a
        clean 1:1 pairing. So the image encoder runs batched (it's the expensive
        part), but the lightweight prompt encoder + mask decoder run per-image."""
        image_embeddings = self._encode_image(images)  # (B, C, H, W)
        input_size = (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size)
        low_res_masks = []
        for i in range(image_embeddings.shape[0]):
            sparse_emb, dense_emb = self._encode_prompt(boxes[i : i + 1])
            masks_i, _iou_pred = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            low_res_masks.append(masks_i)
        low_res_masks = torch.cat(low_res_masks, dim=0)
        return self.sam.postprocess_masks(low_res_masks, input_size, original_size)


def build_sam_model(cfg: dict) -> SAMFineTune:
    sam_cfg = cfg["sam"]
    checkpoint = sam_cfg.get("checkpoint")
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint!r}. This is a manual download - "
            f"see configs/sam_finetune.yaml's `sam.checkpoint` comment for where to get it."
        )
    sam = sam_model_registry[sam_cfg["model_type"]](checkpoint=checkpoint)
    return SAMFineTune(
        sam,
        freeze_image_encoder=sam_cfg.get("freeze_image_encoder", True),
        freeze_prompt_encoder=sam_cfg.get("freeze_prompt_encoder", True),
    )
