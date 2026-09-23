"""
PyTorch Dataset for box-prompted SAM fine-tuning.

Unlike BCTumorSegDataset (src/data/dataset.py), this only ever yields
tumor-positive images (label == "Tumor" with a real mask) - SAM is
box-PROMPTED, it has nothing meaningful to do on a Normal (empty-mask)
image, since there's no box to give it. Tumor presence stays the U-Net/
classifier's job (see configs/sam_finetune.yaml's header comment).

Reuses the same manifest + preprocessing pipeline as the U-Net dataset
(src/data/preprocessing.py) so both models see identical image/mask content
at the same working resolution, up to SAM's own required resize to 1024x1024.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from segment_anything.utils.transforms import ResizeLongestSide

from src.data.preprocessing import PreprocessConfig, preprocess_image, preprocess_mask

SAM_INPUT_SIZE = 1024  # fixed by every SAM ViT image encoder's positional embeddings


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """mask: (H, W) {0., 1.}. Returns [x_min, y_min, x_max, y_max] (inclusive)."""
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _pad_and_jitter_box(box: np.ndarray, size: int, pad_px: int, jitter_px: int,
                         rng: np.random.RandomState) -> np.ndarray:
    """Expands the tight GT box by pad_px, then (if jitter_px > 0) perturbs each
    coordinate independently by +/-jitter_px - teaches the decoder to tolerate an
    imprecise prompt, since no real downstream box source is pixel-perfect."""
    x_min, y_min, x_max, y_max = box
    x_min -= pad_px; y_min -= pad_px
    x_max += pad_px; y_max += pad_px
    if jitter_px > 0:
        x_min += rng.uniform(-jitter_px, jitter_px)
        y_min += rng.uniform(-jitter_px, jitter_px)
        x_max += rng.uniform(-jitter_px, jitter_px)
        y_max += rng.uniform(-jitter_px, jitter_px)
    x_min = np.clip(x_min, 0, size - 1)
    y_min = np.clip(y_min, 0, size - 1)
    x_max = np.clip(x_max, x_min + 1, size)
    y_max = np.clip(y_max, y_min + 1, size)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class SAMTumorSegDataset(Dataset):
    """One sample = one tumor-positive CT-slice PNG + its binary mask + a box
    prompt derived from that same mask (padded, and jittered if `train=True`)."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_root: Path,
        split: str,
        preprocess_cfg: PreprocessConfig,
        box_padding_px: int = 5,
        box_jitter_px: int = 10,
        train: bool = False,
        seed: int = 42,
    ):
        df = manifest_df[manifest_df["split"] == split]
        df = df[(df["label"] == "Tumor") & df["has_mask"].astype(bool)]
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No tumor-positive rows found for split='{split}' in manifest.")
        self.dataset_root = Path(dataset_root)
        self.preprocess_cfg = preprocess_cfg
        self.box_padding_px = box_padding_px
        self.box_jitter_px = box_jitter_px if train else 0
        self.rng = np.random.RandomState(seed)
        self.resize = ResizeLongestSide(SAM_INPUT_SIZE)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.dataset_root / row["image_path"]
        mask_path = self.dataset_root / row["mask_path"]

        image = preprocess_image(image_path, self.preprocess_cfg)  # (H, W) float32 [0, 1]
        mask = preprocess_mask(mask_path, self.preprocess_cfg)     # (H, W) float32 {0., 1.}
        size = self.preprocess_cfg.image_size

        box = _bbox_from_mask(mask)
        box = _pad_and_jitter_box(box, size, self.box_padding_px, self.box_jitter_px, self.rng)
        box_resized = self.resize.apply_boxes(box[None, :], (size, size))[0]

        image_rgb_u8 = np.stack([(image * 255).astype(np.uint8)] * 3, axis=-1)  # (H, W, 3)
        image_resized = self.resize.apply_image(image_rgb_u8)  # (1024, 1024, 3) uint8

        image_t = torch.from_numpy(np.ascontiguousarray(image_resized)).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        box_t = torch.from_numpy(box_resized).float()

        return {
            "image": image_t,           # (3, 1024, 1024) float, raw [0, 255] - sam.preprocess normalizes
            "box": box_t,                # (4,) float, xyxy in the resized-1024 coordinate frame
            "mask": mask_t,               # (1, size, size) float {0., 1.} - loss/metric target
            "label": row["label"],
            "patient_id": row["patient_id"],
            "image_path": str(image_path),
        }
