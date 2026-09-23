#!/usr/bin/env python3
"""
Compare the SAM box-prompted fine-tune against the from-scratch U-Net on the
same tumor-positive validation images.

Usage:
    python scripts/compare_sam_vs_unet.py \
        --unet_checkpoint runs/segmentation/checkpoints/best_model.pth \
        --sam_checkpoint runs/sam_segmentation/checkpoints/best_model.pth

Both models are evaluated ONLY on tumor-positive validation images (label ==
"Tumor" with a real mask) - SAM has nothing to do on Normal images (see
src/data/sam_dataset.py), so this compares against the U-Net's
`tumor_positive_dice`/`tumor_positive_iou` numbers specifically, not its
"overall including Normal" ones (see configs/sam_finetune.yaml's header).

Each model runs at its OWN native working resolution (the U-Net at its
trained image_size, typically 256; SAM at 512 pre-resize -> 1024
internally). Dice/IoU are resolution-normalized (pixel-overlap ratios), so
this is a standard way to compare two different pipelines, but it is not a
pixel-for-pixel identical comparison.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import get_device, resolve_path
from src.data.dataset import load_manifest
from src.data.preprocessing import PreprocessConfig, preprocess_image, preprocess_mask
from src.data.sam_dataset import SAMTumorSegDataset
from src.inference.predict import load_checkpoint as load_unet_checkpoint
from src.models.sam_finetune import build_sam_model
from src.training.metrics import MetricAccumulator
from src.visualization.visualize import make_overlay


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet_checkpoint", type=str, default="runs/segmentation/checkpoints/best_model.pth")
    p.add_argument("--sam_checkpoint", type=str, default="runs/sam_segmentation/checkpoints/best_model.pth")
    p.add_argument("--n_montage_samples", type=int, default=4)
    p.add_argument("--montage_out", type=str, default="runs/sam_vs_unet_comparison.png")
    p.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N tumor-positive validation images - useful for a "
                         "quick code-path check (SAM's ViT-B forward pass is slow on CPU; the full "
                         "validation set is meant for a Kaggle GPU run).")
    return p.parse_args()


@torch.no_grad()
def _run_unet(model, cfg, df, dataset_root, device):
    preprocess_cfg = PreprocessConfig(
        image_size=cfg["image_size"], preserve_aspect_ratio=cfg["preserve_aspect_ratio"],
        image_interpolation=cfg["image_interpolation"], mask_interpolation=cfg["mask_interpolation"],
    )
    accumulator = MetricAccumulator(min_tumor_area_px=cfg["min_tumor_area_px"], compute_hd95=False)
    threshold = cfg["threshold"]
    samples = {}
    for _, row in df.iterrows():
        image_path = dataset_root / row["image_path"]
        mask_path = dataset_root / row["mask_path"]
        image = preprocess_image(image_path, preprocess_cfg)
        gt = preprocess_mask(mask_path, preprocess_cfg)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8)
        accumulator.update(pred, gt.astype(np.uint8), patient_id=row["patient_id"], image_path=str(image_path))
        samples[Path(row["image_path"]).stem] = {"image": image, "gt_mask": gt, "pred_mask": pred}
    return accumulator.summary(), samples


@torch.no_grad()
def _run_sam(model, cfg, df, dataset_root, device):
    preprocess_cfg = PreprocessConfig(
        image_size=cfg["image_size"], preserve_aspect_ratio=cfg["preserve_aspect_ratio"],
        image_interpolation=cfg["image_interpolation"], mask_interpolation=cfg["mask_interpolation"],
    )
    ds = SAMTumorSegDataset(df, dataset_root, "valid", preprocess_cfg,
                             box_padding_px=cfg["sam"]["box_padding_px"], box_jitter_px=0, train=False)
    original_size = (cfg["image_size"], cfg["image_size"])
    accumulator = MetricAccumulator(min_tumor_area_px=cfg["min_tumor_area_px"], compute_hd95=False)
    threshold = cfg["threshold"]
    samples = {}
    for i in range(len(ds)):
        item = ds[i]
        images = item["image"].unsqueeze(0).to(device)
        boxes = item["box"].unsqueeze(0).to(device)
        gt = item["mask"][0].numpy()
        logits = model(images, boxes, original_size)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8)
        accumulator.update(pred, gt.astype(np.uint8), patient_id=item["patient_id"], image_path=item["image_path"])
        stem = Path(item["image_path"]).stem
        # Downsample the 1024x1024 SAM input back to original_size (matches
        # gt/pred resolution) purely for the comparison montage.
        display_image = torch.nn.functional.interpolate(
            images, size=original_size, mode="bilinear", align_corners=False,
        )[0].mean(dim=0).cpu().numpy() / 255.0
        samples[stem] = {"image": display_image, "gt_mask": gt, "pred_mask": pred}
    return accumulator.summary(), samples


def _print_block(name: str, block: dict):
    print(f"  {name:10s} dice={block['dice']:.4f}  iou={block['iou']:.4f}  "
          f"precision={block['precision']}  recall={block['recall']}  n={block['n_images']}")


def _save_montage(unet_samples, sam_samples, out_path, n):
    shared = [s for s in unet_samples if s in sam_samples][:n]
    if not shared:
        print("No shared image stems between U-Net and SAM sample sets - skipping montage.")
        return
    fig, axes = plt.subplots(len(shared), 4, figsize=(14, 3.2 * len(shared)))
    if len(shared) == 1:
        axes = axes[np.newaxis, :]
    for i, stem in enumerate(shared):
        u, s = unet_samples[stem], sam_samples[stem]
        axes[i, 0].imshow(u["image"], cmap="gray"); axes[i, 0].set_title("image (U-Net res)", fontsize=8)
        axes[i, 1].imshow(u["gt_mask"], cmap="gray"); axes[i, 1].set_title("ground truth", fontsize=8)
        axes[i, 2].imshow(make_overlay(u["image"], u["pred_mask"])); axes[i, 2].set_title("U-Net pred", fontsize=8)
        axes[i, 3].imshow(make_overlay(s["image"], s["pred_mask"])); axes[i, 3].set_title("SAM pred", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved comparison montage to {out_path}")


def main():
    args = parse_args()
    device = get_device()

    unet_model, unet_cfg = load_unet_checkpoint(Path(args.unet_checkpoint), device)
    unet_model.eval()

    sam_ckpt = torch.load(args.sam_checkpoint, map_location=device, weights_only=False)
    sam_cfg = sam_ckpt["cfg"]
    sam_model = build_sam_model(sam_cfg).to(device)
    sam_model.load_state_dict(sam_ckpt["model_state"])
    sam_model.eval()
    print(f"Loaded SAM checkpoint: {args.sam_checkpoint} (epoch={sam_ckpt.get('epoch')}, "
          f"val_dice={sam_ckpt.get('val_dice')})")

    dataset_root = Path(sam_cfg["dataset_root"])
    manifest_path = resolve_path(sam_cfg, "manifest_csv")
    df = load_manifest(manifest_path)
    valid_df = df[(df["split"] == "valid") & (df["label"] == "Tumor") & df["has_mask"].astype(bool)]
    if args.limit:
        valid_df = valid_df.iloc[: args.limit]
    print(f"Comparing on {len(valid_df)} tumor-positive validation images.\n")

    unet_summary, unet_samples = _run_unet(unet_model, unet_cfg, valid_df, dataset_root, device)
    sam_summary, sam_samples = _run_sam(sam_model, sam_cfg, valid_df, dataset_root, device)

    print("Tumor-positive segmentation quality (same images, each model's own native resolution):")
    _print_block("U-Net", unet_summary["tumor_positive_segmentation"])
    _print_block("SAM", sam_summary["tumor_positive_segmentation"])

    _save_montage(unet_samples, sam_samples, args.montage_out, args.n_montage_samples)


if __name__ == "__main__":
    main()
