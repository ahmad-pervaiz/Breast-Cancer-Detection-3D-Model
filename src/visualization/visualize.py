"""Plotting (matplotlib only, no seaborn) and prediction-image saving."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def plot_training_curves(log_csv: Path, plot_dir: Path) -> None:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(log_csv)

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(plot_dir / "loss_curve.png", dpi=120); plt.close()

    # Dice curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_dice"], label="train_dice")
    plt.plot(df["epoch"], df["val_dice"], label="val_dice")
    plt.xlabel("epoch"); plt.ylabel("Dice"); plt.title("Training vs Validation Dice")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(plot_dir / "dice_curve.png", dpi=120); plt.close()

    # IoU curve (validation only - not tracked on train per-batch by default)
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["val_iou"], label="val_iou", color="green")
    plt.xlabel("epoch"); plt.ylabel("IoU"); plt.title("Validation IoU")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(plot_dir / "iou_curve.png", dpi=120); plt.close()


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha: float = 0.4) -> np.ndarray:
    """image: (H,W) float [0,1] grayscale. mask: (H,W) {0,1}. Returns (H,W,3) float [0,1]."""
    rgb = np.stack([image, image, image], axis=-1)
    mask_b = mask.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(mask_b, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
    return rgb


def save_prediction_set(
    image: np.ndarray, gt_mask: Optional[np.ndarray], prob_map: np.ndarray,
    pred_mask: np.ndarray, out_dir: Path, stem: str,
) -> None:
    """Saves original/probability/mask/overlay(/gt) PNGs for one sample."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(_to_uint8(image)).save(out_dir / f"{stem}_original.png")
    Image.fromarray(_to_uint8(prob_map)).save(out_dir / f"{stem}_probability.png")
    Image.fromarray(_to_uint8(pred_mask)).save(out_dir / f"{stem}_mask.png")

    overlay = make_overlay(image, pred_mask, color=(1.0, 0.0, 0.0))
    Image.fromarray(_to_uint8(overlay)).save(out_dir / f"{stem}_overlay.png")

    if gt_mask is not None:
        gt_overlay = make_overlay(image, gt_mask, color=(0.0, 1.0, 0.0))
        Image.fromarray(_to_uint8(gt_overlay)).save(out_dir / f"{stem}_gt_overlay.png")


def save_epoch_montage(samples: list, out_path: Path) -> None:
    """samples: list of dicts with keys image, gt_mask, pred_mask (all (H,W) arrays)."""
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, s in enumerate(samples):
        img, gt, pred = s["image"], s["gt_mask"], s["pred_mask"]
        axes[i, 0].imshow(img, cmap="gray"); axes[i, 0].set_title("image", fontsize=8)
        axes[i, 1].imshow(gt, cmap="gray"); axes[i, 1].set_title("ground truth", fontsize=8)
        axes[i, 2].imshow(pred, cmap="gray"); axes[i, 2].set_title("prediction", fontsize=8)
        axes[i, 3].imshow(make_overlay(img, pred)); axes[i, 3].set_title("overlay", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
