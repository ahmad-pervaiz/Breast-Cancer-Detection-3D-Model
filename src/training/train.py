"""
Core training loop for the 2D tumor segmentation U-Net.

Called by scripts/train.py. Kept import-safe without torch installed is NOT
attempted here (torch is a hard dependency of this module, unlike dataset.py).
"""
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.common import get_device, resolve_path, save_config, set_seed
from src.data.dataset import BCTumorSegDataset, build_train_augmentations, load_manifest
from src.data.preprocessing import PreprocessConfig
from src.data.validation import check_integrity, print_patient_split, print_summary, raise_if_broken
from src.models.unet import build_model
from src.training.experiment_tracking import close_clearml, init_clearml, log_checkpoint_artifact, log_epoch_metrics
from src.training.losses import DiceBCELoss
from src.training.metrics import MetricAccumulator
from src.visualization.visualize import plot_training_curves, save_epoch_montage


def _stratified_indices(df, subset_size: int, seed: int) -> list:
    """Half Tumor / half Normal (as available) - so a small smoke-test subset
    still exercises the positive-mask code path, instead of risking an
    all-Normal slice from naive first-N indexing (labels are folder-grouped)."""
    rng = np.random.RandomState(seed)
    half = max(1, subset_size // 2)
    tumor_idx = df.index[df["label"] == "Tumor"].tolist()
    normal_idx = df.index[df["label"] == "Normal"].tolist()
    picked = (
        list(rng.choice(tumor_idx, size=min(half, len(tumor_idx)), replace=False)) if tumor_idx else []
    ) + (
        list(rng.choice(normal_idx, size=min(subset_size - half, len(normal_idx)), replace=False)) if normal_idx else []
    )
    return sorted(picked)[:subset_size]


def _make_loaders(cfg: dict, preprocess_cfg: PreprocessConfig, subset_size: Optional[int] = None):
    dataset_root = Path(cfg["dataset_root"])
    manifest_path = resolve_path(cfg, "manifest_csv")
    df = load_manifest(manifest_path)

    print("\n--- Dataset integrity check (refuses to train on a broken dataset) ---")
    errors = check_integrity(df[df["split"].isin(["train", "valid"])], dataset_root)
    raise_if_broken(errors)
    print("No integrity issues found.")

    print("\n--- Dataset summary ---")
    print_summary(df[df["split"].isin(["train", "valid"])])

    print("\n--- Patient-level split (leakage check) ---")
    print_patient_split(df[df["split"].isin(["train", "valid"])])

    train_aug = build_train_augmentations(cfg["augmentation"])
    train_ds = BCTumorSegDataset(df, dataset_root, "train", preprocess_cfg, augment=train_aug)
    valid_ds = BCTumorSegDataset(df, dataset_root, "valid", preprocess_cfg, augment=None)

    if subset_size:
        train_indices = _stratified_indices(train_ds.df, subset_size, cfg["seed"])
        valid_indices = _stratified_indices(valid_ds.df, subset_size, cfg["seed"])
        train_ds = Subset(train_ds, train_indices)
        valid_ds = Subset(valid_ds, valid_indices)

    print(f"\nTrain samples: {len(train_ds)} | Valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"])
    return train_loader, valid_loader


def _train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp,
                      threshold: float, min_tumor_area_px: int) -> Dict[str, float]:
    model.train()
    total_loss = total_dice_loss = total_bce_loss = 0.0
    n_batches = 0
    # Thresholded, empty-mask-aware accumulator - the SAME metric definition used
    # for validation (see MetricAccumulator), so train_dice and val_dice are
    # actually comparable. The raw soft Dice *loss* (dice_loss, logged separately)
    # is structurally biased low on empty-mask images and must not be read as a
    # Dice score - see the module-level note above train().
    accumulator = MetricAccumulator(min_tumor_area_px=min_tumor_area_px, compute_hd95=False)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss, dice_l, bce_l = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, dice_l, bce_l = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice_loss += dice_l.item()
        total_bce_loss += bce_l.item()
        n_batches += 1

        with torch.no_grad():
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= threshold).astype(np.uint8)
            gts = masks.detach().cpu().numpy().astype(np.uint8)
            for i in range(images.shape[0]):
                accumulator.update(preds[i, 0], gts[i, 0])

    summary = accumulator.summary()
    return {
        "loss": total_loss / n_batches,
        "dice_loss": total_dice_loss / n_batches,
        "bce_loss": total_bce_loss / n_batches,
        "dice": summary["overall_segmentation_including_normal"]["dice"],
        "tumor_positive_dice": summary["tumor_positive_segmentation"]["dice"],
    }


@torch.no_grad()
def _validate_one_epoch(model, loader, criterion, device, threshold, min_tumor_area_px):
    model.eval()
    total_loss = total_dice_loss = total_bce_loss = 0.0
    n_batches = 0
    accumulator = MetricAccumulator(min_tumor_area_px=min_tumor_area_px, compute_hd95=False)
    fixed_samples = []
    n_positive_samples = n_empty_samples = 0
    max_each = 2  # guarantee the montage shows both a tumor and a Normal example, not just whichever sorts first

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss, dice_l, bce_l = criterion(logits, masks)
        total_loss += loss.item()
        total_dice_loss += dice_l.item()
        total_bce_loss += bce_l.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(np.uint8)
        gts = masks.cpu().numpy().astype(np.uint8)

        for i in range(images.shape[0]):
            accumulator.update(preds[i, 0], gts[i, 0],
                                patient_id=batch["patient_id"][i], image_path=batch["image_path"][i])
            is_positive = bool(gts[i, 0].sum() > 0)
            if is_positive and n_positive_samples >= max_each:
                continue
            if not is_positive and n_empty_samples >= max_each:
                continue
            if is_positive:
                n_positive_samples += 1
            else:
                n_empty_samples += 1
            fixed_samples.append({
                "image": images[i, 0].cpu().numpy(),
                "gt_mask": gts[i, 0],
                "pred_mask": preds[i, 0],
            })

    summary = accumulator.summary()
    val_metrics = {
        "loss": total_loss / n_batches,
        "dice_loss": total_dice_loss / n_batches,
        "bce_loss": total_bce_loss / n_batches,
        "dice": summary["overall_segmentation_including_normal"]["dice"],
        "iou": summary["overall_segmentation_including_normal"]["iou"],
        "tumor_positive_dice": summary["tumor_positive_segmentation"]["dice"],
        "tumor_positive_iou": summary["tumor_positive_segmentation"]["iou"],
        "precision": summary["overall_segmentation_including_normal"]["precision"],
        "recall": summary["overall_segmentation_including_normal"]["recall"],
        "detection": summary["image_level_detection"],
    }
    return val_metrics, fixed_samples


def train(cfg: dict, smoke_test: bool = False, subset_size: Optional[int] = None) -> None:
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"

    checkpoint_dir = resolve_path(cfg, "checkpoint_dir")
    log_dir = resolve_path(cfg, "log_dir")
    plot_dir = resolve_path(cfg, "plot_dir")
    prediction_dir = resolve_path(cfg, "prediction_dir") / "validation"
    for d in [checkpoint_dir, log_dir, plot_dir, prediction_dir]:
        d.mkdir(parents=True, exist_ok=True)

    save_config(cfg, log_dir.parent / "config_used.yaml")
    clearml_task = init_clearml(cfg)

    preprocess_cfg = PreprocessConfig(
        image_size=cfg["image_size"],
        preserve_aspect_ratio=cfg["preserve_aspect_ratio"],
        image_interpolation=cfg["image_interpolation"],
        mask_interpolation=cfg["mask_interpolation"],
    )
    train_loader, valid_loader = _make_loaders(cfg, preprocess_cfg, subset_size=subset_size)

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: UNet | in_ch={cfg['model']['in_channels']} out_ch={cfg['model']['out_channels']} "
          f"base_features={cfg['model']['base_features']} depth={cfg['model']['depth']} | params={n_params:,}")

    criterion = DiceBCELoss(dice_weight=cfg["dice_weight"], bce_weight=cfg["bce_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg["lr_scheduler"]["patience"],
        factor=cfg["lr_scheduler"]["factor"], min_lr=cfg["lr_scheduler"]["min_lr"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = 2 if smoke_test else cfg["epochs"]
    early_stop_patience = cfg["early_stopping"]["patience"]
    best_val_dice = -1.0
    epochs_without_improvement = 0

    log_csv_path = log_dir / "training.csv"
    fieldnames = ["epoch", "train_loss", "train_dice", "train_tumor_positive_dice",
                  "val_loss", "val_dice", "val_iou",
                  "val_tumor_positive_dice", "val_tumor_positive_iou", "val_precision", "val_recall",
                  "val_sensitivity", "val_specificity", "learning_rate", "epoch_seconds"]
    with open(log_csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"\n=== Starting training: {epochs} epochs ===\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp,
                                          cfg["threshold"], cfg["min_tumor_area_px"])
        val_metrics, fixed_samples = _validate_one_epoch(
            model, valid_loader, criterion, device, cfg["threshold"], cfg["min_tumor_area_px"]
        )
        scheduler.step(val_metrics["dice"])
        epoch_seconds = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs}  ({epoch_seconds:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
              f"tumor+dice={train_metrics['tumor_positive_dice']}")
        print(f"  Valid: loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} "
              f"iou={val_metrics['iou']:.4f} "
              f"tumor+dice={val_metrics['tumor_positive_dice']}  "
              f"precision={val_metrics['precision']} recall={val_metrics['recall']}")
        det = val_metrics["detection"]
        print(f"  Valid detection: sensitivity={det['sensitivity_recall']} specificity={det['specificity']} "
              f"accuracy={det['accuracy']} f1={det['f1']}")
        print(f"  LR: {current_lr:.2e}")
        log_epoch_metrics(clearml_task, epoch, train_metrics, val_metrics, current_lr)

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_metrics["loss"], "train_dice": train_metrics["dice"],
                "train_tumor_positive_dice": train_metrics["tumor_positive_dice"],
                "val_loss": val_metrics["loss"], "val_dice": val_metrics["dice"], "val_iou": val_metrics["iou"],
                "val_tumor_positive_dice": val_metrics["tumor_positive_dice"],
                "val_tumor_positive_iou": val_metrics["tumor_positive_iou"],
                "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
                "val_sensitivity": det["sensitivity_recall"], "val_specificity": det["specificity"],
                "learning_rate": current_lr, "epoch_seconds": epoch_seconds,
            })

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "cfg": cfg},
                   checkpoint_dir / "last_model.pth")

        val_dice = val_metrics["dice"]
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_dice": val_dice, "cfg": cfg},
                       checkpoint_dir / "best_model.pth")
            print(f"  -> New best model saved (val_dice={val_dice:.4f})")
            log_checkpoint_artifact(clearml_task, "best_model", checkpoint_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            save_epoch_montage(fixed_samples, prediction_dir / f"epoch_{epoch:03d}.png")

        if cfg["early_stopping"]["enabled"] and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping: no val_dice improvement for {early_stop_patience} epochs.")
            break

    if not smoke_test:
        plot_training_curves(log_csv_path, plot_dir)
        print(f"\nSaved plots to {plot_dir}")

    log_checkpoint_artifact(clearml_task, "last_model", checkpoint_dir / "last_model.pth")
    log_checkpoint_artifact(clearml_task, "training_log", log_csv_path)
    if not smoke_test:
        for name in ["loss_curve.png", "dice_curve.png", "iou_curve.png"]:
            log_checkpoint_artifact(clearml_task, name, plot_dir / name)
    close_clearml(clearml_task)

    print(f"\nTraining complete. Best val_dice={best_val_dice:.4f}. "
          f"Checkpoints in {checkpoint_dir}, logs in {log_dir}.")

    if train_metrics["dice"] - best_val_dice > 0.15:
        print("\nWARNING: training Dice substantially exceeds best validation Dice - "
              "possible overfitting given the limited number of independent patients. "
              "Consider stronger regularization/augmentation or more data before trusting this model.")
