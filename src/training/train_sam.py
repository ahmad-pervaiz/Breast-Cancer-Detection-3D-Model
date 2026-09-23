"""
Training loop for the box-prompted SAM decoder fine-tune.

Structurally a copy of src/training/train.py's train() - same CSV logging,
checkpointing, ClearML hooks, montage saving, DiceBCELoss, MetricAccumulator -
so results are directly comparable (see scripts/compare_sam_vs_unet.py).
Two real differences from the U-Net loop:

1. SAMTumorSegDataset only contains tumor-positive images (see its module
   docstring), so every image here has a non-empty GT mask. The
   "overall_segmentation_including_normal" and "tumor_positive_segmentation"
   blocks in MetricAccumulator.summary() are therefore IDENTICAL - there's no
   Normal-image case to average in. Only the tumor-positive number is
   printed/logged, since printing both would misleadingly imply two distinct
   measurements.
2. The optimizer only ever sees `requires_grad=True` params (the mask
   decoder, plus the prompt encoder if `sam.freeze_prompt_encoder=false`) -
   see configs/sam_finetune.yaml.
"""
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import get_device, resolve_path, save_config, set_seed
from src.data.dataset import load_manifest
from src.data.preprocessing import PreprocessConfig
from src.data.sam_dataset import SAMTumorSegDataset
from src.inference.postprocess import remove_small_components
from src.models.sam_finetune import build_sam_model
from src.training.experiment_tracking import close_clearml, init_clearml, log_checkpoint_artifact, log_epoch_metrics
from src.training.losses import DiceBCELoss
from src.training.metrics import MetricAccumulator
from src.visualization.visualize import plot_training_curves, save_epoch_montage


def _make_loaders(cfg: dict, preprocess_cfg: PreprocessConfig, subset_size: Optional[int] = None):
    dataset_root = Path(cfg["dataset_root"])
    manifest_path = resolve_path(cfg, "manifest_csv")
    df = load_manifest(manifest_path)
    sam_cfg = cfg["sam"]

    train_ds = SAMTumorSegDataset(df, dataset_root, "train", preprocess_cfg,
                                   box_padding_px=sam_cfg["box_padding_px"],
                                   box_jitter_px=sam_cfg["box_jitter_px"],
                                   train=True, seed=cfg["seed"])
    valid_ds = SAMTumorSegDataset(df, dataset_root, "valid", preprocess_cfg,
                                   box_padding_px=sam_cfg["box_padding_px"],
                                   box_jitter_px=0, train=False, seed=cfg["seed"])

    if subset_size:
        train_ds.df = train_ds.df.iloc[:subset_size].reset_index(drop=True)
        valid_ds.df = valid_ds.df.iloc[:subset_size].reset_index(drop=True)

    print(f"\nTrain samples (tumor-positive only): {len(train_ds)} | "
          f"Valid samples (tumor-positive only): {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"])
    return train_loader, valid_loader


def _apply_postprocess(preds: np.ndarray, min_component_area_px: int) -> np.ndarray:
    if min_component_area_px <= 0:
        return preds
    for i in range(preds.shape[0]):
        preds[i, 0] = remove_small_components(preds[i, 0], min_component_area_px)
    return preds


def _run_one_epoch(model, loader, device, criterion, threshold: float, min_tumor_area_px: int,
                    original_size: tuple, min_component_area_px: int = 0,
                    optimizer=None, scaler=None, use_amp: bool = False,
                    collect_samples: bool = False) -> Dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = total_dice_loss = total_bce_loss = 0.0
    n_batches = 0
    accumulator = MetricAccumulator(min_tumor_area_px=min_tumor_area_px, compute_hd95=False)
    fixed_samples = []

    for batch in loader:
        images = batch["image"].to(device)
        boxes = batch["box"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train_mode):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, boxes, original_size)
                    loss, dice_l, bce_l = criterion(logits, masks)
            else:
                logits = model(images, boxes, original_size)
                loss, dice_l, bce_l = criterion(logits, masks)

            if train_mode:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        total_dice_loss += dice_l.item()
        total_bce_loss += bce_l.item()
        n_batches += 1

        with torch.no_grad():
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= threshold).astype(np.uint8)
            preds = _apply_postprocess(preds, min_component_area_px)
            gts = masks.detach().cpu().numpy().astype(np.uint8)
            # Downsample the 1024x1024 SAM input back to original_size (the mask
            # resolution) purely for montage display - gt_mask/pred_mask are
            # already at original_size, images isn't.
            display_images = torch.nn.functional.interpolate(
                images, size=original_size, mode="bilinear", align_corners=False,
            ).mean(dim=1).cpu().numpy() / 255.0
            for i in range(images.shape[0]):
                accumulator.update(preds[i, 0], gts[i, 0],
                                    patient_id=batch["patient_id"][i], image_path=batch["image_path"][i])
                if collect_samples and len(fixed_samples) < 4:
                    fixed_samples.append({
                        "image": display_images[i],
                        "gt_mask": gts[i, 0],
                        "pred_mask": preds[i, 0],
                    })

    summary = accumulator.summary()
    metrics = {
        "loss": total_loss / n_batches,
        "dice_loss": total_dice_loss / n_batches,
        "bce_loss": total_bce_loss / n_batches,
        # See module docstring: tumor-positive-only dataset -> these two blocks
        # are identical here. Reported once under the "tumor_positive" name to
        # avoid implying two distinct measurements the way train.py's does.
        "tumor_positive_dice": summary["tumor_positive_segmentation"]["dice"],
        "tumor_positive_iou": summary["tumor_positive_segmentation"]["iou"],
        "precision": summary["tumor_positive_segmentation"]["precision"],
        "recall": summary["tumor_positive_segmentation"]["recall"],
    }
    return metrics, fixed_samples


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
    original_size = (cfg["image_size"], cfg["image_size"])

    model = build_sam_model(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SAM ({cfg['sam']['model_type']}) | total_params={n_total:,} "
          f"trainable_params={n_trainable:,}")

    criterion = DiceBCELoss(dice_weight=cfg["dice_weight"], bce_weight=cfg["bce_weight"],
                             pos_weight=cfg.get("pos_weight")).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
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
    fieldnames = ["epoch", "train_loss", "train_tumor_positive_dice",
                  "val_loss", "val_tumor_positive_dice", "val_tumor_positive_iou",
                  "val_precision", "val_recall", "learning_rate", "epoch_seconds"]
    with open(log_csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"\n=== Starting SAM fine-tune: {epochs} epochs ===\n")

    postprocess_cfg = cfg.get("postprocess", {})
    min_component_area_px = (postprocess_cfg.get("min_component_area_px", 0)
                              if postprocess_cfg.get("remove_small_components", False) else 0)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics, _ = _run_one_epoch(
            model, train_loader, device, criterion, cfg["threshold"], cfg["min_tumor_area_px"],
            original_size, min_component_area_px, optimizer=optimizer, scaler=scaler, use_amp=use_amp,
        )
        val_metrics, fixed_samples = _run_one_epoch(
            model, valid_loader, device, criterion, cfg["threshold"], cfg["min_tumor_area_px"],
            original_size, min_component_area_px, collect_samples=True,
        )
        scheduler.step(val_metrics["tumor_positive_dice"])
        epoch_seconds = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs}  ({epoch_seconds:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f} dice={train_metrics['tumor_positive_dice']:.4f}")
        print(f"  Valid: loss={val_metrics['loss']:.4f} dice={val_metrics['tumor_positive_dice']:.4f} "
              f"iou={val_metrics['tumor_positive_iou']:.4f} "
              f"precision={val_metrics['precision']} recall={val_metrics['recall']}")
        print(f"  LR: {current_lr:.2e}")
        log_epoch_metrics(clearml_task, epoch,
                           {"loss": train_metrics["loss"], "dice": train_metrics["tumor_positive_dice"],
                            "tumor_positive_dice": train_metrics["tumor_positive_dice"]},
                           {"loss": val_metrics["loss"], "dice": val_metrics["tumor_positive_dice"],
                            "iou": val_metrics["tumor_positive_iou"],
                            "tumor_positive_dice": val_metrics["tumor_positive_dice"], "detection": {}},
                           current_lr)

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_metrics["loss"], "train_tumor_positive_dice": train_metrics["tumor_positive_dice"],
                "val_loss": val_metrics["loss"], "val_tumor_positive_dice": val_metrics["tumor_positive_dice"],
                "val_tumor_positive_iou": val_metrics["tumor_positive_iou"],
                "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
                "learning_rate": current_lr, "epoch_seconds": epoch_seconds,
            })

        torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg},
                   checkpoint_dir / "last_model.pth")

        val_dice = val_metrics["tumor_positive_dice"]
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
    close_clearml(clearml_task)

    print(f"\nTraining complete. Best val_dice (tumor-positive)={best_val_dice:.4f}. "
          f"Checkpoints in {checkpoint_dir}, logs in {log_dir}.")
