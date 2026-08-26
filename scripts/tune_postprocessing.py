#!/usr/bin/env python3
"""
Tier 1 model-improvement step (see improving_model.md): sweep classification
threshold and connected-component area filter on the VALIDATION set only,
against an already-trained checkpoint - no retraining, no GPU required.

Runs the model once per validation image (probability maps cached in memory),
then sweeps threshold x min_component_area_px combinations cheaply against
that cache. Picks the combination that maximizes val_dice (the same
model-selection metric used during training) and, if it beats the
threshold=0.5/no-postprocessing baseline, writes it into configs/config.yaml.
Never touches the test set - this is exactly the "tune on validation only"
rule from the project spec.

Usage:
    python scripts/tune_postprocessing.py --config configs/config.yaml \\
        --checkpoint runs/segmentation/checkpoints/best_model.pth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.common import get_device, load_config, resolve_path
from src.data.dataset import load_manifest
from src.data.preprocessing import PreprocessConfig, preprocess_image, preprocess_mask, empty_mask
from src.inference.postprocess import remove_small_components
from src.inference.predict import load_checkpoint
from src.training.metrics import MetricAccumulator

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_COMPONENT_AREAS = [0, 10, 20, 30, 50, 75, 100, 150, 200]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default="runs/segmentation/checkpoints/best_model.pth")
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--apply", action="store_true", default=True,
                    help="Write the winning combo into configs/config.yaml if it beats baseline (default on).")
    p.add_argument("--no-apply", dest="apply", action="store_false")
    return p.parse_args()


@torch.no_grad()
def cache_probability_maps(model, df, dataset_root: Path, preprocess_cfg: PreprocessConfig, device):
    """One forward pass per validation image. Returns list of (prob_map, gt_mask, is_positive)."""
    cached = []
    for _, row in df.iterrows():
        image = preprocess_image(dataset_root / row["image_path"], preprocess_cfg)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        if row["has_mask"] and isinstance(row["mask_path"], str):
            gt = preprocess_mask(dataset_root / row["mask_path"], preprocess_cfg)
        else:
            gt = empty_mask(preprocess_cfg.image_size)

        cached.append((prob, gt.astype(np.uint8)))
    return cached


def evaluate_combo(cached, threshold: float, min_area_px: int) -> dict:
    acc = MetricAccumulator(min_tumor_area_px=20, compute_hd95=False)
    for prob, gt in cached:
        pred = (prob >= threshold).astype(np.uint8)
        if min_area_px > 0:
            pred = remove_small_components(pred, min_area_px)
        acc.update(pred, gt)
    summary = acc.summary()
    return {
        "threshold": threshold,
        "min_component_area_px": min_area_px,
        "val_dice": summary["overall_segmentation_including_normal"]["dice"],
        "val_iou": summary["overall_segmentation_including_normal"]["iou"],
        "tumor_positive_dice": summary["tumor_positive_segmentation"]["dice"],
        "sensitivity": summary["image_level_detection"]["sensitivity_recall"],
        "specificity": summary["image_level_detection"]["specificity"],
    }


def main():
    args = parse_args()
    device = get_device()
    model, train_cfg = load_checkpoint(Path(args.checkpoint), device)

    cfg = load_config(Path(args.config), {"dataset_root": args.dataset_root})
    dataset_root = Path(cfg["dataset_root"])
    df = load_manifest(resolve_path(cfg, "manifest_csv"))
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    print(f"Validation set: {len(valid_df)} images (never touching test/)")

    preprocess_cfg = PreprocessConfig(image_size=train_cfg["image_size"])
    print("Running model once per validation image (caching probability maps)...")
    cached = cache_probability_maps(model, valid_df, dataset_root, preprocess_cfg, device)

    baseline = evaluate_combo(cached, threshold=0.5, min_area_px=0)
    print(f"\nBaseline (threshold=0.5, no post-processing): val_dice={baseline['val_dice']:.4f}")

    print(f"\nSweeping {len(THRESHOLDS)} thresholds x {len(MIN_COMPONENT_AREAS)} component-area "
          f"cutoffs = {len(THRESHOLDS) * len(MIN_COMPONENT_AREAS)} combinations...")
    results = []
    for t in THRESHOLDS:
        for a in MIN_COMPONENT_AREAS:
            results.append(evaluate_combo(cached, t, a))

    results.sort(key=lambda r: r["val_dice"], reverse=True)

    out_csv = resolve_path(cfg, "log_dir").parent / "postprocessing_tuning.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as csv_module
    with open(out_csv, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull sweep results saved to: {out_csv}")

    print("\nTop 5 combinations by val_dice:")
    for r in results[:5]:
        print(f"  threshold={r['threshold']:.2f} min_area={r['min_component_area_px']:>4}px  "
              f"val_dice={r['val_dice']:.4f}  tumor+dice={r['tumor_positive_dice']:.4f}  "
              f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f}")

    best = results[0]
    improvement = best["val_dice"] - baseline["val_dice"]
    print(f"\nBest combo: threshold={best['threshold']}, min_component_area_px={best['min_component_area_px']}")
    print(f"val_dice: {baseline['val_dice']:.4f} -> {best['val_dice']:.4f}  (delta={improvement:+.4f})")

    if improvement <= 0.0005:
        print("\nNo meaningful improvement over baseline - leaving configs/config.yaml unchanged.")
        return

    if args.apply:
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        full_cfg["threshold"] = float(best["threshold"])
        full_cfg["postprocess"]["remove_small_components"] = bool(best["min_component_area_px"] > 0)
        full_cfg["postprocess"]["min_component_area_px"] = int(best["min_component_area_px"])
        with open(args.config, "w") as f:
            yaml.safe_dump(full_cfg, f, sort_keys=False)
        print(f"\nApplied to {args.config}: threshold={best['threshold']}, "
              f"postprocess.remove_small_components={full_cfg['postprocess']['remove_small_components']}, "
              f"min_component_area_px={best['min_component_area_px']}")


if __name__ == "__main__":
    main()
