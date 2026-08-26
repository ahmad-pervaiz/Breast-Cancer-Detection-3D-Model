#!/usr/bin/env python3
"""
Train the 2D tumor segmentation U-Net.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --epochs 30 --batch_size 8
    python scripts/train.py --config configs/config.yaml --smoke_test

--smoke_test runs 2 epochs on a small subset (default 8 samples/split) to
verify the full pipeline (data loading, forward/backward pass, checkpointing)
before committing to a real run. Writes to runs/segmentation_smoke_test/
instead of runs/segmentation/ so it never clobbers a real training run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config
from src.training.train import train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--dataset_root", type=str, default=None,
                    help="Override dataset_root, e.g. /kaggle/input/<dataset-slug> on Kaggle.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true",
                    help="Run a fast 2-epoch, small-subset sanity check instead of full training.")
    p.add_argument("--smoke_test_subset_size", type=int, default=8)
    p.add_argument("--clearml", action="store_true",
                    help="Force-enable ClearML tracking regardless of configs/config.yaml's "
                         "clearml.enabled value. Requires ClearML credentials already configured "
                         "(clearml-init locally, or CLEARML_API_* env vars / Kaggle Secrets remotely).")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "image_size": args.image_size, "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    cfg = load_config(Path(args.config), overrides)
    if args.clearml:
        cfg.setdefault("clearml", {})["enabled"] = True

    if args.smoke_test:
        cfg["checkpoint_dir"] = "runs/segmentation_smoke_test/checkpoints"
        cfg["log_dir"] = "runs/segmentation_smoke_test/logs"
        cfg["plot_dir"] = "runs/segmentation_smoke_test/plots"
        cfg["prediction_dir"] = "runs/segmentation_smoke_test/predictions"
        cfg["batch_size"] = min(cfg["batch_size"], 2)
        cfg["num_workers"] = 0
        cfg["early_stopping"]["enabled"] = False
        print("=== SMOKE TEST MODE: 2 epochs, small subset, separate output dir ===\n")
        train(cfg, smoke_test=True, subset_size=args.smoke_test_subset_size)
    else:
        train(cfg, smoke_test=False)


if __name__ == "__main__":
    main()
