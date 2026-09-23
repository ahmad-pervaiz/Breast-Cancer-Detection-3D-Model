#!/usr/bin/env python3
"""
Fine-tune SAM's mask decoder (box-prompted) for tumor segmentation.

Usage:
    python scripts/train_sam.py --config configs/sam_finetune.yaml
    python scripts/train_sam.py --config configs/sam_finetune.yaml --smoke_test
    python scripts/train_sam.py --config configs/sam_finetune.yaml \
        --dataset_root /kaggle/input/<dataset-slug> --sam_checkpoint /kaggle/input/<ckpt-slug>/medsam_vit_b.pth

Requires the `segment-anything` package (pip install segment-anything) and a
manually-downloaded SAM/MedSAM checkpoint - see configs/sam_finetune.yaml's
`sam.checkpoint` comment.

--smoke_test runs 2 epochs on a small subset (default 4 tumor-positive
samples/split) to verify the full pipeline before committing to a real
(Kaggle-GPU) run - same convention as scripts/train.py. Writes to
runs/sam_segmentation_smoke_test/ instead of runs/sam_segmentation/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config
from src.training.train_sam import train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/sam_finetune.yaml")
    p.add_argument("--dataset_root", type=str, default=None,
                    help="Override dataset_root, e.g. /kaggle/input/<dataset-slug> on Kaggle.")
    p.add_argument("--sam_checkpoint", type=str, default=None,
                    help="Override sam.checkpoint, e.g. a Kaggle input path for the base MedSAM/SAM weights.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true",
                    help="Run a fast 2-epoch, small-subset sanity check instead of full training.")
    p.add_argument("--smoke_test_subset_size", type=int, default=4)
    p.add_argument("--clearml", action="store_true",
                    help="Force-enable ClearML tracking regardless of configs/sam_finetune.yaml's "
                         "clearml.enabled value.")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed,
    }
    cfg = load_config(Path(args.config), overrides)
    if args.sam_checkpoint:
        cfg["sam"]["checkpoint"] = args.sam_checkpoint
    if args.clearml:
        cfg.setdefault("clearml", {})["enabled"] = True

    if args.smoke_test:
        cfg["checkpoint_dir"] = "runs/sam_segmentation_smoke_test/checkpoints"
        cfg["log_dir"] = "runs/sam_segmentation_smoke_test/logs"
        cfg["plot_dir"] = "runs/sam_segmentation_smoke_test/plots"
        cfg["prediction_dir"] = "runs/sam_segmentation_smoke_test/predictions"
        cfg["batch_size"] = min(cfg["batch_size"], 2)
        cfg["num_workers"] = 0
        cfg["early_stopping"]["enabled"] = False
        print("=== SMOKE TEST MODE: 2 epochs, small subset, separate output dir ===\n")
        train(cfg, smoke_test=True, subset_size=args.smoke_test_subset_size)
    else:
        train(cfg, smoke_test=False)


if __name__ == "__main__":
    main()
