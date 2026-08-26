#!/usr/bin/env python3
"""
Standalone visual sanity-check tool (no model needed) - re-generate the
image/mask/overlay montage on demand, e.g. after any manifest change.

Usage:
    python scripts/visualize_masks.py --config configs/config.yaml --out my_montage.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config, resolve_path
from src.data.dataset import load_manifest
from src.data.preprocessing import PreprocessConfig
from src.data.validation import generate_sanity_montage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--dataset_root", type=str, default=None,
                    help="Override dataset_root, e.g. /kaggle/input/<dataset-slug> on Kaggle.")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--samples_per_group", type=int, default=2)
    args = p.parse_args()

    cfg = load_config(Path(args.config), {"dataset_root": args.dataset_root})
    dataset_root = Path(cfg["dataset_root"])
    df = load_manifest(resolve_path(cfg, "manifest_csv"))
    dev_df = df[df["split"].isin(["train", "valid"])]

    preprocess_cfg = PreprocessConfig(image_size=cfg["image_size"])
    out_path = Path(args.out) if args.out else resolve_path(cfg, "log_dir").parent / "sanity_montage.png"
    generate_sanity_montage(dev_df, dataset_root, out_path, preprocess_cfg,
                             samples_per_group=args.samples_per_group, seed=cfg["seed"])


if __name__ == "__main__":
    main()
