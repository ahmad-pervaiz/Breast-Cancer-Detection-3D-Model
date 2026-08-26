#!/usr/bin/env python3
"""
Dataset audit: integrity checks + summary stats + visual sanity-check montage.

Usage:
    python scripts/audit_dataset.py --config configs/config.yaml

Refuses to proceed silently if integrity checks fail (raises with details).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config, resolve_path
from src.data.dataset import load_manifest
from src.data.preprocessing import PreprocessConfig
from src.data.validation import (
    check_integrity, generate_sanity_montage, print_patient_split, print_summary, raise_if_broken,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--dataset_root", type=str, default=None,
                    help="Override dataset_root, e.g. /kaggle/input/<dataset-slug> on Kaggle.")
    args = p.parse_args()

    cfg = load_config(Path(args.config), {"dataset_root": args.dataset_root})
    dataset_root = Path(cfg["dataset_root"])
    manifest_path = resolve_path(cfg, "manifest_csv")

    print(f"Loading manifest: {manifest_path}")
    df = load_manifest(manifest_path)
    dev_df = df[df["split"].isin(["train", "valid"])]  # test has no masks - excluded from mask checks

    print("\n" + "=" * 70)
    print("INTEGRITY CHECK (train + valid only - test has no ground-truth masks)")
    print("=" * 70)
    errors = check_integrity(dev_df, dataset_root)
    total_errors = sum(len(v) for v in errors.values())
    for category, msgs in errors.items():
        print(f"  {category}: {len(msgs)}")
    raise_if_broken(errors)
    print("\nAll integrity checks passed.")

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print_summary(dev_df)

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL SPLIT")
    print("=" * 70)
    print_patient_split(dev_df)

    print("\n" + "=" * 70)
    print("SANITY-CHECK MONTAGE")
    print("=" * 70)
    preprocess_cfg = PreprocessConfig(image_size=cfg["image_size"])
    montage_path = resolve_path(cfg, "log_dir").parent / "sanity_montage.png"
    generate_sanity_montage(dev_df, dataset_root, montage_path, preprocess_cfg, seed=cfg["seed"])

    print("\nAudit complete - dataset is ready for training.")


if __name__ == "__main__":
    main()
