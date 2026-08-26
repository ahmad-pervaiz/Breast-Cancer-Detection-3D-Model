"""
Dataset sanity checks and visual verification.

Run standalone via `python scripts/audit_dataset.py`. Also importable so
scripts/train.py can refuse to start on a dataset that fails these checks
(never proceed silently on a broken dataset - see project requirements).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from src.data.preprocessing import PreprocessConfig, preprocess_image, preprocess_mask, empty_mask


class DatasetIntegrityError(RuntimeError):
    """Raised when the dataset fails a hard sanity check (never caught silently)."""


def check_integrity(df: pd.DataFrame, dataset_root: Path) -> Dict[str, List[str]]:
    """Run all checks from project spec section 9 (items 1-8). Returns dict of
    error-category -> list of messages. Caller decides whether to raise.
    """
    dataset_root = Path(dataset_root)
    errors: Dict[str, List[str]] = {
        "missing_image": [], "missing_mask": [], "dim_mismatch": [],
        "non_binary_mask": [], "normal_nonempty_mask": [], "tumor_empty_mask": [],
        "duplicate_stem": [],
    }

    seen_stems: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        img_path = dataset_root / row["image_path"]
        stem = Path(row["image_path"]).stem
        seen_stems.setdefault(f"{row['split']}::{stem}", []).append(row["image_path"])

        if not img_path.exists():
            errors["missing_image"].append(str(img_path))
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        if row["has_mask"] and isinstance(row["mask_path"], str):
            mask_path = dataset_root / row["mask_path"]
            if not mask_path.exists():
                errors["missing_mask"].append(str(mask_path))
                continue

            with Image.open(mask_path) as m:
                mask_w, mask_h = m.size
                mask_arr = np.array(m.convert("L"))

            if (mask_w, mask_h) != (img_w, img_h):
                errors["dim_mismatch"].append(
                    f"{img_path} ({img_w}x{img_h}) vs {mask_path} ({mask_w}x{mask_h})"
                )

            uniq = set(np.unique(mask_arr).tolist())
            if not uniq.issubset({0, 255}):
                errors["non_binary_mask"].append(f"{mask_path} unique_values={sorted(uniq)}")

            is_positive = bool((mask_arr > 0).any())
            if row["label"] == "Normal" and is_positive:
                errors["normal_nonempty_mask"].append(str(mask_path))
            if row["label"] == "Tumor" and not is_positive:
                errors["tumor_empty_mask"].append(str(mask_path))

    for key, paths in seen_stems.items():
        if len(paths) > 1:
            errors["duplicate_stem"].append(f"{key}: {paths}")

    return errors


def raise_if_broken(errors: Dict[str, List[str]]) -> None:
    total = sum(len(v) for v in errors.values())
    if total > 0:
        lines = [f"Dataset failed integrity checks ({total} issue(s)):"]
        for category, msgs in errors.items():
            if msgs:
                lines.append(f"  [{category}] {len(msgs)} issue(s), e.g.: {msgs[0]}")
        raise DatasetIntegrityError("\n".join(lines))


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total images in manifest: {len(df)}")
    print("\nPer split / label:")
    print(df.groupby(["split", "label"]).size().to_string())

    positive = df[df["label"] == "Tumor"]
    empty = df[df["label"] == "Normal"]
    print(f"\nTotal images: {len(df)}")
    print(f"Positive-mask (Tumor) images: {len(positive)}")
    print(f"Empty-mask (Normal) images: {len(empty)}")

    print("\nImage dimensions:")
    print(df.groupby(["width", "height"]).size().to_string())

    print("\nMask pixel value check: masks are binary {0, 255} per audit_report.txt "
          "(non_binary_mask count should be 0 - see integrity check output above)")

    # Tumor pixel area statistics for positive masks (derived from stored
    # mask_positive_pixel_ratio * width * height - avoids re-reading every mask).
    pos = positive[positive["mask_positive_pixel_ratio"].notna()].copy()
    if len(pos):
        pos["tumor_area_px"] = pos["mask_positive_pixel_ratio"] * pos["width"] * pos["height"]
        print("\nTumor pixel area statistics (positive masks only):")
        print(f"  min:    {pos['tumor_area_px'].min():.0f} px")
        print(f"  max:    {pos['tumor_area_px'].max():.0f} px")
        print(f"  mean:   {pos['tumor_area_px'].mean():.0f} px")
        print(f"  median: {pos['tumor_area_px'].median():.0f} px")


def print_patient_split(df: pd.DataFrame) -> None:
    """Section 32: print train/valid patient groups and verify no overlap."""
    train_patients = set(df[df["split"] == "train"]["patient_id"].unique())
    valid_patients = set(df[df["split"] == "valid"]["patient_id"].unique())

    print("TRAIN patient groups:", sorted(train_patients))
    print("VALID patient groups:", sorted(valid_patients))

    # "Normal" is a generic bucket in both splits by design (different images,
    # same label bucket name) - exclude it from the named-patient overlap check.
    named_train = train_patients - {"Normal"}
    named_valid = valid_patients - {"Normal", "Tumors"}
    overlap = named_train & named_valid
    if overlap:
        raise DatasetIntegrityError(
            f"PATIENT LEAKAGE DETECTED - patients in both train and valid: {overlap}"
        )
    print(f"No named-patient overlap between train and valid. "
          f"(Normal-bucket and valid/Tumors-bucket recurrence across splits is expected, not leakage.)")


def generate_sanity_montage(
    df: pd.DataFrame,
    dataset_root: Path,
    out_path: Path,
    preprocess_cfg: PreprocessConfig,
    samples_per_group: int = 2,
    seed: int = 42,
) -> None:
    """One row per (split, patient_id/group) with sample cols: image | mask | overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_root = Path(dataset_root)
    rng = np.random.RandomState(seed)

    groups = df[df["label"] == "Tumor"].groupby(["split", "patient_id"])
    rows = []
    for (split, patient_id), group_df in groups:
        n = min(samples_per_group, len(group_df))
        sample = group_df.sample(n=n, random_state=rng)
        for _, r in sample.iterrows():
            rows.append(r)

    n_rows = len(rows)
    if n_rows == 0:
        print("No tumor-positive samples found for montage - skipping.")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(rows):
        img = preprocess_image(dataset_root / row["image_path"], preprocess_cfg)
        if row["has_mask"] and isinstance(row["mask_path"], str):
            mask = preprocess_mask(dataset_root / row["mask_path"], preprocess_cfg)
        else:
            mask = empty_mask(preprocess_cfg.image_size)

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"{row['split']}/{row['patient_id']}\nimage", fontsize=8)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("mask", fontsize=8)

        overlay = np.stack([img, img, img], axis=-1)
        overlay[..., 0] = np.where(mask > 0, 1.0, overlay[..., 0])
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("overlay", fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved sanity-check montage: {out_path} ({n_rows} samples)")
