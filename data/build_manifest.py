"""
Build a single manifest CSV for the whole breast-tumor dataset and run an
integrity/quality audit while doing it.

Covers: Train/Train_masks, valid/valid_masks, test (no masks/json).
Skips: TUMOR_Anonymized_DCM (unrelated abdominal CT, excluded by design).

Output:
  manifest.csv        - one row per image, full metadata
  audit_report.txt     - human-readable summary of issues found

Paths in manifest.csv are stored RELATIVE to DATASET_ROOT (e.g. "Train/Normal/x.png"),
not absolute. To use the manifest anywhere else (Kaggle, Colab, another machine),
just join these relative paths against that environment's dataset root, e.g.:
  root = Path("/kaggle/input/<dataset-slug>")   # Kaggle
  full_path = root / row["image_path"]
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("/run/media/ahmad-pervaiz/New Volume/BC_detection_Project/FINAL DATASET")
OUT_DIR = Path("/run/media/ahmad-pervaiz/New Volume/BC_detection_Project/bc_tumor_detection/data")
MANIFEST_CSV = OUT_DIR / "manifest.csv"
REPORT_TXT = OUT_DIR / "audit_report.txt"

SPLITS = {
    "train": ("Train", "Train_masks"),
    "valid": ("valid", "valid_masks"),
    "test":  ("test", None),
}

issues = []  # human-readable strings for the audit report


def log_issue(msg):
    issues.append(msg)


def safe_open_image(path):
    """Try to fully load an image; return (mode, size) or None if corrupt."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            mode, size = im.mode, im.size
            im.load()
        return mode, size
    except Exception as e:
        log_issue(f"CORRUPT image: {path} ({e})")
        return None


def mask_stats(mask_path):
    """Return (is_binary, positive_pixel_ratio) or (None, None) if unreadable."""
    try:
        arr = np.array(Image.open(mask_path).convert("L"))
        uniq = np.unique(arr)
        is_binary = set(uniq.tolist()).issubset({0, 255})
        pos_ratio = float((arr > 0).mean())
        return is_binary, pos_ratio
    except Exception as e:
        log_issue(f"CORRUPT mask: {mask_path} ({e})")
        return None, None


def json_stats(json_path):
    """Return (num_shapes, all_labels) or (None, None) if unreadable."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        labels = sorted({s.get("label", "?") for s in shapes})
        return len(shapes), ",".join(labels)
    except Exception as e:
        log_issue(f"CORRUPT json: {json_path} ({e})")
        return None, None


def infer_patient_and_label(split_name, subfolder_name):
    """Folder-name convention: 'Normal' => Normal, 'Tumors' => Tumor (generic),
    anything else (a named patient folder) => Tumor, patient_id = folder name."""
    if subfolder_name.lower() == "normal":
        return subfolder_name, "Normal"
    else:
        return subfolder_name, "Tumor"


def build():
    rows = []
    seen_stems = {}  # stem -> list of (split, path) to catch cross-folder collisions

    for split, (img_dirname, mask_dirname) in SPLITS.items():
        img_root = DATASET_ROOT / img_dirname
        mask_root = DATASET_ROOT / mask_dirname if mask_dirname else None

        if not img_root.exists():
            log_issue(f"MISSING split dir: {img_root}")
            continue

        for subfolder in sorted(p for p in img_root.iterdir() if p.is_dir()):
            patient_id, label_from_folder = infer_patient_and_label(split, subfolder.name)

            for png_path in sorted(subfolder.glob("*.png")):
                stem = png_path.stem
                seen_stems.setdefault(stem, []).append((split, str(png_path)))

                opened = safe_open_image(png_path)
                mode, size = opened if opened else (None, (None, None))
                width, height = size if opened else (None, None)

                json_path = png_path.with_suffix(".json")
                has_json = json_path.exists()
                num_shapes, shape_labels = (json_stats(json_path) if has_json else (None, None))

                mask_path = None
                has_mask = False
                mask_binary = None
                mask_pos_ratio = None
                if mask_root is not None:
                    candidate = mask_root / subfolder.name / f"{stem}_mask.png"
                    if candidate.exists():
                        mask_path = str(candidate.relative_to(DATASET_ROOT))
                        has_mask = True
                        mask_binary, mask_pos_ratio = mask_stats(candidate)
                    else:
                        log_issue(f"MISSING mask for: {png_path}")

                # Ground-truth label: prefer mask/json evidence over folder name
                if has_mask and mask_pos_ratio is not None:
                    final_label = "Tumor" if mask_pos_ratio > 0 else "Normal"
                elif has_json and num_shapes is not None:
                    final_label = "Tumor" if num_shapes > 0 else "Normal"
                else:
                    final_label = label_from_folder

                if label_from_folder != final_label:
                    log_issue(
                        f"LABEL MISMATCH: {png_path} folder says {label_from_folder}, "
                        f"evidence says {final_label} (mask_pos_ratio={mask_pos_ratio}, num_shapes={num_shapes})"
                    )

                rows.append({
                    "image_path": str(png_path.relative_to(DATASET_ROOT)),
                    "mask_path": mask_path,
                    "json_path": str(json_path.relative_to(DATASET_ROOT)) if has_json else None,
                    "split": split,
                    "patient_id": patient_id,
                    "subfolder": subfolder.name,
                    "label": final_label,
                    "has_mask": has_mask,
                    "has_json": has_json,
                    "width": width,
                    "height": height,
                    "channel_mode": mode,
                    "num_shapes": num_shapes,
                    "shape_labels": shape_labels,
                    "mask_binary_valid": mask_binary,
                    "mask_positive_pixel_ratio": mask_pos_ratio,
                    "is_corrupt": opened is None,
                })

    # cross-folder stem collisions (same filename stem appearing in >1 place)
    for stem, locs in seen_stems.items():
        if len(locs) > 1:
            log_issue(f"DUPLICATE stem '{stem}' appears in: {locs}")

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_CSV, index=False)

    # ---- summary ----
    lines = []
    lines.append(f"Total images in manifest: {len(df)}")
    lines.append("\nPer split / label counts:")
    lines.append(str(df.groupby(["split", "label"]).size()))
    lines.append("\nPer split / patient counts:")
    lines.append(str(df.groupby(["split", "patient_id"]).size()))
    lines.append("\nChannel mode distribution:")
    lines.append(str(df["channel_mode"].value_counts(dropna=False)))
    lines.append("\nImage size distribution (width x height):")
    lines.append(str(df.groupby(["width", "height"]).size()))
    lines.append(f"\nCorrupt images: {int(df['is_corrupt'].sum())}")
    lines.append(f"Missing masks (train/valid only): {sum(1 for i in issues if i.startswith('MISSING mask'))}")
    lines.append(f"Label mismatches (folder vs mask/json evidence): {sum(1 for i in issues if i.startswith('LABEL MISMATCH'))}")
    lines.append(f"Non-binary masks: {int((df['mask_binary_valid'] == False).sum())}")
    lines.append(f"Duplicate stems across folders: {sum(1 for i in issues if i.startswith('DUPLICATE'))}")

    # patient leakage check across splits
    patient_by_split = df.groupby("patient_id")["split"].unique()
    leaks = patient_by_split[patient_by_split.apply(lambda s: len(s) > 1)]
    lines.append(f"\nPatients appearing in more than one split: {len(leaks)}")
    if len(leaks):
        lines.append(str(leaks))

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(REPORT_TXT, "w") as f:
        f.write(summary_text)
        f.write("\n\n---- DETAILED ISSUES ----\n")
        f.write("\n".join(issues) if issues else "None found.")

    print(f"\nWrote manifest: {MANIFEST_CSV} ({len(df)} rows)")
    print(f"Wrote report:   {REPORT_TXT} ({len(issues)} issues logged)")


if __name__ == "__main__":
    build()
