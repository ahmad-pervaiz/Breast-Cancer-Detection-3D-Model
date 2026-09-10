#!/usr/bin/env python
"""Phase-2 Milestone 1: Dataset / DICOM mapping audit.

Project_phase2.txt Section 13 (mask/JSON audit) + Section 39 (DICOM series
report) + Sections 56-57 (the mandatory "first deliverable" inspection that
must happen before any 3D reconstruction code is trusted).

For every configured patient (`configs/phase2_3d.yaml`):
  - matches every PNG <-> JSON <-> DICOM by filename stem (see src/future_3d/mapping.py)
  - rasterizes every JSON into a mask and validates it against the PNG dimensions
  - reports empty/non-empty masks, unknown labels, dimension mismatches
  - discovers which DICOM series the patient's annotated slices actually belong to

Usage:
    python scripts/phase2_audit_dataset.py --config configs/phase2_3d.yaml

Outputs (under <output_root>/audit/):
    audit_report.json   - full structured report (global + per-patient)
    audit_report.csv     - one row per patient, spreadsheet-friendly summary
    slice_mapping.csv    - one row per PNG (every patient), same schema as
                            scripts/phase2_map_slices.py's output (kept in
                            sync here so the audit is fully self-contained)
"""
import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.future_3d.config import PROJECT_ROOT, load_phase2_config
from src.future_3d.labelme_io import labelme_to_mask, load_labelme_json, mask_stats
from src.future_3d.mapping import map_patient_slices, records_to_dicts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_audit")


def audit_patient(patient_code: str, image_dir: Path, dicom_dir: Path, tumor_labels):
    # NOTE: never log `image_dir` itself - it contains the real dataset folder
    # name (e.g. "FINAL DATASET/Train/P1-..."). Only `patient_code` is safe to
    # log/report (see configs/phase2_3d.yaml's PRIVACY NOTE).
    logger.info("Auditing %s", patient_code)

    if not image_dir.exists():
        logger.error("[%s] image_dir does not exist", patient_code)
        return None

    mapping_records = map_patient_slices(patient_code, image_dir, dicom_dir)
    confidence_counts = Counter(r.mapping_confidence for r in mapping_records)

    png_files = {p.name for p in image_dir.iterdir() if p.suffix.lower() == ".png"}
    json_files = {p.name for p in image_dir.iterdir() if p.suffix.lower() == ".json"}
    png_stems = {Path(f).stem for f in png_files}
    json_stems = {Path(f).stem for f in json_files}

    missing_json = sorted(png_stems - json_stems)
    missing_png = sorted(json_stems - png_stems)
    matched_stems = sorted(png_stems & json_stems)

    invalid_json, empty_masks, non_empty_masks = [], [], []
    unknown_labels = Counter()
    dimension_mismatches = []
    total_tumor_px = 0

    for stem in matched_stems:
        json_path = image_dir / f"{stem}.json"
        png_path = image_dir / f"{stem}.png"
        try:
            load_labelme_json(json_path)
        except Exception as exc:
            invalid_json.append({"file": json_path.name, "error": str(exc)})
            continue

        try:
            from PIL import Image
            with Image.open(png_path) as im:
                width, height = im.size
        except Exception as exc:
            dimension_mismatches.append({"file": png_path.name, "error": f"unreadable PNG: {exc}"})
            continue

        result = labelme_to_mask(json_path, height, width, tumor_labels)
        for lbl in result.unknown_labels:
            unknown_labels[lbl] += 1

        count, _ = mask_stats(result.mask)
        total_tumor_px += count
        if count > 0:
            non_empty_masks.append(stem)
        else:
            empty_masks.append(stem)

    series_slices = defaultdict(list)
    for r in mapping_records:
        if r.series_instance_uid:
            series_slices[r.series_instance_uid].append(r)

    series_report = []
    for series_uid, recs in series_slices.items():
        series_report.append({
            "series_instance_uid": series_uid,
            "num_slices": len(recs),
            "study_instance_uid": None,  # filled from inventory in the caller if needed
        })
    series_report.sort(key=lambda s: -s["num_slices"])

    report = {
        "patient_code": patient_code,
        "total_png": len(png_files),
        "total_json": len(json_files),
        "json_png_matches": len(matched_stems),
        "missing_json": missing_json,
        "missing_png": missing_png,
        "invalid_json": invalid_json,
        "empty_masks": len(empty_masks),
        "non_empty_masks": len(non_empty_masks),
        "unknown_labels": dict(unknown_labels),
        "dimension_mismatches": dimension_mismatches,
        "total_tumor_pixels": total_tumor_px,
        "mapping_confidence_counts": dict(confidence_counts),
        "num_dicom_series_used": len(series_slices),
        "dicom_series": series_report,
        "mapping_records": records_to_dicts(mapping_records),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    args = parser.parse_args()

    cfg = load_phase2_config(args.config)
    output_dir = cfg.output_root / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.dicom_inventory_csv.exists():
        logger.error("dicom_inventory.csv not found at %s", cfg.dicom_inventory_csv)
        sys.exit(1)
    if not cfg.dicom_dir.exists():
        logger.error("DICOM directory not found at %s", cfg.dicom_dir)
        sys.exit(1)

    patient_reports = []
    for patient in cfg.patients:
        report = audit_patient(patient.code, patient.image_dir, cfg.dicom_dir, cfg.tumor_labels)
        if report is not None:
            report["split"] = patient.split
            patient_reports.append(report)

    global_report = {
        "config_used": str(args.config),
        "num_patients": len(patient_reports),
        "totals": {
            "total_png": sum(r["total_png"] for r in patient_reports),
            "total_json": sum(r["total_json"] for r in patient_reports),
            "json_png_matches": sum(r["json_png_matches"] for r in patient_reports),
            "empty_masks": sum(r["empty_masks"] for r in patient_reports),
            "non_empty_masks": sum(r["non_empty_masks"] for r in patient_reports),
            "total_tumor_pixels": sum(r["total_tumor_pixels"] for r in patient_reports),
        },
        "mapping_confidence_totals": dict(
            sum((Counter(r["mapping_confidence_counts"]) for r in patient_reports), Counter())
        ),
        "patients": patient_reports,
    }

    json_path = output_dir / "audit_report.json"
    with open(json_path, "w") as f:
        json.dump(global_report, f, indent=2)
    logger.info("Wrote %s", json_path)

    csv_path = output_dir / "audit_report.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "patient_code", "split", "total_png", "total_json", "json_png_matches",
            "missing_json_count", "missing_png_count", "invalid_json_count",
            "empty_masks", "non_empty_masks", "total_tumor_pixels",
            "num_dicom_series_used", "mapping_VERIFIED", "mapping_LIKELY",
            "mapping_UNCERTAIN", "mapping_FAILED", "unknown_labels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in patient_reports:
            mc = r["mapping_confidence_counts"]
            writer.writerow({
                "patient_code": r["patient_code"],
                "split": r["split"],
                "total_png": r["total_png"],
                "total_json": r["total_json"],
                "json_png_matches": r["json_png_matches"],
                "missing_json_count": len(r["missing_json"]),
                "missing_png_count": len(r["missing_png"]),
                "invalid_json_count": len(r["invalid_json"]),
                "empty_masks": r["empty_masks"],
                "non_empty_masks": r["non_empty_masks"],
                "total_tumor_pixels": r["total_tumor_pixels"],
                "num_dicom_series_used": r["num_dicom_series_used"],
                "mapping_VERIFIED": mc.get("VERIFIED", 0),
                "mapping_LIKELY": mc.get("LIKELY", 0),
                "mapping_UNCERTAIN": mc.get("UNCERTAIN", 0),
                "mapping_FAILED": mc.get("FAILED", 0),
                "unknown_labels": ";".join(r["unknown_labels"].keys()),
            })
    logger.info("Wrote %s", csv_path)

    mapping_csv_path = cfg.output_root / "mapping" / "slice_mapping.csv"
    mapping_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_csv_path, "w", newline="") as f:
        fieldnames = [
            "patient_code", "png_file", "json_file", "dicom_file", "series_instance_uid",
            "sop_instance_uid", "instance_number", "z_position", "png_width", "png_height",
            "dicom_width", "dicom_height", "mapping_method", "mapping_confidence", "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in patient_reports:
            for rec in r["mapping_records"]:
                writer.writerow(rec)
    logger.info("Wrote %s", mapping_csv_path)

    logger.info("=== SUMMARY ===")
    for r in patient_reports:
        mc = r["mapping_confidence_counts"]
        logger.info(
            "%-16s png=%-4d json=%-4d matches=%-4d empty=%-4d nonempty=%-4d series=%-2d VERIFIED=%-4d LIKELY=%-3d UNCERTAIN=%-3d FAILED=%-3d",
            r["patient_code"], r["total_png"], r["total_json"], r["json_png_matches"],
            r["empty_masks"], r["non_empty_masks"], r["num_dicom_series_used"],
            mc.get("VERIFIED", 0), mc.get("LIKELY", 0), mc.get("UNCERTAIN", 0), mc.get("FAILED", 0),
        )


if __name__ == "__main__":
    main()
