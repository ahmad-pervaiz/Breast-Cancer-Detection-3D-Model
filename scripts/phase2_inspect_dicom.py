#!/usr/bin/env python
"""Phase-2 Milestone 2: inspect one patient's DICOM series.

Project_phase2.txt Sections 9, 10, 39, 40. Reports, per DICOM series the
patient's annotated PNGs actually map into:
    number of slices, ordering method used, slice spacing (+ irregularities),
    dimensions, pixel spacing, orientation, rescale slope/intercept.

Does not build a volume yet (Milestone 5) - this is inspection/reporting only,
so a series with irregular spacing or an ordering problem is surfaced before
any volume-building code would silently paper over it.

Usage:
    python scripts/phase2_inspect_dicom.py --config configs/phase2_3d.yaml --patient Patient_05

`--patient` takes the anonymized code (see configs/phase2_3d.yaml), not the
real dataset folder name.
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.future_3d.config import load_phase2_config
from src.future_3d.dicom_io import order_slices, read_dicom_header
from src.future_3d.mapping import map_patient_slices

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_inspect_dicom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", required=True)
    args = parser.parse_args()

    cfg = load_phase2_config(args.config)
    patient = next((p for p in cfg.patients if p.code == args.patient), None)
    if patient is None:
        known = ", ".join(p.code for p in cfg.patients)
        logger.error("Unknown patient code %r. Known codes: %s", args.patient, known)
        sys.exit(1)

    logger.info("Loading %s", patient.code)
    records = map_patient_slices(patient.code, patient.image_dir, cfg.dicom_dir)
    by_series = defaultdict(list)
    for r in records:
        if r.dicom_file and r.series_instance_uid:
            by_series[r.series_instance_uid].append(cfg.dicom_dir / r.dicom_file)

    logger.info("Found %d PNG files, %d DICOM series", len(records), len(by_series))
    if len(by_series) > 1:
        logger.warning(
            "%s spans %d distinct DICOM series - NOT a single contiguous acquisition. "
            "Each series must be reconstructed as its own volume; do not merge across series.",
            patient.code, len(by_series),
        )

    report = {"patient_code": patient.code, "num_series": len(by_series), "series": []}

    for series_uid, dcm_paths in by_series.items():
        headers = [read_dicom_header(p) for p in dcm_paths]
        ordered, warnings = order_slices(headers)

        study_uids = {h.study_instance_uid for h in headers}
        rows_set = {h.rows for h in headers}
        cols_set = {h.columns for h in headers}
        spacing_set = {h.pixel_spacing for h in headers}
        thickness_set = {h.slice_thickness for h in headers}
        orientation_set = {h.image_orientation_patient for h in headers}
        slope_set = {h.rescale_slope for h in headers}
        intercept_set = {h.rescale_intercept for h in headers}

        zs = [s.z_projected for s in ordered]
        z_spacing = round(zs[1] - zs[0], 4) if len(zs) > 1 else None
        origin = ordered[0].image_position_patient if ordered[0].image_position_patient else None

        series_report = {
            "series_instance_uid": series_uid,
            "study_instance_uid": list(study_uids),
            "num_slices": len(ordered),
            "rows": list(rows_set),
            "columns": list(cols_set),
            "pixel_spacing_mm": [list(s) for s in spacing_set if s],
            "slice_thickness_mm": list(thickness_set),
            "nominal_z_spacing_mm": z_spacing,
            "image_orientation_patient": [list(o) for o in orientation_set if o],
            "origin_first_slice": list(origin) if origin else None,
            "rescale_slope": list(slope_set),
            "rescale_intercept": list(intercept_set),
            "ordering_warnings": warnings,
        }
        report["series"].append(series_report)

        logger.info(
            "Series %s: %d slices, rows/cols=%s/%s, pixel_spacing=%s mm, "
            "nominal z-spacing=%s mm, slice_thickness=%s mm",
            series_uid[-12:], len(ordered), rows_set, cols_set, spacing_set, z_spacing, thickness_set,
        )
        for w in warnings:
            logger.warning("  [%s] %s", series_uid[-12:], w)
        if len(rows_set) > 1 or len(cols_set) > 1:
            logger.warning("  [%s] inconsistent dimensions within series: rows=%s cols=%s", series_uid[-12:], rows_set, cols_set)
        if len(spacing_set) > 1:
            logger.warning("  [%s] inconsistent pixel spacing within series: %s", series_uid[-12:], spacing_set)

    out_dir = cfg.output_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient.code}_dicom_inspection.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
