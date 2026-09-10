#!/usr/bin/env python
"""Phase-2 Milestones 5-6: build the 3D CT volume + 3D tumor-mask volume.

One volume is built PER (patient, series) - Section 0 item 5 of
Project_phase2.txt found a "patient" is not one contiguous acquisition, so a
whole-patient volume would silently merge physically distinct series. By
default builds every series for the given patient; --series-index restricts
to one (1-based, ordered by descending slice count, matching
scripts/phase2_inspect_dicom.py's numbering).

Usage:
    python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --patient Patient_05 --series-index 1
    python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --all-patients

Outputs (Section 34, adapted): <output_root>/ground_truth/<code>/series_<NN>/
    ct_volume.nii.gz, tumor_mask.nii.gz, reconstruction_metadata.json
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.future_3d.config import Phase2Config, PatientSpec, load_phase2_config
from src.future_3d.dicom_io import order_slices, read_dicom_header
from src.future_3d.mapping import map_patient_slices
from src.future_3d.volume_io import build_series_volume, save_series_volume

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_build_volume")


def build_patient_volumes(cfg: Phase2Config, patient: PatientSpec, series_index_filter=None):
    records = map_patient_slices(patient.code, patient.image_dir, cfg.dicom_dir)

    by_series = defaultdict(list)
    stem_to_json = {}
    for r in records:
        stem_to_json[Path(r.png_file).stem] = (patient.image_dir / r.json_file) if r.json_file else None
        if r.dicom_file and r.series_instance_uid:
            by_series[r.series_instance_uid].append(cfg.dicom_dir / r.dicom_file)

    # Same numbering convention as phase2_inspect_dicom.py: series_01 = most slices.
    series_uids_ordered = sorted(by_series.keys(), key=lambda uid: -len(by_series[uid]))

    written = []
    for idx, series_uid in enumerate(series_uids_ordered, start=1):
        if series_index_filter is not None and idx != series_index_filter:
            continue

        headers = [read_dicom_header(p) for p in by_series[series_uid]]
        ordered, warnings = order_slices(headers)
        method = next((w.split("=", 1)[1] for w in warnings if w.startswith("ordering_method=")), "unknown")
        warnings = [w for w in warnings if not w.startswith("ordering_method=")]

        result = build_series_volume(
            patient_code=patient.code,
            series_uid=series_uid,
            series_index=idx,
            ordered_headers=ordered,
            ordering_method=method,
            ordering_warnings=warnings,
            stem_to_json=stem_to_json,
            tumor_labels=cfg.tumor_labels,
        )
        for w in result.warnings:
            logger.warning("[%s series_%02d] %s", patient.code, idx, w)

        out_dir = cfg.output_root / "ground_truth" / patient.code / f"series_{idx:02d}"
        paths = save_series_volume(result, out_dir)
        logger.info(
            "[%s series_%02d] wrote %s (mask_voxels=%d, tumor_volume=%.2f cm3)",
            patient.code, idx, out_dir, result.tumor_voxel_count, result.tumor_volume_mm3 / 1000.0,
        )
        written.append(paths)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--all-patients", action="store_true", help="Build volumes for every configured patient")
    parser.add_argument("--series-index", type=int, default=None, help="Restrict to one series (1-based); default: all series for the patient")
    args = parser.parse_args()

    if not args.patient and not args.all_patients:
        parser.error("specify --patient CODE or --all-patients")

    cfg = load_phase2_config(args.config)

    if args.all_patients:
        targets = cfg.patients
    else:
        targets = [p for p in cfg.patients if p.code == args.patient]
        if not targets:
            known = ", ".join(p.code for p in cfg.patients)
            logger.error("Unknown patient code %r. Known codes: %s", args.patient, known)
            sys.exit(1)

    total = 0
    for patient in targets:
        written = build_patient_volumes(cfg, patient, series_index_filter=args.series_index)
        total += len(written)

    logger.info("Done: %d series volume(s) written across %d patient(s)", total, len(targets))


if __name__ == "__main__":
    main()
