#!/usr/bin/env python
"""Phase-2 Milestone 4: PNG <-> JSON <-> DICOM slice mapping for one patient.

Usage:
    python scripts/phase2_map_slices.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_map_slices.py --config configs/phase2_3d.yaml --patient Patient_05 --strict-mapping

`--patient` takes the anonymized code (see configs/phase2_3d.yaml), not the
real dataset folder name.

`scripts/phase2_audit_dataset.py` already writes a combined slice_mapping.csv
for every patient in one pass; this script exists for the Section 36 CLI
contract (map one patient in isolation) and to support `--strict-mapping`
(Section 17): with the flag set, non-VERIFIED rows are written to a separate
`*_rejected.csv` instead of the main output, so downstream volume-building
code can consume only verified rows without re-filtering.
"""
import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.future_3d.config import load_phase2_config
from src.future_3d.mapping import VERIFIED, map_patient_slices, records_to_dicts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_map_slices")

FIELDNAMES = [
    "patient_code", "png_file", "json_file", "dicom_file", "series_instance_uid",
    "sop_instance_uid", "instance_number", "z_position", "png_width", "png_height",
    "dicom_width", "dicom_height", "mapping_method", "mapping_confidence", "notes",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", required=True, help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--strict-mapping", action="store_true",
                         help="Split off non-VERIFIED rows into <code>_rejected.csv")
    args = parser.parse_args()

    cfg = load_phase2_config(args.config)
    patient = next((p for p in cfg.patients if p.code == args.patient), None)
    if patient is None:
        known = ", ".join(p.code for p in cfg.patients)
        logger.error("Unknown patient code %r. Known codes: %s", args.patient, known)
        sys.exit(1)

    records = map_patient_slices(patient.code, patient.image_dir, cfg.dicom_dir)
    confidence_counts = Counter(r.mapping_confidence for r in records)
    logger.info("%s: %d PNGs mapped -> %s", patient.code, len(records), dict(confidence_counts))

    out_dir = cfg.output_root / "mapping"
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted = records
    rejected = []
    if args.strict_mapping:
        accepted = [r for r in records if r.mapping_confidence == VERIFIED]
        rejected = [r for r in records if r.mapping_confidence != VERIFIED]
        if rejected:
            logger.warning("--strict-mapping: excluding %d/%d non-VERIFIED slice(s)", len(rejected), len(records))

    out_path = out_dir / f"{patient.code}_slice_mapping.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in records_to_dicts(accepted):
            writer.writerow(rec)
    logger.info("Wrote %s (%d rows)", out_path, len(accepted))

    if rejected:
        rej_path = out_dir / f"{patient.code}_rejected.csv"
        with open(rej_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for rec in records_to_dicts(rejected):
                writer.writerow(rec)
        logger.info("Wrote %s (%d rows)", rej_path, len(rejected))


if __name__ == "__main__":
    main()
