#!/usr/bin/env python
"""Phase-2 Milestone 11: physical tumor measurements (Sections 32-33).

Reads the tumor_mask.nii.gz already written by phase2_build_volume.py and
computes:
    - tumor volume (mm3/cm3), from the actual voxel spacing - whole mask
    - axis-aligned bounding-box extent (mm) - largest connected component only
    - maximum Feret diameter (mm) - largest connected component only, the
      true largest surface-to-surface distance, not a bounding-box stand-in
      (Section 33's explicit warning)

A small disconnected mask fragment (see phase2_build_volume.py's own
fragmentation check) is excluded from the bounding box/diameter so it can't
inflate the reported size - see `excluded_fragment_voxels/fraction` in the
output for what was left out. Does not report anything if spacing/geometry
is missing (Section 32: "Do not report physical measurements if spacing or
slice geometry is uncertain").

Usage:
    python scripts/phase2_measurements.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_measurements.py --config configs/phase2_3d.yaml --all-patients
    python scripts/phase2_measurements.py --config configs/phase2_3d.yaml --all-patients --mode prediction
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import SimpleITK as sitk

from src.future_3d.config import load_phase2_config
from src.future_3d.measurements import compute_measurements

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_measurements")


def measure_series_dir(series_dir: Path) -> bool:
    mask_path = series_dir / "tumor_mask.nii.gz"
    if not mask_path.exists():
        logger.warning("Missing %s - run phase2_build_volume.py first", mask_path)
        return False

    mask_image = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing, origin, direction = mask_image.GetSpacing(), mask_image.GetOrigin(), mask_image.GetDirection()

    if not all(s and s > 0 for s in spacing):
        logger.error("%s: invalid/missing spacing - refusing to report physical measurements (Section 32)", series_dir)
        return False
    if mask_array.sum() == 0:
        logger.error("%s: empty mask - nothing to measure", series_dir)
        return False

    result = compute_measurements(mask_array, spacing, origin, direction)

    out_path = series_dir / "measurements.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    bbox = result["bounding_box"]
    logger.info(
        "%s: total_volume=%.2f cm3 (excluded fragment=%.1f%%), bbox(x,y,z)=(%.1f, %.1f, %.1f)mm, max Feret diameter=%.1fmm",
        series_dir, result["tumor_volume_total_cm3"], 100 * result["excluded_fragment_fraction"],
        bbox["extent_x_mm"], bbox["extent_y_mm"], bbox["extent_z_mm"],
        result["max_feret_diameter"]["max_feret_diameter_mm"],
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--all-patients", action="store_true")
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--mode", choices=["ground_truth", "prediction"], default="ground_truth",
                         help="Which output tree to measure (Section 30: never mixed)")
    args = parser.parse_args()

    if not args.patient and not args.all_patients:
        parser.error("specify --patient CODE or --all-patients")

    cfg = load_phase2_config(args.config)
    codes = [p.code for p in cfg.patients] if args.all_patients else [args.patient]
    known = {p.code for p in cfg.patients}
    for code in codes:
        if code not in known:
            logger.error("Unknown patient code %r. Known codes: %s", code, ", ".join(sorted(known)))
            sys.exit(1)

    mode_dir = "ground_truth" if args.mode == "ground_truth" else "predictions"
    n_ok = 0
    for code in codes:
        patient_dir = cfg.output_root / mode_dir / code
        if not patient_dir.exists():
            logger.warning("[%s] no %s output yet", code, mode_dir)
            continue
        for series_dir in sorted(patient_dir.glob("series_*")):
            if args.series_index is not None and int(series_dir.name.split("_")[1]) != args.series_index:
                continue
            if measure_series_dir(series_dir):
                n_ok += 1

    logger.info("Done: %d series measured", n_ok)


if __name__ == "__main__":
    main()
