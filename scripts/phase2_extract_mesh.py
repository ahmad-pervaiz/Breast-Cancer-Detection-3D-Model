#!/usr/bin/env python
"""Phase-2 Milestone 7: extract a 3D tumor surface mesh (Sections 22-23).

Reads the tumor_mask.nii.gz written by phase2_build_volume.py, runs Marching
Cubes using its actual (possibly anisotropic) voxel spacing, and writes
PLY + STL meshes positioned in the same physical world coordinates as the
CT/mask volumes.

Usage:
    python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --patient Patient_05 --series-index 1
    python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --all-patients
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import SimpleITK as sitk

from src.future_3d.config import load_phase2_config
from src.future_3d.mesh_io import extract_tumor_mesh, save_ply, save_stl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_extract_mesh")


def extract_for_series_dir(series_dir: Path) -> bool:
    mask_path = series_dir / "tumor_mask.nii.gz"
    if not mask_path.exists():
        logger.warning("No tumor_mask.nii.gz in %s - run phase2_build_volume.py first", series_dir)
        return False

    mask_image = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)  # (Z, Y, X)
    spacing = mask_image.GetSpacing()
    origin = mask_image.GetOrigin()
    direction = mask_image.GetDirection()

    if mask_array.sum() == 0:
        logger.error("%s: tumor mask is entirely empty - cannot run marching cubes. Investigate before proceeding.", series_dir)
        return False

    mesh = extract_tumor_mesh(mask_array, spacing, origin, direction)
    save_ply(mesh, series_dir / "tumor_mesh.ply")
    save_stl(mesh, series_dir / "tumor_mesh.stl")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--all-patients", action="store_true")
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--mode", choices=["ground_truth", "prediction"], default="ground_truth")
    args = parser.parse_args()
    mode_dir = "ground_truth" if args.mode == "ground_truth" else "predictions"

    if not args.patient and not args.all_patients:
        parser.error("specify --patient CODE or --all-patients")

    cfg = load_phase2_config(args.config)
    codes = [p.code for p in cfg.patients] if args.all_patients else [args.patient]
    known = {p.code for p in cfg.patients}
    for code in codes:
        if code not in known:
            logger.error("Unknown patient code %r. Known codes: %s", code, ", ".join(sorted(known)))
            sys.exit(1)

    n_ok = 0
    for code in codes:
        ground_truth_dir = cfg.output_root / mode_dir / code
        if not ground_truth_dir.exists():
            logger.warning("[%s] no %s output yet - run phase2_build_volume.py first", code, mode_dir)
            continue
        series_dirs = sorted(ground_truth_dir.glob("series_*"))
        for series_dir in series_dirs:
            if args.series_index is not None:
                idx = int(series_dir.name.split("_")[1])
                if idx != args.series_index:
                    continue
            logger.info("[%s] %s", code, series_dir.name)
            if extract_for_series_dir(series_dir):
                n_ok += 1

    logger.info("Done: %d mesh(es) extracted", n_ok)


if __name__ == "__main__":
    main()
