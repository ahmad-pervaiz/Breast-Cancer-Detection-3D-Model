#!/usr/bin/env python
"""Phase-2 Milestone 8-9 (Section 24): real interactive 3D viewing via PyVista.

The default `phase2_visualize_3d.py` renders static PNGs (headless-safe, no
display required, used for automated validation). THIS script opens PyVista's
actual interactive window - rotate with left-drag, zoom with scroll, pan with
right-drag/middle-drag, matching Section 24's requirement - so it needs a real
display (run it on your desktop, not over a plain SSH/headless session).

Usage (interactive, needs a display):
    python scripts/phase2_interactive_view.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_interactive_view.py --config configs/phase2_3d.yaml --patient Patient_05 --series-index 1 --with-ct

Usage (headless smoke test / CI - saves a screenshot instead of opening a window):
    python scripts/phase2_interactive_view.py --config configs/phase2_3d.yaml --patient Patient_05 --screenshot out.png
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from src.future_3d.config import load_phase2_config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_interactive_view")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", required=True, help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--series-index", type=int, default=1)
    parser.add_argument("--with-ct", action="store_true", help="Also show a semi-transparent CT volume rendering (Section 26), not just the tumor mesh")
    parser.add_argument("--screenshot", type=Path, default=None, help="Headless mode: save a screenshot here instead of opening an interactive window")
    parser.add_argument("--mode", choices=["ground_truth", "prediction"], default="ground_truth")
    args = parser.parse_args()
    mode_dir = "ground_truth" if args.mode == "ground_truth" else "predictions"

    cfg = load_phase2_config(args.config)
    if args.patient not in {p.code for p in cfg.patients}:
        known = ", ".join(sorted(p.code for p in cfg.patients))
        logger.error("Unknown patient code %r. Known codes: %s", args.patient, known)
        sys.exit(1)

    series_dir = cfg.output_root / mode_dir / args.patient / f"series_{args.series_index:02d}"
    mesh_path = series_dir / "tumor_mesh.ply"
    if not mesh_path.exists():
        logger.error("%s not found - run phase2_build_volume.py + phase2_extract_mesh.py first", mesh_path)
        sys.exit(1)

    plotter = pv.Plotter(off_screen=args.screenshot is not None)
    tumor_mesh = pv.read(str(mesh_path))
    plotter.add_mesh(tumor_mesh, color="firebrick", opacity=1.0, smooth_shading=True, label="Tumor")

    if args.with_ct:
        ct_path = series_dir / "ct_volume.nii.gz"
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, Y, X)
        grid = pv.ImageData(
            dimensions=np.array(ct_array.shape[::-1]) + 1,  # point-data dims = cell dims + 1
            spacing=ct_image.GetSpacing(),
            origin=ct_image.GetOrigin(),
        )
        grid.cell_data["HU"] = ct_array.flatten(order="C")
        # Soft-tissue-ish opacity ramp so bone/skin don't fully occlude the tumor (Section 26).
        opacity = [0.0, 0.0, 0.05, 0.15, 0.3]
        plotter.add_volume(grid, scalars="HU", cmap="bone", opacity=opacity, opacity_unit_distance=2.0)

    plotter.add_legend()
    plotter.add_axes()
    plotter.camera_position = "iso"

    if args.screenshot:
        plotter.screenshot(str(args.screenshot))
        logger.info("Wrote %s (headless mode - for a real rotate/zoom/pan window, run this without --screenshot on a machine with a display)", args.screenshot)
    else:
        logger.info("Opening interactive window - left-drag to rotate, scroll to zoom, right-drag to pan. Close the window to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
