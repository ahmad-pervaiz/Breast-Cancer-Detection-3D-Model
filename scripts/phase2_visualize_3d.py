#!/usr/bin/env python
"""Phase-2 Milestones 8-9 (+28): tumor-only and CT+tumor visualization.

Static, headless-safe rendering via matplotlib (mplot3d for the mesh,
imshow for orthogonal slices) - avoids depending on an interactive VTK/
PyVista offscreen stack, which this machine's environment was not confirmed
to support. 3D Slicer (Section 27) remains the recommended tool for
interactive inspection of the saved .nii.gz volumes - these PNGs are the
mandatory "does this look like a coherent tumor, not a fragmented mess"
sanity check (Section 25) before trusting that.

Usage:
    python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --patient Patient_05
    python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --all-patients

Outputs, per series directory:
    3d_tumor.png       Milestone 8 - tumor mesh alone, two viewing angles
    3d_ct_tumor.png     Milestone 9/28 - axial/sagittal/coronal slices (CT +
                        tumor overlay) through the tumor centroid, + 3D tumor
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.future_3d.config import load_phase2_config
from src.future_3d.mesh_io import extract_tumor_mesh

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_visualize_3d")


def _window(ct_slice: np.ndarray, center=40, width=400) -> np.ndarray:
    """Soft-tissue window (Section 40) for display only - never overwrites the saved volume."""
    lo, hi = center - width / 2, center + width / 2
    return np.clip((ct_slice - lo) / (hi - lo), 0, 1)


def render_tumor_only(mesh, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 5))
    for i, (elev, azim) in enumerate([(20, 45), (20, 135)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        poly = Poly3DCollection(mesh.vertices_mm[mesh.faces], alpha=0.9)
        poly.set_facecolor((0.85, 0.25, 0.25))
        poly.set_edgecolor((0.3, 0.05, 0.05, 0.15))
        ax.add_collection3d(poly)
        mins, maxs = mesh.vertices_mm.min(axis=0), mesh.vertices_mm.max(axis=0)
        center, extent = (mins + maxs) / 2, (maxs - mins).max() / 2 * 1.2
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Tumor surface (elev={elev}, azim={azim})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def render_ct_tumor_panels(ct_array, mask_array, spacing_xyz, mesh, out_path: Path) -> None:
    zc, yc, xc = np.round(np.argwhere(mask_array).mean(axis=0)).astype(int)
    sx, sy, sz = spacing_xyz

    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(_window(ct_array[zc]), cmap="gray", aspect=sy / sx)
    ax1.imshow(np.ma.masked_where(mask_array[zc] == 0, mask_array[zc]), cmap="autumn", alpha=0.45, aspect=sy / sx)
    ax1.set_title(f"Axial (z-index={zc})"); ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    sag_ct, sag_mask = ct_array[:, :, xc], mask_array[:, :, xc]
    ax2.imshow(_window(sag_ct), cmap="gray", aspect=sz / sy, origin="lower")
    ax2.imshow(np.ma.masked_where(sag_mask == 0, sag_mask), cmap="autumn", alpha=0.45, aspect=sz / sy, origin="lower")
    ax2.set_title(f"Sagittal (x-index={xc})"); ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    cor_ct, cor_mask = ct_array[:, yc, :], mask_array[:, yc, :]
    ax3.imshow(_window(cor_ct), cmap="gray", aspect=sz / sx, origin="lower")
    ax3.imshow(np.ma.masked_where(cor_mask == 0, cor_mask), cmap="autumn", alpha=0.45, aspect=sz / sx, origin="lower")
    ax3.set_title(f"Coronal (y-index={yc})"); ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    poly = Poly3DCollection(mesh.vertices_mm[mesh.faces], alpha=0.9)
    poly.set_facecolor((0.85, 0.25, 0.25))
    poly.set_edgecolor((0.3, 0.05, 0.05, 0.15))
    ax4.add_collection3d(poly)
    mins, maxs = mesh.vertices_mm.min(axis=0), mesh.vertices_mm.max(axis=0)
    center, extent = (mins + maxs) / 2, (maxs - mins).max() / 2 * 1.2
    ax4.set_xlim(center[0] - extent, center[0] + extent)
    ax4.set_ylim(center[1] - extent, center[1] + extent)
    ax4.set_zlim(center[2] - extent, center[2] + extent)
    ax4.set_title("3D tumor")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def visualize_series_dir(series_dir: Path) -> bool:
    ct_path, mask_path = series_dir / "ct_volume.nii.gz", series_dir / "tumor_mask.nii.gz"
    if not ct_path.exists() or not mask_path.exists():
        logger.warning("Missing volume(s) in %s - run phase2_build_volume.py first", series_dir)
        return False

    ct_image, mask_image = sitk.ReadImage(str(ct_path)), sitk.ReadImage(str(mask_path))
    ct_array, mask_array = sitk.GetArrayFromImage(ct_image), sitk.GetArrayFromImage(mask_image)
    spacing, origin, direction = ct_image.GetSpacing(), ct_image.GetOrigin(), ct_image.GetDirection()

    if mask_array.sum() == 0:
        logger.error("%s: tumor mask is entirely empty - skipping visualization", series_dir)
        return False

    mesh = extract_tumor_mesh(mask_array, spacing, origin, direction)
    render_tumor_only(mesh, series_dir / "3d_tumor.png")
    render_ct_tumor_panels(ct_array, mask_array, spacing, mesh, series_dir / "3d_ct_tumor.png")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_3d.yaml"))
    parser.add_argument("--patient", help="Anonymized patient code, e.g. Patient_05")
    parser.add_argument("--all-patients", action="store_true")
    parser.add_argument("--series-index", type=int, default=None)
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

    n_ok = 0
    for code in codes:
        ground_truth_dir = cfg.output_root / "ground_truth" / code
        if not ground_truth_dir.exists():
            logger.warning("[%s] no ground_truth output yet - run phase2_build_volume.py first", code)
            continue
        for series_dir in sorted(ground_truth_dir.glob("series_*")):
            if args.series_index is not None and int(series_dir.name.split("_")[1]) != args.series_index:
                continue
            logger.info("[%s] %s", code, series_dir.name)
            if visualize_series_dir(series_dir):
                n_ok += 1

    logger.info("Done: %d series visualized", n_ok)


if __name__ == "__main__":
    main()
