#!/usr/bin/env python
"""Phase-2 Milestone 3 (Section 14): mandatory visual validation before 3D.

For one patient, saves original / mask / overlay PNGs for every annotated
slice. Project_phase2.txt Section 14 is explicit that this step is mandatory
and that 3D reconstruction must not proceed if the mask is not visibly
aligned with the tumor in the original image - this script produces the
evidence a human reviews to make that call; it does not make it automatically.

Usage:
    python scripts/phase2_visual_validation.py --config configs/phase2_3d.yaml --patient Patient_05

`--patient` takes the anonymized code (see configs/phase2_3d.yaml), not the
real dataset folder name. Output images are saved under that code only.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.future_3d.config import load_phase2_config
from src.future_3d.labelme_io import labelme_to_mask, mask_stats

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("phase2_visual_validation")

OVERLAY_COLOR = (255, 60, 60)   # red tumor outline/fill tint
OVERLAY_ALPHA = 0.35


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    overlay = image_rgb.astype(np.float32).copy()
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    m = mask.astype(bool)
    overlay[m] = overlay[m] * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA
    return Image.fromarray(overlay.astype(np.uint8))


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

    out_dir = cfg.output_root / "validation" / patient.code
    for sub in ("original", "masks", "overlays"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    png_files = sorted(p for p in patient.image_dir.iterdir() if p.suffix.lower() == ".png")
    n_written, n_empty = 0, 0
    for png_path in png_files:
        json_path = png_path.with_suffix(".json")
        if not json_path.exists():
            logger.warning("Skipping %s: no matching JSON", png_path.name)
            continue

        image = Image.open(png_path).convert("RGB")
        width, height = image.size
        result = labelme_to_mask(json_path, height, width, cfg.tumor_labels)
        count, fraction = mask_stats(result.mask)
        if count == 0:
            n_empty += 1
            logger.warning("Empty mask for annotated slice %s - investigate before 3D", png_path.name)

        image.save(out_dir / "original" / png_path.name)
        Image.fromarray((result.mask * 255).astype(np.uint8)).save(out_dir / "masks" / png_path.name)
        overlay = make_overlay(np.array(image), result.mask)
        overlay.save(out_dir / "overlays" / png_path.name)
        n_written += 1

    logger.info(
        "Wrote %d original/mask/overlay triples for %s to %s (%d empty masks)",
        n_written, patient.code, out_dir, n_empty,
    )
    logger.info("Manually inspect a sample of %s before proceeding to 3D reconstruction (Section 14).", out_dir / "overlays")


if __name__ == "__main__":
    main()
