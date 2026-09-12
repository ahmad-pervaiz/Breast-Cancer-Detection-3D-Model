"""AI-prediction mode (Milestone 12, Sections 29-31).

Wraps the existing, untouched Phase-1 inference code
(`src/inference/predict.py`) to supply 2D masks from the trained model
instead of Labelme JSON - plugged into `volume_io.build_series_volume`
through the exact same `MaskProvider` signature the ground-truth path uses
(`volume_io.make_ground_truth_mask_provider`), per Section 29's requirement
that the downstream pipeline be identical either way.

Phase-1's own `predict_image` returns a mask at the model's training
`image_size` (e.g. 256x256) - this module resizes it (nearest-neighbor, the
same convention `src/data/preprocessing.py` uses for masks) back up to the
DICOM's native Rows/Columns before handing it to the 3D pipeline, since that
pipeline is built around native-resolution slices.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import PreprocessConfig
from src.inference.predict import load_checkpoint, predict_image

from .volume_io import MaskProvider

logger = logging.getLogger(__name__)


class Phase1Predictor:
    """Loads a Phase-1 checkpoint once and runs it per-slice. Does not modify
    or retrain the checkpoint - pure inference, reusing src/inference/predict.py
    exactly as scripts/inference.py does for the 2D pipeline."""

    def __init__(self, checkpoint_path: Path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model, self.cfg = load_checkpoint(Path(checkpoint_path), self.device)

        self.preprocess_cfg = PreprocessConfig(
            image_size=self.cfg["image_size"],
            preserve_aspect_ratio=self.cfg.get("preserve_aspect_ratio", True),
            image_interpolation=self.cfg.get("image_interpolation", "bilinear"),
        )
        self.threshold = self.cfg.get("threshold", 0.5)
        self.min_tumor_area_px = self.cfg.get("min_tumor_area_px", 0)
        postprocess = self.cfg.get("postprocess") or {}
        self.remove_small_components_px = (
            postprocess.get("min_component_area_px", 0) if postprocess.get("remove_small_components") else 0
        )
        logger.info(
            "Loaded Phase-1 checkpoint %s (val_dice=%s, image_size=%d, threshold=%.2f, "
            "remove_small_components_px=%d)",
            checkpoint_path, self.cfg.get("val_dice"), self.preprocess_cfg.image_size,
            self.threshold, self.remove_small_components_px,
        )

    def predict_mask(self, png_path: Path, target_rows: int, target_cols: int) -> np.ndarray:
        """Returns a uint8 {0,1} mask at (target_rows, target_cols) - the
        DICOM's native resolution, not the model's internal image_size."""
        result = predict_image(
            self.model, png_path, self.preprocess_cfg, self.device,
            self.threshold, self.min_tumor_area_px, self.remove_small_components_px,
        )
        pred_mask = result["pred_mask"]  # (image_size, image_size), uint8 {0,1}
        if pred_mask.shape != (target_rows, target_cols):
            im = Image.fromarray(pred_mask * 255)
            im = im.resize((target_cols, target_rows), resample=Image.NEAREST)
            pred_mask = (np.array(im) > 127).astype(np.uint8)
        return pred_mask


def make_ai_prediction_mask_provider(predictor: Phase1Predictor, image_dir: Path) -> MaskProvider:
    """AI-prediction mode: runs the Phase-1 model on each slice's PNG. Returns
    None (-> empty mask, matching the ground-truth path's behavior for a
    missing annotation) only if the PNG itself is missing, which should not
    happen for slices already confirmed VERIFIED by the audit."""

    def _provider(stem: str, rows: int, cols: int) -> Optional[np.ndarray]:
        png_path = image_dir / f"{stem}.png"
        if not png_path.exists():
            logger.warning("No PNG for slice %s - cannot run AI prediction, using empty mask", stem)
            return None
        return predictor.predict_mask(png_path, rows, cols)

    return _provider
