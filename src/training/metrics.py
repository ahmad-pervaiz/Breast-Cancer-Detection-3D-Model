"""
Segmentation + image-level detection metrics.

Key design point (project spec section 15/33): pixel-level segmentation
metrics (Dice/IoU/precision/recall) are reported TWO ways:

  - "tumor_positive": averaged only over images with a non-empty ground-truth
    mask. This is the real measure of segmentation quality.
  - "overall": averaged over ALL images including Normal (empty-mask) ones,
    where a correct empty/empty prediction scores Dice=1.0 by definition.
    This number is reported too, but clearly labeled, because it can look
    artificially high if Normal images dominate the split - never used alone.

Image-level tumor-presence metrics (sensitivity/specificity/precision/
accuracy/F1) are computed separately and are explicitly a detection metric,
not a segmentation metric.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def pixel_confusion(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def dice_from_confusion(tp: int, fp: int, fn: int, eps: float = 1e-6) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0  # both empty - trivially perfect, tracked separately (see module docstring)
    return (2 * tp + eps) / (denom + eps)


def iou_from_confusion(tp: int, fp: int, fn: int, eps: float = 1e-6) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return (tp + eps) / (denom + eps)


def precision_from_confusion(tp: int, fp: int, eps: float = 1e-6) -> Optional[float]:
    if tp + fp == 0:
        return None  # undefined: model predicted nothing positive
    return tp / (tp + fp + eps)


def recall_from_confusion(tp: int, fn: int, eps: float = 1e-6) -> Optional[float]:
    if tp + fn == 0:
        return None  # undefined: no positive ground truth in this image
    return tp / (tp + fn + eps)


def specificity_from_confusion(tn: int, fp: int, eps: float = 1e-6) -> Optional[float]:
    if tn + fp == 0:
        return None
    return tn / (tn + fp + eps)


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    return mask & ~eroded


def hd95(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """95th-percentile symmetric surface distance, in pixels. None if scipy
    unavailable or if HD is undefined (exactly one mask empty)."""
    if not _HAS_SCIPY:
        return None
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return None  # undefined - one mask empty, the other isn't
    pred_bd, gt_bd = _boundary(pred_b), _boundary(gt_b)
    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)
    d1 = dt_gt[pred_bd] if pred_bd.any() else np.array([0.0])
    d2 = dt_pred[gt_bd] if gt_bd.any() else np.array([0.0])
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))


@dataclass
class ImageRecord:
    dice: float
    iou: float
    precision: Optional[float]
    recall: Optional[float]
    specificity: Optional[float]
    hd95: Optional[float]
    gt_positive: bool
    pred_positive: bool
    tumor_pixel_count_pred: int
    tumor_pixel_count_gt: int
    patient_id: str = ""
    image_path: str = ""


class MetricAccumulator:
    def __init__(self, min_tumor_area_px: int = 20, compute_hd95: bool = True):
        self.min_tumor_area_px = min_tumor_area_px
        self.compute_hd95 = compute_hd95
        self.records: List[ImageRecord] = []

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray,
               patient_id: str = "", image_path: str = "") -> ImageRecord:
        tp, fp, fn, tn = pixel_confusion(pred_bin, gt_bin)
        rec = ImageRecord(
            dice=dice_from_confusion(tp, fp, fn),
            iou=iou_from_confusion(tp, fp, fn),
            precision=precision_from_confusion(tp, fp),
            recall=recall_from_confusion(tp, fn),
            specificity=specificity_from_confusion(tn, fp),
            hd95=hd95(pred_bin, gt_bin) if self.compute_hd95 else None,
            gt_positive=bool(gt_bin.sum() > 0),
            pred_positive=bool(pred_bin.sum() >= self.min_tumor_area_px),
            tumor_pixel_count_pred=int(pred_bin.sum()),
            tumor_pixel_count_gt=int(gt_bin.sum()),
            patient_id=patient_id,
            image_path=image_path,
        )
        self.records.append(rec)
        return rec

    def _mean(self, values: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None

    def summary(self) -> dict:
        pos = [r for r in self.records if r.gt_positive]
        all_r = self.records

        def block(records: List[ImageRecord]) -> dict:
            return {
                "n_images": len(records),
                "dice": self._mean([r.dice for r in records]),
                "iou": self._mean([r.iou for r in records]),
                "precision": self._mean([r.precision for r in records]),
                "recall": self._mean([r.recall for r in records]),
                "specificity": self._mean([r.specificity for r in records]),
                "hd95": self._mean([r.hd95 for r in records]),
            }

        # Image-level tumor-presence (detection) confusion matrix.
        tp = sum(1 for r in all_r if r.gt_positive and r.pred_positive)
        fp = sum(1 for r in all_r if not r.gt_positive and r.pred_positive)
        fn = sum(1 for r in all_r if r.gt_positive and not r.pred_positive)
        tn = sum(1 for r in all_r if not r.gt_positive and not r.pred_positive)
        n = len(all_r)
        detection = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity_recall": tp / (tp + fn) if (tp + fn) else None,
            "specificity": tn / (tn + fp) if (tn + fp) else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "accuracy": (tp + tn) / n if n else None,
            "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
        }

        return {
            "tumor_positive_segmentation": block(pos),
            "overall_segmentation_including_normal": block(all_r),
            "image_level_detection": detection,
        }
