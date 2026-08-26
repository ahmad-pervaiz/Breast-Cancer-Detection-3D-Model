"""
Inference: load a trained checkpoint and run it on one image or a folder tree.

IMPORTANT: this module never computes pixel-level Dice/IoU against the test
set, because FINAL DATASET/test has no ground-truth masks (by design - see
project README). When folder-label ground truth is available (test/Normal
vs test/Tumors), it optionally computes IMAGE-LEVEL detection metrics only,
clearly labeled as such - never presented as segmentation quality.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.preprocessing import PreprocessConfig, preprocess_image
from src.models.unet import build_model
from src.visualization.visualize import save_prediction_set


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch={ckpt.get('epoch')}, "
          f"val_dice={ckpt.get('val_dice')})")
    return model, cfg


@torch.no_grad()
def predict_image(
    model: torch.nn.Module, image_path: Path, preprocess_cfg: PreprocessConfig,
    device: torch.device, threshold: float, min_tumor_area_px: int,
) -> Dict:
    image = preprocess_image(image_path, preprocess_cfg)  # (H, W) float32 [0,1]
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_mask = (prob >= threshold).astype(np.uint8)

    tumor_pixel_count = int(pred_mask.sum())
    total_pixels = pred_mask.size
    tumor_detected = tumor_pixel_count >= min_tumor_area_px

    return {
        "image": image,
        "probability_map": prob,
        "pred_mask": pred_mask,
        "tumor_detected": tumor_detected,
        "tumor_pixel_count": tumor_pixel_count,
        "tumor_percentage": 100.0 * tumor_pixel_count / total_pixels,
    }


def run_single_image(
    model, image_path: Path, checkpoint_path: Path, output_dir: Path,
    preprocess_cfg: PreprocessConfig, device: torch.device, threshold: float, min_tumor_area_px: int,
) -> Dict:
    result = predict_image(model, image_path, preprocess_cfg, device, threshold, min_tumor_area_px)
    stem = Path(image_path).stem
    save_prediction_set(result["image"], None, result["probability_map"], result["pred_mask"],
                         output_dir, stem)

    print(f"\n{image_path.name}")
    print(f"  tumor_detected: {result['tumor_detected']}")
    print(f"  tumor_pixel_count: {result['tumor_pixel_count']}")
    print(f"  tumor_percentage_of_image: {result['tumor_percentage']:.3f}%")
    print(f"  Outputs saved to: {output_dir}")
    print("  NOTE: this is an AI-estimated tumor region from a research prototype, "
          "not a clinical diagnosis, and no physical (mm/volume) size is reported yet.")
    return result


def run_folder_inference(
    model, input_root: Path, output_root: Path, preprocess_cfg: PreprocessConfig,
    device: torch.device, threshold: float, min_tumor_area_px: int,
) -> List[Dict]:
    """Runs inference over every PNG under input_root (preserving subfolder
    structure, e.g. Normal/ and Tumors/, in output_root). Returns per-image
    records including folder_label (derived from parent dir name) IF present -
    used only for optional image-level detection metrics, never mask metrics."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    records = []

    image_paths = sorted(input_root.rglob("*.png"))
    print(f"Found {len(image_paths)} images under {input_root}")

    for image_path in image_paths:
        rel = image_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        result = predict_image(model, image_path, preprocess_cfg, device, threshold, min_tumor_area_px)
        save_prediction_set(result["image"], None, result["probability_map"], result["pred_mask"],
                             out_dir, image_path.stem)

        records.append({
            "image_path": str(image_path),
            "folder_label": rel.parent.name,  # e.g. "Normal" / "Tumors" - from folder, not a mask
            "tumor_detected": result["tumor_detected"],
            "tumor_pixel_count": result["tumor_pixel_count"],
            "tumor_percentage": result["tumor_percentage"],
        })

    return records


def image_level_metrics_from_folder_labels(records: List[Dict], positive_label: str = "Tumors") -> Optional[Dict]:
    """Section 20: OPTIONAL image-level detection metrics derived from folder
    names (test/Normal vs test/Tumors), explicitly NOT pixel-level segmentation
    evaluation (no ground-truth masks exist for the test set)."""
    labels = {r["folder_label"] for r in records}
    if not labels & {positive_label, "Normal"}:
        return None  # folder names don't look like a Normal/Tumor split - skip

    tp = fp = fn = tn = 0
    for r in records:
        gt_positive = r["folder_label"] == positive_label
        pred_positive = r["tumor_detected"]
        if gt_positive and pred_positive:
            tp += 1
        elif not gt_positive and pred_positive:
            fp += 1
        elif gt_positive and not pred_positive:
            fn += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    return {
        "note": "IMAGE-LEVEL detection metrics derived from folder labels only - "
                "NOT pixel-level segmentation evaluation (test set has no masks).",
        "n_images": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity_recall": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "accuracy": (tp + tn) / n if n else None,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
    }
