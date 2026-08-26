#!/usr/bin/env python3
"""
Run inference with a trained checkpoint.

Single image:
    python scripts/inference.py \\
        --input /path/to/image.png \\
        --checkpoint runs/segmentation/checkpoints/best_model.pth \\
        --output runs/segmentation/predictions/

Folder (e.g. the held-out test set):
    python scripts/inference.py \\
        --input "FINAL DATASET/test" \\
        --checkpoint runs/segmentation/checkpoints/best_model.pth \\
        --output runs/segmentation/test_predictions/

For a folder input, if subfolders are named "Normal"/"Tumors" (as in the test
set), an OPTIONAL image-level detection metrics report is printed - this is
NOT pixel-level Dice/IoU (the test set has no ground-truth masks) and is
labeled as such in the output.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import get_device, load_config
from src.data.preprocessing import PreprocessConfig
from src.inference.predict import (
    image_level_metrics_from_folder_labels, load_checkpoint, run_folder_inference, run_single_image,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="Single image path or a folder to run over.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/config.yaml",
                    help="Only used for threshold/min_tumor_area_px/image_size defaults "
                         "if not overridden below; the checkpoint's own training config is authoritative.")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min_tumor_area_px", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, train_cfg = load_checkpoint(Path(args.checkpoint), device)

    default_cfg = load_config(Path(args.config))
    threshold = args.threshold if args.threshold is not None else train_cfg.get("threshold", default_cfg["threshold"])
    min_area = args.min_tumor_area_px if args.min_tumor_area_px is not None else train_cfg.get(
        "min_tumor_area_px", default_cfg["min_tumor_area_px"])
    preprocess_cfg = PreprocessConfig(image_size=train_cfg["image_size"])

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        run_single_image(model, input_path, Path(args.checkpoint), output_path,
                          preprocess_cfg, device, threshold, min_area)
    elif input_path.is_dir():
        records = run_folder_inference(model, input_path, output_path, preprocess_cfg,
                                        device, threshold, min_area)
        n_detected = sum(1 for r in records if r["tumor_detected"])
        print(f"\nProcessed {len(records)} images. Tumor detected in {n_detected}.")

        metrics = image_level_metrics_from_folder_labels(records)
        if metrics:
            print("\n" + "=" * 70)
            print(metrics.pop("note"))
            print("=" * 70)
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            with open(output_path / "image_level_detection_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        with open(output_path / "predictions_summary.json", "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nAll outputs saved to: {output_path}")
    else:
        raise FileNotFoundError(f"--input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
