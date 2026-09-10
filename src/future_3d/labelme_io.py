"""Labelme JSON -> binary mask conversion (Project_phase2.txt Section 12).

Verified against the actual dataset (1,352 JSON files, see
runs/phase2_3d/audit/): every shape is a `polygon` with label `"Tumor"` -
no other labels or shape types exist. Matching is still done case-insensitively
against a configurable label list (not hard-coded to "Tumor") so this keeps
working if a differently-cased or synonymous label shows up in data added later.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class MaskResult:
    mask: np.ndarray            # uint8, {0,1}, shape (height, width)
    n_polygons_used: int
    labels_found: List[str] = field(default_factory=list)
    unknown_labels: List[str] = field(default_factory=list)
    shape_types_skipped: List[str] = field(default_factory=list)


def load_labelme_json(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def labelme_to_mask(
    json_path: Path,
    height: int,
    width: int,
    tumor_labels: List[str],
) -> MaskResult:
    """Rasterize every tumor polygon in a Labelme JSON file into one binary mask.

    `height`/`width` must be the corresponding PNG's actual dimensions (the
    mask is generated at that resolution, not inferred from the JSON, since
    Labelme's own `imageHeight`/`imageWidth` fields are not treated as
    authoritative - Section 13 requires `mask.shape == PNG.shape`).
    """
    data = load_labelme_json(json_path)
    shapes = data.get("shapes", [])

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    tumor_label_set = {label.lower() for label in tumor_labels}
    labels_found: List[str] = []
    unknown_labels: List[str] = []
    shape_types_skipped: List[str] = []
    n_used = 0

    for shape in shapes:
        label = shape.get("label", "")
        labels_found.append(label)
        if label.lower() not in tumor_label_set:
            unknown_labels.append(label)
            continue

        shape_type = shape.get("shape_type", "polygon")
        if shape_type != "polygon":
            shape_types_skipped.append(shape_type)
            logger.warning(
                "Skipping non-polygon shape_type=%s in %s (only polygon rasterization implemented)",
                shape_type, json_path,
            )
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            logger.warning("Skipping degenerate polygon (<3 points) in %s", json_path)
            continue

        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, outline=1, fill=1)
        n_used += 1

    mask = (np.array(canvas) > 0).astype(np.uint8)
    return MaskResult(
        mask=mask,
        n_polygons_used=n_used,
        labels_found=labels_found,
        unknown_labels=unknown_labels,
        shape_types_skipped=shape_types_skipped,
    )


def mask_stats(mask: np.ndarray) -> Tuple[int, float]:
    """Returns (tumor_pixel_count, tumor_fraction)."""
    count = int(mask.sum())
    fraction = count / mask.size if mask.size else 0.0
    return count, fraction
