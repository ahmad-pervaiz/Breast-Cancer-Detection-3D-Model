"""
Image/mask preprocessing shared by training, validation, and inference.

Design decisions (verified against the actual data, not assumed - see
scripts/audit_dataset.py output / README "Dataset inspection findings"):

- All source PNGs are 512x512, uint8, [0, 255]. Some are stored as 3-channel
  RGB (Normal-folder and valid/Tumors images) but with R==G==B everywhere -
  i.e. genuinely grayscale content re-encoded as RGB, not color information.
  We therefore convert everything to single-channel grayscale ("L") with no
  information loss, so the model takes in_channels=1 consistently.
- No DICOM windowing is applied: the source is already a PNG export, not a
  raw DICOM intensity array, so arbitrary windowing here would be baseless.
- Masks are already binary (0/255). They are resized with NEAREST interpolation
  ONLY, never bilinear/bicubic, to keep them discrete.
- Images are already square (512x512), so aspect-ratio preservation via padding
  is a documented no-op today, but implemented generically for future data that
  may not be square.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    image_size: int = 256
    preserve_aspect_ratio: bool = True
    image_interpolation: str = "bilinear"   # for images only
    mask_interpolation: str = "nearest"     # for masks only - always nearest in practice


_PIL_RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


def load_grayscale(path: Path) -> np.ndarray:
    """Load any PNG (RGB or L) as a single-channel uint8 grayscale array.

    Verified R==G==B for every RGB source file at audit time, so this is a
    lossless conversion for this dataset's images (see module docstring).
    """
    with Image.open(path) as im:
        im = im.convert("L")
        return np.array(im, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """Load a mask PNG as a single-channel uint8 array with values in {0, 255}."""
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    # Defensive: force strict binarization in case of PNG compression artifacts
    # at edges (should not happen - masks were audited as strictly {0, 255}).
    return np.where(arr > 127, np.uint8(255), np.uint8(0))


def pad_to_square(arr: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Pad a 2D array to square with constant border (centered). No-op if already square."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    size = max(h, w)
    pad_h, pad_w = size - h, size - w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=pad_value)


def resize_array(arr: np.ndarray, size: int, interpolation: str) -> np.ndarray:
    im = Image.fromarray(arr)
    im = im.resize((size, size), resample=_PIL_RESAMPLE[interpolation])
    return np.array(im)


def preprocess_image(path: Path, cfg: PreprocessConfig) -> np.ndarray:
    """Full image preprocessing pipeline -> float32 array in [0, 1], shape (H, W)."""
    arr = load_grayscale(path)
    if cfg.preserve_aspect_ratio:
        arr = pad_to_square(arr, pad_value=0)
    arr = resize_array(arr, cfg.image_size, cfg.image_interpolation)
    return arr.astype(np.float32) / 255.0


def preprocess_mask(path: Path, cfg: PreprocessConfig) -> np.ndarray:
    """Full mask preprocessing pipeline -> float32 array in {0., 1.}, shape (H, W).

    Always uses nearest-neighbor interpolation regardless of cfg.mask_interpolation
    value, to guarantee masks never become non-binary through resizing.
    """
    arr = load_mask(path)
    if cfg.preserve_aspect_ratio:
        arr = pad_to_square(arr, pad_value=0)
    arr = resize_array(arr, cfg.image_size, "nearest")
    return (arr > 127).astype(np.float32)


def empty_mask(size: int) -> np.ndarray:
    """Mask for images with no annotation file (Normal images)."""
    return np.zeros((size, size), dtype=np.float32)
