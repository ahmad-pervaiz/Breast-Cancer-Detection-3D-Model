"""
PyTorch Dataset for 2D breast-tumor CT segmentation.

Reads directly from the already-audited manifest (data/manifest.csv, built by
build_manifest.py) rather than re-walking the FINAL DATASET folders. The
manifest stores paths RELATIVE to dataset_root, so this works unchanged on any
machine/environment as long as dataset_root in the config points at a copy of
the same folder layout (local disk, Kaggle input, Colab, ...).

Train/valid split is taken verbatim from the manifest's "split" column, which
was assigned by folder location (Train/* -> train, valid/* -> valid) - NOT
re-derived or re-shuffled here. No image-level or patient-level resampling is
performed. See scripts/audit_dataset.py for the leakage-safety check.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import PreprocessConfig, preprocess_image, preprocess_mask, empty_mask

try:
    import albumentations as A
    import cv2
    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover - allows dataset import before deps installed
    _HAS_ALBUMENTATIONS = False


def build_train_augmentations(aug_cfg: dict):
    """Conservative, medically-motivated spatial + mild intensity augmentation.

    Spatial transforms (affine) are applied identically to image and mask
    (mask interpolation forced to nearest so it stays binary). Intensity
    transforms (brightness/contrast) apply to the image only - albumentations
    does this automatically based on target type, it never touches the mask.
    """
    if not _HAS_ALBUMENTATIONS or not aug_cfg.get("enabled", True):
        return None

    transforms = [
        A.Affine(
            rotate=(-aug_cfg["rotate_degrees"], aug_cfg["rotate_degrees"]),
            translate_percent={"x": (-aug_cfg["translate_percent"], aug_cfg["translate_percent"]),
                                "y": (-aug_cfg["translate_percent"], aug_cfg["translate_percent"])},
            scale=tuple(aug_cfg["scale_range"]),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg["brightness_limit"],
            contrast_limit=aug_cfg["contrast_limit"],
            p=0.5,
        ),
    ]
    if aug_cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))
    elastic_cfg = aug_cfg.get("elastic_deform", {})
    if elastic_cfg.get("enabled", False):
        # Deliberately mild (small alpha/large sigma = gentle warp) - targets exact-shape
        # memorization (see improving_model.md Tier 3.4) without producing anatomically
        # unrealistic lesion shapes. Mask interpolation forced to nearest, same as Affine.
        transforms.append(A.ElasticTransform(
            alpha=elastic_cfg.get("alpha", 20),
            sigma=elastic_cfg.get("sigma", 5),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=elastic_cfg.get("p", 0.3),
        ))
    return A.Compose(transforms)


class BCTumorSegDataset(Dataset):
    """One sample = one CT-slice PNG + its binary tumor mask (all-zero if Normal)."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_root: Path,
        split: str,
        preprocess_cfg: PreprocessConfig,
        augment: Optional["A.Compose"] = None,
    ):
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in manifest.")
        self.dataset_root = Path(dataset_root)
        self.preprocess_cfg = preprocess_cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.dataset_root / row["image_path"]
        image = preprocess_image(image_path, self.preprocess_cfg)

        if row["has_mask"] and isinstance(row["mask_path"], str):
            mask_path = self.dataset_root / row["mask_path"]
            mask = preprocess_mask(mask_path, self.preprocess_cfg)
        else:
            mask = empty_mask(self.preprocess_cfg.image_size)

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "label": row["label"],
            "patient_id": row["patient_id"],
            "image_path": str(image_path),
        }


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    # is_corrupt is written as the Python bool True/False by build_manifest.py;
    # pandas reads it back as an actual bool dtype, but guard against str "False".
    if df["is_corrupt"].dtype == object:
        df["is_corrupt"] = df["is_corrupt"].astype(str).str.lower() == "true"
    n_corrupt = int(df["is_corrupt"].sum())
    if n_corrupt > 0:
        raise RuntimeError(
            f"Manifest lists {n_corrupt} corrupt image(s). Re-run scripts/audit_dataset.py "
            f"and resolve before training - refusing to silently skip them."
        )
    return df
