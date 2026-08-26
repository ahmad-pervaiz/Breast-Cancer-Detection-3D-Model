# Breast Tumor CT Segmentation — 2D Research Prototype

> **This is a research/educational prototype, not a clinical diagnostic
> system.** Outputs are AI-estimated tumor regions for research use only —
> they are not a medical diagnosis and must not be used to make clinical
> decisions.

## 1. Project purpose

Given a single CT PNG slice, this project:
1. Predicts whether a tumor is present.
2. Segments the tumor at pixel level (binary mask: 0 = background, 1 = tumor).

This is phase one of a larger plan. A **future phase** (not implemented here,
see [`src/future_3d/README.md`](src/future_3d/README.md)) will use the 1,100
DICOM files already available for 7 of the patients to reconstruct 3D tumor
volume and physical dimensions. That is explicitly out of scope right now —
this repo only ever outputs a 2D pixel mask and a pixel count/percentage.

## 2. Dataset

Source: `FINAL DATASET/` (untouched, read-only from this project's point of view —
nothing here ever writes into it). Layout:

```
FINAL DATASET/
├── Train/           Normal/ (no tumor) + 5 patient folders (PNG + LabelMe JSON)
├── Train_masks/      binary PNG masks, mirrors Train/
├── valid/            Normal/ + 2 patient folders + Tumors/ (other patients, annotated, no DICOM)
├── valid_masks/       binary PNG masks, mirrors valid/
├── test/              Normal/ + Tumors/ — held-out, NO masks, NO DICOM
└── TUMOR_Anonymized_DCM/   1,100 DICOM files for the 7 named patients (future 3D phase only)
```

**Patient-level split (fixed, never re-shuffled by this code):**

| Split | Named patients | Generic bucket(s) |
|---|---|---|
| Train | P1-Shukran-S4, P2-Mumtaz-S2, P2-Parveen-S4, P3-Zareena-S2, P5-Kousar-S4 | Normal |
| Valid | P1-Ruqayya, P3-Rabia-S4 | Normal, Tumors (different, DICOM-less patients) |
| Test | *(unknown patients — held out)* | Normal, Tumors |

No image is ever moved between splits, and no slice from a training patient
ever appears in validation, or vice versa — this is checked automatically
by `scripts/audit_dataset.py` on every run and by `train.py` before training starts.

**Test set has no ground-truth masks.** It is used only for final qualitative
inference and an optional *image-level* (folder-label-derived) detection
metric — never for pixel-level Dice/IoU, never for model selection, threshold
tuning, or hyperparameter tuning.

### Dataset inspection findings (verified, not assumed)

- All images/masks: 512×512, 8-bit, PNG.
- Some images are stored as 3-channel RGB (`Normal/` folders, `valid/Tumors/`,
  `test/`) but **R==G==B in every pixel** — genuinely grayscale content, no
  color information lost by converting to single-channel. Other images
  (named patient folders) are already single-channel (`L` mode).
  → All inputs are converted to grayscale (`in_channels=1`) consistently.
- Masks are strictly binary (`{0, 255}`), confirmed across every mask file.
- No DICOM windowing is applied — the source is already a PNG export, not a
  raw DICOM array, so arbitrary windowing here would have no basis.
- A prior `test_masks/` directory (visible in an old `dataset_tree_structure.txt`
  snapshot) was deleted by the user before this project started — correctly,
  since the test set has no annotation JSONs, so any such masks would have
  been meaningless empty placeholders. This project does not regenerate them.

## 3. Project structure

```
bc_tumor_detection/
├── configs/config.yaml        single source of truth for every parameter
├── data/manifest.csv          audited image/mask/label manifest (see below)
├── src/
│   ├── data/                  preprocessing, PyTorch Dataset, integrity checks
│   ├── models/unet.py         configurable U-Net
│   ├── training/              losses, metrics, training loop
│   ├── inference/predict.py   checkpoint loading + single-image/folder inference
│   ├── visualization/         plots, prediction montages, overlays
│   └── future_3d/              placeholder for the later DICOM/3D phase
├── scripts/                   thin CLI entry points (see Usage below)
└── runs/                       all generated outputs (checkpoints, logs, plots,
                                 predictions) — nothing is ever written into
                                 FINAL DATASET
```

`data/manifest.csv` (built by `data/build_manifest.py`) is the single source
of truth for every image/mask/label pair — one row per image, paths stored
**relative to `dataset_root`** so the same CSV works unchanged on another
machine (e.g. Kaggle) by just pointing `dataset_root` elsewhere.

## 4. Installation

The system Python here is 3.14, which has no PyTorch wheels yet. A dedicated
conda environment (`bc_seg`, Python 3.11) was created for this project:

```bash
conda create -n bc_seg python=3.11
conda activate bc_seg
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

This machine has **no NVIDIA GPU** (Intel UHD 620 integrated graphics only),
so training runs on CPU. The code auto-detects CUDA if run elsewhere (e.g.
Kaggle/Colab) and uses it automatically, including mixed precision.

## 5. Usage

Always `conda activate bc_seg` first, and run from the `bc_tumor_detection/` directory.

**Dataset audit** (integrity checks + stats + visual sanity montage):
```bash
python scripts/audit_dataset.py --config configs/config.yaml
```

**Smoke test** (2 epochs, 8-sample subset — verifies the whole pipeline in
under a minute before committing to a real run):
```bash
python scripts/train.py --config configs/config.yaml --smoke_test
```

**Full training:**
```bash
python scripts/train.py --config configs/config.yaml
```

**Inference on one image:**
```bash
python scripts/inference.py \
    --input "FINAL DATASET/valid/P1-Ruqayya/example.png" \
    --checkpoint runs/segmentation/checkpoints/best_model.pth \
    --output runs/segmentation/predictions/single/
```

**Inference on the held-out test set:**
```bash
python scripts/inference.py \
    --input "FINAL DATASET/test" \
    --checkpoint runs/segmentation/checkpoints/best_model.pth \
    --output runs/segmentation/test_predictions/
```

## 6. Outputs

```
runs/segmentation/
├── checkpoints/{best_model.pth, last_model.pth}
├── config_used.yaml            exact config used for that run
├── logs/training.csv           per-epoch loss/Dice/IoU/precision/recall/LR
├── plots/{loss_curve,dice_curve,iou_curve}.png
├── predictions/validation/epoch_XXX.png   fixed-sample progress montages
└── test_predictions/           per-image original/probability/mask/overlay
                                  + predictions_summary.json
                                  + image_level_detection_metrics.json (test only)
```

## 7. Metrics — how to read them

Segmentation metrics (Dice/IoU/precision/recall/HD95) are reported **two ways**:
- **`tumor_positive_segmentation`** — averaged only over images with a real
  tumor mask. This is the actual segmentation-quality number.
- **`overall_segmentation_including_normal`** — averaged over all images,
  where a correct empty/empty prediction scores Dice=1.0 by definition. Look
  at this only alongside the tumor-positive number — on its own it can look
  inflated simply because Normal images are easy.

**Image-level detection metrics** (sensitivity/specificity/precision/accuracy/F1)
are a *separate* concept: "did the model say tumor-present at all", derived
from a minimum-tumor-pixel-area threshold (`min_tumor_area_px` in config), not
from segmentation overlap. On the test set these are the *only* metrics
available (no masks exist there), and are explicitly labeled as
folder-label-derived, not pixel-level evaluation.

Model selection during training uses **validation Dice only** — never
training Dice, never any test-set signal.

## 8. Limitations

- Small number of independent patients (5 train, 2+unnamed-group valid) —
  watch the overfitting warning the training loop prints (train Dice ≫ val
  Dice). Results should be treated as preliminary until validated on more
  patients.
- CPU-only training on this machine is slow; expect long full-training runs.
  Kaggle/Colab GPU is recommended for real training (same code, same config,
  just point `dataset_root` at the mounted copy).
- No physical (mm/volume) tumor size — that requires DICOM pixel spacing and
  is explicitly deferred to the future 3D phase.
- `valid/Tumors` patients are unknown/unnamed and have no DICOM data — usable
  for 2D validation only, never for the future 3D reconstruction phase.

## 9. Future 3D plan (not implemented)

See [`src/future_3d/README.md`](src/future_3d/README.md). Summary:
`DICOM series → slice ordering → 2D model segmentation → 3D mask → physical
voxel dimensions (from DICOM pixel spacing/slice thickness) → tumor volume →
tumor dimensions → 3D visualization`. This uses the 1,100 DICOM files for the
7 patients that have both DICOM and 2D annotations.

---
*Research prototype. Not a clinical diagnostic system. Do not use for medical
decision-making.*
