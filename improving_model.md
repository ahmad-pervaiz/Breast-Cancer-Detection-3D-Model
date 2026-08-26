# Model Improvement Plan & Changelog

Living document. Each idea below is grounded in evidence from an actual run,
not generic advice. When an idea is tried, it moves into the **Changelog**
at the bottom with the before/after numbers — kept even if it *didn't* help,
so we don't re-try dead ends.

Baseline this plan is written against: **Run #1** (see Changelog), Kaggle T4,
`best_val_dice = 0.7664` @ epoch 12, early-stopped @ epoch 27.
Checkpoint: `runs/segmentation/checkpoints/best_model.pth`.

---

## Diagnosed issues (from `runs/segmentation/logs/training.csv` + prediction montages)

### 1. Epochs 1–7: the model collapsed to predicting all-background
`val_dice` in this window (0.003, ~0.00, 0.285, 0.367, 0.104, 0.367, 0.367)
exactly tracks the *fraction of Normal images in the validation set* (286/780
≈ 36.7%), with `precision`/`recall` at exactly 0.0 for 6 of those 7 epochs.
That means the model predicted **zero tumor pixels on every image**, and our
own "empty-vs-empty scores 1.0" convention (correct, for real evaluation)
made that collapse look like partial credit instead of the total failure it
was. It only breaks out at epoch 8. **7 of the run's 27 epochs (26%) were
wasted in this dead state.**

**Root cause**: tumor pixels are a tiny fraction of each image (min 117px,
median 1588px, out of 65,536px at 256×256 — median tumor is ~2.4% of the
image). Unweighted `BCEWithLogitsLoss` has no reason to risk predicting
positive pixels early in training when "predict nothing" is already a low
per-pixel loss almost everywhere. `dice_weight=1.0`/`bce_weight=1.0` doesn't
counteract this fast enough at initialization.

### 2. Epochs 8–27: real learning, but noisy/oscillating, never cleanly improving past epoch 12
`val_dice` bounces between 0.61 and 0.77 for 19 epochs without a clear upward
trend (see `dice_curve.png` — `train_dice` climbs smoothly, `val_dice`
zigzags). `ReduceLROnPlateau` did drop LR twice (1e-4→5e-5 at epoch 18,
→2.5e-5 at epoch 24) but the oscillation continued after each drop. Early
stopping (patience=15) fired without ever beating epoch 12's peak.

### 3. Consistent false-positive pattern in prediction montages (epochs 10, 25)
Every montage I inspected shows the same shape: the model gets the *main*
lesion right (good shape/location match against ground truth), but also
fires a small spurious blob on the **mirror-opposite side of the chest**, or
in the same anterior chest-wall region on genuinely Normal images. This
looks like the model has learned "a lesion is often somewhere around here
anatomically" as a positional shortcut, rather than fully learning tumor
tissue texture vs. normal tissue in that region — plausible with only 5
distinct training patients providing limited "what normal variation looks
like" diversity.

---

## Improvement plan, prioritized by cost

### Tier 1 — no retraining needed (minutes, run against the existing checkpoint)
| # | Idea | Why (grounded in evidence above) | Status |
|---|---|---|---|
| 1.1 | Enable `postprocess.remove_small_components` (currently off by default), tune `min_component_area_px` **on validation only** | Directly targets the small spurious secondary blobs seen in every montage — the real lesion blob is consistently much larger than the false-positive one | Not tried |
| 1.2 | Sweep classification `threshold` (0.3–0.7) on validation, pick the Dice-maximizing value instead of the fixed 0.5 default | Precision (0.54–0.75) and recall (0.42–0.69) trade off a lot epoch to epoch; the fixed 0.5 threshold is a guess, not a tuned choice | Not tried |

### Tier 2 — cheap retrain (~10-20 min on Kaggle T4)
| # | Idea | Why | Status |
|---|---|---|---|
| 2.1 | Add `pos_weight` to `BCEWithLogitsLoss`, computed from the real train-set pixel-level tumor:background ratio | Directly attacks issue #1 (the 7-epoch dead start) — gives the loss a real incentive to risk positive predictions early instead of collapsing to background | Not tried |
| 2.2 | Add LR warmup (e.g. linear warmup over the first ~200 steps) before the configured LR | Same target as 2.1, from a different angle — avoids a large early update pushing the model into the background-collapse basin | Not tried |
| 2.3 | Switch `ReduceLROnPlateau` → `CosineAnnealingLR` (or increase its patience) | The plateau scheduler reacts to noise it can't distinguish from real stalling; a smooth schedule may reduce the epoch-8-27 oscillation (issue #2) | Not tried |
| 2.4 | Raise `early_stopping.patience` (currently 15) | The run may have stopped before genuinely converging past epoch 12 — several later epochs (20: 0.748, 27: 0.719) came close without beating it | Not tried |

### Tier 3 — more involved (architecture/data)
| # | Idea | Why | Status |
|---|---|---|---|
| 3.1 | Swap the from-scratch U-Net encoder for a pretrained one (e.g. ResNet34 via `segmentation_models_pytorch`) | Only 5 training patients is very little data for learning texture discrimination from scratch (issue #3); ImageNet-pretrained features are a standard, well-evidenced fix for small medical datasets | Not tried |
| 3.2 | Train at full 512×512 instead of downsampled 256×256 | Median tumor is already small (1588px); downsampling to 256 shrinks it further, likely hurting small-lesion recall specifically. Kaggle T4 has memory headroom (batch 16 was comfortable at 256) | Not tried |
| 3.3 | Patient-level k-fold cross-validation across the 5 train + 2 named valid patients | With only 7 named patients total, a single train/valid split's 0.7664 could be somewhat lucky/unlucky in which patients landed in validation. K-fold gives a real confidence interval | Not tried |
| 3.4 | Ablate the current augmentation config (rotate/translate/scale/brightness/contrast) — try training with it off | Currently unknown whether augmentation is helping or contributing to the oscillation in issue #2 — worth an A/B | Not tried |

### Tier 4 — evaluation / infrastructure
| # | Idea | Why | Status |
|---|---|---|---|
| 4.1 | Run `scripts/inference.py` against the held-out `test/` set (image-level detection metrics only — no fake masks) | Would confirm whether the mirror-side false-positive pattern (issue #3) also shows up on genuinely unseen patients, or was specific to this validation set | Not tried |
| 4.2 | Route all future experiment variants through ClearML (already wired, see `WALKTHROUGH.md` §9) | Makes comparing Tier 2/3 variants against this baseline a dashboard lookup instead of manually diffing CSVs | Not tried |

---

## Changelog

### Run #1 — baseline (2026-08-26, Kaggle T4)
- Config: `configs/config.yaml` defaults (image_size=256, batch_size=16 on Kaggle,
  lr=1e-4, dice_weight=bce_weight=1.0, threshold=0.5, no post-processing).
- Result: `best_val_dice = 0.7664` @ epoch 12, `val_iou = 0.6805` at that epoch.
  Early-stopped @ epoch 27 (patience=15, no improvement past epoch 12).
- Detection metrics (image-level, epoch 27): sensitivity 0.957, specificity 0.969,
  accuracy 0.962, F1 0.969.
- Issues diagnosed above (epochs 1-7 collapse, epochs 8-27 oscillation, mirror-side
  false positives). This is the baseline every future entry compares against.
