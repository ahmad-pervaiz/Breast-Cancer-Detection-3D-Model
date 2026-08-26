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
| 1.1 | Enable `postprocess.remove_small_components` (currently off by default), tune `min_component_area_px` **on validation only** | Directly targets the small spurious secondary blobs seen in every montage — the real lesion blob is consistently much larger than the false-positive one | **Done — helped, see changelog** |
| 1.2 | Sweep classification `threshold` (0.3–0.7) on validation, pick the Dice-maximizing value instead of the fixed 0.5 default | Precision (0.54–0.75) and recall (0.42–0.69) trade off a lot epoch to epoch; the fixed 0.5 threshold is a guess, not a tuned choice | **Done — helped, see changelog** |

### Tier 2 — cheap retrain (~10-20 min on Kaggle T4)
| # | Idea | Why | Status |
|---|---|---|---|
| 2.1 | Add `pos_weight` to `BCEWithLogitsLoss` | Directly attacks issue #1 (the 7-epoch dead start) — gives the loss a real incentive to risk positive predictions early instead of collapsing to background | **Tried @ 2.0 — hurt, see changelog. Reverted to off.** |
| 2.2 | Add LR warmup (e.g. linear warmup over the first ~200 steps) before the configured LR | Same target as 2.1, from a different angle — avoids a large early update pushing the model into the background-collapse basin | Not tried |
| 2.3 | Switch `ReduceLROnPlateau` → `CosineAnnealingLR` (or increase its patience) | The plateau scheduler reacts to noise it can't distinguish from real stalling; a smooth schedule may reduce the epoch-8-27 oscillation (issue #2) | Not tried |
| 2.4 | Raise `early_stopping.patience` (currently 15) | The run may have stopped before genuinely converging past epoch 12 — several later epochs (20: 0.748, 27: 0.719) came close without beating it | Not tried |

### Tier 3 — more involved (architecture/data)
| # | Idea | Why | Status |
|---|---|---|---|
| 3.1 | Swap the from-scratch U-Net encoder for a pretrained one (e.g. ResNet34 via `segmentation_models_pytorch`) | Only 5 training patients is very little data for learning texture discrimination from scratch (issue #3); ImageNet-pretrained features are a standard, well-evidenced fix for small medical datasets | Not tried |
| 3.2 | Train at full 512×512 instead of downsampled 256×256 | Median tumor is already small (1588px); downsampling to 256 shrinks it further, likely hurting small-lesion recall specifically. Kaggle T4 has memory headroom (batch 16 was comfortable at 256) | Not tried |
| 3.3 | Patient-level k-fold cross-validation across the 5 train + 2 named valid patients | With only 7 named patients total, a single train/valid split's 0.7664 could be somewhat lucky/unlucky in which patients landed in validation. K-fold gives a real confidence interval | Not tried |
| 3.4 | Boost augmentation (wider rotate/translate/scale, + horizontal flip, + ElasticTransform) and raise `weight_decay` + bottleneck spatial dropout | Run #2 revealed genuine overfitting (train_dice 0.88 vs val_dice 0.70) now that the metric bug is fixed — direct regularization is now the clear priority, ahead of the rest of Tier 3 | **Done — helped a lot. val_dice 0.7914→0.8018, train/val gap 0.18→0.067. See changelog** |

### Tier 4 — evaluation / infrastructure
| # | Idea | Why | Status |
|---|---|---|---|
| 4.1 | Run `scripts/inference.py` against the held-out `test/` set (image-level detection metrics only — no fake masks) | Would confirm whether the mirror-side false-positive pattern (issue #3) also shows up on genuinely unseen patients, or was specific to this validation set | **Done — see changelog. Major finding: one test patient (Amna) is a likely anatomical-level outlier, not a model failure** |
| 4.2 | Route all future experiment variants through ClearML (already wired, see `WALKTHROUGH.md` §9) | Makes comparing Tier 2/3 variants against this baseline a dashboard lookup instead of manually diffing CSVs | Not tried |

---

## Changelog

### Test-set inference (2026-08-26, local, Run #3's best_model.pth, epoch 31)
First real look at genuinely unseen patients (`FINAL DATASET/test/`, 859
images, 7 Tumor-side + 5 Normal-side patients never seen in train/valid).
Image-level detection metrics only, per the no-fake-masks rule (no test
masks exist). Full per-image output: `runs/segmentation/test_predictions/`.

**Headline numbers**: sensitivity 0.978, specificity 0.687, precision 0.742,
accuracy 0.827, F1 0.844 (tp=403, fp=140, fn=9, tn=307).

**Tumor detection generalizes genuinely well**: 94.1%-100% caught per
patient across all 7 Tumor-side test patients (5 of 7 at a perfect 100%,
worst case Rehana 94.1%). This is the core clinical task and it holds up on
real unseen patients.

**Specificity's poor 0.687 is driven almost entirely by one patient.**
Per-patient false-positive rate on the 5 Normal-side test patients:

| Patient | FP rate |
|---|---|
| **Amna** | **96.9%** (93/96) |
| Khalida | 22.3% |
| Nazeer | 18.8% |
| Nusrat | 21.3% |
| Musarrat | **0.0%** (0/128) |

Inspected several of Amna's false-positive predictions directly
(`runs/segmentation/test_predictions/Normal/Amna *_original.png`) against a
correctly-classified Musarrat image. **Amna's scans are visibly a different
anatomical level** - shoulder/clavicle region, no heart or both-lung view -
versus the mid-chest level (heart + both lungs + ribs) that every training
image, every validation image, and every other test patient uses. The model
is firing on shoulder/vascular tissue texture it was never trained to
distinguish from tumor. This looks like a data-scope mismatch in what got
included as "Normal" test data, not a core model failure - flagged for the
user to review the source scans; not something to be fixed by retraining.

**Excluding Amna: specificity = 0.866** (47 FP / 351 Normal images) - a more
representative read of real-world specificity on the task's actual intended
anatomical domain. Still meaningfully below validation's ~0.99, a real and
expected generalization gap (validation patients are more similar to
training patients than a genuinely new patient will be), but a very
different picture than the raw 0.687 suggests.

**Net read**: the model is in substantially better shape than the raw
headline specificity implies. Recommend the user check whether Amna's scans
belong in this test set's scope before drawing conclusions from the
uncorrected number.

### Tier 1 — post-processing + threshold tuning (2026-08-26, local, no retraining)
Ran `scripts/tune_postprocessing.py` against Run #1's `best_model.pth`: one
forward pass per validation image (780 images, cached), then swept 9
thresholds × 9 `min_component_area_px` cutoffs (81 combos) purely on
validation. Full sweep: `runs/segmentation/postprocessing_tuning.csv`.

| | threshold | min_component_area_px | val_dice | val_iou | tumor+dice | sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| Baseline | 0.50 | 0 (off) | 0.7664 | 0.6805 | 0.6656 | 0.972 | 0.951 |
| **Winner** | **0.60** | **100px** | **0.7914** | **0.7127** | **0.6707** | 0.943 | **1.000** |

**val_dice +0.025, val_iou +0.032, specificity reached a perfect 1.000** —
every false positive on a Normal validation image was eliminated by the
100px component filter, confirming the "mirror-side spurious blob" pattern
diagnosed above really was small, filterable noise rather than a deeper
model problem. Small trade-off: sensitivity dropped 0.972→0.943 (a couple
more missed true positives) — worth watching on the test set (task 4.1).

Applied automatically to `configs/config.yaml`: `threshold: 0.6`,
`postprocess.remove_small_components: true`, `postprocess.min_component_area_px: 100`.
This is now the default for all future inference and for Tier 2's retrain.

### Run #2 — pos_weight=2.0 (2026-08-26, Kaggle T4) — regressed, reverted
- Change from Run #1: `BCEWithLogitsLoss(pos_weight=2.0)` (local experiment,
  not yet in shared code at the time - formalized into `losses.py` afterward).
- Result: `best_val_dice = 0.6962` @ epoch 12 (**-7.0%** vs Run #1's 0.7664),
  `val_iou = 0.6029` (**-7.8%**). Precision 0.7175→0.5658 (**-15.2%**) for
  only +2.4% recall (0.6944→0.7185) — a bad trade, net loss on both Dice and
  IoU. Detection accuracy 96.41%→94.48%.
- **First real look at train/val gap since the metric fix**: `train_dice`
  reached 0.8815 while `val_dice` peaked at 0.6962 then dropped to 0.54-0.62
  — genuine overfitting, invisible before the train_dice computation was
  fixed (see the earlier fix commit). With only 5 training patients, this is
  a real and expected risk, not a fluke.
- **Decision**: revert `pos_weight` to off (null). The mechanism worked
  exactly as theorized (more willingness to predict positive pixels → more
  recall) but the specific value tried gave up too much precision for too
  little gain. Left available as a real config option (`pos_weight` in
  `configs/config.yaml`, wired through `losses.py`) for a milder value later
  if the overfitting-focused round below doesn't fully resolve issue #1.

### Overfitting-attack round (2026-08-26, config pushed, retrain pending)
Direct response to Run #2's overfitting finding — four changes at once,
all reasoned through in this session:
- `weight_decay`: 1e-5 → **1e-2** (direct L2 regularization)
- New `model.bottleneck_dropout_p: 0.15` (`nn.Dropout2d`, deepest U-Net layer
  only — see `src/models/unet.py`)
- Augmentation boosted: rotate ±10°→±15°, translate 5%→10%, scale ±5%→±10%,
  **horizontal_flip enabled** (reasoned: Run #1's mirror-side false-positive
  pattern suggests a positional shortcut; training on left/right flips
  directly counters that, and breast/chest-wall anatomy is mirror-symmetric
  enough for this to be valid), **new ElasticTransform** (alpha=20, sigma=5,
  p=0.3 — mild warp against exact-shape memorization)
- `pos_weight` confirmed reverted to null (see Run #2 above)
- Also fixed a related gap while making these changes: `postprocess.
  remove_small_components` (Tier 1's tuned result) was only ever applied in
  `scripts/inference.py`, never during training's own validation loop - so
  `best_model.pth` selection didn't reflect what real inference achieves.
  Now applied consistently in both `_train_one_epoch` and `_validate_one_epoch`.
- **Result: done.** 46 epochs (early-stopped, patience=15 from best epoch 31).
  `best_val_dice = 0.8018` @ epoch 31, `val_iou = 0.7126` — beats Tier 1's
  post-processed baseline (0.7914/0.7127; both numbers now include the same
  threshold=0.6/100px-filter post-processing, so this is apples-to-apples).
  At the best epoch: precision 0.7061, recall 0.7475, sensitivity 0.9838,
  specificity 0.9930.
- **Overfitting: substantially better.** Train/val gap at the best epoch is
  only **0.067** (train_dice 0.869 vs val_dice 0.802) — down from Run #2's
  0.18 gap (0.88 vs 0.70). The regularization bundle (weight_decay, dropout,
  augmentation) did its job. The gap does creep back up over the 15
  post-peak epochs (train_dice climbs to 0.886 by epoch 46 while val_dice
  drifts in a 0.74-0.79 band) — early stopping is correctly catching this
  rather than a longer run helping further.
- **Issue #1 (epochs 1-10 dead start) is still present, unresolved** — this
  round deliberately didn't touch it (pos_weight, its fix, was reverted after
  Run #2). `val_dice` sits flat at exactly the Normal-image-fraction (0.367)
  through epoch ~10 with sensitivity=0/specificity=1.0, then breaks out
  sharply around epoch 11-12 — same signature as Run #1, just ~3 epochs
  shorter this time (10 vs 7... actually slightly longer). A milder
  `pos_weight` (e.g. 1.2-1.3, well short of 2.0) or LR warmup (Tier 2.2,
  still untried) remains the right fix to try next, now that it can be
  layered onto a version of the model that no longer overfits as badly.
- **Issue #3 (mirror-side/positional false positives): mixed picture.**
  Tumor-positive predictions (epoch 45 montage) no longer show the
  contralateral spurious blob seen in Run #1 - clean single-lesion
  predictions matching ground truth shape well. But Normal-image false
  positives didn't disappear - one epoch-45 sample shows two fairly
  substantial bilateral blobs (large enough to survive the 100px filter).
  Note this montage is from epoch 45 (a late post-peak checkpoint saved for
  visual tracking), not the actual best_model.pth from epoch 31 - the
  deployed checkpoint's specificity (0.993) suggests this is not
  representative of production behavior, but it's a real caveat until
  confirmed by test-set inference (task 4.1, still pending).
- Superseded checkpoint from the Tier-1-tuned run (best_val_dice=0.7914)
  archived at `runs/segmentation_run1_archive/` rather than deleted.

### Run #1 — baseline (2026-08-26, Kaggle T4)
- Config: `configs/config.yaml` defaults (image_size=256, batch_size=16 on Kaggle,
  lr=1e-4, dice_weight=bce_weight=1.0, threshold=0.5, no post-processing).
- Result: `best_val_dice = 0.7664` @ epoch 12, `val_iou = 0.6805` at that epoch.
  Early-stopped @ epoch 27 (patience=15, no improvement past epoch 12).
- Detection metrics (image-level, epoch 27): sensitivity 0.957, specificity 0.969,
  accuracy 0.962, F1 0.969.
- Issues diagnosed above (epochs 1-7 collapse, epochs 8-27 oscillation, mirror-side
  false positives). This is the baseline every future entry compares against.
