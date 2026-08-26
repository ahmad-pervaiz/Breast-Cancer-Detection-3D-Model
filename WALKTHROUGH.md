# Kaggle GPU Training — Step-by-Step Walkthrough

This walks through training this project's U-Net on a free Kaggle GPU,
end to end: upload the dataset, run the notebook, get a trained checkpoint
back onto your machine. No prior Kaggle experience assumed.

> This machine has no NVIDIA GPU, so full training here would take ~15 hours
> (measured: ~18 min/epoch × 50 epochs). On a Kaggle T4, measured **~20s/epoch**
> — 50 epochs finishes in ~17 minutes. See `scripts/benchmark_timing.py` if you
> want to re-measure this for yourself on either machine.

---

## 1. Push the code to GitHub (one-time)

If you haven't already:

```bash
cd bc_tumor_detection
gh auth login                # interactive - opens a browser device-code flow
gh repo create ahmad-pervaiz/Breast-Cancer-Detection-3D-Model --public --source=. --remote=origin
git push -u origin main
```

If the repo already exists (e.g. created on github.com first), just:

```bash
git remote add origin https://github.com/ahmad-pervaiz/Breast-Cancer-Detection-3D-Model.git
git push -u origin main
```

**The `FINAL DATASET/` folder is never pushed** — it isn't part of this repo
at all (the repo only contains `bc_tumor_detection/`, and the dataset lives
outside it). The data goes to Kaggle separately, as a Kaggle Dataset (step 2).

---

## 2. Upload the dataset to Kaggle

1. Go to [kaggle.com](https://kaggle.com) → sign in (or create a free account).
2. **Datasets → New Dataset**.
3. Upload `kaggle_upload/breast_tumor_dataset.zip` (342MB — built already;
   regenerate with the zip command in that folder's README if the dataset
   changes). Kaggle unzips it automatically on upload.
4. Title it anything (e.g. "Breast Tumor CT Segmentation Dataset"). Keep it
   **Private** unless you specifically want it public.
5. Click **Create**. Wait for the upload/processing to finish (a progress bar
   shows on the dataset page).
6. Note the dataset's **slug** (visible in its URL, e.g.
   `kaggle.com/datasets/<your-username>/breast-tumor-dataset` → slug is
   `breast-tumor-dataset`). You won't need to type this manually — the
   notebook auto-detects it — but it's useful for sanity-checking.

---

## 3. Create the Kaggle Notebook

**Option A — upload the prepared notebook (recommended):**
1. **Code → New Notebook → File → Import Notebook**.
2. Upload `kaggle/bc_tumor_segmentation_kaggle.ipynb` from this repo.

**Option B — start blank and copy cells:** open a new notebook and paste in
the cells from `kaggle/bc_tumor_segmentation_kaggle.ipynb` (view it on
GitHub or locally) in order.

---

## 4. Configure the notebook environment

In the notebook's right-hand **Settings** panel:

- **Accelerator**: `GPU T4 x2` (or `GPU P100` if offered) — not "None".
- **Internet**: `On` — required, the notebook `git clone`s the repo.
- **Add Data** → search your uploaded dataset by name → **Add**. It will
  appear under `/kaggle/input/<slug>/` once attached.

Without the GPU and Internet toggles, the notebook's cell 1 (environment
check) and cell 2 (git clone) will fail loudly — that's intentional, better
than silently running slow on CPU.

---

## 5. Run the notebook

Run cells top to bottom (**Run All**, or step through individually to watch
each stage):

1. **Environment check** — confirms GPU is attached (`nvidia-smi` output +
   `torch.cuda.is_available() == True`). If this says `False`, go back to
   step 4 — the accelerator setting didn't take.
2. **Clone repo** — pulls the latest code from GitHub into `/kaggle/working/`.
3. **Install deps** — Kaggle's base image already has PyTorch+CUDA, pandas,
   numpy, matplotlib, scipy; this just adds albumentations/opencv/pyyaml.
4. **Locate dataset** — auto-detects `/kaggle/input/<slug>/` by finding the
   folder containing a `Train/` subdirectory. Prints the path it found.
5. **Dataset audit** — re-runs the same integrity checks that passed locally
   (corruption, binary masks, label consistency, patient-leakage assertion).
   Should print "Audit complete - dataset is ready for training." If it
   doesn't, **stop and read the error** — don't proceed to training on a
   flagged dataset.
6. **Smoke test** — 2 epochs, tiny subset, confirms the pipeline actually
   runs on this GPU before committing real time to it. Should take well
   under a minute.
7. **Full training** — the real run. Default is 50 epochs, batch_size 16
   (raised from the CPU default of 4, since the GPU has room). Expect
   roughly 35–60 minutes total; watch the per-epoch printout (loss, Dice,
   IoU, sensitivity/specificity) as it goes. Early stopping may end it
   sooner if validation Dice plateaus (patience=15 epochs by default).
8. **View curves** — loss/Dice/IoU plots and a validation prediction montage,
   rendered inline so you can eyeball progress without leaving the notebook.
9. **Zip results** — packages `runs/segmentation/` (checkpoints, logs, plots,
   prediction montages) into `/kaggle/working/segmentation_results.zip`.

---

## 6. Download the trained model

1. Click **Save Version** (top right) → **Save & Run All** (or just
   **Quick Save** if you already ran everything interactively).
2. Once it finishes, open the notebook's **Output** tab.
3. Download `segmentation_results.zip`.

---

## 7. Bring it back to your local project

```bash
cd bc_tumor_detection
unzip ~/Downloads/segmentation_results.zip -d runs/segmentation
```

This restores `runs/segmentation/checkpoints/best_model.pth` (and
`last_model.pth`, logs, plots) locally, matching the layout the rest of the
project expects. From here, run inference exactly as documented in the main
`README.md`:

```bash
conda activate bc_seg
python scripts/inference.py \
    --input "FINAL DATASET/test" \
    --checkpoint runs/segmentation/checkpoints/best_model.pth \
    --output runs/segmentation/test_predictions/
```

---

## 8. Reading the results — what to trust, what not to

- **`val_dice` in the training log/plots** is what selected `best_model.pth`
  — never `train_dice`, never anything test-set-derived.
- The **test set has no masks** — `test_predictions/` gives you per-image
  masks/overlays and an *optional* image-level detection report (derived
  from the `Normal`/`Tumors` folder names), never a Dice/IoU number. If you
  see a Dice number reported against the test set anywhere, something is
  wrong — it means fake ground truth crept in somewhere, which this project
  explicitly refuses to generate.
- Watch for the **overfitting warning** the training loop prints if training
  Dice pulls far ahead of validation Dice — expected risk given only 5 named
  training patients.

---

## 9. (Optional) ClearML experiment tracking

Logs every epoch's metrics and uploads `best_model.pth`/`last_model.pth`/plots
as artifacts to a ClearML dashboard, so you don't have to dig through
`training.csv` or re-download a zip to compare runs.

**One-time setup:**
1. Sign up free at [app.clear.ml](https://app.clear.ml).
2. **Settings → Workspace → Create new credentials** — copy the config block shown.
3. Locally: `pip install clearml && clearml-init`, paste the block when prompted
   (writes to `~/clearml.conf` — never paste credentials into a chat or commit them).
4. On Kaggle: notebook menu → **Add-ons → Secrets** → add two secrets named
   `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` with the values from
   step 2. The notebook's cell 6.5 picks these up automatically.

**Using it:**
- Locally: `python scripts/train.py --config configs/config.yaml --clearml`
- On Kaggle: the training cell auto-detects the Secrets and adds `--clearml`
  for you — nothing to edit.
- Without either of the above, training runs exactly as before (ClearML is
  fully optional — see `configs/config.yaml`'s `clearml.enabled: false` default).

---

## 10. Iterating

To try a different config (more epochs, different image size, different
loss weights): edit `configs/config.yaml`, commit, push to GitHub, then in
the Kaggle notebook re-run cell 2 (`git clone` — it'll no-op if the folder
exists; delete `/kaggle/working/Breast-Cancer-Detection-3D-Model` first if
you need a hard refresh) followed by the training cells again. Each Kaggle
session gets a fresh `/kaggle/working/`, so a `git pull` inside an existing
clone works too if you're continuing within the same session.

---

*Research prototype. Not a clinical diagnostic system.*
