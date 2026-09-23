# SAM Box-Prompted Fine-Tune — Walkthrough

Companion to `WALKTHROUGH.md` (that one is the from-scratch U-Net) and
`PHASE2_WALKTHROUGH.md` (3D reconstruction). This one fine-tunes SAM
(Segment Anything)'s mask decoder — see `improving_model.md` Tier 5 for the
full reasoning. Full design plan: this session's plan file, summarized here.

## Honest status right now

| Step | Status |
|---|---|
| Design (what's frozen, box-prompt strategy, metric to compare against) | ✅ Done — see `configs/sam_finetune.yaml` header comment |
| Code (`src/data/sam_dataset.py`, `src/models/sam_finetune.py`, `src/training/train_sam.py`, `scripts/train_sam.py`, `scripts/compare_sam_vs_unet.py`) | ✅ Written, and CPU smoke-tested this session with a random-init checkpoint — confirmed the forward/backward/checkpointing/logging pipeline runs end-to-end without crashing |
| Real training | ❌ **Not done.** A random-init checkpoint produces near-zero Dice by construction — this needs a real pretrained SAM/MedSAM checkpoint, and a real GPU run (this dev machine has no CUDA — SAM's ViT-B encoder is far too slow on CPU for real training, ~50s/epoch on just 2 images in the smoke test) |
| Comparison against the U-Net | ❌ Not done — `scripts/compare_sam_vs_unet.py` is written and smoke-tested (ran correctly on 3 images with the throwaway checkpoint), but has no real numbers yet |

**What's left is entirely "run it for real," not "write more code."** Follow steps 1-5 below.

## 1. Get a SAM checkpoint (one-time, manual)

Two options — pick one:

**A. Vanilla SAM ViT-B** — Meta's original weights, pretrained on natural
images only (not medical). Simplest, fully scriptable:
```bash
mkdir -p checkpoints
wget -O checkpoints/medsam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**B. MedSAM ViT-B (recommended)** — same architecture, already fine-tuned on
medical images with box prompts, so this project's fine-tune becomes
domain-adaptation on top of a much better starting point instead of
adapting from natural images. Its checkpoint is hosted on Google Drive (not
reliably scriptable) — get the download link from
[github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)'s
README, download it by hand, and save it as `checkpoints/medsam_vit_b.pth`
(the path `configs/sam_finetune.yaml` already points at).

Either way, `checkpoints/` is gitignored — this file never gets committed.

## 2. Install the one new dependency

```bash
conda activate bc_seg
pip install segment-anything
```
(Already verified installable — see `requirements.txt`'s "SAM fine-tuning" section.)

## 3. Local smoke test (optional but recommended)

Confirms your checkpoint loads and the pipeline still runs in your actual
environment before spending Kaggle GPU time:
```bash
python scripts/train_sam.py --config configs/sam_finetune.yaml --smoke_test
```
Expect this to take a few minutes on CPU (SAM's encoder is heavy) and to
produce a near-meaningless Dice if you skipped step 1 — that's expected,
it's only checking that nothing crashes.

## 4. Real training on Kaggle

Same pattern as `WALKTHROUGH.md`, new notebook:
1. Upload `kaggle/sam_finetune_kaggle.ipynb` as a new Kaggle Notebook (or
   import it into your existing one).
2. Attach the same `breast_tumor_dataset.zip` Kaggle Dataset you already use
   for U-Net training.
3. (Optional, for the MedSAM path) Upload your downloaded
   `medsam_vit_b.pth` as its own Kaggle Dataset and attach it too — the
   notebook auto-detects any attached `.pth` file with "medsam" in its name
   and uses it; otherwise it downloads vanilla SAM ViT-B itself.
4. Run all cells. Enable GPU (T4 x2 or P100) and Internet in Settings first.
5. Download `sam_segmentation_results.zip` from the Output tab when done.

## 5. Compare against the U-Net baseline

Unzip the Kaggle result locally into `runs/sam_segmentation/`, then:
```bash
python scripts/compare_sam_vs_unet.py \
    --unet_checkpoint runs/segmentation/checkpoints/best_model.pth \
    --sam_checkpoint runs/sam_segmentation/checkpoints/best_model.pth
```
This runs both models on the same tumor-positive validation images and
prints a side-by-side Dice/IoU table, plus saves a comparison montage to
`runs/sam_vs_unet_comparison.png`. On CPU this will be slow over the full
~494-image validation set — add `--limit 20` (or similar) for a quick look,
drop it for the real number.

**Compare against the U-Net's `tumor_positive_dice`/`tumor_positive_iou`**
(currently 0.8018 / 0.7126 in `improving_model.md`), not its "overall
including Normal" numbers — SAM only ever segments tumor-positive images
(see `configs/sam_finetune.yaml`'s header comment for why).

## 6. Record the result

Whatever you find — better, worse, or mixed — replace the "Run #5" entry in
`improving_model.md`'s changelog with the real numbers, same discipline as
every other run there (Run #4 is a good example of writing up a negative
result honestly). If it helps, it's your next CV bullet; if it doesn't, the
changelog entry is exactly the kind of "diagnosed and ruled out, here's why"
detail that already makes this project's documentation stand out.
