# Phase 2 — 3D Reconstruction Walkthrough

Companion to `WALKTHROUGH.md` (that one is Phase-1/Kaggle training; this one
is the 3D CT reconstruction/visualization/measurement work in
`src/future_3d/` + `scripts/phase2_*.py`). Full spec: `Project_phase2.txt`.

## Is this "done"? Honest status against the original 12-milestone plan

| # | Milestone | Status |
|---|---|---|
| 1 | Dataset/DICOM audit | ✅ Done, all 7 patients |
| 2 | Inspect one patient's DICOM series | ✅ Done, all 7 patients |
| 3 | Labelme JSON → mask + overlay validation | ✅ Parser done for all; overlay images generated + eyeballed for `Patient_05` only |
| 4 | PNG↔DICOM slice mapping table | ✅ Done, 100% VERIFIED confidence, all 7 |
| 5 | Build 3D CT volume | ✅ Done, all 26 (patient, series) volumes |
| 6 | Build 3D tumor-mask volume | ✅ Done, same 26 |
| 7 | Extract 3D tumor mesh (PLY/STL) | ✅ Done, same 26 |
| 8 | Visualize tumor only | ✅ Static PNG for all 26 + real interactive PyVista option |
| 9 | Visualize CT + tumor | ✅ Static 4-panel PNG for all 26 + PyVista volume rendering |
| 10 | 3D Slicer compatibility | ⚠️ **Not verified** — 3D Slicer isn't installed on this machine. The files are standard NIfTI, which Slicer reads natively, but nobody has actually opened one yet |
| 11 | Physical measurements | ✅ Done — volume, bounding box, and true max Feret diameter, all 26 series |
| 12 | Phase-1 AI-prediction mode | ✅ Done — Phase-1's checkpoint run through the identical pipeline, all 26 series, both modes never mixed |

**So: 11 of 12 milestones are done and validated, for all 7 patients, both
ground-truth and AI-prediction modes.** The one gap — Milestone 10 — isn't a
missing feature, it's a missing tool: 3D Slicer needs a human with a GUI to
actually open a file in it, which this session can't do. Everything that can
be verified by running code and inspecting the output has been.

## What "3D visualization" concretely means right now

Three different things exist under that name — worth being precise about
which one you're looking at:

1. **Static PNGs** (`3d_tumor.png`, `3d_ct_tumor.png` in every series folder)
   — matplotlib, no rotation, generated automatically by the pipeline for
   validation.
2. **A real interactive window** — `scripts/phase2_interactive_view.py`
   (PyVista-based): rotate/zoom/pan with the mouse, tumor mesh alone or with
   a semi-transparent CT volume rendering around it. **Needs a real display**
   (run it on your desktop, not a headless SSH session).
3. **3D Slicer** (external, free download) — opens the same `.nii.gz` files,
   gives axial/sagittal/coronal + 3D views together, is the tool the
   original spec asks for as independent validation. Not yet tried on this
   machine (see Milestone 10 above).

## Step 1 — reproduce the pipeline from scratch (if you haven't already)

```bash
cd bc_tumor_detection
conda activate bc_seg

# One-time: configs/phase2_3d.yaml is gitignored (real patient folder names -
# this repo is public). Copy the template and fill in real folder names:
cp configs/phase2_3d.example.yaml configs/phase2_3d.yaml
# edit the `id:` / `image_dir:` fields in configs/phase2_3d.yaml now

python scripts/phase2_audit_dataset.py --config configs/phase2_3d.yaml

# Ground-truth mode (Labelme JSON annotations):
python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_measurements.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --all-patients

# AI-prediction mode (Phase-1's trained model instead of Labelme JSON) -
# identical downstream steps, just add --mode prediction (and --checkpoint
# on the build step):
python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --all-patients \
    --mode prediction --checkpoint runs/segmentation/checkpoints/best_model.pth
python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --all-patients --mode prediction
python scripts/phase2_measurements.py --config configs/phase2_3d.yaml --all-patients --mode prediction
python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --all-patients --mode prediction
```

Every script also takes `--patient Patient_05` (or any code) instead of
`--all-patients` to run just one.

## Step 2 — actually look at the result

**Fastest — the static PNGs already exist**, no extra command needed:
```
runs/phase2_3d/ground_truth/Patient_05/series_01/3d_tumor.png
runs/phase2_3d/ground_truth/Patient_05/series_01/3d_ct_tumor.png
runs/phase2_3d/predictions/Patient_05/series_01/3d_ct_tumor.png    # AI-mode version, same patient
```
Open these like any image file.

**Real interactive 3D (recommended — this is the "wow" one):**
```bash
python scripts/phase2_interactive_view.py \
    --config configs/phase2_3d.yaml --patient Patient_05 --with-ct
```
A window opens: left-drag rotates, scroll zooms, right-drag pans. Close the
window to exit. Drop `--with-ct` to see just the tumor mesh by itself
(cleaner, faster). Add `--mode prediction` to view the AI-predicted
reconstruction instead of ground truth. Pick a different series with
`--series-index N` (see `runs/phase2_3d/ground_truth/<code>/` for what
exists — most patients have several).

If you're on a remote/SSH session with no display, add
`--screenshot out.png` instead — it renders off-screen and saves a picture
rather than opening a window (this is how it was smoke-tested in this
session, since this environment also has no GUI).

**3D Slicer (do this to actually close out Milestone 10):**
1. Download from [slicer.org](https://www.slicer.org) (free).
2. **Add Data** → load both `ct_volume.nii.gz` and `tumor_mask.nii.gz` from
   the same `series_NN` folder.
3. Right-click the mask volume in the Data module → **Convert to Segmentation**
   (or use the **Segmentations** module) so it renders as an overlay/3D
   surface rather than a second grayscale volume.
4. Check the axial/sagittal/coronal views line up, and switch to the 3D view
   to see the tumor + optionally a volume-rendered CT body.

## Ground truth vs. AI prediction — how close is the model in 3D?

Real numbers, not estimates, comparing `measurements.json` across both
modes. `Patient_05` (the single-series, most-validated case):

| | Ground truth | AI-predicted | Difference |
|---|---|---|---|
| Volume | 102.3 cm³ | 119.2 cm³ | +16% (over-predicts) |
| Max Feret diameter | 126.1 mm | 125.4 mm | -0.6% (essentially identical) |

Across all 26 series, the model **consistently over-predicts volume** —
expected, given `improving_model.md`'s own finding that this checkpoint's
sensitivity (97.8%) is well ahead of its specificity (86.6%): it errs toward
calling more pixels tumor. Max diameter agreement is usually close (within
~15%), except the two sparse `Patient_07` scout-like series (2-11 slices at
14-29mm spacing) where the model — trained on ordinary multi-slice CT —
predicts erratically on an atypical export; not a 3D-pipeline problem, a
data-domain-mismatch one, same pattern `improving_model.md` already found
for one 2D test patient (the "Amna" anatomical-outlier finding).

No formal 3D Dice/IoU/Hausdorff comparison was built — Section 31 of the
spec explicitly defers that past the first milestone ("prove both 3D masks
are correctly aligned" first, which the volume/diameter comparison above
and the visual side-by-side do).

## Where everything lives

```
runs/phase2_3d/
├── audit/                          Milestone 1: audit_report.json/csv, PHASE2_FIRST_DELIVERABLE.md
├── mapping/                          Milestone 4: slice_mapping.csv
├── validation/Patient_05/             Milestone 3: original/mask/overlay PNGs (one patient)
├── ground_truth/<code>/series_NN/       Labelme-JSON-based reconstruction, per (patient, series)
└── predictions/<code>/series_NN/         Phase-1-model-based reconstruction, same structure, never mixed with the above

# each series_NN/ (either tree) holds:
    ct_volume.nii.gz                  the CT, spacing/origin/direction preserved
    tumor_mask.nii.gz                   the 3D tumor mask, same geometry
    tumor_mesh.ply / .stl                 the extracted surface mesh
    3d_tumor.png / 3d_ct_tumor.png          static validation renders
    reconstruction_metadata.json             slice count, spacing, tumor volume,
                                               fragmentation check (num_connected_components etc.)
    measurements.json                          volume, bounding box, max Feret diameter
```

`Patient_05`/`series_01` is the most thoroughly checked (personally
inspected slice-by-slice this session, in both modes). All 26 series across
all 7 patients built and rendered without errors in both modes, but only a
few were individually eyeballed — worth a look at `3d_ct_tumor.png` for any
patient you care most about before trusting it fully, per the spec's own
Section 44 rule ("don't declare it correct just because a mesh was generated").

## Known limitations, stated plainly

- Two of `Patient_07`'s series have only 2-3 slices at 14-29mm spacing —
  almost certainly scout/localizer images, not full diagnostic stacks. They
  built without crashing in both modes, but don't trust their volume/diameter
  numbers, and the AI mode performs noticeably worse there specifically.
- A handful of series show a small mask fragment not explained by a known
  slice-spacing gap (see `unexplained_component_splits_z_index` in each
  series' `reconstruction_metadata.json`) — always small relative to the
  main mass, excluded from the Milestone 11 bounding-box/diameter numbers
  (see `excluded_fragment_voxels`/`fraction` in `measurements.json`), never
  hidden.
- `RescaleSlope`/`RescaleIntercept` = 1/0 on every file checked so far,
  meaning pixel values may not be true Hounsfield Units — noted, not fixed
  (doesn't affect geometry/shape, only absolute intensity/windowing).
- Milestone 10 (3D Slicer) genuinely unverified — see above.

## What's actually left

Just Milestone 10, and it needs you: install 3D Slicer and load a
`ct_volume.nii.gz` + `tumor_mask.nii.gz` pair per the instructions above.
Everything else in the original 12-milestone plan has real, inspected,
working code and output behind it.
