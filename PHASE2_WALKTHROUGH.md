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
| 8 | Visualize tumor only | ✅ Static PNG for all 26 + real interactive PyVista option (see below) |
| 9 | Visualize CT + tumor | ✅ Static 4-panel PNG for all 26 + PyVista volume rendering |
| 10 | 3D Slicer compatibility | ⚠️ **Not verified** — 3D Slicer isn't installed on this machine. The files are standard NIfTI, which Slicer reads natively, but nobody has actually opened one yet |
| 11 | Physical measurements | ⚠️ **Partial** — `tumor_volume_cm3` + voxel count saved per series; the dedicated dimensions/max-diameter script (Section 33) is not built |
| 12 | Phase-1 AI-prediction mode | ❌ **Not started** — everything so far uses Labelme ground truth only |

**So: the core reconstruction pipeline (1-9) is real, done, and validated for
all 7 patients — not just claimed.** What's *not* done yet is (a) an
independent confirmation in actual 3D Slicer, (b) a proper measurements
script beyond the basic volume number, and (c) the AI-prediction mode. If
"completed" means the full original plan end-to-end, it isn't — about 3 of
12 milestones remain.

## What "3D visualization" concretely means right now

Three different things exist under that name — worth being precise about
which one you're looking at:

1. **Static PNGs** (`3d_tumor.png`, `3d_ct_tumor.png` in every series folder)
   — matplotlib, no rotation, generated automatically by the pipeline for
   validation. These are what I looked at earlier in this conversation to
   confirm the reconstructions were coherent, not fragmented garbage.
2. **A real interactive window** — `scripts/phase2_interactive_view.py`
   (new, PyVista-based): rotate/zoom/pan with the mouse, tumor mesh alone or
   with a semi-transparent CT volume rendering around it. **Needs a real
   display** (run it on your desktop, not a headless SSH session).
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
python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --all-patients
```

Each of the last three also takes `--patient Patient_05` (or any code) to
run just one patient instead of all 7.

## Step 2 — actually look at the result

**Fastest — the static PNGs already exist**, no extra command needed:
```
runs/phase2_3d/ground_truth/Patient_05/series_01/3d_tumor.png
runs/phase2_3d/ground_truth/Patient_05/series_01/3d_ct_tumor.png
```
Open these like any image file.

**Real interactive 3D (recommended — this is the "wow" one):**
```bash
python scripts/phase2_interactive_view.py \
    --config configs/phase2_3d.yaml --patient Patient_05 --with-ct
```
A window opens: left-drag rotates, scroll zooms, right-drag pans. Close the
window to exit. Drop `--with-ct` to see just the tumor mesh by itself
(cleaner, faster). Pick a different series with `--series-index N` (see
`runs/phase2_3d/ground_truth/<code>/` for what exists — most patients have
several).

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

## Where everything lives

```
runs/phase2_3d/
├── audit/                          Milestone 1: audit_report.json/csv, PHASE2_FIRST_DELIVERABLE.md
├── mapping/                          Milestone 4: slice_mapping.csv
├── validation/Patient_05/             Milestone 3: original/mask/overlay PNGs (one patient)
└── ground_truth/<code>/series_NN/       Milestones 5-9, per (patient, series):
    ├── ct_volume.nii.gz                  the CT, spacing/origin/direction preserved
    ├── tumor_mask.nii.gz                   the 3D tumor mask, same geometry
    ├── tumor_mesh.ply / .stl                 the extracted surface mesh
    ├── 3d_tumor.png / 3d_ct_tumor.png          static validation renders
    └── reconstruction_metadata.json             everything numeric: slice count,
                                                    spacing, tumor volume, and the
                                                    automated fragmentation check
                                                    (num_connected_components etc.)
```

`Patient_05`/`series_01` is the most thoroughly checked (it's the one I
personally inspected slice-by-slice this session). All 26 series across all
7 patients built and rendered without errors, but only a few were
individually eyeballed — worth a look at `3d_ct_tumor.png` for any patient
you care most about before trusting it fully, per the spec's own Section 44
rule ("don't declare it correct just because a mesh was generated").

## Known limitations, stated plainly

- Two of `Patient_07`'s series have only 2-3 slices at 14-29mm spacing —
  almost certainly scout/localizer images, not full diagnostic stacks. They
  built without crashing, but don't trust their `tumor_volume_cm3`.
- A handful of series show a small mask fragment not explained by a known
  slice-spacing gap (see `unexplained_component_splits_z_index` in each
  series' `reconstruction_metadata.json`) — always small relative to the
  main mass, flagged rather than hidden, not yet individually reviewed.
- `RescaleSlope`/`RescaleIntercept` = 1/0 on every file checked so far,
  meaning pixel values may not be true Hounsfield Units — noted, not fixed
  (doesn't affect geometry/shape, only absolute intensity/windowing).

## Suggested next step

Milestone 11 (a real measurements script — X/Y/Z extent, max diameter, with
precise terminology per Section 33) is the natural next piece: it's
self-contained, doesn't touch anything already built, and turns
`tumor_volume_cm3` into the fuller "how big is it, in what direction" answer
a clinician would actually want. Milestone 12 (swap in Phase-1's AI
predictions instead of Labelme ground truth) is the other major remaining
piece, and depends on nothing here changing. Say which one (or both) and
I'll continue.
