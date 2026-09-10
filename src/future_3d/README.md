# Phase 2 — 3D CT tumor reconstruction, visualization & measurement

Full specification: `Project_phase2.txt` (project root) — read Section 0
("VERIFIED FINDINGS ADDENDUM") first; it corrects the original plan against
the real dataset.

**Patients are referred to only by anonymized code (`Patient_01`...`Patient_07`)
everywhere in this module's outputs, reports, and docs — never by the real
dataset folder name. The code <-> real-folder-name mapping exists in exactly
one place, `configs/phase2_3d.yaml`, which is used only to locate source
files on disk and must never itself be copied into a report, image, or
video.**

**Status: Milestones 1-9 done for all 7 patients, all 26 (patient, series)
volumes.** Milestone 3's overlay visual validation is done for `Patient_05`
only (the others rely on the automated per-series fragmentation check below
instead of a full manual overlay review - see Known limitations). Milestones
10 (3D Slicer load-check - files are compatible but not yet opened in Slicer
to confirm), 11 (formal measurements.py with dimensions/diameter), and 12
(Phase-1 AI-prediction mode) are not started.

## Setup (first time on any machine)

`configs/phase2_3d.yaml` is gitignored (this repo is public and that file
maps anonymized codes to real patient folder names). Copy the template and
fill in the real folder names for your own `FINAL DATASET/` copy:
```bash
cp configs/phase2_3d.example.yaml configs/phase2_3d.yaml
# then edit the `id:`/`image_dir:` fields in configs/phase2_3d.yaml
```

## What's here

```
src/future_3d/
├── config.py       Phase2Config loader (configs/phase2_3d.yaml)
├── dicom_io.py       DICOM header reading + physical slice ordering
├── labelme_io.py      Labelme JSON -> binary mask rasterization
├── mapping.py          PNG <-> JSON <-> DICOM matching + confidence rating
├── volume_io.py         CT + tumor-mask NIfTI volume construction, per (patient, series)
└── mesh_io.py             Marching Cubes + PLY/STL export

scripts/
├── phase2_audit_dataset.py     Milestone 1: full dataset/DICOM audit
├── phase2_map_slices.py         Milestone 4: per-patient slice_mapping.csv
├── phase2_inspect_dicom.py       Milestone 2: per-series ordering/spacing report
├── phase2_visual_validation.py    Milestone 3 (Section 14): mandatory mask/overlay images
├── phase2_build_volume.py          Milestones 5-6: ct_volume.nii.gz + tumor_mask.nii.gz
├── phase2_extract_mesh.py           Milestone 7: tumor_mesh.ply / .stl
└── phase2_visualize_3d.py            Milestones 8-9: 3d_tumor.png / 3d_ct_tumor.png
```

Run from `bc_tumor_detection/`, `conda activate bc_seg` first. All `--patient`
flags take the anonymized code; every build/mesh/visualize script also takes
`--all-patients` to run the whole pipeline in one call:
```bash
python scripts/phase2_audit_dataset.py --config configs/phase2_3d.yaml
python scripts/phase2_inspect_dicom.py --config configs/phase2_3d.yaml --patient Patient_05
python scripts/phase2_map_slices.py --config configs/phase2_3d.yaml --patient Patient_05 --strict-mapping
python scripts/phase2_visual_validation.py --config configs/phase2_3d.yaml --patient Patient_05

python scripts/phase2_build_volume.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_extract_mesh.py --config configs/phase2_3d.yaml --all-patients
python scripts/phase2_visualize_3d.py --config configs/phase2_3d.yaml --all-patients
```

Outputs land under `runs/phase2_3d/` (audit/, mapping/, validation/,
ground_truth/<code>/series_NN/), following the same convention as the
Phase-1 `runs/segmentation/` tree — nothing is ever written into
`FINAL DATASET/`. Each `ground_truth/<code>/series_NN/` directory holds:
`ct_volume.nii.gz`, `tumor_mask.nii.gz`, `tumor_mesh.ply`, `tumor_mesh.stl`,
`3d_tumor.png`, `3d_ct_tumor.png`, `reconstruction_metadata.json`.

## Verified findings (see Project_phase2.txt Section 0 for full detail)

- **Mapping is solved by filename stem alone.** Every PNG/JSON/DICOM triple
  shares one filename stem (e.g. `aa85df1c.png`/`.json`/`.dcm` — this stem is
  an anonymization hash, not a patient identifier). 100% VERIFIED confidence
  across all 1,352 annotated slices, 7 patients. DICOM `PatientID` is blank
  (stripped by anonymization) — identity comes only from the code<->folder
  mapping in `configs/phase2_3d.yaml`.
- **7 patients have DICOM, not 5**: 5 from the original `Train/` list plus 2
  more from `valid/` that an earlier draft of the top-level `README.md`
  incorrectly called "DICOM-less" (now corrected there).
- **Every patient's annotated slices span multiple DICOM series** (2-7 per
  patient) — a "patient" is not one contiguous acquisition:

  | Code | Series count | Slices per series |
  |---|---|---|
  | Patient_01 | 4 | 9, 91, 93, 89 |
  | Patient_02 | 4 | 19, 10, 75, 75 |
  | Patient_03 | 5 | 85, 6, 8, 63, 78 |
  | Patient_04 | 3 | 31, 32, 34 |
  | **Patient_05** | **1** | **60** |
  | Patient_06 | 2 | 39, 39 |
  | Patient_07 | 7 | 20, 82, 11, 12, 3, 2, 34 |

  `Patient_05` is the only single-series case and is `first_patient` in
  `configs/phase2_3d.yaml` for that reason.
- **Only one Labelme label exists: `"Tumor"`**, all `polygon` shapes. 0 empty
  masks, 0 invalid JSON, 0 dimension mismatches across every annotated slice.
- **Z-spacing is not always uniform even within one series** — gaps up to
  10x the median step were found (e.g. `Patient_05`: median 0.8mm, but a few
  2.4mm/8.0mm gaps). Volume-building code must check this per series rather
  than assuming `SliceThickness` is the true step.
- **Visual validation (mandatory, Section 14) passed for `Patient_05`**: the
  rasterized tumor mask overlays the same anatomical region across
  consecutive slices, forming one coherent shape (not fragmented/jumping) —
  see `runs/phase2_3d/validation/Patient_05/overlays/`.
- **All 26 (patient, series) volumes built successfully** (Milestones 5-6),
  zero failures. `Patient_05`'s mask_voxels (188,408) exactly matches the
  independently-computed 2D audit's `total_tumor_pixels` sum - the 2D→3D
  stacking loses/duplicates nothing.
- **Fragmentation is checked automatically, not just eyeballed (Section 44).**
  Every built mask is 3D-connected-component-labeled; a split is auto-checked
  against the known irregular z-spacing gaps. 22/26 series are one single
  connected component; 2 more are multi-component but *every* split lands
  exactly on a missing-slice gap (explained); only 3/26 series have any
  split NOT explained by a gap, and even there the unexplained fragment is
  small (largest single component still 87.9-97.2%+ of the mask). See each
  series' `reconstruction_metadata.json` (`num_connected_components`,
  `unexplained_component_splits_z_index`).
- Two `Patient_07` series (`series_06`, `series_07`) have only 2-3 slices at
  14-29mm nominal spacing - almost certainly scout/localizer exports, not
  full diagnostic stacks. Their volumes were built (nothing crashes), but
  their `tumor_volume_cm3` should NOT be treated as a reliable physical
  measurement (Section 32) given how coarsely they're sampled.

## Next milestone (not started)

Milestone 10: open a saved CT+mask pair in 3D Slicer to confirm axial/
sagittal/coronal/3D alignment independently of this module's own rendering
(the files are NIfTI with correct spacing/origin/direction, so they *should*
load correctly, but this hasn't been done). Then Milestone 11 (a dedicated
`measurements.py` with documented X/Y/Z extent and max-diameter definitions,
per Section 33's terminology requirement - `tumor_volume_cm3` alone is
already saved per series but is not the full Milestone 11 deliverable) and
Milestone 12 (Phase-1 AI-prediction mode, replacing Labelme JSON with the
2D model's predicted masks through the same downstream pipeline).
