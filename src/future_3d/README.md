# Future 3D module (placeholder — not implemented)

This directory is reserved for the future phase of this project:

```
DICOM series -> slice ordering -> 2D model segmentation -> 3D mask
    -> physical voxel dimensions -> tumor volume -> tumor dimensions
```

**Out of scope for the current 2D segmentation phase.** Nothing here is
implemented yet, deliberately — see the top-level README's "Future 3D plan"
section for the intended design.

Inputs already available for when this phase starts:
- `FINAL DATASET/TUMOR_Anonymized_DCM/` — 1,100 DICOM files for the 7
  patients that have both DICOM and 2D annotated PNG/mask data
  (`P1-Shukran-S4`, `P2-Mumtaz-S2`, `P2-Parveen-S4`, `P3-Zareena-S2`,
  `P5-Kousar-S4`, `P1-Ruqayya`, `P3-Rabia-S4`).
- `dicom_inventory.csv` (project root) — per-file DICOM metadata: patient ID,
  study/series UIDs, rows/columns, pixel spacing, slice thickness.

Planned steps (not started):
1. Match each DICOM slice to its corresponding PNG/mask by filename stem.
2. Order slices per series using `SeriesInstanceUID` + slice position/instance number.
3. Stack the 2D model's predicted masks into a 3D volume per series.
4. Use `pixel_spacing_row/col` and `slice_thickness` to convert voxel counts
   to physical volume (mm³) and tumor extent (max diameter, mm).
5. 3D visualization / mesh export, prediction video.

Do not add code here until the 2D pipeline (`../models`, `../training`,
`../inference`) is validated and finalized.
