# kaggle_upload/

Not part of the git repo (gitignored — this whole folder is a local staging
area, not source code). Contains `breast_tumor_dataset.zip`, the file you
upload to Kaggle as a Dataset (see `../WALKTHROUGH.md` step 2).

## Regenerating the zip

Needed if `FINAL DATASET/` or `data/manifest.csv` changes. From the
`bc_tumor_detection/` directory:

```bash
cd ../ && cd "FINAL DATASET"
zip -r -q "../bc_tumor_detection/kaggle_upload/breast_tumor_dataset.zip" \
    Train Train_masks valid valid_masks test -x "valid/*.py"
cd ../bc_tumor_detection
cp data/manifest.csv kaggle_upload/manifest.csv
cd kaggle_upload && zip -q breast_tumor_dataset.zip manifest.csv && rm manifest.csv
```

This excludes `TUMOR_Anonymized_DCM/` (not needed for the 2D phase) and the
3 stray helper scripts under `valid/`. The zip's top-level layout is
`Train/`, `Train_masks/`, `valid/`, `valid_masks/`, `test/`, `manifest.csv` —
so once Kaggle mounts it at `/kaggle/input/<slug>/`, the manifest's paths
(already relative) resolve directly, no editing needed.
