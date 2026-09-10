"""Phase-2 configuration loading.

Mirrors the style of `src/common.py` (Phase-1) but is kept fully separate -
Phase-2 has its own config file (`configs/phase2_3d.yaml`) and its own
`PROJECT_ROOT`-relative path resolution, per Project_phase2.txt Section 35
("Do not hard-code important paths throughout Python files").
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PatientSpec:
    patient_id: str    # real dataset folder name - INTERNAL USE ONLY (path resolution).
                        # Never write this into a report, filename, log, or visualization -
                        # use `code` for anything that is saved or displayed.
    code: str           # anonymized identifier (e.g. "Patient_05") - use this everywhere else
    split: str
    image_dir: Path   # absolute path to the patient's PNG+JSON folder


@dataclass
class Phase2Config:
    dataset_root: Path
    dicom_dir: Path
    dicom_inventory_csv: Path
    patients: List[PatientSpec]
    first_patient: str
    tumor_labels: List[str]
    strict_mapping: bool
    output_root: Path
    raw: Dict[str, Any]   # full parsed YAML, for anything not modeled above


def load_phase2_config(config_path: Path) -> Phase2Config:
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    dataset_root = Path(raw["dataset_root"])
    dicom_dir = dataset_root / raw["dicom_dir"]
    dicom_inventory_csv = (dataset_root / raw["dicom_inventory_csv"]).resolve()

    patients = [
        PatientSpec(
            patient_id=p["id"],
            code=p["code"],
            split=p["split"],
            image_dir=dataset_root / p["image_dir"],
        )
        for p in raw["patients"]
    ]

    output_root = Path(raw["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    # first_patient is specified by anonymized code in the config; resolved
    # here but exposed as-is (still a code, never the real id).
    first_patient_code = raw["first_patient"]
    if first_patient_code not in {p.code for p in patients}:
        raise ValueError(f"first_patient={first_patient_code!r} does not match any patient 'code' in {config_path}")

    return Phase2Config(
        dataset_root=dataset_root,
        dicom_dir=dicom_dir,
        dicom_inventory_csv=dicom_inventory_csv,
        patients=patients,
        first_patient=first_patient_code,
        tumor_labels=[label.lower() for label in raw["tumor_labels"]],
        strict_mapping=bool(raw.get("strict_mapping", False)),
        output_root=output_root,
        raw=raw,
    )
