"""PNG <-> JSON <-> DICOM slice mapping (Project_phase2.txt Sections 7, 15-17).

VERIFIED finding (see runs/phase2_3d/audit/ and the addendum in
Project_phase2.txt): every annotated PNG/JSON stem in the 7 named patient
folders has an exact-filename-stem match against a `.dcm` file in
`TUMOR_Anonymized_DCM/` (e.g. `aa85df1c.png` / `aa85df1c.json` /
`aa85df1c.dcm`), and the counts match exactly per patient. This is Method 1
("filename relationship") from Section 15, confirmed rather than assumed -
so it is used as the primary mapping method, cross-checked by dimension
comparison (Method 3). Methods 2/4/5 were not needed given how clean Method 1
turned out to be, and are not implemented here.
"""
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .dicom_io import DicomSliceInfo, read_dicom_header

logger = logging.getLogger(__name__)

VERIFIED = "VERIFIED"
LIKELY = "LIKELY"
UNCERTAIN = "UNCERTAIN"
FAILED = "FAILED"


@dataclass
class MappingRecord:
    patient_code: str   # anonymized code (e.g. "Patient_05") - never the real dataset folder name
    png_file: str
    json_file: Optional[str]
    dicom_file: Optional[str]
    series_instance_uid: Optional[str]
    sop_instance_uid: Optional[str]
    instance_number: Optional[int]
    z_position: Optional[float]
    png_width: Optional[int]
    png_height: Optional[int]
    dicom_width: Optional[int]
    dicom_height: Optional[int]
    mapping_method: str
    mapping_confidence: str
    notes: str = ""


def map_patient_slices(
    patient_code: str,
    image_dir: Path,
    dicom_dir: Path,
) -> List[MappingRecord]:
    """Builds one MappingRecord per PNG file found in a patient's folder.

    `patient_code` is the anonymized code (e.g. "Patient_05"), used only to
    tag output rows - it is never used to look anything up (that's what
    `image_dir`/`dicom_dir`, resolved from the real folder name upstream in
    config.py, are for). PNG/JSON/DICOM filenames themselves are random
    anonymization hashes (e.g. `aa85df1c`), not patient-identifying.

    Does not consult a DICOM-per-study series filter - the file is looked up
    directly by stem in `dicom_dir` (flat, 1,100 files, globally unique
    stems - verified in the audit). If pydicom cannot open a matched file,
    or dimensions disagree, confidence is downgraded rather than the record
    being dropped, so every PNG is accounted for in the output table.
    """
    records: List[MappingRecord] = []
    png_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")

    for png_path in png_files:
        stem = png_path.stem
        json_path = png_path.with_suffix(".json")
        dicom_path = dicom_dir / f"{stem}.dcm"

        json_file = json_path.name if json_path.exists() else None
        if json_file is None:
            logger.warning("[%s] %s has no matching JSON annotation", patient_code, png_path.name)

        try:
            with Image.open(png_path) as im:
                png_width, png_height = im.size
        except Exception as exc:
            logger.error("[%s] could not open PNG %s: %s", patient_code, png_path.name, exc)
            records.append(MappingRecord(
                patient_code=patient_code, png_file=png_path.name, json_file=json_file,
                dicom_file=None, series_instance_uid=None, sop_instance_uid=None,
                instance_number=None, z_position=None, png_width=None, png_height=None,
                dicom_width=None, dicom_height=None, mapping_method="filename_stem",
                mapping_confidence=FAILED, notes=f"PNG unreadable: {exc}",
            ))
            continue

        if not dicom_path.exists():
            records.append(MappingRecord(
                patient_code=patient_code, png_file=png_path.name, json_file=json_file,
                dicom_file=None, series_instance_uid=None, sop_instance_uid=None,
                instance_number=None, z_position=None, png_width=png_width, png_height=png_height,
                dicom_width=None, dicom_height=None, mapping_method="filename_stem",
                mapping_confidence=FAILED, notes="No DICOM file with matching stem",
            ))
            continue

        try:
            header: DicomSliceInfo = read_dicom_header(dicom_path)
        except Exception as exc:
            logger.error("[%s] could not read DICOM %s: %s", patient_code, dicom_path.name, exc)
            records.append(MappingRecord(
                patient_code=patient_code, png_file=png_path.name, json_file=json_file,
                dicom_file=dicom_path.name, series_instance_uid=None, sop_instance_uid=None,
                instance_number=None, z_position=None, png_width=png_width, png_height=png_height,
                dicom_width=None, dicom_height=None, mapping_method="filename_stem",
                mapping_confidence=FAILED, notes=f"DICOM unreadable: {exc}",
            ))
            continue

        dims_match = (header.columns == png_width) and (header.rows == png_height)
        confidence = VERIFIED if (dims_match and json_file is not None) else LIKELY if dims_match else UNCERTAIN
        notes = "" if dims_match else f"dimension mismatch: PNG={png_width}x{png_height} DICOM={header.columns}x{header.rows}"
        if json_file is None and dims_match:
            notes = (notes + "; " if notes else "") + "no JSON annotation for this slice"

        records.append(MappingRecord(
            patient_code=patient_code,
            png_file=png_path.name,
            json_file=json_file,
            dicom_file=dicom_path.name,
            series_instance_uid=header.series_instance_uid,
            sop_instance_uid=header.sop_instance_uid,
            instance_number=header.instance_number,
            z_position=header.image_position_patient[2] if header.image_position_patient else header.slice_location,
            png_width=png_width,
            png_height=png_height,
            dicom_width=header.columns,
            dicom_height=header.rows,
            mapping_method="filename_stem",
            mapping_confidence=confidence,
            notes=notes,
        ))

    return records


def records_to_dicts(records: List[MappingRecord]) -> List[Dict]:
    return [asdict(r) for r in records]
