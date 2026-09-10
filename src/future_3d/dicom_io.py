"""DICOM inspection and slice-ordering helpers (Project_phase2.txt Sections 8-10, 39-40).

Uses pydicom directly for per-file metadata inspection (headers only, no pixel
data, for speed - `stop_before_pixels=True`). SimpleITK is reserved for actual
volume construction (Milestone 5+) where it is the better tool for preserving
spacing/origin/direction; it is not needed for the audit/mapping milestones
this module currently implements.

VERIFIED (see runs/phase2_3d/audit/): the `patient_id` DICOM tag is blank on
every one of the 1,100 anonymized files (anonymization stripped it). Patient
identity is therefore established via the filename-stem join against the
PNG/JSON folders (see `mapping.py`), never via the DICOM PatientID tag.
"""
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom

logger = logging.getLogger(__name__)


@dataclass
class DicomSliceInfo:
    file_path: Path
    sop_instance_uid: Optional[str]
    study_instance_uid: Optional[str]
    series_instance_uid: Optional[str]
    instance_number: Optional[int]
    slice_location: Optional[float]
    image_position_patient: Optional[Tuple[float, float, float]]
    image_orientation_patient: Optional[Tuple[float, ...]]
    rows: Optional[int]
    columns: Optional[int]
    pixel_spacing: Optional[Tuple[float, float]]
    slice_thickness: Optional[float]
    rescale_slope: Optional[float]
    rescale_intercept: Optional[float]
    z_projected: Optional[float] = None   # filled in by order_slices()


def load_dicom_inventory(csv_path: Path) -> Dict[str, dict]:
    """Loads dicom_inventory.csv, keyed by file stem (no extension)."""
    inventory: Dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["file_name"]).stem
            inventory[stem] = row
    return inventory


def read_dicom_header(path: Path) -> DicomSliceInfo:
    """Reads DICOM tags needed for ordering/geometry, without decoding pixel data."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)

    def _get(tag, cast=None):
        val = getattr(ds, tag, None)
        if val is None:
            return None
        if cast is None:
            return val
        try:
            return cast(val)
        except (TypeError, ValueError):
            return None

    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    ps = getattr(ds, "PixelSpacing", None)

    return DicomSliceInfo(
        file_path=path,
        sop_instance_uid=_get("SOPInstanceUID", str),
        study_instance_uid=_get("StudyInstanceUID", str),
        series_instance_uid=_get("SeriesInstanceUID", str),
        instance_number=_get("InstanceNumber", int),
        slice_location=_get("SliceLocation", float),
        image_position_patient=tuple(float(v) for v in ipp) if ipp is not None else None,
        image_orientation_patient=tuple(float(v) for v in iop) if iop is not None else None,
        rows=_get("Rows", int),
        columns=_get("Columns", int),
        pixel_spacing=tuple(float(v) for v in ps) if ps is not None else None,
        slice_thickness=_get("SliceThickness", float),
        rescale_slope=_get("RescaleSlope", float),
        rescale_intercept=_get("RescaleIntercept", float),
    )


def _slice_normal(iop: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Cross product of the row/column direction cosines -> slice-stack normal."""
    r = iop[0:3]
    c = iop[3:6]
    return (
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    )


def order_slices(slices: List[DicomSliceInfo]) -> Tuple[List[DicomSliceInfo], List[str]]:
    """Orders slices along the true anatomical (physical) axis.

    Per Project_phase2.txt Section 10: prefer ImagePositionPatient projected
    onto the ImageOrientationPatient-derived normal over InstanceNumber, which
    is only supporting information. Falls back to SliceLocation, then
    InstanceNumber, and reports which method was actually used plus any
    spacing irregularities found (never silently assumes uniform spacing).
    """
    warnings: List[str] = []

    have_ipp_iop = all(s.image_position_patient and s.image_orientation_patient for s in slices)
    if have_ipp_iop:
        normal = _slice_normal(slices[0].image_orientation_patient)
        for s in slices:
            s.z_projected = sum(a * b for a, b in zip(s.image_position_patient, normal))
        method = "ImagePositionPatient_projected"
    elif all(s.slice_location is not None for s in slices):
        for s in slices:
            s.z_projected = s.slice_location
        method = "SliceLocation"
        warnings.append("ImagePositionPatient/ImageOrientationPatient unavailable for all slices; used SliceLocation")
    elif all(s.instance_number is not None for s in slices):
        for s in slices:
            s.z_projected = float(s.instance_number)
        method = "InstanceNumber_fallback"
        warnings.append("No spatial position tags available; fell back to InstanceNumber (NOT a verified physical order)")
    else:
        return slices, ["FAILED: no ordering information available (no IPP, no SliceLocation, no InstanceNumber)"]

    ordered = sorted(slices, key=lambda s: s.z_projected)

    zs = [s.z_projected for s in ordered]
    diffs = [round(zs[i + 1] - zs[i], 4) for i in range(len(zs) - 1)]
    nonpositive = [d for d in diffs if d <= 0]
    if nonpositive:
        warnings.append(f"{len(nonpositive)} non-increasing z-step(s) after sorting - possible duplicate slice positions")

    unique_diffs = sorted(set(diffs))
    if len(unique_diffs) > 1:
        median = sorted(diffs)[len(diffs) // 2] if diffs else 0.0
        irregular = [d for d in diffs if median and abs(d - median) > 0.01]
        if irregular:
            warnings.append(
                f"Irregular slice spacing: median step={median}mm, "
                f"{len(irregular)}/{len(diffs)} steps deviate (values seen: {unique_diffs}). "
                "Volume is NOT uniformly spaced - document/handle before treating as a regular grid."
            )

    logger.info("Ordered %d slices using method=%s (%d warning(s))", len(ordered), method, len(warnings))
    return ordered, [f"ordering_method={method}"] + warnings
