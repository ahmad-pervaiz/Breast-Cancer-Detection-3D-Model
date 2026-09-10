"""3D CT + tumor-mask volume construction (Milestones 5-6, Sections 19-21).

Builds one volume PER (patient, series) - see Project_phase2.txt Section 0
item 5: a "patient" is not one contiguous acquisition, so a whole-patient
volume would silently merge physically distinct series.

Deliberately does NOT use `sitk.ImageSeriesReader` end-to-end: that reader
infers z-spacing from file order without surfacing whether the underlying
positions are actually uniform, and Section 0 item 6 found several series
have irregular z-spacing (some gaps 3-10x the median step). Pixel data is
read directly (rescale slope/intercept applied explicitly), and geometry
(spacing/origin/direction) is set explicitly from the same verified
`order_slices()` ordering already used for mapping/inspection, with the
irregularity captured in the returned result rather than silently absorbed.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

from .dicom_io import DicomSliceInfo

logger = logging.getLogger(__name__)


@dataclass
class SeriesVolumeResult:
    patient_code: str
    series_index: int              # 1-based, ordered by descending slice count
    series_uid: str
    ct_image: sitk.Image
    mask_image: sitk.Image
    num_slices: int
    size_xyz: Tuple[int, int, int]
    spacing_xyz: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]
    ordering_method: str
    nominal_z_spacing_mm: float
    irregular_gap_count: int
    gap_values_mm: List[float]
    slices_missing_json: int
    tumor_voxel_count: int
    voxel_volume_mm3: float
    tumor_volume_mm3: float
    num_connected_components: int
    component_voxel_sizes: List[int]
    unexplained_component_splits: List[int]   # z-indices of boundaries with a component change NOT at an irregular gap
    warnings: List[str] = field(default_factory=list)


def _analyze_fragmentation(mask_array: np.ndarray, diffs: List[float], nominal_z: float) -> dict:
    """Section 25/44: a fragmented mesh must be explained, not waved through.

    Labels 3D-connected components (26-connectivity) in the mask and checks
    every slice boundary where the dominant component changes against the
    z-spacing gap list - a split that lands exactly on an irregular (missing-
    slice) gap is an explained, documented data-completeness limit; a split
    at a normal-spacing boundary is NOT explained and should be looked at.
    """
    if mask_array.sum() == 0:
        return {"num_components": 0, "component_sizes": [], "unexplained_splits": []}

    labeled, n = ndimage.label(mask_array, structure=np.ones((3, 3, 3)))
    sizes = sorted((int(s) for s in ndimage.sum(mask_array, labeled, range(1, n + 1))), reverse=True)

    def _dominant_component(z: int) -> int:
        vals, counts = np.unique(labeled[z][mask_array[z] > 0], return_counts=True)
        return int(vals[np.argmax(counts)]) if len(vals) else 0

    unexplained = []
    for i, d in enumerate(diffs):
        if mask_array[i].sum() == 0 or mask_array[i + 1].sum() == 0:
            continue
        if _dominant_component(i) != _dominant_component(i + 1):
            is_gap = nominal_z and abs(d - nominal_z) > 0.01
            if not is_gap:
                unexplained.append(i)

    return {"num_components": n, "component_sizes": sizes, "unexplained_splits": unexplained}


def _read_hu_slice(header: DicomSliceInfo) -> np.ndarray:
    """Reads one DICOM's pixel array and applies RescaleSlope/Intercept explicitly."""
    ds = pydicom.dcmread(str(header.file_path))
    pixels = ds.pixel_array
    slope = header.rescale_slope if header.rescale_slope is not None else 1.0
    intercept = header.rescale_intercept if header.rescale_intercept is not None else 0.0
    if slope == 1.0 and intercept == 0.0:
        return pixels  # no-op, keeps native dtype (int16 here - see Section 0 item 8 uncertainty)
    return (pixels.astype(np.float32) * slope) + intercept


def _direction_from_iop(iop: Tuple[float, ...]) -> Tuple[float, ...]:
    """Row cosines + column cosines + slice-normal (cross product) -> flat 9-tuple,
    the standard DICOM-to-ITK direction matrix convention."""
    row = np.array(iop[0:3], dtype=np.float64)
    col = np.array(iop[3:6], dtype=np.float64)
    normal = np.cross(row, col)
    return tuple(row) + tuple(col) + tuple(normal)


def build_series_volume(
    patient_code: str,
    series_uid: str,
    series_index: int,
    ordered_headers: List[DicomSliceInfo],
    ordering_method: str,
    ordering_warnings: List[str],
    stem_to_json: Dict[str, Optional[Path]],
    tumor_labels: List[str],
) -> SeriesVolumeResult:
    """Builds the CT volume and the geometrically-matched tumor-mask volume for
    one already-ordered series. `ordered_headers` must be in the final physical
    slice order (see dicom_io.order_slices) and share one Rows/Columns/PixelSpacing."""
    from .labelme_io import labelme_to_mask  # local import: avoids a hard PIL dependency at module load

    n = len(ordered_headers)
    rows = ordered_headers[0].rows
    cols = ordered_headers[0].columns
    if any(h.rows != rows or h.columns != cols for h in ordered_headers):
        raise ValueError(f"[{patient_code}] series {series_uid}: inconsistent Rows/Columns within series - cannot build one volume")

    pixel_spacing = ordered_headers[0].pixel_spacing or (1.0, 1.0)
    iop = ordered_headers[0].image_orientation_patient
    origin = ordered_headers[0].image_position_patient or (0.0, 0.0, 0.0)

    zs = [h.z_projected for h in ordered_headers]
    diffs = [round(zs[i + 1] - zs[i], 4) for i in range(n - 1)] if n > 1 else []
    nominal_z = sorted(diffs)[len(diffs) // 2] if diffs else (ordered_headers[0].slice_thickness or 1.0)
    gap_values = sorted({d for d in diffs if nominal_z and abs(d - nominal_z) > 0.01})
    irregular_gap_count = sum(1 for d in diffs if nominal_z and abs(d - nominal_z) > 0.01)

    warnings = list(ordering_warnings)
    if irregular_gap_count:
        warnings.append(
            f"{irregular_gap_count}/{len(diffs)} z-step(s) deviate from the nominal {nominal_z}mm spacing "
            f"(values: {gap_values}mm) - volume built with a single nominal spacing; physical distance at "
            "these gaps is understated in the regular-grid representation (no slices fabricated to fill them)."
        )

    # spacing_xyz: DICOM PixelSpacing = [row_spacing(Y), col_spacing(X)]; SimpleITK spacing = (X, Y, Z)
    spacing_xyz = (float(pixel_spacing[1]), float(pixel_spacing[0]), float(nominal_z))
    direction = _direction_from_iop(iop) if iop else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if iop is None:
        warnings.append("No ImageOrientationPatient - assumed identity direction (axial, unverified)")

    logger.info(
        "[%s] series_%02d: building volume, %d slices, size=%dx%dx%d, spacing=%s mm",
        patient_code, series_index, n, cols, rows, n, spacing_xyz,
    )

    ct_array = np.stack([_read_hu_slice(h) for h in ordered_headers], axis=0)  # (Z, Y, X)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing(spacing_xyz)
    ct_image.SetOrigin(tuple(float(v) for v in origin))
    ct_image.SetDirection(direction)

    mask_slices = []
    slices_missing_json = 0
    for h in ordered_headers:
        json_path = stem_to_json.get(h.file_path.stem)
        if json_path is None:
            mask_slices.append(np.zeros((rows, cols), dtype=np.uint8))
            slices_missing_json += 1
            continue
        result = labelme_to_mask(json_path, rows, cols, tumor_labels)
        mask_slices.append(result.mask)
    if slices_missing_json:
        warnings.append(f"{slices_missing_json}/{n} slice(s) had no matching Labelme JSON - filled with an empty (all-background) mask")

    mask_array = np.stack(mask_slices, axis=0).astype(np.uint8)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.SetSpacing(ct_image.GetSpacing())
    mask_image.SetOrigin(ct_image.GetOrigin())
    mask_image.SetDirection(ct_image.GetDirection())

    voxel_volume_mm3 = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    tumor_voxel_count = int(mask_array.sum())
    tumor_volume_mm3 = tumor_voxel_count * voxel_volume_mm3

    fragmentation = _analyze_fragmentation(mask_array, diffs, nominal_z)
    if fragmentation["num_components"] > 1:
        largest_frac = fragmentation["component_sizes"][0] / tumor_voxel_count if tumor_voxel_count else 0.0
        if fragmentation["unexplained_splits"]:
            warnings.append(
                f"Mask has {fragmentation['num_components']} disconnected 3D components (largest covers "
                f"{largest_frac:.1%} of voxels); {len(fragmentation['unexplained_splits'])} split(s) at "
                f"z-index(es) {fragmentation['unexplained_splits']} occur at NORMAL spacing (not an irregular "
                "gap) - NOT explained by missing slices, worth visual inspection (Section 44)."
            )
        else:
            warnings.append(
                f"Mask has {fragmentation['num_components']} disconnected 3D components (largest covers "
                f"{largest_frac:.1%} of voxels), but every split lands exactly on an irregular z-spacing gap "
                "(missing native slices) - explained by data completeness, not a mapping/geometry error."
            )

    return SeriesVolumeResult(
        patient_code=patient_code,
        series_index=series_index,
        series_uid=series_uid,
        ct_image=ct_image,
        mask_image=mask_image,
        num_slices=n,
        size_xyz=(cols, rows, n),
        spacing_xyz=spacing_xyz,
        origin_xyz=tuple(float(v) for v in origin),
        direction=direction,
        ordering_method=ordering_method,
        nominal_z_spacing_mm=nominal_z,
        irregular_gap_count=irregular_gap_count,
        gap_values_mm=gap_values,
        slices_missing_json=slices_missing_json,
        tumor_voxel_count=tumor_voxel_count,
        voxel_volume_mm3=voxel_volume_mm3,
        tumor_volume_mm3=tumor_volume_mm3,
        num_connected_components=fragmentation["num_components"],
        component_voxel_sizes=fragmentation["component_sizes"],
        unexplained_component_splits=fragmentation["unexplained_splits"],
        warnings=warnings,
    )


def save_series_volume(result: SeriesVolumeResult, out_dir: Path) -> Dict[str, Path]:
    """Saves ct_volume.nii.gz + tumor_mask.nii.gz + reconstruction_metadata.json (Sections 21, 46)."""
    import json as _json

    out_dir.mkdir(parents=True, exist_ok=True)
    ct_path = out_dir / "ct_volume.nii.gz"
    mask_path = out_dir / "tumor_mask.nii.gz"
    sitk.WriteImage(result.ct_image, str(ct_path))
    sitk.WriteImage(result.mask_image, str(mask_path))

    meta = {
        "patient_code": result.patient_code,
        "series_index": result.series_index,
        # SeriesInstanceUID is DICOM metadata, not a patient name - kept for traceability
        # against dicom_inventory.csv, consistent with the rest of this module.
        "series_instance_uid": result.series_uid,
        "num_slices": result.num_slices,
        "size_xyz": list(result.size_xyz),
        "spacing_xyz_mm": list(result.spacing_xyz),
        "origin_xyz_mm": list(result.origin_xyz),
        "direction": list(result.direction),
        "ordering_method": result.ordering_method,
        "nominal_z_spacing_mm": result.nominal_z_spacing_mm,
        "irregular_gap_count": result.irregular_gap_count,
        "gap_values_mm": result.gap_values_mm,
        "slices_missing_json": result.slices_missing_json,
        "mask_voxels": result.tumor_voxel_count,
        "voxel_volume_mm3": result.voxel_volume_mm3,
        "tumor_volume_mm3": result.tumor_volume_mm3,
        "tumor_volume_cm3": result.tumor_volume_mm3 / 1000.0,
        "num_connected_components": result.num_connected_components,
        "component_voxel_sizes": result.component_voxel_sizes,
        "unexplained_component_splits_z_index": result.unexplained_component_splits,
        "warnings": result.warnings,
    }
    meta_path = out_dir / "reconstruction_metadata.json"
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    return {"ct_volume": ct_path, "tumor_mask": mask_path, "metadata": meta_path}
