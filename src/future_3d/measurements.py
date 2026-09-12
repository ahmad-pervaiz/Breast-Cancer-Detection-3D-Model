"""Physical tumor measurements (Milestone 11, Sections 32-33).

Two measurements only, both precisely named per Section 33's explicit
warning against casually calling a bounding-box dimension "the diameter":

- `bounding_box_extent_mm`: axis-aligned extent along the volume's own
  row/column/slice axes (Y/X/Z) - NOT necessarily anatomical
  craniocaudal/AP/transverse axes, since that mapping isn't established here.
- `max_feret_diameter_mm`: the actual maximum Feret diameter - the largest
  Euclidean distance between any two points on the tumor surface - computed
  from the mesh's convex hull (the true maximum always occurs between two
  hull vertices, so this is exact, not an approximation, while avoiding an
  O(n^2) pass over every mesh vertex).

Volume was already computed in volume_io.py (voxel count x voxel volume);
repeated here from the mask array directly so this module works standalone
from a saved tumor_mask.nii.gz.
"""
import logging
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from .mesh_io import TumorMesh, extract_tumor_mesh

logger = logging.getLogger(__name__)


def bounding_box_extent_mm(mask_array: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Dict:
    """mask_array is (Z, Y, X). Returns extent in mm along each array axis,
    labeled by axis role, not assumed anatomical direction."""
    zs, ys, xs = np.where(mask_array > 0)
    if len(zs) == 0:
        raise ValueError("Empty mask - no bounding box to compute")

    sx, sy, sz = spacing_xyz
    # +1 voxel: bounding box extent spans from the first voxel's near edge to
    # the last voxel's far edge, not center-to-center.
    extent_x = (xs.max() - xs.min() + 1) * sx
    extent_y = (ys.max() - ys.min() + 1) * sy
    extent_z = (zs.max() - zs.min() + 1) * sz

    return {
        "extent_x_mm": float(extent_x),   # along image columns
        "extent_y_mm": float(extent_y),   # along image rows
        "extent_z_mm": float(extent_z),   # along the slice-stacking axis
        "voxel_bbox": {
            "x": [int(xs.min()), int(xs.max())],
            "y": [int(ys.min()), int(ys.max())],
            "z": [int(zs.min()), int(zs.max())],
        },
        "note": "Axis-aligned extents along the volume's own row/column/slice "
                "axes - NOT verified to correspond to anatomical "
                "craniocaudal/anteroposterior/transverse directions. This is "
                "a bounding-box extent, not a diameter (Section 33).",
    }


def max_feret_diameter_mm(mesh: TumorMesh) -> Dict:
    """Exact maximum Feret diameter: largest pairwise distance between any two
    points on the tumor surface, found via the mesh's convex hull."""
    verts = mesh.vertices_mm
    if len(verts) < 4:
        raise ValueError("Too few mesh vertices to compute a convex hull")

    hull = ConvexHull(verts)
    hull_points = verts[hull.vertices]
    n = len(hull_points)

    distances = pdist(hull_points)   # condensed pairwise-distance form
    max_dist = float(distances.max())
    pair = _condensed_index_to_pair(int(np.argmax(distances)), n)

    return {
        "max_feret_diameter_mm": max_dist,
        "endpoint_a_mm": hull_points[pair[0]].tolist(),
        "endpoint_b_mm": hull_points[pair[1]].tolist(),
        "num_hull_vertices_considered": n,
        "note": "Maximum Feret diameter: the exact largest distance between "
                "any two points on the tumor surface (via convex hull, not "
                "an approximation). Distinct from the bounding-box extents "
                "above (Section 33).",
    }


def _condensed_index_to_pair(k: int, n: int) -> Tuple[int, int]:
    """Inverse of scipy.spatial.distance.pdist's condensed-matrix indexing."""
    i = 0
    remaining = k
    step = n - 1
    while remaining >= step:
        remaining -= step
        step -= 1
        i += 1
    j = i + 1 + remaining
    return i, j


def largest_connected_component(mask_array: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """26-connectivity, matching volume_io.py's fragmentation check. Returns
    (mask restricted to the largest component, stats about what was dropped)."""
    labeled, n = ndimage.label(mask_array, structure=np.ones((3, 3, 3)))
    total = int(mask_array.sum())
    if n <= 1:
        return mask_array, {"num_components": n, "excluded_voxels": 0, "excluded_fraction": 0.0}

    sizes = ndimage.sum(mask_array, labeled, range(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    restricted = (labeled == largest_label).astype(mask_array.dtype)
    excluded = total - int(restricted.sum())
    return restricted, {
        "num_components": n,
        "excluded_voxels": excluded,
        "excluded_fraction": excluded / total if total else 0.0,
    }


def compute_measurements(
    mask_array: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    direction: Tuple[float, ...],
) -> Dict:
    """Section 44/45: a small disconnected fragment (e.g. an unexplained
    speck flagged by volume_io.py's fragmentation check) must not be allowed
    to silently inflate the reported bounding box / max diameter - that would
    describe the gap between two separate blobs as if it were tumor extent.
    Volume is reported for the WHOLE mask (a legitimate total-burden number);
    bounding box and max Feret diameter are computed on the LARGEST connected
    component only, with what was excluded reported alongside, never hidden.
    """
    voxel_volume_mm3 = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    total_voxel_count = int(mask_array.sum())
    total_volume_mm3 = total_voxel_count * voxel_volume_mm3

    restricted_mask, frag_info = largest_connected_component(mask_array)
    restricted_voxel_count = int(restricted_mask.sum())
    restricted_volume_mm3 = restricted_voxel_count * voxel_volume_mm3

    mesh = extract_tumor_mesh(restricted_mask, spacing_xyz, origin_xyz, direction)

    return {
        "tumor_volume_total_mm3": total_volume_mm3,
        "tumor_volume_total_cm3": total_volume_mm3 / 1000.0,
        "mask_voxels_total": total_voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        "measurement_basis": "largest_connected_component",
        "largest_component_voxels": restricted_voxel_count,
        "largest_component_volume_cm3": restricted_volume_mm3 / 1000.0,
        "num_connected_components": frag_info["num_components"],
        "excluded_fragment_voxels": frag_info["excluded_voxels"],
        "excluded_fragment_fraction": frag_info["excluded_fraction"],
        "bounding_box": bounding_box_extent_mm(restricted_mask, spacing_xyz),
        "max_feret_diameter": max_feret_diameter_mm(mesh),
        "note": "bounding_box and max_feret_diameter are computed on the "
                "largest connected component only (excluded_fragment_voxels/"
                "fraction show what was left out) so a small disconnected "
                "speck cannot inflate the reported size. tumor_volume_total "
                "includes every voxel regardless of fragmentation.",
    }
