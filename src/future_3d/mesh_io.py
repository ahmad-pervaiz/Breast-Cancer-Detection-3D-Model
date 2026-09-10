"""3D surface extraction + mesh export (Milestone 7, Sections 22-23).

Marching Cubes via scikit-image, run on the tumor-mask array with the
volume's actual (possibly anisotropic) voxel spacing - never assumes cubic
voxels. Mesh vertices are then placed in the same physical world coordinate
system as the saved CT/mask NIfTI volumes (origin + direction applied), so
the exported mesh visually aligns with the CT volume when both are loaded
together (e.g. in 3D Slicer, Section 27).

PLY/STL writers are hand-rolled (numpy + stdlib only) rather than adding a
mesh library dependency - Section 54: "only install libraries that are
actually necessary".
"""
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import measure

logger = logging.getLogger(__name__)


@dataclass
class TumorMesh:
    vertices_mm: np.ndarray   # (N, 3) float32, world-space mm (x, y, z)
    faces: np.ndarray          # (M, 3) int32, vertex indices
    normals: np.ndarray         # (N, 3) float32, per-vertex


def extract_tumor_mesh(
    mask_array: np.ndarray,          # (Z, Y, X) uint8/bool
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    direction: Tuple[float, ...],
) -> TumorMesh:
    """Runs marching_cubes on a (Z, Y, X) mask and returns vertices in world mm."""
    if mask_array.sum() == 0:
        raise ValueError("Empty mask - marching cubes needs at least one foreground voxel")

    # skimage.measure.marching_cubes wants array-order spacing (Z, Y, X) to match
    # the array's own axis order (Section 22: must NOT assume isotropic voxels).
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    verts_zyx, faces, normals_zyx, _values = measure.marching_cubes(
        mask_array.astype(np.float32), level=0.5, spacing=spacing_zyx,
    )

    # verts_zyx columns are (z_mm, y_mm, x_mm) offsets from the volume's first
    # voxel - reorder to (x, y, z), then rotate by the direction matrix and
    # translate by the origin to land in the same world space as the NIfTI volumes.
    local_xyz = verts_zyx[:, [2, 1, 0]]
    R = np.array(direction, dtype=np.float64).reshape(3, 3)
    world_xyz = local_xyz @ R.T + np.array(origin_xyz, dtype=np.float64)

    normals_xyz = normals_zyx[:, [2, 1, 0]] @ R.T

    logger.info("Marching cubes: %d vertices, %d faces", len(world_xyz), len(faces))
    return TumorMesh(
        vertices_mm=world_xyz.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals_xyz.astype(np.float32),
    )


def save_ply(mesh: TumorMesh, path: Path) -> None:
    """ASCII PLY (vertices + per-vertex normals + triangle faces)."""
    path = Path(path)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices_mm)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {len(mesh.faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, n in zip(mesh.vertices_mm, mesh.normals):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    logger.info("Wrote %s", path)


def save_stl(mesh: TumorMesh, path: Path) -> None:
    """Binary STL (per-facet normal, computed from triangle winding - STL has no
    shared per-vertex normals)."""
    path = Path(path)
    verts = mesh.vertices_mm
    faces = mesh.faces

    with open(path, "wb") as f:
        header = b"Phase-2 tumor mesh (marching cubes)".ljust(80, b"\x00")
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))
    logger.info("Wrote %s", path)
