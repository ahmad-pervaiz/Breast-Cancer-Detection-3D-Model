"""
Optional prediction post-processing - connected-component filtering.

Off by default (project spec: "default should be conservative/off, because
small tumors may be clinically meaningful"). See improving_model.md Tier 1
for why this was investigated: prediction montages from Run #1 consistently
showed a correct main lesion blob plus a small spurious secondary blob
(often mirror-opposite side of the chest, or on genuinely Normal images) -
exactly the shape a connected-component area filter is suited to clean up,
without touching a large, real lesion.
"""
import numpy as np
from scipy import ndimage


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Zero out any connected foreground component smaller than min_area_px.
    min_area_px <= 0 is a no-op (returns mask unchanged)."""
    if min_area_px <= 0 or not mask.any():
        return mask

    labeled, num_components = ndimage.label(mask)
    if num_components == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_components + 1))
    keep_labels = set(np.where(sizes >= min_area_px)[0] + 1)

    if len(keep_labels) == num_components:
        return mask  # nothing removed
    out = np.isin(labeled, list(keep_labels)).astype(mask.dtype)
    return out
