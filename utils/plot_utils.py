"""Shared calibrated scale bars and reference intensity normalization."""

import numpy as np
from matplotlib.colors import PowerNorm
from microfilm.microplot import Microimage


def add_scalebar(ax, width: int, pixel_size: float, stride: int = 1):
    """Use the ingestion montage's calibrated, readable 1/2/5 scale bar."""
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("Scale bar requires positive pixel_size_um")
    target = width * pixel_size / 5
    magnitude = 10 ** np.floor(np.log10(target))
    length = max(value for value in (1, 2, 5) if value <= target / magnitude)
    Microimage(images=None, ax=ax).add_scalebar(
        unit="um",
        scalebar_size_in_units=float(length * magnitude),
        scalebar_unit_per_pix=pixel_size * stride,
        scalebar_font_size=10,
    )


def reference_norm(array):
    """Normalize using the supplied array's P1–P99.5 and gamma=0.7."""
    low, high = np.percentile(array, (1, 99.5))
    return PowerNorm(0.7, low, max(high, low + 1e-6), clip=True)
