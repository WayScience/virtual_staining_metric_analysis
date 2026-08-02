"""
Utilities for applying degradation to reference images.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .degrade_spec import (
    DegradationSpec,
    _build_transform,
)


def _derive_page_seed(
    reference_patch_path: str,
    page_index: int,
    base_seed: int,
) -> int:
    """Derive a stable 32-bit seed independent of processing order."""
    payload = f"{base_seed}|{reference_patch_path}|{page_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _validate_reference_image(image: np.ndarray, path: Path) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D reference image at {path}, found {image.shape}.")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Expected a floating-point reference image at {path}.")
    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError(f"Reference image contains non-finite values: {path}")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"Reference image values fall outside [0, 1]: {path}")
    return image


def build_degradation_stack(
    reference_image: np.ndarray,
    reference_patch_path: str,
    specs: Sequence[DegradationSpec],
    base_seed: int = 42,
) -> np.ndarray:
    """Generate all degradation pages for one reference image."""
    image = _validate_reference_image(reference_image, Path(reference_patch_path))
    pages = []
    for spec in specs:
        transform = _build_transform(spec)
        transform.set_random_seed(
            _derive_page_seed(reference_patch_path, spec.page_index, base_seed)
        )
        page = np.asarray(transform(image=image)["image"], dtype=np.float32)
        if page.shape != image.shape:
            raise ValueError(
                f"Page {spec.page_index} changed shape from {image.shape} to {page.shape}."
            )
        if not np.isfinite(page).all() or page.min() < 0.0 or page.max() > 1.0:
            raise ValueError(f"Page {spec.page_index} contains invalid intensity values.")
        pages.append(page)
    return np.stack(pages, axis=0)
