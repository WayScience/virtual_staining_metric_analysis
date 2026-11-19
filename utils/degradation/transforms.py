"""Explicit transform contracts; raw input/output intensity units, fixed geometry."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from utils.fov_ingestion.writer import digest


@dataclass(frozen=True)
class TransformSpec:
    name: str
    parameters: dict
    implementation: str  # Explicit code version, e.g. my_module.my_transform:v1.
    software: dict       # Record relevant dependency versions.
    function: Callable  # function(image_float32, *, rng, **parameters) -> YX array
    base_seed: int = 42
    mask_policy: str = 'fixed_reference'

    def descriptor(self):
        if not self.name or not self.implementation:
            raise ValueError('Transform name and implementation version are required')
        if self.mask_policy != 'fixed_reference':
            raise ValueError('This runner supports fixed-coordinate transforms only')
        return json.loads(json.dumps({
            'name': self.name, 'parameters': self.parameters,
            'implementation': self.implementation, 'software': self.software,
            'base_seed': self.base_seed, 'mask_policy': self.mask_policy,
            'input_scale': 'raw', 'output_scale': 'raw', 'output_dtype': 'float32',
            'seed_policy': 'sha256(base_seed,reference_image_id,variant_id):uint64',
        }, allow_nan=False))

    @property
    def variant_id(self):
        return 'variant_' + digest(self.descriptor())[:24]


def gaussian_blur(image, *, rng, sigma_px, mode='reflect', truncate=4.0):
    """Blur raw intensities without clipping, normalization, or integer rounding."""
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError('sigma_px must be positive and finite')
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError('truncate must be positive and finite')
    if mode not in ('reflect', 'mirror', 'nearest', 'wrap'):
        raise ValueError('Unsupported boundary mode')
    return gaussian_filter(image, sigma=float(sigma_px), output=np.float32,
                           mode=mode, truncate=float(truncate))


def gaussian_spec(sigma_px, *, mode='reflect', truncate=4.0):
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError('sigma_px must be positive and finite')
    return TransformSpec(
        name='gaussian_blur', parameters={'sigma_px': float(sigma_px), 'mode': mode, 'truncate': float(truncate)},
        implementation='scipy.ndimage.gaussian_filter:raw-f32-v1',
        software={'scipy': scipy.__version__, 'numpy': np.__version__}, function=gaussian_blur,
    )


def gaussian_pilot_specs():
    # Three entries from the user's existing six-level sigma schedule.
    return [(level, gaussian_spec(0.8 * np.sqrt(iterations)))
            for level, iterations in [('low', 10), ('mid', 30), ('high', 60)]]
