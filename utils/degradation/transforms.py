"""SciPy transforms on a fixed output grid, with raw collection input/output units.

Gaussian blur deliberately retains the pilot's reflect/truncate boundaries, not
the non-identical legacy OpenCV boundaries. Elastic deformation preserves the
output grid, not structure coordinates: reference masks stay anchored deliberately.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter

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
    working_scale: str = 'raw'

    def __post_init__(self):
        if self.working_scale not in ('raw', 'unit'):
            raise ValueError("working_scale must be 'raw' or 'unit'")

    def descriptor(self):
        if not self.name or not self.implementation:
            raise ValueError('Transform name and implementation version are required')
        if self.mask_policy != 'fixed_reference':
            raise ValueError('This runner keeps reference masks anchored on the output grid')
        descriptor = {
            'name': self.name, 'parameters': self.parameters,
            'implementation': self.implementation, 'software': self.software,
            'base_seed': self.base_seed, 'mask_policy': self.mask_policy,
            'input_scale': 'raw', 'output_scale': 'raw', 'output_dtype': 'float32',
            'seed_policy': 'sha256(base_seed,reference_image_id,variant_id):uint64',
        }
        # Do not change raw descriptors: their bytes determine existing pilot IDs.
        if self.working_scale != 'raw':
            descriptor['working_scale'] = self.working_scale
        return json.loads(json.dumps(descriptor, allow_nan=False))

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


SEVERITY_LEVELS = ('trace', 'mild', 'moderate', 'marked', 'severe', 'extreme')


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def elastic_grid(image, *, rng, displacement_fraction, num_steps=5):
    """Independent smooth dy/dx displacements; masks are intentionally not warped."""
    _positive(displacement_fraction, 'displacement_fraction')
    _positive_integer(num_steps, 'num_steps')
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not image.size:
        raise ValueError('elastic_grid requires a nonempty 2D image')
    step = min(image.shape) / num_steps
    coordinates = np.indices(image.shape, dtype=np.float64)
    for axis in range(2):
        field = gaussian_filter(rng.uniform(-1, 1, image.shape), sigma=step * 0.25)
        peak = np.max(np.abs(field))
        if peak > 0:
            field /= peak
        coordinates[axis] += field * (displacement_fraction * step)
    return map_coordinates(image, coordinates, order=1, mode='reflect',
                           prefilter=False, output=np.float32)


def gauss_noise(image, *, rng, std_fraction):
    """Add independent per-pixel unit-scale normal noise, clipping only this transform."""
    _positive(std_fraction, 'std_fraction')
    return np.clip(np.asarray(image, dtype=np.float32) +
                   rng.normal(0, std_fraction, np.shape(image)), 0, 1).astype(np.float32)


def random_gamma(image, *, rng, gamma):
    """Apply a fixed gamma exponent to clipped unit-scale intensities."""
    _positive(gamma, 'gamma')
    return np.power(np.clip(np.asarray(image, dtype=np.float32), 0, 1), gamma).astype(np.float32)


def _morphology_width(iterations, kernel_size):
    _positive_integer(iterations, 'iterations')
    _positive_integer(kernel_size, 'kernel_size')
    if kernel_size % 2 != 1:
        raise ValueError('kernel_size must be odd')
    return 1 + iterations * (kernel_size - 1)


def dilate(image, *, rng, iterations, kernel_size=3):
    """Repeated square dilation, with nearest borders matching OpenCV extrema."""
    width = _morphology_width(iterations, kernel_size)
    return maximum_filter(np.asarray(image, dtype=np.float32), size=width,
                          mode='nearest', output=np.float32)


def erode(image, *, rng, iterations, kernel_size=3):
    """Repeated square erosion, with nearest borders matching OpenCV extrema."""
    width = _morphology_width(iterations, kernel_size)
    return minimum_filter(np.asarray(image, dtype=np.float32), size=width,
                          mode='nearest', output=np.float32)


def elastic_spec(displacement_fraction, *, num_steps=5):
    _positive(displacement_fraction, 'displacement_fraction')
    _positive_integer(num_steps, 'num_steps')
    return TransformSpec(
        name='elastic_grid',
        parameters={'displacement_fraction': float(displacement_fraction),
                    'num_steps': int(num_steps)},
        implementation='scipy.ndimage.map_coordinates:elastic-grid-f32-v1',
        software={'scipy': scipy.__version__, 'numpy': np.__version__}, function=elastic_grid,
    )


def noise_spec(std_fraction):
    _positive(std_fraction, 'std_fraction')
    return TransformSpec(
        name='gauss_noise', parameters={'std_fraction': float(std_fraction)},
        implementation='numpy.random.Generator.normal:unit-f32-v1',
        software={'numpy': np.__version__}, function=gauss_noise, working_scale='unit',
    )


def gamma_spec(gamma):
    _positive(gamma, 'gamma')
    return TransformSpec(
        name='random_gamma', parameters={'gamma': float(gamma)},
        implementation='numpy.power:unit-f32-v1',
        software={'numpy': np.__version__}, function=random_gamma, working_scale='unit',
    )


def dilate_spec(iterations, *, kernel_size=3):
    _morphology_width(iterations, kernel_size)
    return TransformSpec(
        name='dilate', parameters={'iterations': int(iterations), 'kernel_size': int(kernel_size)},
        implementation='scipy.ndimage.maximum_filter:nearest-f32-v1',
        software={'scipy': scipy.__version__, 'numpy': np.__version__}, function=dilate,
    )


def erode_spec(iterations, *, kernel_size=3):
    _morphology_width(iterations, kernel_size)
    return TransformSpec(
        name='erode', parameters={'iterations': int(iterations), 'kernel_size': int(kernel_size)},
        implementation='scipy.ndimage.minimum_filter:nearest-f32-v1',
        software={'scipy': scipy.__version__, 'numpy': np.__version__}, function=erode,
    )


def degradation_specs() -> list[tuple[str, TransformSpec]]:
    """Six families, six severity levels each, in legacy family-major page order."""
    families = [
        (elastic_spec, [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (noise_spec, [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]),
        (gaussian_spec, [0.8 * np.sqrt(n) for n in (10, 20, 30, 40, 50, 60)]),
        (gamma_spec, np.geomspace(1.0, 3.0, 6)),
        (dilate_spec, range(1, 7)),
        (erode_spec, range(1, 7)),
    ]
    return [(level, factory(value)) for factory, values in families
            for level, value in zip(SEVERITY_LEVELS, values)]
