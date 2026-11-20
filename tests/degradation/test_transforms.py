"""Focused numerical and compatibility checks for the shared SciPy backend."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter

from utils.degradation.degrade import degrade_collection
from utils.degradation.transforms import (
    SEVERITY_LEVELS,
    TransformSpec,
    degradation_specs,
    dilate_spec,
    elastic_spec,
    erode_spec,
    gamma_spec,
    gaussian_pilot_specs,
    noise_spec,
)
from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import digest

from .test_backend import inputs as inputs  # noqa: PLC0414 -- register reused pytest fixture
from .test_backend import reference as reference  # noqa: PLC0414 -- register reused pytest fixture


def test_catalog_and_unchanged_pilot_descriptors():
    catalog = degradation_specs()
    assert len(catalog) == 36
    assert SEVERITY_LEVELS == ('trace', 'mild', 'moderate', 'marked', 'severe', 'extreme')
    names = ['elastic_grid', 'gauss_noise', 'gaussian_blur', 'random_gamma', 'dilate', 'erode']
    schedules = [
        [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        [0.8 * np.sqrt(n) for n in (10, 20, 30, 40, 50, 60)],
        np.geomspace(1, 3, 6), list(range(1, 7)), list(range(1, 7)),
    ]
    for family, name in enumerate(names):
        group = catalog[family * 6:(family + 1) * 6]
        assert [level for level, _ in group] == list(SEVERITY_LEVELS)
        assert [spec.name for _, spec in group] == [name] * 6
        assert [next(iter(spec.parameters.values())) for _, spec in group] == list(schedules[family])
    for _, spec in catalog:
        assert spec.working_scale == ('unit' if spec.name in ('gauss_noise', 'random_gamma') else 'raw')
        assert ('working_scale' in spec.descriptor()) == (spec.working_scale == 'unit')
    for (level, spec), index in zip(gaussian_pilot_specs(), (12, 14, 17)):
        expected = {
            'name': 'gaussian_blur', 'parameters': spec.parameters,
            'implementation': 'scipy.ndimage.gaussian_filter:raw-f32-v1',
            'software': {'scipy': scipy.__version__, 'numpy': np.__version__},
            'base_seed': 42, 'mask_policy': 'fixed_reference', 'input_scale': 'raw',
            'output_scale': 'raw', 'output_dtype': 'float32',
            'seed_policy': 'sha256(base_seed,reference_image_id,variant_id):uint64',
        }
        assert level in ('low', 'mid', 'high')
        assert spec.descriptor() == expected == catalog[index][1].descriptor()
        assert spec.variant_id == 'variant_' + digest(expected)[:24]


def test_unit_noise_gamma_formulas():
    image = np.linspace(-0.2, 1.2, 600, dtype=np.float32).reshape(20, 30)
    spec = noise_spec(0.1)
    expected = np.clip(image + np.random.default_rng(7).normal(0, 0.1, image.shape), 0, 1)
    actual = spec.function(image, rng=np.random.default_rng(7), **spec.parameters)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    gamma = gamma_spec(2)
    actual = gamma.function(image, rng=None, **gamma.parameters)
    np.testing.assert_array_equal(actual, np.clip(image, 0, 1) ** 2)
    assert actual.dtype == np.float32


@pytest.mark.parametrize('factory,filter_function', [(dilate_spec, maximum_filter), (erode_spec, minimum_filter)])
def test_morphology_matches_repeated_extrema(factory, filter_function):
    image = np.random.default_rng(0).normal(300, 100, (13, 17)).astype(np.float32)
    expected = image.copy()
    for iterations in range(1, 7):
        expected = filter_function(expected, size=3, mode='nearest')
        spec = factory(iterations)
        actual = spec.function(image, rng=None, **spec.parameters)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32


def test_elastic_field_formula_and_reproducibility():
    image = np.arange(19 * 23, dtype=np.float32).reshape(19, 23)
    spec = elastic_spec(0.6)
    rng = np.random.default_rng(12)
    coordinates = np.indices(image.shape, dtype=np.float64)
    step = min(image.shape) / 5
    for axis in range(2):
        field = gaussian_filter(rng.uniform(-1, 1, image.shape), sigma=step * 0.25)
        coordinates[axis] += field / np.max(np.abs(field)) * (0.6 * step)
    expected = map_coordinates(image, coordinates, order=1, mode='reflect', prefilter=False)
    actual = spec.function(image, rng=np.random.default_rng(12), **spec.parameters)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.shape == image.shape
    assert not np.array_equal(actual, image)


@pytest.mark.parametrize('factory,args,kwargs', [
    (elastic_spec, (np.nan,), {}), (elastic_spec, (0,), {}),
    (elastic_spec, (0.1,), {'num_steps': 0}), (elastic_spec, (0.1,), {'num_steps': 1.5}),
    (noise_spec, (-0.1,), {}), (noise_spec, (np.inf,), {}), (gamma_spec, (0,), {}),
    (dilate_spec, (0,), {}), (erode_spec, (1,), {'kernel_size': 2}),
])
def test_invalid_parameters(factory, args, kwargs):
    with pytest.raises(ValueError):
        factory(*args, **kwargs)


def test_working_scale_validation_and_positional_compatibility():
    spec = TransformSpec('identity', {}, 'identity-v1', {}, lambda image, rng: image,
                         123, 'fixed_reference')
    assert spec.working_scale == 'raw' and spec.base_seed == 123
    with pytest.raises(ValueError, match='working_scale'):
        replace(spec, working_scale='normalized')


def test_collection_unit_scaling_and_anchored_masks(tmp_path, reference):
    def unbounded(image, *, rng):
        return image * 8 - 1

    specs = [noise_spec(0.1), gamma_spec(2), elastic_spec(0.6),
             TransformSpec('unbounded', {}, 'test-v1', {}, unbounded, working_scale='unit')]
    for index, spec in enumerate(specs):
        out = tmp_path / f'derived-{index}'
        degrade_collection(reference, out, spec, workers=2)
        derived = MicroscopyCollection(out)
        for row in reference.images.loc[reference.images.role.eq('intensity')].to_dict('records'):
            fid, key = row['fov_id'], ('intensity', row['channel'])
            source = reference.read_fov(fid)[key].astype(np.float32)
            unit = spec.working_scale == 'unit'
            image = source / row['normalization_divisor'] if unit else source
            seed = int(digest([spec.base_seed, row['image_id'], spec.variant_id])[:16], 16)
            expected = spec.function(image, rng=np.random.default_rng(seed), **spec.parameters)
            if unit:
                expected *= row['normalization_divisor']
            actual = derived.read_fov(fid)[key]
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.float32
            if spec.name == 'unbounded':
                assert actual.min() < 0 and actual.max() > row['normalization_divisor']
            for kind in ('cells', 'nuclei'):
                key = ('labels', kind)
                np.testing.assert_array_equal(reference.read_fov(fid)[key], derived.read_fov(fid)[key])
            for before, after in zip(reference.object_tables(fid), derived.object_tables(fid)):
                pd.testing.assert_frame_equal(before, after)
        assert derived.images.loc[derived.images.role.eq('intensity'), 'intensity_scale'].eq('raw').all()


def test_invalid_normalization_divisor_rejected_before_writing(tmp_path, reference):
    index = reference.images.index[reference.images.role.eq('intensity')][0]
    for bad in (0, -1, np.inf, np.nan):
        reference.images.loc[index, 'normalization_divisor'] = bad
        with pytest.raises(ValueError, match='normalization_divisor'):
            degrade_collection(reference, tmp_path / 'invalid', noise_spec(0.1), dry_run=True)
    assert not (tmp_path / 'invalid').exists()
