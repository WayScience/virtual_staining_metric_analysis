import importlib.util

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy.ndimage import gaussian_filter

from utils.degradation.degrade import degrade_collection
from utils.degradation.transforms import (
    TransformSpec,
    gaussian_blur,
    gaussian_pilot_specs,
    gaussian_spec,
)
from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.ingest import ingest


@pytest.fixture
def inputs(tmp_path):
    outlines = tmp_path / 'outlines' / 'P'
    outlines.mkdir(parents=True)
    rows = []
    for site in ('01', '02'):
        image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) * 31
        path = tmp_path / f'{site}.tiff'
        tifffile.imwrite(path, image)
        for name, lo, hi in [('Cells', 5, 25), ('Nuclei', 10, 20)]:
            boundary = np.zeros((32, 32), np.uint8)
            boundary[lo:hi, lo] = boundary[lo:hi, hi - 1] = 255
            boundary[lo, lo:hi] = boundary[hi - 1, lo:hi] = 255
            rgb = np.zeros((32, 32, 3), np.uint8)
            rgb[..., 1] = boundary
            tifffile.imwrite(outlines / f'{name}Outlines_P_A01_{site}.tiff', rgb, photometric='rgb')
        rows.append({'Metadata_Plate': 'P', 'Metadata_Well': 'A01', 'Metadata_Site': site,
                     'cell_type': 'test', 'seeding_density': '100', 'FileName_DNA': str(path)})
    return pd.DataFrame(rows), outlines.parent


@pytest.fixture
def reference(tmp_path, inputs):
    frame, outlines = inputs
    root = tmp_path / 'reference'
    ingest(frame, outlines, root, dataset_id='test', pixel_size_um=.593, workers=2)
    return MicroscopyCollection(root)


def test_reference_and_three_blurs(tmp_path, reference):
    assert set(reference.metadata().cell_type) == {'test'}
    for level, transform in gaussian_pilot_specs():
        out = tmp_path / level
        degrade_collection(reference, out, transform, workers=2)
        derived = MicroscopyCollection(out)
        assert set(derived.fovs.fov_id) == set(reference.fovs.fov_id)
        for fid in reference.fovs.fov_id:
            src = reference.read_fov(fid)
            dst = derived.read_fov(fid)
            assert src[('intensity', 'DNA')].dtype == np.uint16
            actual = dst[('intensity', 'DNA')]
            expected = gaussian_filter(src[('intensity', 'DNA')].astype(np.float32),
                                       sigma=transform.parameters['sigma_px'], mode='reflect', truncate=4)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.float32 and actual.max() > 1
            for kind in ('cells', 'nuclei'):
                np.testing.assert_array_equal(src[('labels', kind)], dst[('labels', kind)])
                assert dst[('labels', kind)].dtype == np.uint32
            for old, new in zip(reference.object_tables(fid), derived.object_tables(fid)):
                pd.testing.assert_frame_equal(old, new)
            np.testing.assert_allclose(derived.read_fov(fid, normalize=True)[('intensity', 'DNA')], actual / 65535)
            rows = derived.images.loc[derived.images.fov_id.eq(fid) & derived.images.role.eq('intensity')]
            assert set(rows.reference_image_id).issubset(set(reference.images.image_id))
            assert not set(rows.image_id).intersection(reference.images.image_id)
        # No copied masks in the local pixel dataset.
        from ome_arrow import OMEArrowDataset
        fid = derived.fovs.iloc[0].fov_id
        ds = OMEArrowDataset(out / 'fovs' / fid / 'pixels.ome-arrow')
        label_id = derived.images.loc[derived.images.role.eq('labels'), 'image_id'].iloc[0]
        with pytest.raises((KeyError, ValueError)):
            ds.read_image(label_id)
        resumed = degrade_collection(reference, out, transform, workers=2)
        assert resumed.resumed.all()


def test_reject_changed_configuration(tmp_path, reference):
    out = tmp_path / 'derived'
    degrade_collection(reference, out, gaussian_spec(1))
    with pytest.raises(ValueError, match='different configuration'):
        degrade_collection(reference, out, gaussian_spec(2))
    with pytest.raises(ValueError, match='missing requested channels'):
        degrade_collection(reference, tmp_path / 'bad', gaussian_spec(1), channels=['absent'])
    with pytest.raises(ValueError, match='non-nested'):
        degrade_collection(reference, reference.root, gaussian_spec(1))


def test_failure_cleanup_and_resume(tmp_path, reference):
    out = tmp_path / 'retry'
    state = {'count': 0, 'fail': True}
    def sometimes_fail(image, *, rng):
        state['count'] += 1
        if state['fail'] and state['count'] == 2:
            raise RuntimeError('simulated interruption')
        return image
    spec = TransformSpec('identity', {}, 'identity-v1', {}, sometimes_fail)
    with pytest.raises(RuntimeError, match='interruption'):
        degrade_collection(reference, out, spec)
    assert not (out / '_SUCCESS').exists()
    # Keep the lock inode stable across writers, but release flock on failure.
    import fcntl
    lock = out / '.ingest.lock'
    inode = lock.stat().st_ino
    with lock.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
    assert len(list((out / 'fovs').glob('fov_*'))) == 1
    assert not list((out / 'fovs').glob('.*'))
    state['fail'] = False
    result = degrade_collection(reference, out, spec)
    assert result.resumed.sum() == 1
    assert (out / '_SUCCESS').exists()
    assert lock.stat().st_ino == inode


def test_brightfield_division_before_degradation(tmp_path, inputs):
    frame, outlines = inputs
    frame = frame.rename(columns={'FileName_DNA': 'FileName_Brightfield'})
    field = np.full((32, 32), .25, dtype=np.float32)
    field[:, 16:] = 2
    illumination = tmp_path / 'brightfield_illumination.npy'
    np.save(illumination, field)
    root = tmp_path / 'corrected_brightfield'
    ingest(frame, outlines, root, dataset_id='brightfield',
           illumination={('P', 'Brightfield'): (illumination, 'divide')},
           output_channels={'Brightfield': 'IllumBrightfield'})
    ref = MicroscopyCollection(root)
    destination = tmp_path / 'brightfield_blur'
    degrade_collection(ref, destination, gaussian_spec(2))
    variant = MicroscopyCollection(destination)
    for collection in (ref, variant):
        registry = collection.images.loc[collection.images.role.eq('intensity')]
        assert set(registry.channel) == {'IllumBrightfield'}
        assert registry.dtype.eq('float32').all()
        assert registry.intensity_scale.eq('raw').all()
        assert registry.normalization_divisor.eq(65535).all()
        assert registry.illumination_method.eq('divide').all()
    for row in ref.metadata().itertuples(index=False):
        raw = tifffile.imread(row.FileName_Brightfield)
        corrected = np.clip(raw.astype(np.float32) / field, 0, 65535)
        assert corrected.max() == 65535  # Exercise clipping before blur, not afterward.
        key = ('intensity', 'IllumBrightfield')
        np.testing.assert_array_equal(ref.read_fov(row.fov_id)[key], corrected)
        blurred = variant.read_fov(row.fov_id)[key]
        np.testing.assert_array_equal(blurred, gaussian_filter(corrected, 2))
        np.testing.assert_allclose(variant.read_fov(row.fov_id, normalize=True)[key],
                                   blurred / np.float32(65535))


def test_stochastic_determinism_and_invalid_output(tmp_path, reference):
    def noise(image, *, rng, std):
        return image + rng.normal(0, std, image.shape)
    spec = TransformSpec('noise', {'std': 10}, 'test-noise-v1', {'numpy': np.__version__}, noise)
    degrade_collection(reference, tmp_path / 'a', spec, workers=1)
    degrade_collection(reference, tmp_path / 'b', spec, workers=2, fov_ids=list(reversed(reference.fovs.fov_id)))
    a, b = MicroscopyCollection(tmp_path / 'a'), MicroscopyCollection(tmp_path / 'b')
    for fid in a.fovs.fov_id:
        np.testing.assert_array_equal(a.read_fov(fid)[('intensity', 'DNA')], b.read_fov(fid)[('intensity', 'DNA')])
    bad = TransformSpec('bad', {}, 'bad-v1', {}, lambda image, rng: image[:-1])
    with pytest.raises(ValueError, match='changed shape'):
        degrade_collection(reference, tmp_path / 'bad', bad)
    assert not (tmp_path / 'bad' / '_SUCCESS').exists()


def test_gaussian_scale_and_constant():
    image = np.random.default_rng(0).uniform(0, 65535, (100, 100)).astype(np.float32)
    kw = {'rng': None, 'sigma_px': 3.0}
    raw = gaussian_blur(image, **kw)
    norm = gaussian_blur(image / 65535, **kw)
    np.testing.assert_allclose(raw / 65535, norm, atol=2e-7)
    np.testing.assert_allclose(gaussian_blur(np.full((20, 20), 1234, np.float32), **kw), 1234)


def test_original_ingestion_compatibility(tmp_path, inputs):
    # Supplied via environment only in development; tests remain portable.
    import os
    original = os.environ.get('ORIGINAL_INGEST')
    if not original:
        pytest.skip('Set ORIGINAL_INGEST to the supplied pre-refactor source')
    spec = importlib.util.spec_from_file_location('original_ingest', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame, outlines = inputs
    root = tmp_path / 'legacy'
    module.ingest(frame, outlines, root, dataset_id='test', pixel_size_um=.593)
    before = (root / '_SUCCESS').read_text()
    result = ingest(frame, outlines, root, dataset_id='test', pixel_size_um=.593)
    assert result.resumed.all()
    assert (root / '_SUCCESS').read_text() == before
    legacy = MicroscopyCollection(root)
    degrade_collection(legacy, tmp_path / 'derived', gaussian_spec(2))


@pytest.mark.parametrize('module', ['utils.degradation.cli', 'utils.degradation'])
def test_cli_pilot(tmp_path, reference, module):
    import subprocess
    import sys
    destination = tmp_path / 'cli_pilot'
    subprocess.run([
        sys.executable, '-m', module, 'pilot',
        '--reference', str(reference.root), '--output-root', str(destination),
        '--limit', '1', '--channels', 'DNA', '--workers', '2',
    ], check=True, capture_output=True, text=True)
    roots = sorted(destination.glob('*variant_*'))
    assert len(roots) == 3
    cohorts = [MicroscopyCollection(root).fovs.fov_id.tolist() for root in roots]
    assert cohorts[0] == cohorts[1] == cohorts[2] and len(cohorts[0]) == 1


def test_full_workflow_and_report(tmp_path, reference):
    import subprocess
    import sys

    from utils.pedcp_workflow import discover_degradation_variants

    destination, results = tmp_path / 'production', tmp_path / 'results'
    command = [sys.executable, '-m', 'utils.degradation',
               '--reference', str(reference.root), '--output-root', str(destination),
               '--results-dir', str(results), '--limit', '1', '--channels', 'DNA']
    subprocess.run([*command, '--dry-run'], check=True, capture_output=True, text=True)
    assert not destination.exists() and not results.exists()
    subprocess.run(command, check=True, capture_output=True, text=True)
    variants = discover_degradation_variants(reference, destination)
    assert len(variants) == 36
    cohorts = [collection.fovs.fov_id.tolist() for _, collection in variants]
    assert all(ids == cohorts[0] for ids in cohorts) and len(cohorts[0]) == 1
    assert len(pd.read_csv(destination / 'degradation_catalog.csv')) == 36
    assert len(list(results.glob('montage_*_DNA.png'))) == 1
    # Resume, then render the same artifact independently without regeneration.
    resumed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert 'True' in resumed.stdout
    subprocess.run([*command[:3], 'report', *command[3:]], check=True,
                   capture_output=True, text=True)
