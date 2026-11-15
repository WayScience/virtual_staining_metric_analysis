"""Reference collection -> callable transform -> immutable derived collection."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .collection import MicroscopyCollection
from .transforms import TransformSpec, gaussian_pilot_specs, gaussian_spec
from .writer import ShardPayload, digest, run_collection


def degrade_collection(reference, output, transform: TransformSpec, *, channels=None,
                       fov_ids=None, workers=1, dry_run=False):
    """Write one variant. Transform must preserve coordinates and raw intensity units.

    function(image_float32, *, rng, **parameters) returns a finite, same-shape
    array. Masks are linked, never transformed. Each image gets an independent
    reproducible RNG. Use a new implementation version whenever code changes.
    """
    ref = reference if isinstance(reference, MicroscopyCollection) else MicroscopyCollection(reference)
    if ref.spec.get('kind') == 'degradation':
        raise ValueError('Supply the original reference, not a degraded collection')
    output = Path(output).resolve()
    if output == ref.root or output in ref.root.parents or ref.root in output.parents:
        raise ValueError('Reference and output must be separate, non-nested directories')
    if workers < 1:
        raise ValueError('workers must be positive')
    ids = list(ref.fovs.fov_id if fov_ids is None else fov_ids)
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(ref.spec['items']):
        raise ValueError('Select nonempty, unique FOV IDs from the reference')
    descriptor = transform.descriptor()  # Snapshot caller-owned dictionaries.
    variant_id = 'variant_' + digest(descriptor)[:24]
    dataset_id = ref.spec['dataset_id'] + ':' + variant_id
    selected = ref.images.loc[ref.images.fov_id.isin(ids) & ref.images.role.eq('intensity')]
    channels = sorted(selected.channel.unique()) if channels is None else list(channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError('Select distinct, nonempty channels')
    for fov_id in ids:
        records = ref.images.loc[ref.images.fov_id.eq(fov_id)]
        found = set(records.loc[records.role.eq('intensity'), 'channel'])
        if set(channels) - found:
            raise ValueError(f'{fov_id}: missing requested channels')
        if set(records.loc[records.role.eq('labels'), 'channel']) != {'cells', 'nuclei'}:
            raise ValueError(f'{fov_id}: requires reference cells and nuclei')
        if not records.loc[records.role.eq('intensity'), 'intensity_scale'].eq('raw').all():
            raise ValueError('This runner expects raw-scale reference intensities')
    relative_ref = os.path.relpath(ref.root, output)
    spec = {
        'version': 'degrade-v1', 'kind': 'degradation', 'dataset_id': dataset_id,
        'variant_id': variant_id, 'transform': descriptor, 'channels': channels,
        'reference_dataset_id': ref.spec['dataset_id'],
        'reference_collection_uri': relative_ref, 'reference_fingerprint': ref.fingerprint,
        'items': {fov_id: digest(['degrade-v1', ref.spec['items'][fov_id], descriptor, channels]) for fov_id in ids},
    }
    if dry_run:
        return pd.DataFrame({'fov_id': ids, 'variant_id': variant_id, 'channels': [channels] * len(ids)})

    def produce(fov_id):
        start = time.perf_counter()
        reference_rows = ref.images.loc[ref.images.fov_id.eq(fov_id)]
        planes = ref.read_fov(fov_id, channels=channels, include_masks=False)
        read_seconds = time.perf_counter() - start
        arrays, registry, seeds = {}, [], {}
        start = time.perf_counter()
        for row in reference_rows.loc[reference_rows.role.eq('intensity') & reference_rows.channel.isin(channels)].to_dict('records'):
            reference_id = row['image_id']
            image_id = 'img_' + digest([reference_id, variant_id])[:24]
            seed = int(digest([descriptor['base_seed'], reference_id, variant_id])[:16], 16)
            source = planes[('intensity', row['channel'])]
            image = np.array(source, dtype=np.float32, copy=True)
            if not np.isfinite(image).all():
                raise ValueError(f'Cannot represent source as finite float32: {reference_id}')
            transformed = np.asarray(transform.function(
                image, rng=np.random.default_rng(seed), **descriptor['parameters']), dtype=np.float32)
            if transformed.shape != source.shape or not np.isfinite(transformed).all():
                raise ValueError(f'Transform changed shape or produced nonfinite pixels: {reference_id}')
            arrays[image_id] = transformed
            seeds[image_id] = seed
            registry.append({**row, 'image_id': image_id, 'reference_image_id': reference_id,
                             'dataset_id': dataset_id, 'reference_dataset_id': ref.spec['dataset_id'],
                             'variant': 'degradation', 'variant_id': variant_id,
                             'dtype': 'float32', 'intensity_scale': 'raw',
                             'ome_arrow_uri': f'fovs/{fov_id}/pixels.ome-arrow'})
        transform_seconds = time.perf_counter() - start
        for row in reference_rows.loc[reference_rows.role.eq('labels')].to_dict('records'):
            # Same mask image IDs; pointers remain valid when the whole tree is moved.
            registry.append({**row, 'reference_dataset_id': ref.spec['dataset_id'],
                             'variant_id': 'reference',
                             'ome_arrow_uri': os.path.relpath(ref.root / row['ome_arrow_uri'], output)})
        objects, relationships = ref.object_tables(fov_id)
        fov = ref.fovs.loc[ref.fovs.fov_id.eq(fov_id)].copy()
        fov['dataset_id'] = dataset_id
        fov['reference_dataset_id'] = ref.spec['dataset_id']
        fov['variant_id'] = variant_id
        if (ref.root / '_SUCCESS').read_text().strip() != ref.fingerprint:
            raise RuntimeError('Reference changed during generation')
        return ShardPayload(
            arrays=arrays, registry=pd.DataFrame(registry), fov=fov,
            objects=objects, relationships=relationships,
            provenance={'reference_collection_uri': relative_ref,
                        'reference_fingerprint': ref.fingerprint,
                        'reference_fov_fingerprint': ref.spec['items'][fov_id],
                        'fov_id': fov_id, 'variant_id': variant_id, 'transform': descriptor,
                        'image_seeds': seeds, 'mask_policy': 'fixed_reference'},
            statistics={'read_seconds': read_seconds, 'transform_seconds': transform_seconds},
        )

    return run_collection(output, spec, produce, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--channels', nargs='+', help='Default: all intensity channels, including BF')
    parser.add_argument('--sigmas', nargs='+', type=float, help='Override pilot sigma values in pixels')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--fov-ids-file', help='One FOV ID per line; optional frozen cohort')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ref = MicroscopyCollection(args.reference)
    ids = (Path(args.fov_ids_file).read_text().splitlines() if args.fov_ids_file else list(ref.fovs.fov_id))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        ids = list(np.random.default_rng(args.sample_seed).choice(ids, size=min(args.limit, len(ids)), replace=False))
    variants = ([(f'sigma_{sigma:g}', gaussian_spec(sigma)) for sigma in args.sigmas]
                if args.sigmas else gaussian_pilot_specs())
    # Sequential variants bound memory; workers parallelize FOVs within each variant.
    for level, transform in variants:
        destination = Path(args.output_root) / f'{level}_{transform.variant_id}'
        print(f'{level}: {transform.parameters} -> {destination}', flush=True)
        results = degrade_collection(ref, destination, transform, channels=args.channels,
                                     fov_ids=ids, workers=args.workers, dry_run=args.dry_run)
        print(results.to_string(index=False))


if __name__ == '__main__':
    main()
