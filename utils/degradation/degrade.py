"""
Orchestator for applying degradation transforms to some image data. 
Reference collection + callable transform -> derived collection.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.degradation.transforms import TransformSpec
from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import ShardPayload, digest, run_collection


def degrade_collection(
    reference: MicroscopyCollection | str, 
    output: str | Path, 
    transform: TransformSpec, 
    *, 
    channels: list[str] | None = None,
    fov_ids: list[str] | None = None, 
    workers: int = 1, 
    dry_run: bool = False
) -> pd.DataFrame | None:
    """
    Write one variant. Transform must preserve coordinates and raw intensity units.

    function(image_float32, *, rng, **parameters) returns a finite, same-shape
    array. Masks are linked, never transformed. Each image gets an independent
    reproducible RNG. Use a new implementation version whenever code changes.

    :param reference: Original microscopy collection.
    :param output: Directory to write degraded collection.
    :param transform: Transform specification.
    :param channels: Channels to degrade.
    :param fov_ids: Field of view IDs to process.
    :param workers: Number of parallel workers.
    :param dry_run: If True, do not write output, just return planned operations.
    :return: DataFrame summarizing the planned degradation.
    """
    ref = reference if isinstance(reference, MicroscopyCollection) else MicroscopyCollection(reference)

    # Validate reference and output directories.
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

    # Prepare degradation id to uniquely identify this variant.
    descriptor = transform.descriptor()  # Snapshot caller-owned dictionaries.
    variant_id = 'variant_' + digest(descriptor)[:24]
    dataset_id = ref.spec['dataset_id'] + ':' + variant_id
    selected = ref.images.loc[ref.images.fov_id.isin(ids) & ref.images.role.eq('intensity')]
    channels = sorted(selected.channel.unique()) if channels is None else list(channels)

    # Validate channels foundable in reference images against requested 
    # degradation channels.
    if not channels or len(channels) != len(set(channels)):
        raise ValueError('Select distinct, nonempty channels')
    for fov_id in ids:
        records = ref.images.loc[ref.images.fov_id.eq(fov_id)]
        found = set(records.loc[records.role.eq('intensity'), 'channel'])
        if set(channels) - found:
            raise ValueError(f'{fov_id}: missing requested channels {sorted(set(channels) - found)}; '
                             f'available channels: {sorted(found)}')
        if set(records.loc[records.role.eq('labels'), 'channel']) != {'cells', 'nuclei'}:
            raise ValueError(f'{fov_id}: requires reference cells and nuclei')
        if not records.loc[records.role.eq('intensity'), 'intensity_scale'].eq('raw').all():
            raise ValueError('This runner expects raw-scale reference intensities')
    
    # Compute relative path from output to reference root.
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

    def produce(fov_id: str) -> ShardPayload:
        """
        Produce shard (containing degraded images for a single field of view)
            to be supplied to the writer
        """

        start = time.perf_counter()
        reference_rows = ref.images.loc[ref.images.fov_id.eq(fov_id)]
        planes = ref.read_fov(fov_id, channels=channels, include_masks=False)
        read_seconds = time.perf_counter() - start
        arrays, registry, seeds = {}, [], {}
        start = time.perf_counter()

        # Iterate over each intensity channel of the reference rows for the current field of view.
        for row in reference_rows.loc[reference_rows.role.eq('intensity') & reference_rows.channel.isin(channels)].to_dict('records'):
            reference_id = row['image_id']

            # degraded fov id from original image id and transform
            image_id = 'img_' + digest([reference_id, variant_id])[:24]            
            # Deterministic seed for the transformation based on the base seed, reference image ID, and variant ID.
            seed = int(digest([descriptor['base_seed'], reference_id, variant_id])[:16], 16)
            source = planes[('intensity', row['channel'])]
            
            # Convert the source image to a float32 for processing
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
        
        # Append label rows to the registry without modification, 
        #   maintaining references to the original dataset.
        for row in reference_rows.loc[reference_rows.role.eq('labels')].to_dict('records'):
            # Same mask image IDs; pointers remain valid when the whole tree is moved.
            registry.append({**row, 'reference_dataset_id': ref.spec['dataset_id'],
                             'variant_id': 'reference',
                             'ome_arrow_uri': os.path.relpath(ref.root / row['ome_arrow_uri'], output)})
        
        # Retrieve object and relationship tables for the current field of view.
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
