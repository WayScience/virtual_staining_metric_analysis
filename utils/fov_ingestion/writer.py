"""Storage-only backend shared by reference ingestion and derived collections."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass
class ShardPayload:
    # Only locally owned pixels. Registry may additionally link external masks.
    arrays: dict[str, np.ndarray]
    registry: pd.DataFrame
    fov: pd.DataFrame
    objects: pd.DataFrame
    relationships: pd.DataFrame
    provenance: dict
    statistics: dict = field(default_factory=dict)


def write_shard(output, fov_id, fingerprint, produce: Callable[[], ShardPayload]):
    """Resume or stage, verify, then atomically commit a complete FOV."""
    from ome_arrow import OMEArrowDataset, from_numpy, write_ome_arrow_dataset

    if Path(fov_id).name != fov_id or fov_id in ('.', '..'):
        raise ValueError('Invalid FOV ID')
    parent = Path(output) / 'fovs'
    dest = parent / fov_id
    if dest.exists():
        done = json.loads((dest / 'complete.json').read_text())
        if done['fingerprint'] != fingerprint:
            raise ValueError(f'{fov_id}: inputs changed; use a new output collection')
        return {**done, 'resumed': True}
    tmp = parent / ('.' + fov_id + '.' + uuid.uuid4().hex)
    tmp.mkdir()
    start = time.perf_counter()
    try:
        payload = produce()
        if payload.registry.image_id.duplicated().any():
            raise ValueError('Duplicate image IDs in shard')
        registry = payload.registry.set_index('image_id')
        if len(payload.fov) != 1 or payload.fov.iloc[0]['fov_id'] != fov_id:
            raise ValueError('FOV metadata identity mismatch')
        shape = (int(payload.fov.iloc[0]['height']), int(payload.fov.iloc[0]['width']))
        size = payload.fov.iloc[0]['pixel_size_um']
        size = None if pd.isna(size) else float(size)
        records = []
        local_uri = f'fovs/{fov_id}/pixels.ome-arrow'
        local_ids = set(payload.registry.loc[payload.registry.ome_arrow_uri.eq(local_uri), 'image_id'])
        if local_ids != set(payload.arrays):
            raise ValueError('Local pixel arrays and registry differ')
        for image_id, arr in payload.arrays.items():
            row = registry.loc[image_id]
            if arr.ndim != 2 or arr.shape != shape or arr.dtype.kind not in 'buif' or not np.isfinite(arr).all():
                raise ValueError(f'Invalid pixels for {image_id}')
            if str(arr.dtype) != row['dtype'] or row['fov_id'] != fov_id:
                raise ValueError(f'Pixel metadata mismatch: {image_id}')
            if row['role'] == 'labels' and (arr.dtype.kind not in 'ui' or np.any(arr < 0)):
                raise ValueError('Labels must be nonnegative integers')
            records.append(from_numpy(
                arr, dim_order='YX', image_id=image_id, name=row['channel'],
                image_type='label' if row['role'] == 'labels' else 'image',
                channel_names=[row['channel']], clamp_to_uint16=False, chunk_encoding='bytes',
                physical_size_x=size or 1., physical_size_y=size or 1.,
                physical_size_unit='µm' if size else 'pixel',
            ))
        if not records:
            raise ValueError('Cannot write an empty pixel shard')
        write_ome_arrow_dataset(records, tmp / 'pixels.ome-arrow', pixel_dtype=None, compression='zstd')
        stored = OMEArrowDataset(tmp / 'pixels.ome-arrow')
        for image_id, arr in payload.arrays.items():
            actual = stored.read_image(image_id)[0, 0, 0]
            if actual.dtype != arr.dtype or not np.array_equal(actual, arr):
                raise RuntimeError(f'Pixel round-trip failed: {image_id}')
        del stored
        for name, table in [('fov', payload.fov), ('image_registry', payload.registry),
                            ('objects', payload.objects), ('relationships', payload.relationships)]:
            table.to_parquet(tmp / f'{name}.parquet', index=False)
        (tmp / 'provenance.json').write_text(json.dumps(payload.provenance, indent=2))
        done = {**payload.statistics, 'fov_id': fov_id, 'fingerprint': fingerprint,
                'elapsed_seconds': time.perf_counter() - start,
                'bytes_written': sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file()),
                'resumed': False}
        (tmp / 'complete.json').write_text(json.dumps(done, indent=2))
        tmp.rename(dest)
        return done
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def run_collection(output, spec, produce, *, workers=1):
    """One coordinator owns a collection; producer receives one FOV ID."""
    if workers < 1 or not spec['items']:
        raise ValueError('Positive workers and nonempty FOV selection required')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.ingest.lock'  # Compatible with the existing ingestion lock.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        config = output / 'collection.json'
        if config.exists():
            if json.loads(config.read_text()) != spec:
                raise ValueError('Output has a different configuration/selection; use a new directory')
        else:
            unexpected = [p for p in output.iterdir() if p != lock]
            if unexpected:
                raise ValueError('Output is nonempty but has no collection.json')
            tmp = output / 'collection.json.tmp'
            tmp.write_text(json.dumps(spec, indent=2))
            tmp.replace(config)
        (output / 'fovs').mkdir(exist_ok=True)
        success = output / '_SUCCESS'
        success.unlink(missing_ok=True)
        def process(fov_id):
            return write_shard(output, fov_id, spec['items'][fov_id], lambda: produce(fov_id))
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(process, spec['items']):
                results.append(result)
                print(f"[{len(results)}/{len(spec['items'])}] {result['fov_id']} "
                      f"{result['elapsed_seconds']:.2f}s resumed={result['resumed']}", flush=True)
        for filename in ('fov.parquet', 'image_registry.parquet', 'objects.parquet', 'relationships.parquet'):
            paths = [output / 'fovs' / fov_id / filename for fov_id in spec['items']]
            schema = pa.unify_schemas([pq.read_schema(p) for p in paths], promote_options='permissive')
            name = 'fovs.parquet' if filename == 'fov.parquet' else filename
            tmp = output / (name + '.tmp')
            with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
                for path in paths:
                    writer.write_table(pq.read_table(path).cast(schema))
            tmp.replace(output / name)
        pd.DataFrame(results).to_parquet(output / 'timings.parquet.tmp', index=False)
        (output / 'timings.parquet.tmp').replace(output / 'timings.parquet')
        (output / '_SUCCESS.tmp').write_text(digest(spec))
        (output / '_SUCCESS.tmp').replace(success)
        return pd.DataFrame(results)
    finally:
        lock.unlink()
