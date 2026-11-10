"""Immutable, per-FOV OME-Arrow ingestion. No intensity normalization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from scipy import ndimage as ndi

VERSION = "ingest-v0"
KEYS = ['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']
REL_COLUMNS = ['nucleus_id','cell_id','overlap_pixels','nucleus_overlap_fraction','is_best_match']
OBJECT_COLUMNS = ['object_id','fov_id','mask_set_id','compartment','label','centroid_y','centroid_x','y0','x0','y1','x1','area_pixels','touches_border']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def token(value):
    if pd.isna(value):
        raise ValueError('Missing identity field')
    if isinstance(value, (float, np.floating)) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    if not s or '/' in s or '\\' in s or s in {'.', '..'}:
        raise ValueError(f'Invalid identity field: {value!r}')
    return s


def read_inputs(inputs):
    if isinstance(inputs, pd.DataFrame):
        return inputs.copy()
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    frames = []
    for path in inputs:
        p = Path(path)
        if p.suffix.lower() in {'.parquet', '.pq'}:
            frames.append(pd.read_parquet(p))
        else:
            frames.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    if not frames:
        raise ValueError('No LoadData inputs supplied')
    return pd.concat(frames, ignore_index=True)


def plan(inputs, outline_root, *, dataset_id, image_root=None, channels=None, pixel_size_um=None, skip_missing_outlines=False):
    """Resolve every path before reading pixels; reject duplicate FOV identities.

    CSV identifiers preserve leading zeros. DataFrame/parquet values must retain
    the spelling used in filenames. Relative image paths require image_root.
    """
    if not dataset_id:
        raise ValueError('dataset_id must be a stable, nonempty acquisition namespace')
    df = read_inputs(inputs)
    missing = set(KEYS) - set(df)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    channels = list(channels) if channels is not None else sorted(c[9:] for c in df if c.startswith('FileName_'))
    if not channels or len(channels) != len(set(channels)) or set(channels) & {'CellsOutlines', 'NucleiOutlines'}:
        raise ValueError('Supply distinct channel names or FileName_* columns')
    if pixel_size_um is not None and (not np.isfinite(pixel_size_um) or pixel_size_um <= 0):
        raise ValueError('pixel_size_um must be positive')
    root = Path(outline_root).expanduser().resolve()
    rows, seen, errors = [], set(), []
    skipped = 0
    for index, row in df.iterrows():
        try:
            plate, well, site = (token(row[k]) for k in KEYS)
            key = (plate, well, site)
            if key in seen:
                raise ValueError(f'Duplicate FOV {key}; include only one acquisition per dataset')
            seen.add(key)
            paths = {}
            for ch in channels:
                filename = row.get('FileName_' + ch)
                if pd.isna(filename) or not str(filename).strip():
                    raise ValueError(f'Missing FileName_{ch}')
                p = Path(str(filename)).expanduser()
                if not p.is_absolute():
                    directory = row.get('PathName_' + ch, '')
                    if pd.isna(directory):
                        directory = ''
                    p = Path(str(directory)) / p
                if not p.is_absolute():
                    if image_root is None:
                        raise ValueError(f'Relative image path requires image_root: {p}')
                    p = Path(image_root) / p
                paths[ch] = str(p.resolve())
            for kind in ('Cells', 'Nuclei'):
                paths[kind + 'Outlines'] = str(root / plate / f'{kind}Outlines_{plate}_{well}_{site}.tiff')
            # Missing intensity images remain errors, even for excluded FOVs.
            for ch in channels:
                if not Path(paths[ch]).is_file():
                    raise FileNotFoundError(f'Missing intensity image: {paths[ch]}')
            missing_outlines = [paths[k] for k in ('CellsOutlines', 'NucleiOutlines')
                                if not Path(paths[k]).is_file()]
            if missing_outlines:
                if skip_missing_outlines:
                    skipped += 1
                    continue
                raise FileNotFoundError('Missing outlines: ' + ', '.join(missing_outlines))
            # Preserve every original LoadData field as JSON without coercing IDs.
            metadata = json.loads(row.to_json())
            stats = {k: {'size': Path(p).stat().st_size, 'mtime_ns': Path(p).stat().st_mtime_ns} for k,p in paths.items()}
            item = {
                'dataset_id': dataset_id,
                'fov_id': 'fov_' + digest([dataset_id, *key])[:24],
                'plate': plate,
                'well': well,
                'site': site,
                'paths': paths,
                'channels': channels,
                'metadata': metadata,
                'source_stats': stats,
                'pixel_size_um': pixel_size_um,
            }
            item['fingerprint'] = digest([VERSION, item])
            rows.append(item)
        except (ValueError, TypeError, OSError) as exc:
            errors.append(f'row {index}: {exc}')
    if skipped:
        warnings.warn(
            f'Skipped {skipped} FOVs missing cell and/or nuclear outline files; '
            f'{len(rows)} FOVs passed preflight.', stacklevel=2)
    if errors:
        raise ValueError('Preflight failed:\n' + '\n'.join(errors[:50]) + (f'\n... {len(errors)} errors total' if len(errors)>50 else ''))
    if df.empty:
        raise ValueError('Empty LoadData table')
    return rows


def read_2d(path):
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f'Expected a single 2D grayscale TIFF, got {arr.shape}: {path}')
    if arr.dtype.kind not in 'buif' or not np.isfinite(arr).all():
        raise ValueError(f'Unsupported or nonfinite pixels: {path}')
    return arr


def read_outline(path):
    """Read binary outlines as YX bool, including colored RGB exports.

    RGB channels must be empty or encode the same boundary. Alpha is ignored.
    Non-RGB stacks are accepted only when all pages are exact duplicates.
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        axes = tif.series[0].axes
        is_rgb = tif.pages[0].photometric == 2
    if arr.dtype.kind not in 'buif' or not np.isfinite(arr).all():
        raise ValueError(f'Unsupported or nonfinite outline pixels: {path}')
    if is_rgb:
        if arr.ndim != 3 or axes not in ('YXS', 'SYX'):
            raise ValueError(f'Unsupported RGB outline layout {axes} {arr.shape}: {path}')
        planes = np.moveaxis(arr, axes.index('S'), 0)
        if planes.shape[0] not in (3, 4):
            raise ValueError(f'Expected RGB or RGBA outline: {path}')
        planes = planes[:3]  # An opaque alpha channel is not a boundary.
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 3 and axes.endswith('YX'):
            if not np.all(arr == arr[0]):
                raise ValueError(f'Outline pages differ; expected one 2D boundary: {path}')
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f'Unsupported outline layout {axes} {arr.shape}: {path}')
        planes = arr[None, ...]
    boundary = np.zeros(planes.shape[1:], dtype=bool)
    found = False
    for plane in planes:
        values = np.unique(plane)
        if np.any(values < 0) or np.count_nonzero(values) > 1:
            raise ValueError(f'Expected zero background and one positive outline value: {path}')
        mask = plane > 0
        if not mask.any():
            continue
        if found and not np.array_equal(boundary, mask):
            raise ValueError(f'RGB outline channels contain different boundaries: {path}')
        boundary = mask
        found = True
    return boundary


def outlines_to_labels(outline):
    """Closed interiors only, 4-connected; outline pixels remain background.

    Open contours (including edge-cut objects) do not enclose an interior.
    No morphology, boundary expansion, size filtering or re-segmentation.
    """
    if outline.ndim != 2 or len(np.unique(outline)) > 2:
        raise ValueError('Outlines must be 2D binary images (zero background)')
    boundary = outline > 0
    labels, _ = ndi.label(ndi.binary_fill_holes(boundary) & ~boundary)
    return labels.astype(np.uint32)


def objects_for(labels, compartment, fov_id, mask_set_id):
    rows = []
    for label, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        yy, xx = np.nonzero(labels[box] == label)
        y0, x0 = box[0].start, box[1].start
        rows.append({
            'object_id': f'{mask_set_id}:{compartment}:{label}',
            'fov_id': fov_id,
            'mask_set_id': mask_set_id,
            'compartment': compartment,
            'label': label,
            'centroid_y': float(yy.mean() + y0),
            'centroid_x': float(xx.mean() + x0),
            'y0': y0,
            'x0': x0,
            'y1': box[0].stop,
            'x1': box[1].stop,
            'area_pixels': len(yy),
            'touches_border': bool(
                y0 == 0 or x0 == 0
                or box[0].stop == labels.shape[0] or box[1].stop == labels.shape[1]
            ),
        })
    return rows


def relationships(nuclei, cells, mask_set_id):
    rows = []
    sizes = np.bincount(nuclei.ravel())
    valid = (nuclei > 0) & (cells > 0)
    pairs, counts = np.unique(np.stack([nuclei[valid],cells[valid]], axis=1), axis=0, return_counts=True)
    best = {}
    for (n,c), count in zip(pairs, counts):
        if n not in best or count > best[n][1]:
            best[n] = (c,count)
    for (n,c), count in zip(pairs, counts):
        rows.append({
            'nucleus_id': f'{mask_set_id}:nuclei:{n}',
            'cell_id': f'{mask_set_id}:cells:{c}',
            'overlap_pixels': int(count),
            'nucleus_overlap_fraction': float(count / sizes[n]),
            'is_best_match': bool(best[n][0] == c),
        })
    return rows


def write_table(rows, path, columns=None):
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(path, index=False)


def _process(item, output):
    from ome_arrow import OMEArrowDataset, from_numpy, write_ome_arrow_dataset
    fov_id = item['fov_id']
    parent = Path(output) / 'fovs'
    dest = parent / fov_id
    if dest.exists():
        done = json.loads((dest / 'complete.json').read_text())
        if done['fingerprint'] != item['fingerprint']:
            raise ValueError(f'{fov_id}: inputs changed; use a new output collection')
        return {**done, 'resumed': True}
    tmp = parent / ('.' + fov_id + '.' + uuid.uuid4().hex)
    tmp.mkdir()
    start = time.perf_counter()
    try:
        arrays = {ch: read_2d(item['paths'][ch]) for ch in item['channels']}
        shape = next(iter(arrays.values())).shape
        masks = {kind: outlines_to_labels(read_outline(item['paths'][prefix+'Outlines'])) for kind,prefix in [('cells','Cells'),('nuclei','Nuclei')]}
        if any(a.shape != shape for a in [*arrays.values(),*masks.values()]):
            raise ValueError(f'{fov_id}: image/mask shapes differ')
        mask_set_id = 'mask_' + digest([fov_id, {k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in masks.items()}, 'interiors-4conn-v1'])[:24]
        records, registry, expected = [], [], {}
        for role, data in [('intensity',arrays),('labels',masks)]:
            for name, arr in data.items():
                image_id = 'img_' + digest([fov_id,role,name,mask_set_id if role=='labels' else 'reference'])[:24]
                # Pixel units explicitly mark unknown physical calibration.
                size = item['pixel_size_um']
                record = from_numpy(arr, dim_order='YX', image_id=image_id, name=name,
                    image_type='label' if role=='labels' else 'image', channel_names=[name],
                    clamp_to_uint16=False, chunk_encoding='bytes',
                    physical_size_x=size or 1., physical_size_y=size or 1., physical_size_unit='µm' if size else 'pixel')
                records.append(record)
                expected[image_id] = arr
                registry.append({
                    'image_id': image_id,
                    'fov_id': fov_id,
                    'dataset_id': item['dataset_id'],
                    'reference_image_id': image_id,
                    'mask_set_id': mask_set_id,
                    'role': role,
                    'channel': name,
                    'variant': 'reference',
                    'dtype': str(arr.dtype),
                    'intensity_scale': 'raw' if role == 'intensity' else 'labels',
                    'normalization_divisor': 65535. if role == 'intensity' else None,
                    'ome_arrow_uri': f'fovs/{fov_id}/pixels.ome-arrow',
                    'height': shape[0],
                    'width': shape[1],
                })
        write_ome_arrow_dataset(records, tmp/'pixels.ome-arrow', pixel_dtype=None, compression='zstd')
        stored = OMEArrowDataset(tmp/'pixels.ome-arrow')
        for image_id, arr in expected.items():
            actual = stored.read_image(image_id)[0,0,0]
            if actual.dtype != arr.dtype or not np.array_equal(actual, arr):
                raise RuntimeError(f'Pixel round-trip failed: {image_id}')
        del stored
        objs = [r for kind,mask in masks.items() for r in objects_for(mask,kind,fov_id,mask_set_id)]
        rels = relationships(masks['nuclei'], masks['cells'], mask_set_id)
        write_table(registry,tmp/'image_registry.parquet')
        write_table(objs,tmp/'objects.parquet',OBJECT_COLUMNS)
        write_table(rels,tmp/'relationships.parquet',REL_COLUMNS)
        fov = {k:item[k] for k in ('dataset_id','fov_id','plate','well','site','pixel_size_um')}
        fov.update({k: v for k, v in item['metadata'].items() if k.startswith('Metadata_')})
        fov.update(height=shape[0],width=shape[1],mask_set_id=mask_set_id,metadata_json=json.dumps(item['metadata']))
        write_table([fov],tmp/'fov.parquet')
        (tmp/'provenance.json').write_text(json.dumps(item,indent=2))
        # Detect source changes during ingestion (size/mtime, not content hashes).
        for key,p in item['paths'].items():
            st = Path(p).stat()
            if {'size':st.st_size,'mtime_ns':st.st_mtime_ns} != item['source_stats'][key]:
                raise RuntimeError(f'Source changed during ingestion: {p}')
        done = {
            'fov_id': fov_id,
            'fingerprint': item['fingerprint'],
            'elapsed_seconds': time.perf_counter() - start,
            'nuclei': int(masks['nuclei'].max()),
            'cells': int(masks['cells'].max()),
            'unmatched_nuclei': int(masks['nuclei'].max()) - len({r['nucleus_id'] for r in rels}),
            'bytes_written': sum(p.stat().st_size for p in tmp.rglob('*') if p.is_file()),
            'resumed': False,
        }
        (tmp/'complete.json').write_text(json.dumps(done,indent=2))
        tmp.rename(dest)
        return done
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def ingest(inputs, outline_root, output, *, dataset_id, image_root=None, channels=None, pixel_size_um=None, workers=1, limit=None, dry_run=False, skip_missing_outlines=False):
    """Preflight all inputs, then ingest. Rerun identical command to resume.

    A single coordinator owns output; workers process disjoint FOVs. Completed
    shards are immutable. Root catalogs become visible after complete success.
    """
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('workers and limit must be positive')
    items = plan(inputs,outline_root,dataset_id=dataset_id,image_root=image_root,channels=channels,pixel_size_um=pixel_size_um,
                 skip_missing_outlines=skip_missing_outlines)
    if limit is not None:
        items = items[:limit]
    if not items:
        return pd.DataFrame(columns=['fov_id', 'plate', 'well', 'site', 'channels'])
    if dry_run:
        return pd.DataFrame([{'fov_id':i['fov_id'],'plate':i['plate'],'well':i['well'],'site':i['site'],'channels':i['channels']} for i in items])
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock = output/'.ingest.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        (output/'fovs').mkdir(exist_ok=True)
        spec = {'version':VERSION,'dataset_id':dataset_id,'items':{i['fov_id']:i['fingerprint'] for i in items}}
        config = output/'collection.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise ValueError('Output belongs to a different input selection/configuration; choose another output directory')
        if not config.exists():
            config.write_text(json.dumps(spec,indent=2))
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda i:_process(i,output),items):
                results.append(result)
                print(f"[{len(results)}/{len(items)}] {result['fov_id']} {result['elapsed_seconds']:.2f}s resumed={result['resumed']}",flush=True)
        for filename in ('fov.parquet','image_registry.parquet','objects.parquet','relationships.parquet'):
            paths = [output/'fovs'/i['fov_id']/filename for i in items]
            schema = pa.unify_schemas([pq.read_schema(p) for p in paths], promote_options='permissive')
            name = 'fovs.parquet' if filename=='fov.parquet' else filename
            tmp = output/(name+'.tmp')
            with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
                for path in paths:
                    writer.write_table(pq.read_table(path).cast(schema))
            tmp.replace(output/name)
        pd.DataFrame(results).to_parquet(output/'timings.parquet',index=False)
        (output/'_SUCCESS').write_text(digest(spec))
        return pd.DataFrame(results)
    finally:
        lock.unlink()


def load_plane(root, image_id, *, normalize=False):
    """Load a YX plane from a completed collection; optionally /65535 intensities."""
    from ome_arrow import OMEArrowDataset
    rows = pq.read_table(Path(root)/'image_registry.parquet', filters=[('image_id','=',image_id)]).to_pylist()
    if len(rows)!=1:
        raise KeyError(image_id)
    row = rows[0]
    arr = OMEArrowDataset(Path(root)/row['ome_arrow_uri']).read_image(image_id)[0,0,0]
    if normalize:
        if row['role'] != 'intensity':
            raise ValueError('Segmentation labels must not be normalized')
        arr = arr.astype(np.float32)/np.float32(65535)
    return arr
