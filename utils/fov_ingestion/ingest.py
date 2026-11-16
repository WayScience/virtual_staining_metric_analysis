"""Immutable, per-FOV OME-Arrow ingestion. No intensity normalization."""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
from scipy import ndimage as ndi

from utils.fov_ingestion.writer import ShardPayload, run_collection

VERSION = "ingest-v1"
KEYS = ['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']
REL_COLUMNS = ['nucleus_id','cell_id','overlap_pixels','nucleus_overlap_fraction','is_best_match']
OBJECT_COLUMNS = ['object_id','fov_id','mask_set_id','compartment','label','centroid_y','centroid_x','y0','x0','y1','x1','area_pixels','touches_border']


def digest(value: dict) -> str:
    """Compute a SHA-256 digest of a dictionary."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def token(value: str | float) -> str:
    """Normalize and validate an identity field for identity columns."""
    if pd.isna(value):
        raise ValueError('Missing identity field')
    if isinstance(value, (float, np.floating)) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    if not s or '/' in s or '\\' in s or s in {'.', '..'}:
        raise ValueError(f'Invalid identity field: {value!r}')
    return s


def read_inputs(inputs: pd.DataFrame | str | Path | list[str | Path]) -> pd.DataFrame:
    """Read input metadata from a DataFrame or file paths"""
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


def plan(
    inputs: pd.DataFrame | str | Path | list[str | Path], 
    outline_root: str | Path, 
    *, 
    dataset_id: str, 
    image_root: str | Path | None = None, 
    channels: list[str] | None = None, 
    pixel_size_um: float | None = None, 
    skip_missing_outlines: bool = False,
    illumination: dict | None = None
):
    """
    Resolve all image+segmentation masks file path and validate against duplicate FOVs.

    :param inputs: CSV or Parquet paths or a DataFrame containing FOV metadata
    :param outline_root: Root directory containing segmentation mask files
    :param dataset_id: Stable, nonempty acquisition namespace
    :param image_root: Root directory for relative image paths
    :param channels: List of channel names corresponding to FileName_* columns
    :param pixel_size_um: Physical pixel size in micrometers
    :param skip_missing_outlines: Whether to skip FOVs missing outlines
    :return: List of resolved FOV rows with absolute paths
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
            corrections = {}
            if illumination is not None:
                for ch in channels:
                    if (plate, ch) not in illumination:
                        raise ValueError(f'Missing illumination mapping: {plate}/{ch}')
                    path, method = illumination[plate, ch]
                    path = Path(path).resolve(strict=True)
                    st = path.stat()
                    if method not in ('subtract', 'divide'):
                        raise ValueError(f'Invalid illumination method: {method}')
                    _illumination_array(str(path), st.st_size, st.st_mtime_ns, method)
                    paths[f'illumination:{ch}'] = str(path)
                    corrections[ch] = {'method': method, 'divisor': 65535.}
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
                'illumination': corrections,
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


def read_2d(path: str | Path):
    """
    Read a single 2D grayscale TIFF image. 
    This project expects CellPainting images to be strictly 2D and grayscale.

    :param path: Path to the TIFF file
    :return: 2D numpy array of image pixels
    :raises ValueError: If the image is not 2D, not grayscale, or contains nonfinite pixels
    """
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f'Expected a single 2D grayscale TIFF, got {arr.shape}: {path}')
    if arr.dtype.kind not in 'buif' or not np.isfinite(arr).all():
        raise ValueError(f'Unsupported or nonfinite pixels: {path}')
    return arr


def read_outline(path: str | Path) -> np.ndarray:
    """
    Read binary outlines as YX bool. 
    The pre-computed profiling data for this project saves outlines as RGB TIFFs (only G channel is used),
        and this function does best guess 3 channel binarization to extract the outline.

    :param path: Path to the outline TIFF file
    :return: 2D boolean array where True indicates outline pixels
    :raises ValueError: If the outline is not valid or contains inconsistent RGB channels
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


def outlines_to_labels(outline: np.ndarray) -> np.ndarray:
    """
    Convert binary outlines to labeled interiors.
    Helps converts the outlines as instance segmentation masks.
    Assumes outlines are closed, 4-connected and no two objects can overlap and
        have outlines intersecting each other, which is the case for this project's data.

    :param outline: 2D binary array of outline pixels
    :return: 2D array of labeled interiors (uint32)
    :raises ValueError: If the input is not a 2D binary image
    """
    if outline.ndim != 2 or len(np.unique(outline)) > 2:
        raise ValueError('Outlines must be 2D binary images (zero background)')
    boundary = outline > 0
    labels, _ = ndi.label(ndi.binary_fill_holes(boundary) & ~boundary)
    return labels.astype(np.uint32)


def objects_for(labels: np.ndarray, compartment: str, fov_id: str, mask_set_id: str) -> list[dict]:
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


def relationships(nuclei: np.ndarray, cells: np.ndarray, mask_set_id: str) -> list[dict]:
    """
    Compute relationships between nuclei and cells based on overlap.

    :param nuclei: 2D array of labeled nuclei
    :param cells: 2D array of labeled cells
    :param mask_set_id: Identifier for the mask set
    :return: List of dictionaries describing nucleus-cell relationships
    """
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


def write_table(rows: list[dict], path: str, columns: list[str] | None = None):
    """
    Write a list of dictionaries to a Parquet file.

    :param rows: List of dictionaries representing table rows
    :param path: Path to the output Parquet file
    :param columns: Optional list of column names to include in the output
    """
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(path, index=False)


@lru_cache(maxsize=36)
def _illumination_array(path, size, mtime_ns, method):
    field = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if field.ndim != 2 or not field.size or not np.isfinite(field).all():
        raise ValueError(f'Invalid illumination array: {path}')
    if method == 'divide' and np.any(field <= 0):
        raise ValueError(f'Nonpositive division field: {path}')
    field.setflags(write=False)
    return field


def _correct_plane(image, field, method):
    if image.dtype != np.uint16 or image.shape != field.shape:
        raise ValueError('Correction requires uint16 TIFF and matching YX illumination shape')
    result = image.astype(np.float32)
    if method == 'subtract':
        result -= np.float32(65535) * field
    else:
        result /= field
    return np.clip(result, 0, 65535, out=result)


def _reference_payload(item: dict) -> ShardPayload:
    """TIFF/outline adapter; storage is delegated to the shared writer."""
    fov_id = item['fov_id']
    arrays = {ch: read_2d(item['paths'][ch]) for ch in item['channels']}
    for ch, correction in item.get('illumination', {}).items():
        key = f'illumination:{ch}'
        stats = item['source_stats'][key]
        field = _illumination_array(item['paths'][key], stats['size'],
                                    stats['mtime_ns'], correction['method'])
        arrays[ch] = _correct_plane(arrays[ch], field, correction['method'])
    shape = next(iter(arrays.values())).shape
    masks = {kind: outlines_to_labels(read_outline(item['paths'][prefix+'Outlines'])) for kind,prefix in [('cells','Cells'),('nuclei','Nuclei')]}
    if any(a.shape != shape for a in [*arrays.values(),*masks.values()]):
        raise ValueError(f'{fov_id}: image/mask shapes differ')
    mask_set_id = 'mask_' + digest([fov_id, {k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in masks.items()}, 'interiors-4conn-v1'])[:24]
    registry, pixels = [], {}
    for role, data in [('intensity', arrays), ('labels', masks)]:
        for name, arr in data.items():
            image_id = 'img_' + digest([fov_id, role, name, mask_set_id if role == 'labels' else 'reference'])[:24]
            pixels[image_id] = arr
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
                'illumination_method': item.get('illumination', {}).get(name, {}).get('method') if role == 'intensity' else None,
                'normalization_divisor': 65535. if role == 'intensity' else None,
                'ome_arrow_uri': f'fovs/{fov_id}/pixels.ome-arrow',
                'height': shape[0],
                'width': shape[1],
            })
    objs = [r for kind,mask in masks.items() for r in objects_for(mask,kind,fov_id,mask_set_id)]
    rels = relationships(masks['nuclei'], masks['cells'], mask_set_id)
    fov = {k:item[k] for k in ('dataset_id','fov_id','plate','well','site','pixel_size_um')}
    fov.update({k: str(v) for k, v in item['metadata'].items() if k.startswith('Metadata_')})
    fov.update(height=shape[0],width=shape[1],mask_set_id=mask_set_id,metadata_json=json.dumps(item['metadata']))
    for key, path in item['paths'].items():
        st = Path(path).stat()
        if {'size': st.st_size, 'mtime_ns': st.st_mtime_ns} != item['source_stats'][key]:
            raise RuntimeError(f'Source changed during ingestion: {path}')
    return ShardPayload(
        arrays=pixels, registry=pd.DataFrame(registry), fov=pd.DataFrame([fov]),
        objects=pd.DataFrame(objs, columns=OBJECT_COLUMNS),
        relationships=pd.DataFrame(rels, columns=REL_COLUMNS), provenance=item,
        statistics={'nuclei': int(masks['nuclei'].max()), 'cells': int(masks['cells'].max()),
                    'unmatched_nuclei': int(masks['nuclei'].max()) - len({r['nucleus_id'] for r in rels})},
    )


def ingest(
    inputs: list[Path | str], 
    outline_root: Path | str, 
    output: Path | str, 
    *, 
    dataset_id: str, 
    image_root: Path | str | None = None, 
    channels: list[str] | None = None, 
    pixel_size_um: float | None = None, 
    workers: int = 1, 
    limit: int | None = None, 
    dry_run: bool = False, 
    skip_missing_outlines: bool = False,
    illumination: dict | None = None
) -> pd.DataFrame:
    """
    Preflight all inputs, then ingest. Rerun identical command to resume.
    
    :param inputs: List of input files or directories to ingest.
    :param outline_root: Root directory containing outline files.
    :param output: Directory to write the ingested output.
    :param dataset_id: Identifier for the dataset.
    :param image_root: Root directory containing image files (optional).
    :param channels: List of channels to ingest (optional).
    :param pixel_size_um: Pixel size in micrometers (optional).
    :param workers: Number of worker threads to use.
    :param limit: Limit the number of items to ingest (optional).
    :param dry_run: If True, only perform a dry run without writing output.
    :param skip_missing_outlines: If True, skip items with missing outlines.
    :return: A pandas DataFrame containing ingestion results.
    """
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('workers and limit must be positive')

    # Preflight all inputs to determine what needs to be ingested
    items = plan(
        inputs,
        outline_root,
        dataset_id=dataset_id,
        image_root=image_root,
        channels=channels,
        pixel_size_um=pixel_size_um,
        skip_missing_outlines=skip_missing_outlines,
        illumination=illumination
    )
    
    if limit is not None:
        items = items[:limit]
    if not items:
        return pd.DataFrame(columns=['fov_id', 'plate', 'well', 'site', 'channels'])
    if dry_run:
        return pd.DataFrame([{'fov_id':i['fov_id'],'plate':i['plate'],'well':i['well'],'site':i['site'],'channels':i['channels']} for i in items])
    spec = {'version': VERSION, 'dataset_id': dataset_id,
            'items': {i['fov_id']: i['fingerprint'] for i in items}}
    by_id = {i['fov_id']: i for i in items}
    return run_collection(output, spec, lambda fov_id: _reference_payload(by_id[fov_id]), workers=workers)


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
