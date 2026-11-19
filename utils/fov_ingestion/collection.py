"""Read completed collections without exposing their per-FOV storage layout.

Catalogs resolve image IDs to pixel shards, including externally linked masks in
derived collections. Source TIFFs and outlines are not needed for these reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.fov_ingestion.writer import digest


class MicroscopyCollection:
    """Validated collection catalogs with lazy pixel reads (no full-FOV cache)."""

    def __init__(self, root):
        """Check completion and catalog identities before making data available."""
        # A completion marker must agree with the exact collection specification.
        self.root = Path(root).resolve()
        self.spec = json.loads((self.root / 'collection.json').read_text())
        self.fingerprint = (self.root / '_SUCCESS').read_text().strip()
        if self.fingerprint != digest(self.spec):
            raise ValueError('Collection completion marker does not match configuration')
        if 'reference_collection_uri' in self.spec:
            # Linked masks are valid only while their source collection is unchanged.
            reference_root = (self.root / self.spec['reference_collection_uri']).resolve()
            if (reference_root / '_SUCCESS').read_text().strip() != self.spec['reference_fingerprint']:
                raise ValueError('Linked reference collection changed')
        # Load metadata eagerly, but leave the much larger pixel arrays on disk.
        self.fovs = pd.read_parquet(self.root / 'fovs.parquet')
        self.images = pd.read_parquet(self.root / 'image_registry.parquet')
        if self.fovs.fov_id.duplicated().any() or self.images.image_id.duplicated().any():
            raise ValueError('Duplicate FOV/image identifiers')
        if set(self.fovs.fov_id) != set(self.spec['items']):
            raise ValueError('FOV catalog differs from collection specification')

    def metadata(self):
        """Expose all LoadData fields, including non-Metadata_* columns."""
        # Canonical catalog columns take precedence over the preserved source JSON.
        result = self.fovs.copy()
        extra = pd.json_normalize(result.metadata_json.map(json.loads))
        for column in extra:
            if column not in result:
                result[column] = extra[column].to_numpy()
        return result

    def read_fov(self, fov_id, *, channels=None, include_masks=True, normalize=False):
        """Return full YX planes keyed by (role, channel), opening each URI once.

        ``channels`` selects intensity planes; labels remain unless include_masks
        is false. Normalization is opt-in and applies only to intensities. Calls
        load full planes afresh, so repeated crop readers should cache externally.
        """
        from ome_arrow import OMEArrowDataset

        # Resolve the requested planes first, before opening any pixel shard.
        rows = self.images.loc[self.images.fov_id.eq(fov_id)]
        if rows.empty:
            raise KeyError(fov_id)
        if channels is not None:
            absent = set(channels) - set(rows.loc[rows.role.eq('intensity'), 'channel'])
            if absent:
                raise ValueError(f'Channels absent from {fov_id}: {sorted(absent)}')
            rows = rows.loc[rows.role.eq('labels') | rows.channel.isin(channels)]
        if not include_masks:
            rows = rows.loc[rows.role.eq('intensity')]
        # Group by URI because one FOV may combine local images and linked labels.
        result = {}
        for uri, group in rows.groupby('ome_arrow_uri', sort=False):
            dataset = OMEArrowDataset(self.root / uri)
            for row in group.itertuples(index=False):
                arr = dataset.read_image(row.image_id)[0, 0, 0]
                # Label integers encode object identities, not intensity values.
                if normalize and row.role == 'intensity':
                    arr = arr.astype(np.float32) / np.float32(row.normalization_divisor)
                result[(row.role, row.channel)] = arr
        return result

    def object_tables(self, fov_id):
        """Read object geometry and nucleus-cell overlaps for a cataloged FOV."""
        if fov_id not in self.spec['items']:
            raise KeyError(fov_id)
        # Objects retain their canonical reference identities in derived collections.
        folder = self.root / 'fovs' / fov_id
        return (pd.read_parquet(folder / 'objects.parquet'),
                pd.read_parquet(folder / 'relationships.parquet'))
