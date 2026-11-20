"""
Full degradation pipeline.
"""

import json
from pathlib import Path

import pandas as pd

from utils.degradation.degrade import degrade_collection
from utils.degradation.transforms import degradation_specs
from utils.fov_ingestion.collection import MicroscopyCollection
from utils.pedcp_workflow import discover_degradation_variants


def run(reference, output_root, *, channels=None, fov_ids=None, workers=1, dry_run=False):
    """Process variants sequentially; parallelism stays bounded to FOV workers."""
    reference = (reference if isinstance(reference, MicroscopyCollection)
                 else MicroscopyCollection(reference))
    catalog = []
    for level, transform in degradation_specs():
        destination = Path(output_root) / transform.name / level
        print(f'{transform.name} / {level}: {transform.parameters} -> {destination}', flush=True)
        result = degrade_collection(reference, destination, transform, channels=channels,
                                    fov_ids=fov_ids, workers=workers, dry_run=dry_run)
        print(result.to_string(index=False), flush=True)
        catalog.append({'transform': transform.name, 'level': level,
                        'variant_id': transform.variant_id, 'collection': str(destination),
                        'descriptor_json': json.dumps(transform.descriptor(), sort_keys=True)})
    frame = pd.DataFrame(catalog)

    if not dry_run:
        # using csv for the sake of human readability and easy inspection
        # this is a very small catalog to begin with, with string identifiers for transforms and levels
        # for plotting so parquet wouldn't provide much additional benefit
        frame.to_csv(Path(output_root) / 'degradation_catalog.csv', index=False)
    return frame


def report(reference, output_root, output_dir, *, fov_id=None, channels=None,
           crop_size=256, seed=42):
    """
    Report for the degradation module is simply montages of the degradations.
    As the exact same processed data is used and already reported by 0.data_ingestion.    
    """
    from utils.degradation.plots import save_degradation_montages

    reference = (reference if isinstance(reference, MicroscopyCollection)
                 else MicroscopyCollection(reference))
    variants = discover_degradation_variants(reference, output_root)
    candidates = reference.metadata().sort_values('fov_id')
    for _, collection in variants:
        candidates = candidates.loc[candidates.fov_id.isin(collection.fovs.fov_id)]
    if candidates.empty:
        raise ValueError('No FOV is shared by the reference and all degradation variants')
    if fov_id is None:
        fov_id = candidates.sample(n=1, random_state=seed).iloc[0].fov_id
    elif fov_id not in set(candidates.fov_id):
        raise ValueError(f'{fov_id}: FOV is not shared by all degradation variants')
    if channels is None:
        channels = variants[0][1].spec['channels']
        
    return save_degradation_montages(reference, variants, fov_id, output_dir,
                                    channels=channels, crop_size=crop_size)
