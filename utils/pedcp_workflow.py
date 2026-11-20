"""
Shared locations and read-only discovery for the corrected PedCP blur workflow.

Constructing paths requires no source mounts or prepared metadata. Completed
collections carry the metadata needed by downstream consumers.
"""
from dataclasses import dataclass
from pathlib import Path

from utils.degradation.transforms import gaussian_pilot_specs
from utils.fov_ingestion.collection import MicroscopyCollection
from utils.pedcp_data_integration.artifacts import Artifacts
from utils.project_config import ProjectConfig


@dataclass(frozen=True)
class WorkflowPaths:
    config: ProjectConfig

    @property
    def reference(self) -> Path:
        return Artifacts(self.config).collection

    @property
    def blur(self) -> Path:
        return self.reference.parent / 'degradation' / 'blur'

    @property
    def degradations(self) -> Path:
        return self.reference.parent / 'degradation' / 'full'

    @property
    def morphem_profiles(self) -> Path:
        return self.reference.parent / 'profiling' / 'morphem_blur_pilot'

    @property
    def cp_profiles(self) -> Path:
        return self.reference.parent / 'profiling' / 'cp_blur_pilot'

    @property
    def morphem_embedding(self) -> Path:
        return self.morphem_profiles.parent / 'morphem_blur_balanced_all_cell_types'

    @property
    def morphem_per_channel(self) -> Path:
        return self.morphem_profiles.parent / 'morphem_blur_per_channel'

    @property
    def cp_embedding(self) -> Path:
        return self.cp_profiles / 'analysis_embedding'

    @property
    def cp_per_channel(self) -> Path:
        return self.cp_profiles / 'analysis_embedding_per_channel'

    @property
    def cp_single_cell(self) -> Path:
        return self.cp_profiles / 'analysis_single_cell_knn_pilot'


def discover_blur_variants(reference, root, channels=None) -> list[MicroscopyCollection]:
    """Find exactly one completed low/mid/high variant of this reference.

    Match transform semantics, not software-dependent directory hashes. Incomplete
    children and unrelated transforms are ignored; stale references, unavailable
    channels and ambiguous matches are reported rather than silently profiled.
    """
    import json

    reference = (reference if isinstance(reference, MicroscopyCollection)
                 else MicroscopyCollection(reference))
    root = Path(root)
    expected = dict(gaussian_pilot_specs())
    matches = {level: [] for level in expected}
    rejected = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.name.startswith('.') or not (path / '_SUCCESS').is_file():
            continue
        try:
            spec = json.loads((path / 'collection.json').read_text())
            transform = spec.get('transform', {})
            levels = [level for level, wanted in expected.items()
                      if transform.get('name') == wanted.name
                      and transform.get('parameters') == wanted.parameters
                      and transform.get('implementation') == wanted.implementation
                      and transform.get('mask_policy') == 'fixed_reference']
            if not levels or spec.get('kind') != 'degradation':
                continue
            if spec.get('reference_fingerprint') != reference.fingerprint:
                raise ValueError('different reference fingerprint')
            collection = MicroscopyCollection(path)
            required = set(channels if channels is not None else spec.get('channels', []))
            for fid in collection.spec['items']:
                rows = collection.images.loc[collection.images.fov_id.eq(fid)
                                             & collection.images.role.eq('intensity')]
                if required - set(rows.channel):
                    raise ValueError(f'{fid}: missing channels {sorted(required - set(rows.channel))}')
            matches[levels[0]].append(collection)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append(f'{path.name}: {exc}')
    problems = [f'{level}: expected one completed variant, found {len(found)}'
                for level, found in matches.items() if len(found) != 1]
    if problems:
        detail = '; '.join([*problems, *rejected])
        raise ValueError(f'Cannot discover blur pilot in {root}: {detail}. '
                         'Generate the pilot or supply explicit --variants paths.')
    return [matches[level][0] for level in expected]


def discover_degradation_variants(reference, root):
    """Open the production catalog by readable name and validate its semantics.

    Dependency versions may differ from today's environment; saved implementation
    and parameters must still match. No hash-directory guessing or pilot mixing.
    """
    from utils.degradation.transforms import degradation_specs

    reference = (reference if isinstance(reference, MicroscopyCollection)
                 else MicroscopyCollection(reference))
    variants = []
    for level, wanted in degradation_specs():
        path = Path(root) / wanted.name / level
        collection = MicroscopyCollection(path)
        descriptor = collection.spec.get('transform', {})
        if (collection.spec.get('kind') != 'degradation'
                or collection.spec.get('reference_fingerprint') != reference.fingerprint
                or (collection.root / collection.spec['reference_collection_uri']).resolve()
                != reference.root):
            raise ValueError(f'{path}: not a degradation of this reference')
        for field in ('name', 'parameters', 'implementation', 'mask_policy', 'working_scale',
                      'base_seed', 'input_scale', 'output_scale', 'output_dtype', 'seed_policy'):
            if descriptor.get(field) != wanted.descriptor().get(field):
                raise ValueError(f'{path}: transform {field} differs from the production catalog')
        variants.append((level, collection))
    return variants
