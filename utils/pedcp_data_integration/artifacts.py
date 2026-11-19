"""Data ingestion artifact write"""

import os
import tempfile
from pathlib import Path

from utils.pedcp_data_integration.recipe import COHORTS, DATASET_ID, SOURCE_COMMIT, TRAIN_DENSITIES


class Artifacts:
    def __init__(self, config, density=None):
        """Resolve one collection; the legacy reference paths and identity are unchanged."""
        if density is not None and density not in TRAIN_DENSITIES:
            raise ValueError(f"Unknown training density: {density}")
        self.density = density
        self.name = "reference" if density is None else f"density_{density}"
        self.dataset_id = DATASET_ID if density is None else f"pedcp90-training-density-{density}"
        self.root = config.metadata_root / "pedcp90"
        self.raw = self.root / "raw" / SOURCE_COMMIT
        self.export_parquet = self.root / "loaddata_for_degradation.parquet"
        self.collection = config.large_output_root / "pedcp90" / "reference"
        self.figures = Path(__file__).resolve().parents[2] / "0.data_ingestion" / "results"
        if density is not None:
            self.export_parquet = self.root / "training" / self.name / "loaddata.parquet"
            self.collection = config.large_output_root / "pedcp90" / "training" / self.name
            self.figures = self.figures / "training" / self.name


def selected_artifacts(config, cohort="all") -> dict[str, Artifacts]:
    """Expand the fixed CLI selection in deterministic reference/density order."""
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}; expected one of {COHORTS}")
    densities = ([None] if cohort != "training" else []) + (
        list(TRAIN_DENSITIES) if cohort != "reference" else []
    )
    return {a.name: a for a in (Artifacts(config, density) for density in densities)}


def atomic_write(path: Path, callback) -> None:
    """Atomically write to a file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        callback(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
