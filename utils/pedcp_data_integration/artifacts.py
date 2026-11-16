"""Data ingestion artifact write"""

import os
import tempfile
from pathlib import Path

from utils.pedcp_data_integration.recipe import SOURCE_COMMIT


class Artifacts:
    def __init__(self, config):
        """Initialize artifact paths based on the given configuration."""
        self.root = config.metadata_root / "pedcp90"
        self.raw = self.root / "raw" / SOURCE_COMMIT
        self.export_parquet = self.root / "loaddata_for_degradation.parquet"
        self.collection = config.large_output_root / "pedcp90" / "reference"
        self.figures = Path(__file__).resolve().parents[2] / "0.data_ingestion" / "results"


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
