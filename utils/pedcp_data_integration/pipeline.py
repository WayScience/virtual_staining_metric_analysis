"""Data ingestion pipeline wrangling and setting up pedcp90 for subsequent analysis"""

from pathlib import Path

import pandas as pd

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.ingest import ingest
from utils.pedcp_data_integration import plots
from utils.pedcp_data_integration.artifacts import Artifacts
from utils.pedcp_data_integration.download import download_metadata
from utils.pedcp_data_integration.file_index import prepare_file_index as prepare
from utils.pedcp_data_integration.recipe import (
    CHANNELS,
    DATASET_ID,
    ILLUMINATION_METHODS,
    PIXEL_SIZE_UM,
)


def ingest_prepared(config, dry_run=False) -> pd.DataFrame:
    """Ingest prepared profiling metadata indexing tiff files"""
    artifacts = Artifacts(config)
    if config.outline_root is None or not config.outline_root.is_dir():
        raise ValueError("Ingest requires an existing outline_root directory")
    prepared = pd.read_parquet(artifacts.export_parquet)
    illumination = {
        (plate, channel): (
            artifacts.raw / "illumination" / plate /
            f"{plate}_Illum{channel.removeprefix('Orig')}.npy", method
        )
        for plate in prepared.Metadata_Plate.unique()
        for channel, method in ILLUMINATION_METHODS.items()
    }
    results = ingest(
        inputs=[artifacts.export_parquet],
        outline_root=config.outline_root,
        output=artifacts.collection,
        dataset_id=DATASET_ID,
        illumination=illumination,
        channels=list(CHANNELS),
        image_root=config.image_root,
        pixel_size_um=PIXEL_SIZE_UM,
        workers=config.workers,
        skip_missing_outlines=True,
        dry_run=dry_run,
    )
    if results.empty:
        raise ValueError("No eligible FOVs with both outlines; ingestion is empty")
    if dry_run:
        print(f"Preflight: {len(results)} FOVs")
        return results
    collection = MicroscopyCollection(artifacts.collection)
    counts = (
        pd.DataFrame(
            {
                "Prepared": prepared.Metadata_Plate.value_counts(),
                "Written": collection.fovs.plate.value_counts(),
            }
        )
        .fillna(0)
        .sort_index()
    )
    plots.plot_counts(
        counts, artifacts.figures / "outline_retention.png", "Outline availability and ingestion"
    )
    print(f"Ingestion: {len(results)} FOVs, {int(results.resumed.sum())} resumed")
    return results


def report(config) -> list[Path]:
    """Generate preview montages for the ingested data."""
    artifacts = Artifacts(config)
    collection = MicroscopyCollection(artifacts.collection)
    return plots.save_montages(
        collection, config.preview_count, config.preview_seed, artifacts.figures
    )


def run(config) -> list[Path]:
    """Run the full data ingestion pipeline and generate reports."""
    download_metadata(config)
    prepare(config)
    ingest_prepared(config)
    return report(config)
