"""Data ingestion pipeline wrangling and setting up pedcp90 for subsequent analysis"""

import warnings
from pathlib import Path

import pandas as pd

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.ingest import KEYS, ingest, token
from utils.pedcp_data_integration import plots
from utils.pedcp_data_integration.artifacts import selected_artifacts
from utils.pedcp_data_integration.download import download_metadata
from utils.pedcp_data_integration.file_index import prepare_file_index as prepare
from utils.pedcp_data_integration.file_index import training_coverage
from utils.pedcp_data_integration.recipe import (
    CHANNELS,
    CORRECTED_CHANNELS,
    ILLUMINATION_METHODS,
    PIXEL_SIZE_UM,
)


def ingest_prepared(config, dry_run=False, *, cohort="all") -> dict[str, pd.DataFrame]:
    """Write selected independent collections, retaining per-collection worker parallelism."""
    outputs = selected_artifacts(config, cohort)
    if config.outline_root is None or not config.outline_root.is_dir():
        raise ValueError("Ingest requires an existing outline_root directory")
    return {
        name: _ingest_collection(config, artifacts, dry_run=dry_run)
        for name, artifacts in outputs.items()
    }


def _ingest_collection(config, artifacts, *, dry_run):
    """Use the unchanged TIFF/outline adapter for one prepared subset."""
    prepared = pd.read_parquet(artifacts.export_parquet)
    illumination = {
        (plate, channel): (
            artifacts.raw
            / "illumination"
            / plate
            / f"{plate}_Illum{channel.removeprefix('Orig')}.npy",
            method,
        )
        for plate in prepared.Metadata_Plate.unique()
        for channel, method in ILLUMINATION_METHODS.items()
    }
    results = ingest(
        inputs=[artifacts.export_parquet],
        outline_root=config.outline_root,
        output=artifacts.collection,
        dataset_id=artifacts.dataset_id,
        illumination=illumination,
        channels=list(CHANNELS),
        output_channels=CORRECTED_CHANNELS,
        image_root=config.image_root,
        pixel_size_um=PIXEL_SIZE_UM,
        workers=config.workers,
        skip_missing_outlines=True,
        dry_run=dry_run,
    )
    if results.empty:
        raise ValueError(
            f"{artifacts.name}: no eligible FOVs with both outlines; ingestion is empty"
        )
    collection = None if dry_run else MicroscopyCollection(artifacts.collection)
    if artifacts.density is not None:
        if dry_run:
            # Preflight uses canonical string identities; leave prepared metadata untouched.
            keys = pd.MultiIndex.from_frame(prepared[KEYS].map(token))
            eligible = pd.MultiIndex.from_frame(results[["plate", "well", "site"]])
            written = prepared.loc[keys.isin(eligible)]
        else:
            written = collection.metadata()
        coverage = training_coverage(written)
        counts = pd.concat(
            {
                "Prepared": training_coverage(prepared),
                "Eligible" if dry_run else "Written": coverage,
            },
            axis=1,
        )
        print(f"{artifacts.name}: outline retention\n{counts.to_string()}")
        missing = coverage.index[coverage.FOVs.eq(0)].tolist()
        if missing:
            warnings.warn(
                f"{artifacts.name}: no FOVs with both outlines at timepoints {missing}",
                stacklevel=2,
            )
    if dry_run:
        print(f"{artifacts.name}: preflight {len(results)} FOVs")
        return results
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
    if artifacts.density is not None:
        counts = pd.DataFrame(
            {
                "Prepared": training_coverage(prepared).FOVs,
                "Written": coverage.FOVs,
            }
        )
    plots.plot_counts(
        counts, artifacts.figures / "outline_retention.png", "Outline availability and ingestion"
    )
    print(f"{artifacts.name}: ingestion {len(results)} FOVs, {int(results.resumed.sum())} resumed")
    return results


def report(config, *, cohort="all") -> list[Path]:
    """Generate selected previews without requiring source images or outlines."""
    paths = []
    for artifacts in selected_artifacts(config, cohort).values():
        collection = MicroscopyCollection(artifacts.collection)
        paths.extend(
            plots.save_montages(
                collection, config.preview_count, config.preview_seed, artifacts.figures
            )
        )
    return paths


def run(config, *, cohort="all") -> list[Path]:
    """Run the full data ingestion pipeline and generate reports."""
    download_metadata(config)
    prepare(config, cohort=cohort)
    ingest_prepared(config, cohort=cohort)
    return report(config, cohort=cohort)
