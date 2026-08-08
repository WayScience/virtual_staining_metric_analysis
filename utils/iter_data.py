from collections.abc import Iterator
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

NEEDED_COLUMNS = [
    "record_id",
    "channel",
    "Metadata_Plate",
    "Metadata_Well",
    "transform_name",
    "parameter_name",
    "parameter_value",
    "metric_value",
]


def iter_lance_fragments(
    dataset_path: Path,
) -> Iterator[tuple[int, int, pa.Table]]:
    dataset = lance.dataset(dataset_path)

    fragments = sorted(
        dataset.get_fragments(),
        key=lambda fragment: fragment.fragment_id,
    )

    for position, fragment in enumerate(fragments):
        yield (
            position,
            fragment.fragment_id,
            fragment.scanner(
                blob_handling="all_binary",
            ).to_table(),
        )


def iter_metric_transform_frames(
    metric_subdirs: list[Path],
    *,
    columns: list[str] | None = None,
) -> Iterator[tuple[Path, str, pd.DataFrame]]:
    """Yield one pandas frame per metric directory and transform."""

    if columns is None:
        columns = NEEDED_COLUMNS

    for subdir in metric_subdirs:
        metric_files = sorted((subdir / "parquet").glob("*.parquet"))

        if not metric_files:
            continue

        dataset = ds.dataset(metric_files, format="parquet")

        missing_columns = set(columns).difference(dataset.schema.names)
        if missing_columns:
            raise ValueError(f"{subdir} is missing required columns: {sorted(missing_columns)}")

        # Read only transform_name to discover the available groups.
        transform_column = dataset.to_table(
            columns=["transform_name"],
        )["transform_name"]

        transform_names = sorted(
            value for value in pc.unique(transform_column).to_pylist() if value is not None
        )

        for transform_name in transform_names:
            frame = dataset.to_table(
                columns=list(columns),
                filter=(ds.field("transform_name") == transform_name),
                use_threads=True,
            ).to_pandas()

            yield subdir, transform_name, frame
