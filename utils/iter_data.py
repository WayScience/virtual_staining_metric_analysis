from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import pandas as pd
import pyarrow as pa
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
    group_columns: str | list[str] = "transform_name",
) -> Iterator[tuple[Path, Any, pd.DataFrame]]:
    """Yield one pandas frame per metric directory and optional grouping."""

    if columns is None:
        columns = NEEDED_COLUMNS

    if isinstance(group_columns, str):
        group_columns = [group_columns]
    else:
        group_columns = list(group_columns)

    for subdir in metric_subdirs:
        metric_files = sorted((subdir / "parquet").glob("*.parquet"))

        if not metric_files:
            continue

        dataset = ds.dataset(metric_files, format="parquet")

        required_columns = set(columns) | set(group_columns)
        missing_columns = required_columns.difference(dataset.schema.names)
        if missing_columns:
            raise ValueError(f"{subdir} is missing required columns: {sorted(missing_columns)}")

        # No additional grouping: yield the entire metric dataset.
        if not group_columns:
            frame = dataset.to_table(
                columns=list(columns),
                use_threads=True,
            ).to_pandas()

            yield subdir, None, frame
            continue

        # Discover unique grouping combinations.
        group_frame = (
            dataset.to_table(columns=group_columns)
            .to_pandas()
            .dropna()
            .drop_duplicates()
            .sort_values(group_columns)
        )

        for group_values in group_frame.itertuples(index=False, name=None):
            filter_expr = None

            for column, value in zip(group_columns, group_values):
                expr = ds.field(column) == value
                filter_expr = expr if filter_expr is None else filter_expr & expr

            frame = dataset.to_table(
                columns=list(columns),
                filter=filter_expr,
                use_threads=True,
            ).to_pandas()

            group_key = group_values[0] if len(group_columns) == 1 else group_values

            yield subdir, group_key, frame
