"""
Helper utilities for writing normalized images along with metadata as parquet files.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

_IMAGE_COLUMNS = {
    "record_id",
    "dataset_index",
    "channel",
    "pixels",
    "shape",
    "axes",
    "dtype",
    "byte_order",
    "source_image_file",
}


def make_record_id(
    dataset_index: int,
    channel: str,
) -> str:
    """Return a stable identifier for one image-channel record."""
    if dataset_index < 0:
        raise ValueError("dataset_index must be non-negative.")

    channel = str(channel)
    if not channel:
        raise ValueError("channel must not be empty.")

    return f"{dataset_index:09d}:{channel}"


def _image_record_schema(image_metadata: pd.DataFrame) -> pa.Schema:
    """
    Combine fixed image-record fields with the existing metadata schema.
    """
    conflicts = _IMAGE_COLUMNS.intersection(image_metadata.columns)
    conflicts.discard("dataset_index")

    if conflicts:
        raise ValueError(f"Image metadata conflicts with image-record columns: {sorted(conflicts)}")

    metadata = image_metadata.drop(
        columns=["dataset_index"],
        errors="ignore",
    )
    metadata_fields = pa.Table.from_pandas(
        metadata,
        preserve_index=False,
    ).schema

    return pa.schema(
        [
            # Fixed image-record field
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("dataset_index", pa.int64(), nullable=False),  # dataset __get_item__ index
            pa.field("channel", pa.string(), nullable=False),  # channel name from original loaddata
            pa.field("pixels", pa.large_binary(), nullable=False),
            pa.field("shape", pa.list_(pa.int32()), nullable=False),
            pa.field("axes", pa.string(), nullable=False),  # axis order of the image array
            pa.field("dtype", pa.string(), nullable=False),  # data type of the image array
            pa.field("byte_order", pa.string(), nullable=False),
            pa.field(
                "source_image_file", pa.string(), nullable=False
            ),  # original source image file from loaddata
            # Plus any additional metadata fields from metadata
            *metadata_fields,
        ],
        metadata={
            b"pixel_encoding": b"C-contiguous little-endian float32",
        },
    )


def _write_parquet_shard(
    rows: list[dict[str, Any]],
    path: Path,
    schema: pa.Schema,
    overwrite: bool,
) -> None:
    """
    Atomically write one Parquet shard.

    Existing shards are retained when overwrite=False, matching the previous
    TIFF writer's behavior for existing output files.
    """
    if path.exists() and not overwrite:
        return

    table = pa.Table.from_pylist(rows, schema=schema)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
        row_group_size=len(rows),
    )
    temporary_path.replace(path)


def write_normalized_images_parquet(
    dataset: Any,
    image_metadata: pd.DataFrame,
    output_dir: str | Path,
    image_indices: Sequence[int] | None = None,
    *,
    overwrite: bool = False,
    shard_size: int = 128,
    index_filename: str = "index.parquet",
) -> pd.DataFrame:
    """
    Write normalized, multi-channel images as Parquet records containing image bytes
    and their corresponding image metadata.

    Each Parquet row represents one image of a specific channel. Pixels are stored as
    C-contiguous, little-endian float32 bytes and can be reconstructed using
    the accompanying shape, dtype, and byte-order fields.

    :params dataset: Dataset returning ``(input_stack, target_stack)`` for each image.
    :params image_metadata: Metadata ordered so that row position matches
    the dataset __get_item__ index.
    :params output_dir: Root output directory.
    :params image_indices: Image indices to write. All dataset images are used when omitted.
    :params overwrite: Rewrite existing Parquet shards when True.
    :params shard_size: Maximum number of image-channel records per Parquet shard.
    :params index_filename: Name of the lightweight Parquet index.
    :returns: One row per image channel, including metadata and its Parquet shard location.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be positive.")

    output_dir = Path(output_dir)
    parquet_root = output_dir / "parquet"
    parquet_root.mkdir(parents=True, exist_ok=True)

    if image_indices is None:
        image_indices = range(len(dataset))

    image_indices = list(image_indices)
    if not image_indices:
        raise ValueError("image_indices did not contain any images.")

    selected_metadata = image_metadata.iloc[image_indices]
    schema = _image_record_schema(selected_metadata)

    input_channels = list(dataset.input_channel_keys)
    target_channels = list(dataset.target_channel_keys)

    shard_rows: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []
    shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_rows, shard_index

        if not shard_rows:
            return

        shard_path = parquet_root / f"part-{shard_index:06d}.parquet"
        _write_parquet_shard(
            shard_rows,
            shard_path,
            schema,
            overwrite,
        )

        shard_rows = []
        shard_index += 1

    for image_dataset_index in tqdm(
        image_indices,
        desc="Writing normalized images",
    ):
        metadata = image_metadata.iloc[image_dataset_index]

        if int(metadata["dataset_index"]) != image_dataset_index:
            raise ValueError("image_metadata is not ordered by dataset_index.")

        input_stack, target_stack = dataset[image_dataset_index]
        metadata_values = metadata.drop(
            labels=["dataset_index"],
            errors="ignore",
        ).to_dict()

        for channel_names, image_stack in (
            (input_channels, input_stack),
            (target_channels, target_stack),
        ):
            if image_stack.shape[0] != len(channel_names):
                raise ValueError("Tensor channel count does not match the dataset channel keys.")

            for channel_position, channel in enumerate(channel_names):
                channel = str(channel)
                record_id = make_record_id(
                    dataset_index=image_dataset_index,
                    channel=channel,
                )

                image = image_stack[channel_position]
                if hasattr(image, "detach"):
                    image = image.detach()
                if hasattr(image, "cpu"):
                    image = image.cpu()
                if hasattr(image, "numpy"):
                    image = image.numpy()

                image = np.asarray(
                    image,
                    dtype=np.dtype("<f4"),
                    order="C",
                )

                if image.ndim != 2:
                    raise ValueError(f"Expected a 2D {channel} image, found shape {image.shape}.")

                current_shard = parquet_root / f"part-{shard_index:06d}.parquet"
                row_in_shard = len(shard_rows)

                shard_rows.append(
                    {
                        "record_id": record_id,
                        "dataset_index": image_dataset_index,
                        "channel": channel,
                        "pixels": image.tobytes(order="C"),
                        "shape": list(image.shape),
                        "axes": "YX",
                        "dtype": "float32",
                        "byte_order": "little",
                        "source_image_file": str(metadata[channel]),
                        **metadata_values,
                    }
                )

                index_records.append(
                    {
                        "record_id": record_id,
                        "dataset_index": image_dataset_index,
                        "channel": channel,
                        "parquet_path": current_shard.relative_to(output_dir).as_posix(),
                        "row_in_shard": row_in_shard,
                        "source_image_file": str(metadata[channel]),
                    }
                )

                if len(shard_rows) == shard_size:
                    flush_shard()

    flush_shard()

    file_index = pd.DataFrame.from_records(index_records)
    write_index = file_index.merge(
        selected_metadata,
        on="dataset_index",
        how="left",
        validate="many_to_one",
    )

    index_path = output_dir / index_filename
    temporary_index_path = index_path.with_name(f".{index_path.name}.tmp")
    write_index.to_parquet(temporary_index_path, index=False)
    temporary_index_path.replace(index_path)

    print(
        f"Indexed {len(write_index):,} image records "
        f"across {shard_index:,} Parquet shards in {index_path}"
    )

    return write_index
