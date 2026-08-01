"""
Utilities for writing reference images to Parquet files with metadata
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from tqdm.auto import tqdm

from .encoding import encode_pixels
from .parquet_writer import Writer

PARQUET_WRITE_ROOT = "parquet"
IMAGE_RECORD_COLUMNS = {
    "record_id",
    "channel",
    "pixels",
    "shape",
    "axes",
    "dtype",
    "byte_order",
    "source_image_file",
}


def _image_record_schema(image_metadata: pd.DataFrame) -> pa.Schema:
    """
    Definition of image-record schema determined by the fixed image-record fields
        and the provided metadata schema. The typing related to image metadata
        columns are less strict and best effort inferred by pyarrow.
    Should nonetheless be sufficient for writing parquet records associated
        with just the image metadata.
    """
    conflicts = IMAGE_RECORD_COLUMNS.intersection(image_metadata.columns)
    if conflicts:
        raise ValueError(f"Image metadata conflicts with image-record columns: {sorted(conflicts)}")

    metadata_fields = pa.Table.from_pandas(
        image_metadata,
        preserve_index=False,
    ).schema

    return pa.schema(
        [
            # Fixed image-record field
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("channel", pa.string(), nullable=False),  # channel name from original loaddata
            pa.field("pixels", pa.large_binary(), nullable=False),
            pa.field("shape", pa.list_(pa.int32()), nullable=False),
            pa.field("axes", pa.string(), nullable=False),  # axis order of the image array
            pa.field("dtype", pa.string(), nullable=False),  # data type of the image array
            pa.field("byte_order", pa.string(), nullable=False),
            pa.field(
                "source_image_file", pa.string(), nullable=False
            ),  # original source image file from loaddata
            # Plus all the columns from the provided image metadata
            *metadata_fields,
        ],
        metadata={
            b"pixel_encoding": b"C-contiguous little-endian float32",
        },
    )


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array, ensuring that the tensor is detached
        and moved to the CPU if necessary.
    """
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    return image


def write_reference_images(
    path: Path,
    dataset: torch.utils.data.Dataset,
    metadata: pd.DataFrame,
) -> None:
    """
    Main function to write reference images from a dataset to Parquet files,
        validated against schema with fixed schema fields and provided metadata,
        generated one time for the entire dataset.
        The write uses fixed shard size and does not overwrite existing shards.

    :param path: Directory where Parquet files will be written.
    :param dataset: PyTorch dataset containing reference images.
    :param metadata: DataFrame containing metadata for the reference images.
    """
    SHARD_SIZE = 128  # Fixed shard size for writing reference images to parquet files

    if not path.exists():
        raise ValueError(f"Provided path does not exist: {path}")

    if len(metadata) != len(dataset):
        raise ValueError(
            f"Length mismatch: metadata has {len(metadata)} items, "
            f"dataset has {len(dataset)} items."
        )

    write_path = path / PARQUET_WRITE_ROOT
    write_path.mkdir(parents=True, exist_ok=True)

    with Writer(
        output_dir=write_path,
        schema=_image_record_schema(metadata),
        overwrite=False,
        shardsize=SHARD_SIZE,
    ) as writer:
        progress = tqdm(
            enumerate(metadata.to_dict(orient="records")),
            total=len(metadata),
            desc="Writing reference images to parquet",
        )

        for i, row in progress:
            sample = dataset[i]
            _, target_image = sample
            target_image_np = _tensor_to_numpy(target_image)

            for j, channel in enumerate(dataset.target_channel_keys):
                record_id = f"{i:09d}:{channel!s}"

                writer.add_row(
                    {
                        "record_id": record_id,
                        "channel": channel,
                        "source_image_file": str(row[channel]),
                        **row,
                        **encode_pixels(
                            target_image_np[j],
                            axes="YX",
                            require_finite=True,
                        ),
                    }
                )
