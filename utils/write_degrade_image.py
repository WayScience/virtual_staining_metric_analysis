"""
Utilities for applying degradation and writing the resulting images as
    Parquet files based on reference images and degradation specs.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import lance
from tqdm.auto import tqdm

from utils.apply_degradation import DegradationSpec, build_degradation_stack

from .encoding import decode_pixels, encode_pixels
from .iter_data import iter_lance_fragments
from .parquet_writer import LanceWriter, Writer
from .write_reference_image import PARQUET_WRITE_ROOT, SHARD_SIZE


def _fingerprint_degradation_specs(
    specs: list[DegradationSpec],
) -> str:
    """
    Return an order-sensitive SHA-256 fingerprint for degradation specs.

    The fingerprint is stable when:

    - the specifications have the same values;
    - the specifications appear in the same order;
    - dictionary keys appear in any order.
    """

    def json_default(value: Any) -> Any:
        """Convert supported non-standard values to JSON-compatible values."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Cannot fingerprint value of type {type(value).__module__}.{type(value).__qualname__}"
        )

    document = {
        # Allows the serialization format to be intentionally changed later.
        "fingerprint_format_version": 1,
        "specs": [asdict(spec) for spec in specs],
    }

    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _degrade_record_schema(ref_schema: pa.Schema, base_seed: int) -> pa.Schema:
    """
    Create a new schema for degraded records based on the reference schema.

    :param ref_schema: The schema of the reference records.
    :param base_seed: Base random seed used to derive per-page transform seeds.
    :return: A new schema for degraded records.
    """

    metadata = {
        b"record_kind": b"degradation_stack",
        b"pixel_encoding": b"C-contiguous little-endian float32",
        b"albumentations_version": A.__version__.encode(),
        b"opencv_version": cv2.__version__.encode(),
        b"base_seed": str(base_seed).encode(),
    }

    degrade_schema = ref_schema.append(pa.field("spec_fingerprint", pa.string())).with_metadata(
        metadata
    )

    return degrade_schema


def write_degraded_images(
    *,
    specs: list[DegradationSpec],
    output_dir: Path,
    base_seed: int,
    overwrite: bool = False,
    reference_shards: list[Path] | None = None,
    reference_lance_dir: Path | None = None,
) -> None:
    """
    Write degraded images to Parquet files based on reference shards and degradation specs.
    Operates on a per-shard basis, reading reference images, applying degradations, and writing results.
    """

    if reference_shards is None and reference_lance_dir is None:
        raise ValueError("Either reference_shards or reference_lance_dir must be provided.")
    if reference_shards is not None and reference_lance_dir is not None:
        raise ValueError("Only one of reference_shards or reference_lance_dir should be provided.")
    if reference_shards is not None:
        print("Writing degraded images from reference parquet shards.")
        write_path = output_dir / PARQUET_WRITE_ROOT
        _Writer = Writer
        progress = tqdm(
            reference_shards,
            total=len(reference_shards),
            desc="Writing degraded images",
        )

        def process_item(item):
            table = pq.read_table(item)
            return table
    else:
        print("Writing degraded images from Lance reference")
        write_path = output_dir
        _Writer = LanceWriter
        dataset = lance.dataset(reference_lance_dir)
        fragment_count = len(dataset.get_fragments())
        progress = tqdm(
            iter_lance_fragments(reference_lance_dir),
            desc="Writing degraded images",
            total=fragment_count,
        )

        def process_item(item):
            _, _, table = item
            return table

    write_path.mkdir(parents=True, exist_ok=True)

    with _Writer(
        output_dir=write_path,
        schema=None,
        overwrite=overwrite,
        shardsize=SHARD_SIZE,
    ) as writer:
        fingerprint = _fingerprint_degradation_specs(specs)

        for item in progress:
            ref_tab = process_item(item)
            degrade_schema = _degrade_record_schema(ref_tab.schema, base_seed)
            references = ref_tab.to_pylist()

            for reference in references:
                image = decode_pixels(
                    reference["pixels"],
                    reference["shape"],
                    reference["dtype"],
                    reference["byte_order"],
                )
                degrade_stack = build_degradation_stack(
                    image,
                    str(reference["record_id"]),  # used to salt seed for degradation
                    specs,
                    base_seed=base_seed,
                )

                new_row = {
                    **reference,
                    "spec_fingerprint": fingerprint,
                    **encode_pixels(
                        degrade_stack,
                        axes="SYX",
                        require_finite=True,
                    ),
                }

                writer.add_row(new_row, schema=degrade_schema)
