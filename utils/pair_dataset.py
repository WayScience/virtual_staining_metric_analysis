from collections.abc import Iterator, Sequence
from itertools import chain, zip_longest
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import lance
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from utils.encoding import decode_pixel_record

PIXEL_COLUMNS = (
    "pixels",
    "shape",
    "axes",
    "dtype",
    "byte_order",
)

_MISSING = object()


class PairedLanceSample(NamedTuple):
    reference_image: Tensor
    degraded_stack: Tensor
    reference_metadata: dict[str, Any]
    degradation_metadata: dict[str, Any]


def _projected_columns(
    metadata_columns: Sequence[str] | None,
    *,
    pair_key: str,
) -> list[str] | None:
    """
    Return the minimum required Lance projection.

    None means that all columns, including all metadata columns, are read.
    """
    if metadata_columns is None:
        return None

    return list(
        dict.fromkeys(
            (
                *PIXEL_COLUMNS,
                pair_key,
                *metadata_columns,
            )
        )
    )


def _to_float32_tensor(
    image: npt.NDArray[np.float32],
) -> Tensor:
    """
    Convert a decoded NumPy image to a contiguous, writable float32 tensor.

    decode_pixel_record() may return an array backed by immutable Python bytes.
    In that case a copy is required before exposing the storage to PyTorch.
    """
    array = np.asarray(image, dtype=np.float32)

    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(
            array,
            dtype=np.float32,
            order="C",
            copy=True,
        )

    return torch.from_numpy(array)


def _scanner_rows(
    dataset: lance.LanceDataset,
    fragment: lance.LanceFragment,
    *,
    columns: list[str] | None,
    batch_size: int,
    batch_readahead: int,
) -> Iterator[dict[str, Any]]:
    """
    Stream Python records from exactly one Lance fragment.

    Conversion to Python dictionaries is bounded by scanner batch_size rather
    than the complete fragment size.
    """
    batches = dataset.scanner(
        columns=columns,
        fragments=[fragment],
        batch_size=batch_size,
        batch_readahead=batch_readahead,
        fragment_readahead=1,
        scan_in_order=True,
        # Set to "all_binary" here if pixels later become Lance Blob columns.
        blob_handling=None,
    ).to_batches()

    return chain.from_iterable(batch.to_pylist() for batch in batches)


class PairedLanceImageDataset(IterableDataset[PairedLanceSample]):
    """
    Stream corresponding records from mirrored reference and degradation
    Lance datasets.

    Pairing assumptions
    -------------------
    1. The datasets have the same number of physical fragments.
    2. Fragment position i in the reference dataset corresponds to fragment
       position i in the degraded dataset.
    3. Rows have the same order within corresponding fragments.
    4. pair_key uniquely identifies corresponding rows.

    Independent compaction or fragment rewriting of only one dataset violates
    these assumptions and causes validation to fail.
    """

    def __init__(
        self,
        reference_uri: str | Path,
        degradation_uri: str | Path,
        *,
        pair_key: str = "record_id",
        reference_metadata_columns: Sequence[str] | None = None,
        degradation_metadata_columns: Sequence[str] | None = None,
        scan_batch_size: int = 8,
        batch_readahead: int = 2,
        validate_pairing: bool = True,
    ) -> None:
        super().__init__()

        if scan_batch_size <= 0:
            raise ValueError("scan_batch_size must be positive.")
        if batch_readahead <= 0:
            raise ValueError("batch_readahead must be positive.")

        # Store only serializable paths/configuration. Do not retain an open
        # LanceDataset object on the Dataset instance.
        self.reference_uri = str(reference_uri)
        self.degradation_uri = str(degradation_uri)

        self.pair_key = pair_key
        self.scan_batch_size = scan_batch_size
        self.batch_readahead = batch_readahead
        self.validate_pairing = validate_pairing

        self.reference_columns = _projected_columns(
            reference_metadata_columns,
            pair_key=pair_key,
        )
        self.degradation_columns = _projected_columns(
            degradation_metadata_columns,
            pair_key=pair_key,
        )

    def __len__(self) -> int:
        """
        Return the total reference record count.

        The Lance handle is temporary and is not retained across processes.
        """
        return lance.dataset(self.reference_uri).count_rows()

    def __iter__(self) -> Iterator[PairedLanceSample]:
        # Open the datasets independently in each DataLoader worker.
        reference_dataset = lance.dataset(self.reference_uri)
        degradation_dataset = lance.dataset(self.degradation_uri)

        reference_fragments = reference_dataset.get_fragments()
        degradation_fragments = degradation_dataset.get_fragments()

        if len(reference_fragments) != len(degradation_fragments):
            raise ValueError(
                "Reference/degradation fragment count mismatch: "
                f"{len(reference_fragments)} reference fragments and "
                f"{len(degradation_fragments)} degradation fragments."
            )

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker_info.id
            worker_count = worker_info.num_workers

        for fragment_position in range(
            worker_id,
            len(reference_fragments),
            worker_count,
        ):
            reference_fragment = reference_fragments[fragment_position]
            degradation_fragment = degradation_fragments[fragment_position]

            reference_rows = _scanner_rows(
                reference_dataset,
                reference_fragment,
                columns=self.reference_columns,
                batch_size=self.scan_batch_size,
                batch_readahead=self.batch_readahead,
            )
            degradation_rows = _scanner_rows(
                degradation_dataset,
                degradation_fragment,
                columns=self.degradation_columns,
                batch_size=self.scan_batch_size,
                batch_readahead=self.batch_readahead,
            )

            paired_rows = zip_longest(
                reference_rows,
                degradation_rows,
                fillvalue=_MISSING,
            )

            for fragment_row, pair in enumerate(paired_rows):
                reference_record, degradation_record = pair

                if reference_record is _MISSING:
                    raise ValueError(
                        "Degradation fragment contains more rows than its "
                        f"reference fragment at fragment position "
                        f"{fragment_position}."
                    )

                if degradation_record is _MISSING:
                    raise ValueError(
                        "Reference fragment contains more rows than its "
                        f"degradation fragment at fragment position "
                        f"{fragment_position}."
                    )

                if self.validate_pairing:
                    reference_key = reference_record.get(self.pair_key)
                    degradation_key = degradation_record.get(self.pair_key)

                    if reference_key != degradation_key:
                        raise ValueError(
                            "Reference/degradation record mismatch at "
                            f"fragment position {fragment_position}, "
                            f"row {fragment_row}: "
                            f"{self.pair_key}={reference_key!r} versus "
                            f"{degradation_key!r}."
                        )

                reference_image, reference_metadata = decode_pixel_record(reference_record)
                degraded_stack, degradation_metadata = decode_pixel_record(degradation_record)

                yield PairedLanceSample(
                    reference_image=_to_float32_tensor(reference_image),
                    degraded_stack=_to_float32_tensor(degraded_stack),
                    reference_metadata=reference_metadata,
                    degradation_metadata=degradation_metadata,
                )


class PairedLanceBatch(TypedDict):
    reference_image: Tensor
    degraded_stack: Tensor
    reference_metadata: list[dict[str, Any]]
    degradation_metadata: list[dict[str, Any]]


def collate_paired_lance_samples(
    samples: Sequence[PairedLanceSample],
) -> PairedLanceBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample sequence.")

    return {
        "reference_image": torch.stack([sample.reference_image for sample in samples]),
        "degraded_stack": torch.stack([sample.degraded_stack for sample in samples]),
        "reference_metadata": [sample.reference_metadata for sample in samples],
        "degradation_metadata": [sample.degradation_metadata for sample in samples],
    }
