from pathlib import Path

from collections.abc import Iterator
from dataclasses import dataclass

import lance
import pyarrow as pa


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
