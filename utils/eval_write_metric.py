from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metric_spec import MetricSpec
from .parquet_writer import Writer

# anything lower than float32 can cause precision issues due to low SNR in data
EVAL_DTYPE = torch.float32


@dataclass(frozen=True)
class _PreparedBatch:
    degraded: Tensor
    reference: Tensor
    metadata: Sequence[dict[str, Any]]
    batch_size: int
    page_count: int


def _prep_metrics(metric_specs: Mapping[str, MetricSpec], device: torch.device) -> None:
    for spec in metric_specs.values():
        metric = spec.metric
        if isinstance(metric, torch.nn.Module):
            metric.to(device=device, dtype=EVAL_DTYPE)
            metric.eval()


def _prep_batch(batch: Mapping[str, Any]) -> _PreparedBatch:

    degraded = batch["degraded_stack"].to(dtype=EVAL_DTYPE)
    reference = batch["reference_image"].to(dtype=EVAL_DTYPE)
    metadata = batch["degradation_metadata"]

    # page count is number of degraded variant with respect to each reference
    batch_size, page_count, height, width = degraded.shape

    degraded_flat = degraded.reshape(batch_size * page_count, 1, height, width)
    reference_flat = (
        reference[:, None, :, :]
        .expand(-1, page_count, -1, -1)  # broadcast ref to match page count of degraded
        .reshape(batch_size * page_count, 1, height, width)
    )

    return _PreparedBatch(
        degraded=degraded_flat,
        reference=reference_flat,
        metadata=metadata,
        batch_size=batch_size,
        page_count=page_count,
    )


def _build_output_schema(
    metadata_row: Mapping[str, Any],
    catalog_row: Mapping[str, Any],
) -> pa.Schema:

    metadata_columns = set(metadata_row)
    catalog_columns = set(catalog_row)

    if collisions := metadata_columns.intersection(catalog_columns):
        raise ValueError(
            f"Degradation metadata conflicts with catalog columns: {sorted(collisions)}."
        )

    example_row = {**metadata_row, **catalog_row, "metric_value": 0.0}
    schema = pa.Table.from_pylist([example_row]).schema
    metric_index = schema.get_field_index("metric_value")

    return schema.set(
        metric_index,
        pa.field("metric_value", pa.float32(), nullable=False),
    ).with_metadata({b"record_kind": b"image_metric"})


def _open_metric_writers(
    stack: ExitStack,
    metric_names: Sequence[str],
    output_dirs: Mapping[str, Path],
    schema: pa.Schema,
    *,
    overwrite: bool,
    shard_size: int = 128,
) -> dict[str, Writer]:

    return {
        name: stack.enter_context(
            Writer(
                output_dir=output_dirs[name],
                schema=schema,
                overwrite=overwrite,
                compression="zstd",
                shardsize=shard_size,
            )
        )
        for name in metric_names
    }


def _match_input_channels(
    tensor: Tensor,
    spec: MetricSpec,
) -> Tensor:
    if spec.input_channels == 1:
        return tensor
    if spec.input_channels > 1:  # some natural image metrics require 3-channel input
        return tensor.expand(-1, spec.input_channels, -1, -1)
    raise ValueError(f"Metrics must accept >=1 channels, not {spec.input_channels}.")


def _evaluate_metric(
    metric: Any,
    spec: MetricSpec,
    degraded: Tensor,
    reference: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    total_images = degraded.shape[0]
    value_chunks: list[Tensor] = []
    kwargs = dict(spec.kwargs)

    for start in range(0, total_images, spec.page_batch_size):
        stop = min(
            start + spec.page_batch_size,
            total_images,
        )

        _degraded = degraded[start:stop].to(device=device, non_blocking=True)
        _reference = reference[start:stop].to(device=device, non_blocking=True)
        _degraded = _match_input_channels(_degraded, spec)
        _reference = _match_input_channels(_reference, spec)

        values = metric(_degraded, _reference, **kwargs)
        values = spec.aggregate_samples(values)
        expected_size = stop - start

        if values.ndim != 1 or values.shape[0] != expected_size:
            raise ValueError(
                f"Metric {spec.name!r} produced shape "
                f"{tuple(values.shape)} for {expected_size} images."
            )

        value_chunks.append(values.detach().to(device="cpu"))

    return torch.cat(value_chunks)


def _write_metric_rows(
    writer: Writer,
    metadata_rows: Sequence[dict[str, Any]],
    catalog_rows: Sequence[dict[str, Any]],
    scores: Tensor,
) -> None:
    score_rows = scores.tolist()

    for metadata_row, record_scores in zip(
        metadata_rows,
        score_rows,
        strict=True,
    ):
        for catalog_row, metric_value in zip(
            catalog_rows,
            record_scores,
            strict=True,
        ):
            writer.add_row(
                {
                    **metadata_row,
                    **catalog_row,
                    "metric_value": metric_value,
                }
            )


def evaluate_lance_metrics_to_parquet(
    metric_specs: Mapping[str, MetricSpec],
    loader: DataLoader,
    degradation_catalog: pd.DataFrame,
    output_root: Path,
    *,
    overwrite: bool = False,
    device: torch.device | str = "cpu",
    shard_size: int = 128,
) -> dict[str, Path]:
    """Evaluate metrics over paired Lance batches and write Parquet outputs."""

    device = torch.device(device)
    _prep_metrics(metric_specs, device)

    output_dirs = {name: output_root / name / "parquet" for name in metric_specs}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    catalog_rows = pa.Table.from_pandas(degradation_catalog, preserve_index=False).to_pylist()

    writers: dict[str, Writer] | None = None

    with (
        ExitStack() as stack,
        tqdm(
            loader,
            desc="Evaluating Lance metrics",
            unit="batch",
            total=len(loader),
        ) as progress,
        torch.inference_mode(),
    ):
        for raw_batch in progress:
            batch: _PreparedBatch = _prep_batch(raw_batch)

            if writers is None:
                writers = _open_metric_writers(
                    stack,
                    list(metric_specs),
                    output_dirs,
                    _build_output_schema(batch.metadata[0], catalog_rows[0]),
                    overwrite=overwrite,
                    shard_size=shard_size,
                )

            for name, spec in metric_specs.items():
                values = _evaluate_metric(
                    spec.metric,
                    spec,
                    batch.degraded,
                    batch.reference,
                    device=device,
                )

                scores = values.reshape(
                    batch.batch_size,
                    batch.page_count,
                )

                _write_metric_rows(
                    writers[name],
                    batch.metadata,
                    catalog_rows,
                    scores,
                )

    return output_dirs
