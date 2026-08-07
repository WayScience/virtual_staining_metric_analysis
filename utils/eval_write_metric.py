"""
Helper module for orchestrating metric evaluation and results writing.
Main helper function is `evaluate_lance_metrics_to_parquet`,
    which accepts a torch DataLoader of reference and degraded image stacks
    along with metadata, and write parquet shards of metric results to disk.
"""

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
    """
    Data class for holding broadcasted and flattened batch data,
        along with original metadata and batch/page counts for reshaping back.
    """

    degraded: Tensor
    reference: Tensor
    metadata: Sequence[dict[str, Any]]
    batch_size: int
    page_count: int


def _prep_metrics(metric_specs: Mapping[str, MetricSpec], device: torch.device) -> None:
    """
    Helper function to move metrics to proper device and dtype for evaluation.
    """
    for spec in metric_specs.values():
        metric = spec.metric
        if isinstance(metric, torch.nn.Module):
            metric.to(device=device, dtype=EVAL_DTYPE)
            metric.eval()


def _prep_batch(batch: Mapping[str, Any]) -> _PreparedBatch:
    """
    Prepare a batch of paired images for metric evaluation by broadcasting and flattening.
    This is needed because the reference slice needs to be expanded to
        match the number of degraded variants for each reference image,
        and certain metrics require duplication along the channel dimension,
        so the channel dimension must be available.
    """

    # V is the page count, corresponding to the number of degraded variants
    # for each reference image
    degraded = batch["degraded_stack"].to(dtype=EVAL_DTYPE)  # N, V, H, W
    reference = batch["reference_image"].to(dtype=EVAL_DTYPE)  # N, 1, H, W
    metadata = batch["degradation_metadata"]  # N rows

    batch_size, page_count, height, width = degraded.shape

    # flatten along N, V so make the channel dimension available.
    degraded_flat = degraded.reshape(batch_size * page_count, 1, height, width)
    # broadcast the reference image to match the page count of degraded images,
    # then flatten along N, V
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
    """Schema builder for metric output. Contracts the metric value dtype."""

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
    """
    Helper function for initializing Parquet writers for each metric,
        using the provided schema and output directories.
    """

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
    """
    Helper for duplicating channel dimensions to needed number.
    Typically the expansion is either 1->1 = no-op or 1->3 = RGB duplication.
    However, any positive integer is allowed for flexibility.

    :param tensor: Input tensor of shape (N * V, 1, H, W) (from _prep_batch)
    :param spec: MetricSpec containing the required input channel count.
    :return: Tensor of shape (N * V, C, H, W) where C is spec.input_channels.
    """
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
    """
    Helper function to evaluate a metric over a batch of paired images,
        returning a 1D tensor of metric values for each image in the batch.

    :param metric: The metric function or module to evaluate.
    :param spec: The MetricSpec containing metric configuration.
    :param degraded: A tensor of degraded images, shape (N*V, C, H, W).
    :param reference: A tensor of reference images, shape (N*V, C, H, W).
    :param device: The device to perform evaluation on.
    :return: A 1D tensor of metric values, shape (N*V, ).
    """

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
    """
    Helper function to write metric evaluation results to a Parquet writer.
    Each row combines metadata, catalog information, and the computed metric value.
    """

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
    """
    Evaluate metrics over paired batches from torch dataloader and write Parquet outputs.

    :param metric_specs: Mapping of metric names to MetricSpec objects.
    :param loader: DataLoader yielding batches of paired reference and degraded images.
    :param degradation_catalog: DataFrame containing catalog information for degraded images.
    :param output_root: Root directory for output Parquet files.
    :param overwrite: Whether to overwrite existing output files.
    :param device: Device to perform metric evaluation on.
    :param shard_size: Number of rows per Parquet shard.
    :return: Mapping of metric names to output Parquet directory paths.
    """

    device = torch.device(device)

    # move metrics to the specified device and set them to evaluation mode
    _prep_metrics(metric_specs, device)

    output_dirs = {name: output_root / name / "parquet" for name in metric_specs}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    # convert the degradation catalog to a list of dictionaries for easier row-wise access
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
            # initial flattening and broadcasting of the batch for metric evaluation
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
                # evaluate the metric over the prepared batch, additional
                # channel-wise broadcasting is handled inside _evaluate_metric
                values = _evaluate_metric(
                    spec.metric,
                    spec,
                    batch.degraded,
                    batch.reference,
                    device=device,
                )

                # reshape the flat values back to (batch_size, page_count) for writing
                scores = values.reshape(
                    batch.batch_size,
                    batch.page_count,
                )

                # write 1 row per (reference, degraded-variant/page, metric) combination
                # attaching metadata associated with reference (batch.metadata),
                # metadata associated with degraded variant (catalog_rows),
                # and the computed metric value
                _write_metric_rows(
                    writers[name],
                    batch.metadata,
                    catalog_rows,
                    scores,
                )

    return output_dirs
