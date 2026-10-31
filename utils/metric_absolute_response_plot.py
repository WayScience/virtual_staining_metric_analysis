"""Visualization helpers for metric absolute response analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import tifffile as tiff


def load_absolute_response_artifacts(
    analysis_output_dir: str | Path,
    *,
    results_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load artifacts and reject normalized metrics that do not match the raw analysis run."""
    analysis_output_dir = Path(analysis_output_dir)
    results_output_dir = (
        analysis_output_dir if results_output_dir is None else Path(results_output_dir)
    )
    artifact_paths = {
        "raw metrics": analysis_output_dir / "metric_results.parquet",
        "normalized metrics": results_output_dir / "normalized_metric_results.parquet",
        "configuration": analysis_output_dir / "configuration.json",
        "reference image": analysis_output_dir / "reference_iteration_000.tiff",
    }
    missing = [f"{label}: {path}" for label, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing metric absolute response artifacts in {analysis_output_dir}: "
            f"{', '.join(missing)}. Run notebook 1.9 first."
        )

    raw_results = pd.read_parquet(artifact_paths["raw metrics"])
    normalized_results = pd.read_parquet(artifact_paths["normalized metrics"])
    run_keys = ["reference_channel", "transform_name", "iteration", "metric_name"]
    run_columns = [*run_keys, "metric_value"]
    try:
        pd.testing.assert_frame_equal(
            raw_results[run_columns].sort_values(run_keys).reset_index(drop=True),
            normalized_results[run_columns].sort_values(run_keys).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
        )
    except (AssertionError, KeyError, ValueError) as error:
        raise ValueError(
            "Normalized metrics do not match the raw results for the loaded analysis run. "
            "Run notebook 1.9 for the configured reference channel before plotting."
        ) from error
    with artifact_paths["configuration"].open() as config_file:
        configuration = json.load(config_file)
    return raw_results, normalized_results, configuration


def _ordered_present(values: pd.Series, requested_order: Sequence[str]) -> list[str]:
    present = set(values.unique())
    ordered = [value for value in requested_order if value in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _save_and_show(
    figure: Figure,
    output_path: str | Path | None,
    *,
    dpi: int,
    show: bool,
) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def metric_legend_handles(
    metric_order: Sequence[str],
    metric_labels: Mapping[str, str],
    metric_palette: Mapping[str, str],
) -> list[Line2D]:
    """Build consistent legend handles for metric response plots."""
    return [
        Line2D(
            [0],
            [0],
            color=metric_palette[metric_name],
            linewidth=2.2,
            label=metric_labels[metric_name],
        )
        for metric_name in metric_order
    ]


def plot_normalized_trajectories(
    data: pd.DataFrame,
    *,
    metric_order: Sequence[str],
    metric_labels: Mapping[str, str],
    metric_palette: Mapping[str, str],
    transform_order: Sequence[str],
    transform_labels: Mapping[str, str],
    output_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> Figure:
    """Plot normalized metric trajectories in one panel per degradation."""
    transforms = _ordered_present(data["transform_name"], transform_order)
    columns = 2
    rows = int(np.ceil(len(transforms) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 3.15 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, transform_name in zip(axes.flat, transforms, strict=False):
        transform_data = data.loc[data["transform_name"] == transform_name]
        for metric_name in metric_order:
            metric_data = transform_data.loc[
                transform_data["metric_name"] == metric_name
            ].sort_values("iteration")
            if metric_data.empty:
                continue
            axis.plot(
                metric_data["iteration"],
                metric_data["normalized_metric_value"],
                color=metric_palette[metric_name],
                linewidth=1.8,
            )
        axis.set_title(transform_labels.get(transform_name, transform_name).replace("\n", " "))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Cumulative iterations")
        axis.set_ylabel("Normalized quality score")
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(transforms) :]:
        axis.set_visible(False)

    figure.legend(
        handles=metric_legend_handles(metric_order, metric_labels, metric_palette),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14, hspace=0.32)
    _save_and_show(figure, output_path, dpi=dpi, show=show)
    return figure


def summarize_normalized_ranges(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observed normalized range for each metric and degradation."""
    return (
        data.groupby(["transform_name", "metric_name"], sort=False, observed=True)[
            "normalized_metric_value"
        ]
        .agg(empirical_low="min", empirical_high="max")
        .reset_index()
    )


def plot_normalized_ranges(
    data: pd.DataFrame,
    *,
    metric_order: Sequence[str],
    metric_labels: Mapping[str, str],
    metric_palette: Mapping[str, str],
    transform_order: Sequence[str],
    transform_labels: Mapping[str, str],
    output_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """Plot each metric's observed span within its documented score range."""
    summary = summarize_normalized_ranges(data)
    transforms = _ordered_present(summary["transform_name"], transform_order)
    figure, axes = plt.subplots(
        1,
        len(transforms),
        figsize=(1.25 * len(transforms), 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for index, (axis, transform_name) in enumerate(zip(axes, transforms, strict=True)):
        transform_summary = summary.loc[summary["transform_name"] == transform_name]
        plot_data = (
            pd.DataFrame({"metric_name": metric_order})
            .merge(transform_summary, on="metric_name", how="left", validate="one_to_one")
            .reset_index(drop=True)
        )
        axis.axhspan(0.0, 1.0, color="#BDBDBD", alpha=0.10, zorder=0)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6, zorder=1)
        axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6, zorder=1)

        for position, row in plot_data.iterrows():
            low = row["empirical_low"]
            high = row["empirical_high"]
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            metric_name = row["metric_name"]
            color = metric_palette[metric_name]
            axis.vlines(position, low, high, color=color, linewidth=3.2, zorder=3)
            axis.scatter(
                [position, position],
                [low, high],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

        axis.set_title(
            transform_labels.get(transform_name, transform_name).replace("\n", " "),
            fontsize=10,
            rotation=35,
        )
        axis.set_xticks([])
        axis.set_xlim(-0.6, len(metric_order) - 0.4)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8, zorder=0)
        if index == 0:
            axis.set_ylabel("Position in documented metric range")
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)

    figure.legend(
        handles=metric_legend_handles(metric_order, metric_labels, metric_palette),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
    )
    figure.subplots_adjust(left=0.11, right=0.995, top=0.78, bottom=0.24, wspace=0.08)
    _save_and_show(figure, output_path, dpi=dpi, show=show)
    return figure, summary


def plot_initial_response_vs_mae(
    data: pd.DataFrame,
    *,
    metric_order: Sequence[str],
    metric_labels: Mapping[str, str],
    metric_palette: Mapping[str, str],
    transform_order: Sequence[str],
    transform_labels: Mapping[str, str],
    output_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """Compare each metric's first-step normalized response with MAE."""
    comparison = data.loc[
        data["iteration"] == 1,
        ["transform_name", "metric_name", "clipped_metric_value", "normalized_metric_value"],
    ].copy()
    mae_reference = (
        comparison.loc[comparison["metric_name"] == "mae"]
        .set_index("transform_name")["normalized_metric_value"]
    )
    comparison["mae_normalized_reference"] = comparison["transform_name"].map(mae_reference)
    comparison["score_difference_from_mae"] = (
        comparison["normalized_metric_value"] - comparison["mae_normalized_reference"]
    )
    comparison["stronger_response_than_mae"] = comparison["score_difference_from_mae"] < 0.0

    transforms = _ordered_present(comparison["transform_name"], transform_order)
    figure, axes = plt.subplots(
        1,
        len(transforms),
        figsize=(1.25 * len(transforms), 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    minimum = float(comparison["normalized_metric_value"].min())
    ceiling_span = 1.0 - minimum
    y_limits = (max(0.0, minimum - 0.04 * ceiling_span), 1.0 + 0.02 * ceiling_span)

    for index, (axis, transform_name) in enumerate(zip(axes, transforms, strict=True)):
        plot_data = (
            pd.DataFrame({"metric_name": metric_order})
            .merge(
                comparison.loc[comparison["transform_name"] == transform_name],
                on="metric_name",
                how="left",
                validate="one_to_one",
            )
            .reset_index(drop=True)
        )
        axis.axhline(
            float(mae_reference.loc[transform_name]),
            color=metric_palette["mae"],
            linestyle="--",
            linewidth=1.6,
            zorder=2,
        )
        for position, row in plot_data.iterrows():
            metric_name = row["metric_name"]
            is_mae = metric_name == "mae"
            axis.scatter(
                position,
                row["normalized_metric_value"],
                s=75 if is_mae else 45,
                color=metric_palette[metric_name],
                edgecolor="black" if is_mae else "white",
                linewidth=1.0 if is_mae else 0.6,
                marker="D" if is_mae else "o",
                zorder=4 if is_mae else 3,
            )
        axis.set_title(
            transform_labels.get(transform_name, transform_name).replace("\n", " "),
            fontsize=10,
            rotation=35,
        )
        axis.set_xticks([])
        axis.set_xlim(-0.6, len(metric_order) - 0.4)
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", alpha=0.25)
        if index == 0:
            axis.set_ylabel("First-step normalized quality score")
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="D" if metric_name == "mae" else "o",
            color="none",
            markerfacecolor=metric_palette[metric_name],
            markeredgecolor="black" if metric_name == "mae" else "white",
            markeredgewidth=1.0,
            markersize=8 if metric_name == "mae" else 7,
            label=metric_labels[metric_name],
        )
        for metric_name in metric_order
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=metric_palette["mae"],
            linestyle="--",
            linewidth=1.6,
            label="MAE reference",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
    )
    figure.subplots_adjust(left=0.10, right=0.995, top=0.80, bottom=0.24, wspace=0.08)
    _save_and_show(figure, output_path, dpi=dpi, show=show)
    return figure, comparison


def plot_snapshot_response(
    data: pd.DataFrame,
    analysis_output_dir: str | Path,
    transform_name: str,
    snapshot_iterations: Sequence[int],
    *,
    metric_order: Sequence[str],
    metric_labels: Mapping[str, str],
    metric_palette: Mapping[str, str],
    transform_label: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> Figure:
    """Plot saved degradation snapshots above their normalized metric trajectories."""
    analysis_output_dir = Path(analysis_output_dir)
    snapshot_paths = [
        analysis_output_dir / "reference_iteration_000.tiff",
        *[
            analysis_output_dir / transform_name / f"iteration_{iteration:03d}.tiff"
            for iteration in snapshot_iterations
        ],
    ]
    missing = [path for path in snapshot_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing snapshots for {transform_name}: {missing}")

    images = [tiff.imread(path) for path in snapshot_paths]
    labels = ["Reference", *[f"Iteration {iteration}" for iteration in snapshot_iterations]]
    display_min = float(images[0].min())
    display_max = float(images[0].max())
    maximum_iteration = max(snapshot_iterations)
    curve_data = data.loc[
        (data["transform_name"] == transform_name)
        & data["iteration"].between(1, maximum_iteration)
    ]

    figure = plt.figure(figsize=(13, 4.2))
    outer_grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.15), hspace=0.08)
    montage_grid = outer_grid[0].subgridspec(1, len(images), wspace=0.015)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        image_axis = figure.add_subplot(montage_grid[0, index])
        image_axis.imshow(
            image,
            cmap="gray",
            vmin=display_min,
            vmax=display_max,
            interpolation="nearest",
        )
        image_axis.set_title(label, fontsize=8, fontweight="bold" if index == 0 else "normal")
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        for spine in image_axis.spines.values():
            spine.set_linewidth(1.5 if index == 0 else 0.5)
            spine.set_color("black" if index == 0 else "#BDBDBD")

    curve_axis = figure.add_subplot(outer_grid[1])
    for metric_name in metric_order:
        metric_data = curve_data.loc[curve_data["metric_name"] == metric_name].sort_values(
            "iteration"
        )
        if metric_data.empty:
            continue
        curve_axis.plot(
            metric_data["iteration"],
            metric_data["normalized_metric_value"],
            color=metric_palette[metric_name],
            linewidth=2.0,
            label=metric_labels[metric_name],
        )
    for iteration in snapshot_iterations:
        curve_axis.axvline(iteration, color="#BDBDBD", linewidth=0.6, alpha=0.45, zorder=0)

    label = (transform_label or transform_name).replace("\n", " ").lower()
    curve_axis.set_xlim(1, maximum_iteration)
    curve_axis.set_ylim(0.0, 1.0)
    curve_axis.set_xlabel(f"Cumulative {label} iterations")
    curve_axis.set_ylabel("Normalized quality score")
    curve_axis.grid(axis="y", alpha=0.25)
    curve_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=7,
        frameon=True,
    )
    figure.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.20)
    _save_and_show(figure, output_path, dpi=dpi, show=show)
    return figure