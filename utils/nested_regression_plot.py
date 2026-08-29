from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


def _read_summary(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Accept either an existing DataFrame or a CSV path."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def compute_burden_df(
    data: pd.DataFrame | str | Path,
    confounder_label: str,
    *,
    metric_order: Sequence[str] | None = None,
    transform_order: Sequence[str] | None = None,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """Calculate confounder burden from a nested-regression summary."""
    df = _read_summary(data)

    required = {
        "metric_name",
        "transform_name",
        "restricted_r2_mean",
        "partial_r2_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["metric_name"] = df["metric_name"].replace({"l1": "mae"})
    df["transform_name"] = df["transform_name"].astype(str).str.strip().str.lower()
    df["confounder"] = confounder_label
    df["burden"] = (df["partial_r2_mean"] + eps) / (df["restricted_r2_mean"] + eps)

    if metric_order is None:
        metric_order = list(df["metric_name"].drop_duplicates())
    if transform_order is None:
        transform_order = list(df["transform_name"].drop_duplicates())

    df["metric_name"] = pd.Categorical(
        df["metric_name"],
        categories=metric_order,
        ordered=True,
    )
    df["transform_name"] = pd.Categorical(
        df["transform_name"],
        categories=transform_order,
        ordered=True,
    )

    return df


def plot_nested_r2_multi(
    df: pd.DataFrame,
    *,
    partial_terms: Sequence[str] | None = None,
    restricted_term: str = "Degradation severity",
    metric_labels: Mapping[str, str] | None = None,
    metric_colors: Mapping[str, str] | None = None,
    metric_order: Sequence[str] | None = None,
    transform_labels: Mapping[str, str] | None = None,
    transform_markers: Mapping[str, str] | None = None,
    figsize_per_panel: tuple[float, float] = (5, 5),
    marker_size: float = 7,
    capsize: float = 2,
    sharey: bool = True,
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot restricted R² against partial R² for each metric × ablation."""
    plot_df = df.copy()
    if metric_colors is not None:
        plot_df = plot_df[plot_df["metric_name"].isin(metric_colors)]
    if transform_markers is not None:
        plot_df = plot_df[plot_df["transform_name"].isin(transform_markers)]
    if partial_terms is None:
        partial_terms = list(plot_df["partial_term"].drop_duplicates())
    else:
        partial_terms = [term for term in partial_terms if term in set(plot_df["partial_term"])]

    if not partial_terms:
        raise ValueError("No requested partial terms are present in the data.")

    n_panels = len(partial_terms)
    fig, axes_array = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        sharey=sharey,
        squeeze=False,
    )
    axes = list(axes_array[0])
    restricted_label = r"Restricted $R^2$" + f"\n({restricted_term})"

    for ax, partial_term in zip(axes, partial_terms):
        panel_df = plot_df[plot_df["partial_term"] == partial_term]
        for row in panel_df.itertuples(index=False):
            x = row.restricted_r2_mean
            y = row.partial_r2_mean
            xerr = [[x - row.restricted_r2_lower], [row.restricted_r2_upper - x]]
            yerr = [[y - row.partial_r2_lower], [row.partial_r2_upper - y]]

            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt=(transform_markers or {}).get(row.transform_name, "o"),
                markersize=marker_size,
                color=metric_colors[row.metric_name],
                markeredgecolor="black",
                markeredgewidth=0.5,
                elinewidth=1,
                capsize=capsize,
                linestyle="none",
            )

        ax.set_xlabel("")
        ax.set_title(partial_term, fontsize=13, fontweight="normal", loc="left")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Partial $R^2$\n(Confounding)", fontsize=13)

    present_metrics = [
        metric for metric in (metric_order or []) if metric in set(plot_df["metric_name"])
    ]
    metric_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=marker_size,
            markerfacecolor=metric_colors[metric],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=(metric_labels or {}).get(metric, metric),
        )
        for metric in present_metrics
    ]

    present_transforms = set(plot_df["transform_name"])
    ablation_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markersize=marker_size,
            color="black",
            label=(transform_labels or {}).get(transform, transform).replace("\n", " "),
        )
        for transform, marker in (transform_markers or {}).items()
        if transform in present_transforms
    ]

    metric_legend = fig.legend(
        handles=metric_handles,
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(0.87, 0.9),
        frameon=False,
    )
    fig.add_artist(metric_legend)
    fig.supxlabel(restricted_label)
    fig.legend(
        handles=ablation_handles,
        title="Degradation",
        loc="lower left",
        bbox_to_anchor=(0.87, 0.1),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    if output_path is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_burden_heatmaps(
    confounder_data: Mapping[str, pd.DataFrame | str | Path],
    *,
    metric_order: Sequence[str] | None = None,
    metric_labels: Mapping[str, str] | None = None,
    transform_labels: Mapping[str, str] | None = None,
    transform_order: Sequence[str] | None = None,
    eps: float = 1e-4,
    fill_limit: float | None = None,
    annotate: bool = True,
    annotation_fmt: str = ".2f",
    figsize: tuple[float, float] | None = None,
    title: str | None = r"Burden (partial $R^2$ / restricted $R^2$)",
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plot burden heatmaps for multiple confounders side by side."""
    burden_frames = [
        compute_burden_df(
            data,
            confounder_label=confounder,
            metric_order=metric_order,
            transform_order=transform_order,
            eps=eps,
        )
        for confounder, data in confounder_data.items()
    ]
    burden_df = pd.concat(burden_frames, ignore_index=True)

    if burden_df.empty:
        raise ValueError("No burden data available to plot.")

    if metric_order is None:
        metric_order = list(burden_df["metric_name"].cat.categories)
    if transform_order is None:
        transform_order = list(burden_df["transform_name"].cat.categories)
    if metric_labels is None:
        metric_labels = {metric: metric for metric in metric_order}
    if transform_labels is None:
        transform_labels = {transform: transform for transform in transform_order}

    if fill_limit is None:
        max_abs_burden = burden_df["burden"].abs().max()
        if not np.isfinite(max_abs_burden):
            raise ValueError("No finite burden values found.")
        fill_limit = float(np.ceil(max_abs_burden))

    if fill_limit <= 0:
        fill_limit = 1.0

    burden_cmap = LinearSegmentedColormap.from_list(
        "burden",
        ["#2166AC", "white", "#B2182B"],
    )
    norm = TwoSlopeNorm(vmin=-fill_limit, vcenter=0, vmax=fill_limit)

    n_panels = len(confounder_data)
    if figsize is None:
        figsize = (5 * n_panels, 5)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for panel_i, (ax, confounder) in enumerate(zip(axes, confounder_data)):
        panel_df = burden_df.loc[burden_df["confounder"].eq(confounder)]
        heatmap_df = panel_df.pivot_table(
            index="metric_name",
            columns="transform_name",
            values="burden",
            observed=False,
            aggfunc="first",
        ).reindex(index=metric_order, columns=transform_order)
        values = heatmap_df.to_numpy(dtype=float)

        ax.imshow(
            values,
            cmap=burden_cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
        )

        ax.set_xticks(np.arange(len(transform_order) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(metric_order) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(np.arange(len(transform_order)))
        ax.set_xticklabels(
            [transform_labels[name] for name in transform_order],
            fontsize=11,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlabel("")
        ax.set_title(confounder, fontsize=13, fontweight="normal", loc="left")

        if annotate:
            for row_i in range(values.shape[0]):
                for col_i in range(values.shape[1]):
                    value = values[row_i, col_i]
                    if np.isfinite(value):
                        ax.text(
                            col_i,
                            row_i,
                            format(value, annotation_fmt),
                            ha="center",
                            va="center",
                            fontsize=11,
                            color="black",
                        )

        if panel_i == n_panels - 1:
            ax.set_yticks(np.arange(len(metric_order)))
            ax.set_yticklabels(
                [metric_labels[m] for m in metric_order],
                fontsize=11,
            )
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelright=True, labelleft=False, right=False)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)

    fig.supxlabel("Degradation transform")

    if title is not None:
        fig.suptitle(title, fontsize=13)

    if output_path is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes, burden_df
