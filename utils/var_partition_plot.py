import math
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


class RadarComponentSpec(TypedDict):
    columns: Collection[str]
    color: str
    label: NotRequired[str]
    alpha: NotRequired[float]


def _prep_radar_plot(
    anova_df: pd.DataFrame,
    component_specs: Mapping[str, RadarComponentSpec],
) -> pd.DataFrame:
    required_columns = {
        "metric_name",
        "transform_name",
        "term",
        "eta2",
    }
    missing_columns = required_columns.difference(anova_df.columns)
    if missing_columns:
        raise ValueError(
            f"variance_partition_df is missing required columns: {sorted(missing_columns)}"
        )

    if not component_specs:
        raise ValueError("component_specs must define at least one component.")

    term_to_component: dict[str, str] = {}
    for component, spec in component_specs.items():
        for term in spec["columns"]:
            if term in term_to_component:
                raise ValueError(f"ANOVA term {term!r} is assigned to multiple components.")
            term_to_component[term] = component

    anova_df = anova_df.copy()
    anova_df["term"] = anova_df["term"].astype(str)
    anova_df["eta2"] = pd.to_numeric(
        anova_df["eta2"],
        errors="coerce",
    ).fillna(0.0)
    anova_df["component"] = anova_df["term"].map(term_to_component).fillna("unmapped")

    unmapped_df = anova_df.loc[
        anova_df["component"].eq("unmapped"),
        ["term", "eta2"],
    ]

    if not unmapped_df.empty:
        unmapped_summary = (
            unmapped_df.groupby("term", as_index=False)["eta2"]
            .sum()
            .sort_values("eta2", ascending=False)
        )

        unmapped_pct = 100.0 * unmapped_summary["eta2"].sum()

        print(
            f"Warning: {unmapped_pct:.3f}% of total eta2 is assigned "
            "to unmapped terms and will be excluded:"
        )

        print(unmapped_summary.to_string(index=False))

    radar_comp = (
        anova_df.loc[
            anova_df["component"].ne("unmapped"),
            [
                "metric_name",
                "transform_name",
                "component",
                "eta2",
            ],
        ]
        .groupby(
            [
                "metric_name",
                "transform_name",
                "component",
            ],
            as_index=False,
        )["eta2"]
        .sum()
    )

    radar_comp["pct"] = 100.0 * radar_comp["eta2"]

    radar_wide = radar_comp.pivot_table(
        index=[
            "metric_name",
            "transform_name",
        ],
        columns="component",
        values="pct",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    for component in component_specs:
        if component not in radar_wide.columns:
            radar_wide[component] = 0.0

    return radar_wide.loc[:, ["metric_name", "transform_name", *component_specs]]


def plot_anova_radar(
    anova_df: pd.DataFrame,
    *,
    component_specs: Mapping[str, RadarComponentSpec],
    metric_labels: Mapping[str, str] | None = None,
    metric_order: Sequence[str] | None = None,
    transform_labels: Mapping[str, str] | None = None,
    transform_order: Sequence[str] | None = None,
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
    legend_adjust: float = 0.15,
) -> tuple[Figure, np.ndarray]:
    """Plot ordered ANOVA component groups as stacked radar charts."""
    component_order = list(component_specs)
    metric_labels = dict(metric_labels or {})
    transform_labels = dict(transform_labels or {})

    radar_wide = _prep_radar_plot(
        anova_df=anova_df,
        component_specs=component_specs,
    )

    if transform_order is None:
        transform_order = sorted(radar_wide["transform_name"].astype(str).unique().tolist())
    else:
        transform_order = list(transform_order)

    metrics_present = [
        metric for metric in (metric_order or []) if metric in radar_wide["metric_name"].unique()
    ]

    if not metrics_present:
        metrics_present = sorted(radar_wide["metric_name"].astype(str).unique().tolist())

    if not metrics_present or not transform_order:
        raise ValueError("No metrics or transforms remain for plotting.")

    angles = np.linspace(
        0,
        2 * np.pi,
        len(transform_order),
        endpoint=False,
    )

    def close_cycle(values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.concatenate([values, values[:1]])

    angles_closed = close_cycle(angles)

    def fill_band(
        ax: Axes,
        lower_closed: np.ndarray,
        upper_closed: np.ndarray,
        color: str,
        alpha: float,
        zorder: int,
    ) -> None:
        theta_polygon = np.concatenate([angles_closed, angles_closed[::-1]])
        radial_polygon = np.concatenate(
            [
                upper_closed,
                lower_closed[::-1],
            ]
        )

        ax.fill(
            theta_polygon,
            radial_polygon,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=alpha,
            zorder=zorder,
        )

    n_metrics = len(metrics_present)
    ncols = min(4, n_metrics)
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3.2 * ncols,
            3.3 * nrows + 0.7,
        ),
        subplot_kw={
            "polar": True,
        },
    )
    axes = np.atleast_1d(axes).ravel()

    for i, metric_name in enumerate(metrics_present):
        ax = axes[i]

        metric_df = (
            radar_wide.loc[
                radar_wide["metric_name"].eq(metric_name),
                [
                    "transform_name",
                    *component_order,
                ],
            ]
            .groupby(
                "transform_name",
                as_index=False,
            )[component_order]
            .mean()
            .set_index("transform_name")
            .reindex(transform_order)
            .fillna(0.0)
            .reset_index()
        )

        lower_boundary = np.zeros(
            len(transform_order),
            dtype=float,
        )

        for index, component in enumerate(component_order):
            spec = component_specs[component]
            zorder = len(component_order) - index
            upper_boundary = lower_boundary + metric_df[component].to_numpy(dtype=float)

            fill_band(
                ax=ax,
                lower_closed=close_cycle(lower_boundary),
                upper_closed=close_cycle(upper_boundary),
                color=spec["color"],
                alpha=spec.get("alpha", 0.55),
                zorder=zorder,
            )

            ax.plot(
                angles_closed,
                close_cycle(upper_boundary),
                color=spec["color"],
                linewidth=1.5,
                zorder=zorder + 0.5,
            )

            lower_boundary = upper_boundary

        # Put first degradation at the top and proceed clockwise.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles)
        ax.set_xticklabels(
            [transform_labels.get(name, name) for name in transform_order],
            fontsize=8.5,
        )

        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.5)

        ax.set_rlabel_position(15)
        ax.grid(alpha=0.3)

        ax.set_title(
            metric_labels.get(metric_name, metric_name),
            fontsize=10,
            pad=17,
        )

    # Hide unused panels.
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=spec["color"],
            edgecolor=spec["color"],
            alpha=spec.get("alpha", 0.55),
            label=spec.get("label", component),
        )
        for component, spec in component_specs.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -legend_adjust),
        ncol=len(component_specs),
        frameon=False,
        title="Variance component",
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()

    fig.subplots_adjust(
        bottom=0.14,
        top=0.90,
        wspace=0.35,
        hspace=0.45,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_anova_variance_partition(
    anova_df: pd.DataFrame,
    *,
    row_cols: Sequence[str] = ("metric_name", "transform_name"),
    term_col: str = "term",
    value_col: str = "eta2",
    term_order: Sequence[str] | None = None,
    term_labels: Mapping[str, str] | None = None,
    term_display_labels: Mapping[str, str] | None = None,
    term_colors: Mapping[str, str] | None = None,
    hatch_groups: Mapping[
        str,
        tuple[Collection[str], str],
    ]
    | None = None,
    unhatched_legend_label: str | None = None,
    row_orders: Mapping[str, Sequence[Any]] | None = None,
    row_value_labels: Mapping[
        str,
        Mapping[Any, str],
    ]
    | None = None,
    row_label_fn: Callable[[tuple[Any, ...]], str] | None = None,
    xlabel: str = r"Variance explained ($\eta^2$)",
    ylabel: str | None = None,
    figsize_width: float = 12,
    row_height: float = 0.32,
    minimum_height: float = 6,
    bar_width: float = 0.85,
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
    legend_adjust: float = 0.15,
    hatch_legend_adjust: float = 0.15,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot stacked ANOVA variance partitions from canonical term names."""
    row_cols = tuple(row_cols)

    if term_labels is not None and term_display_labels is not None:
        raise ValueError("Specify only one of term_labels or term_display_labels.")

    required_columns = {
        *row_cols,
        term_col,
        value_col,
    }

    missing = required_columns.difference(anova_df.columns)
    if missing:
        raise ValueError(f"ANOVA table is missing required columns: {sorted(missing)}")

    term_labels = dict(term_labels or term_display_labels or {})
    term_colors = dict(term_colors or {})
    hatch_groups = dict(hatch_groups or {})
    row_orders = dict(row_orders or {})
    row_value_labels = dict(row_value_labels or {})

    work_df = anova_df.loc[
        :,
        [*row_cols, term_col, value_col],
    ].copy()

    work_df[value_col] = pd.to_numeric(
        work_df[value_col],
        errors="coerce",
    )

    work_df = work_df.dropna(subset=[*row_cols, term_col, value_col])

    if work_df.empty:
        raise ValueError("No complete ANOVA rows remain for plotting.")

    plot_wide = work_df.pivot_table(
        index=list(row_cols),
        columns=term_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )

    if plot_wide.empty:
        raise ValueError("The ANOVA table produced no plottable groups.")

    # Normalize the index so downstream logic always sees tuples.
    if not isinstance(plot_wide.index, pd.MultiIndex):
        plot_wide.index = pd.MultiIndex.from_arrays(
            [plot_wide.index],
            names=row_cols,
        )

    # Order rows.
    row_index = plot_wide.index.to_frame(index=False)
    sort_columns: list[str] = []

    for column in row_cols:
        text_col = f"_text_{column}"
        row_index[text_col] = row_index[column].astype(str)

        if column in row_orders:
            rank = {value: i for i, value in enumerate(row_orders[column])}
            rank_col = f"_rank_{column}"

            row_index[rank_col] = row_index[column].map(rank).fillna(len(rank))

            sort_columns.extend([rank_col, text_col])
        else:
            sort_columns.append(text_col)

    row_index = row_index.sort_values(sort_columns, kind="stable").loc[:, list(row_cols)]

    ordered_index = pd.MultiIndex.from_frame(
        row_index,
        names=row_cols,
    )

    plot_wide = plot_wide.reindex(ordered_index)

    # Order terms.
    observed_terms = list(plot_wide.columns)

    if term_order is None:
        terms = sorted(observed_terms, key=str)
    else:
        preferred = [term for term in term_order if term in observed_terms]
        remaining = sorted(
            set(observed_terms).difference(preferred),
            key=str,
        )
        terms = preferred + remaining

    plot_wide = plot_wide.loc[:, terms]

    # Row labels.
    if row_label_fn is not None:
        row_labels = [row_label_fn(values) for values in plot_wide.index]
    else:
        row_labels = [
            " | ".join(
                row_value_labels.get(column, {}).get(value, str(value))
                for column, value in zip(row_cols, values)
            )
            for values in plot_wide.index
        ]

    # Term colors.
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colors = [
        term_colors.get(
            term,
            fallback_colors[i % len(fallback_colors)],
        )
        for i, term in enumerate(terms)
    ]

    plot_values = plot_wide.copy()
    plot_values.columns = [term_labels.get(term, str(term)) for term in terms]

    figure_height = max(
        minimum_height,
        row_height * len(row_labels),
    )

    fig, ax = plt.subplots(figsize=(figsize_width, figure_height))

    plot_values.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=colors,
        width=bar_width,
        edgecolor="white",
        linewidth=0.35,
    )

    # Hatching.
    term_hatches = {
        term: hatch for _, (group_terms, hatch) in hatch_groups.items() for term in group_terms
    }

    for container, term in zip(ax.containers, terms):
        hatch = term_hatches.get(term)

        if hatch is not None:
            for patch in container.patches:
                patch.set_hatch(hatch)
                patch.set_edgecolor("white")
                patch.set_linewidth(0.35)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else " | ".join(row_cols))

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()

    # Term legend.
    term_legend = ax.legend(
        title="Term",
        loc="upper center",
        bbox_to_anchor=(0.4, -0.07),
        ncol=min(4, max(1, len(terms))),
        frameon=False,
    )

    # Optional hatch legend.
    hatch_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch,
            label=label,
        )
        for label, (_, hatch) in hatch_groups.items()
    ]

    if unhatched_legend_label is not None:
        hatch_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                label=unhatched_legend_label,
            )
        )

    if hatch_handles:
        ax.legend(
            handles=hatch_handles,
            title="Shading",
            loc="upper center",
            bbox_to_anchor=(0.5, -hatch_legend_adjust),
            ncol=min(3, len(hatch_handles)),
            frameon=False,
        )
        ax.add_artist(term_legend)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_adjust)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, plot_wide
