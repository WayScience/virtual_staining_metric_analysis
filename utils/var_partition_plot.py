from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def plot_anova_variance_partition(
    anova_df: pd.DataFrame,
    *,
    row_cols: Sequence[str] = ("metric_name", "transform_name"),
    term_col: str = "term",
    value_col: str = "eta2",
    term_order: Sequence[str] | None = None,
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

    required_columns = {
        *row_cols,
        term_col,
        value_col,
    }

    missing = required_columns.difference(anova_df.columns)
    if missing:
        raise ValueError(f"ANOVA table is missing required columns: {sorted(missing)}")

    term_display_labels = dict(term_display_labels or {})
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
    plot_values.columns = [term_display_labels.get(term, str(term)) for term in terms]

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

    return fig, ax, plot_wide
