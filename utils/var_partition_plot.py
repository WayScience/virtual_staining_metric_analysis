from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def plot_anova_variance_partition(
    anova_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
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
    row_group_col: str | None = None,
    row_group_colors: Mapping[Any, str] | None = None,
    row_group_label_x: float = -0.24,
    row_group_line_x: float = -0.12,
    row_group_label_color: str = "black",
    row_group_label_fontsize: float | None = None,
    row_group_label_fontweight: str | None = None,
    row_group_linewidth: float = 2.5,
    row_group_line_gap: float = 0.0,
    row_group_gap: float = 0.0,
    row_labels_side: Literal["left", "right", "inside"] = "left",
    row_label_x: float = 0.99,
    row_label_color: str = "black",
    row_label_fontsize: float | None = None,
    row_label_outline_width: float = 2.0,
    show_y_axis: bool = True,
    xlabel: str = r"Variance explained ($\eta^2$)",
    xlabel_fontsize: float | None = None,
    ylabel: str | None = None,
    figsize_width: float = 12,
    figsize_height: float | None = None,
    row_height: float = 0.32,
    minimum_height: float = 6,
    bar_width: float = 0.85,
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
    show_legend: bool = True,
    legend_adjust: float = 0.15,
    hatch_legend_adjust: float = 0.15,
    title: str | None = None,
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
    row_group_colors = dict(row_group_colors or {})

    if row_group_col is not None and row_group_col not in row_cols:
        raise ValueError(f"row_group_col must be one of row_cols: {list(row_cols)}")

    if row_labels_side not in {"left", "right", "inside"}:
        raise ValueError("row_labels_side must be 'left', 'right', or 'inside'.")

    if row_group_gap < 0:
        raise ValueError("row_group_gap must be non-negative.")

    if not 0 <= row_group_line_gap < bar_width:
        raise ValueError("row_group_line_gap must be non-negative and less than bar_width.")

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
        preferred = [term for term in term_labels if term in observed_terms]
        remaining = sorted(
            set(observed_terms).difference(preferred),
            key=str,
        )
        terms = preferred + remaining
    else:
        preferred = [term for term in term_order if term in observed_terms]
        remaining = sorted(
            set(observed_terms).difference(preferred),
            key=str,
        )
        terms = preferred + remaining

    plot_wide = plot_wide.loc[:, terms]

    # Row labels.
    if row_label_fn is not None and row_group_col is not None:
        raise ValueError("row_label_fn cannot be combined with row_group_col.")

    if row_label_fn is not None:
        row_labels = [row_label_fn(values) for values in plot_wide.index]
    else:
        label_columns = [column for column in row_cols if column != row_group_col]
        row_labels = [
            " | ".join(
                row_value_labels.get(column, {}).get(value, str(value))
                for column, value in zip(row_cols, values)
                if column in label_columns
            )
            for values in plot_wide.index
        ]

    row_positions = np.arange(len(row_labels), dtype=float)
    group_values: list[Any] | None = None

    if row_group_col is not None:
        group_level = row_cols.index(row_group_col)
        group_values = [values[group_level] for values in plot_wide.index]
        group_numbers = np.zeros(len(group_values), dtype=float)

        for row_number in range(1, len(group_values)):
            group_numbers[row_number] = group_numbers[row_number - 1] + (
                group_values[row_number] != group_values[row_number - 1]
            )

        row_positions += group_numbers * row_group_gap

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

    if figsize_height is None:
        figure_height = max(
            minimum_height,
            row_height * len(row_labels),
        )
    else:
        figure_height = figsize_height

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_width, figure_height))
    else:
        fig = ax.figure

    plot_values.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=colors,
        width=bar_width,
        edgecolor="white",
        linewidth=0.35,
    )

    if row_group_gap:
        for container in ax.containers:
            for row_number, patch in enumerate(container.patches):
                patch.set_y(patch.get_y() + row_positions[row_number] - row_number)

    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

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

    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else " | ".join(row_cols))

    ax.set_yticks(row_positions)

    if row_labels_side == "inside":
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False, right=False, length=0)

        text_effects = None
        if row_label_outline_width > 0:
            text_effects = [
                path_effects.Stroke(linewidth=row_label_outline_width, foreground="white"),
                path_effects.Normal(),
            ]

        for row_position, row_label in zip(row_positions, row_labels):
            ax.text(
                row_label_x,
                row_position,
                row_label,
                color=row_label_color,
                fontsize=row_label_fontsize,
                ha="right",
                va="center",
                transform=ax.get_yaxis_transform(),
                path_effects=text_effects,
                zorder=20,
            )
    else:
        ax.set_yticklabels(row_labels)
        ax.yaxis.set_ticks_position(row_labels_side)
        ax.tick_params(
            axis="y",
            labelleft=row_labels_side == "left",
            labelright=row_labels_side == "right",
        )

    ax.set_ylim(row_positions[-1] + 0.5, -0.5)

    if group_values is not None:
        group_start = 0

        for row_number in range(1, len(group_values) + 1):
            group_ends = row_number == len(group_values) or (
                group_values[row_number] != group_values[group_start]
            )
            if not group_ends:
                continue

            group_value = group_values[group_start]
            group_end = row_number - 1
            group_center = (row_positions[group_start] + row_positions[group_end]) / 2
            group_color = row_group_colors.get(group_value, "black")
            line_inset = row_group_line_gap / 2

            ax.text(
                row_group_label_x,
                group_center,
                row_value_labels.get(row_group_col, {}).get(group_value, str(group_value)),
                color=row_group_label_color,
                fontsize=row_group_label_fontsize,
                fontweight=row_group_label_fontweight,
                ha="right",
                va="center",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )
            ax.plot(
                [row_group_line_x, row_group_line_x],
                [
                    row_positions[group_start] - bar_width / 2 + line_inset,
                    row_positions[group_end] + bar_width / 2 - line_inset,
                ],
                color=group_color,
                linewidth=row_group_linewidth,
                solid_capstyle="butt",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )
            group_start = row_number

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

    if show_legend:
        term_legend = ax.legend(
            title="Term",
            loc="upper center",
            bbox_to_anchor=(0.4, -0.07),
            ncol=min(4, max(1, len(terms))),
            frameon=False,
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

    if not show_y_axis:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, right=False, length=0)
        ax.set_ylabel("")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    if owns_figure:
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

    if show and owns_figure:
        plt.show()
    elif not show and owns_figure:
        plt.close(fig)

    return fig, ax, plot_wide
