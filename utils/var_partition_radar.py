from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import matplotlib.patheffects as path_effects
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
) -> tuple[Figure, np.ndarray]:
    """Plot one stacked radar per metric, with degradations on the radial axes."""
    return _plot_anova_radar(
        anova_df,
        component_specs=component_specs,
        panel_by="metric_name",
        panel_order=metric_order,
        panel_labels=metric_labels,
        radial_order=transform_order,
        radial_labels=transform_labels,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )


def plot_anova_radar_by_degradation(
    anova_df: pd.DataFrame,
    *,
    component_specs: Mapping[str, RadarComponentSpec],
    metric_labels: Mapping[str, str] | None = None,
    metric_order: Sequence[str] | None = None,
    transform_labels: Mapping[str, str] | None = None,
    transform_order: Sequence[str] | None = None,
    layout: Literal["grid", "single_row"] = "grid",
    output_path: Path | str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot one stacked radar per degradation, with metrics on the radial axes."""
    return _plot_anova_radar(
        anova_df,
        component_specs=component_specs,
        panel_by="transform_name",
        panel_order=transform_order,
        panel_labels=transform_labels,
        radial_order=metric_order,
        radial_labels=metric_labels,
        layout=layout,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )


def _plot_anova_radar(
    anova_df: pd.DataFrame,
    *,
    component_specs: Mapping[str, RadarComponentSpec],
    panel_by: Literal["metric_name", "transform_name"],
    panel_order: Sequence[str] | None,
    panel_labels: Mapping[str, str] | None,
    radial_order: Sequence[str] | None,
    radial_labels: Mapping[str, str] | None,
    layout: Literal["grid", "single_row"] = "grid",
    output_path: Path | str | None,
    dpi: int,
    show: bool,
) -> tuple[Figure, np.ndarray]:
    """Render either orientation of the ANOVA stacked radar grid."""
    component_order = list(component_specs)
    panel_labels = dict(panel_labels or {})
    radial_labels = dict(radial_labels or {})

    radar_wide = _prep_radar_plot(
        anova_df=anova_df,
        component_specs=component_specs,
    )

    radial_by = "transform_name" if panel_by == "metric_name" else "metric_name"
    observed_panels = radar_wide[panel_by].astype(str).unique().tolist()
    observed_radials = radar_wide[radial_by].astype(str).unique().tolist()

    panels = (
        [value for value in panel_order if value in observed_panels]
        if panel_order is not None
        else sorted(observed_panels)
    )
    radials = list(radial_order) if radial_order is not None else sorted(observed_radials)

    if not panels:
        panels = sorted(observed_panels)

    if not panels or not radials:
        raise ValueError("No metrics or transforms remain for plotting.")
    if layout == "grid" and len(panels) > 7:
        raise ValueError("Radar grids support at most seven panels plus the legend panel.")

    angles = np.linspace(
        0,
        2 * np.pi,
        len(radials),
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

    if layout == "single_row":
        nrows = 1
        ncols = len(panels) + 1
        figsize = (3.2 * ncols, 3.65)
    else:
        nrows = 2
        ncols = 4
        figsize = (12.8, 7.3)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={
            "polar": True,
        },
    )
    axes = np.atleast_1d(axes).ravel()

    for i, panel_value in enumerate(panels):
        ax = axes[i]

        panel_df = (
            radar_wide.loc[
                radar_wide[panel_by].eq(panel_value),
                [
                    radial_by,
                    *component_order,
                ],
            ]
            .groupby(
                radial_by,
                as_index=False,
            )[component_order]
            .mean()
            .set_index(radial_by)
            .reindex(radials)
            .fillna(0.0)
            .reset_index()
        )

        lower_boundary = np.zeros(
            len(radials),
            dtype=float,
        )

        for index, component in enumerate(component_order):
            spec = component_specs[component]
            zorder = len(component_order) - index
            upper_boundary = lower_boundary + panel_df[component].to_numpy(dtype=float)

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
            [radial_labels.get(name, name) for name in radials],
            fontsize=9 if radial_by == "metric_name" else 10,
        )
        for tick_label in ax.get_xticklabels():
            tick_label.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )
        ax.tick_params(axis="x", pad=4)

        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)

        ax.set_rlabel_position(15)
        ax.grid(alpha=0.3)

        ax.set_title(
            panel_labels.get(panel_value, panel_value).replace("\n", " "),
            fontsize=13,
            fontweight="normal",
            pad=14,
        )

    for j in range(len(panels), len(axes) - 1):
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

    legend_ax = axes[-1]
    legend_ax.set_axis_off()
    component_legend = legend_ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
        frameon=False,
        title="Variance component",
        fontsize=10.5,
        title_fontsize=12,
        alignment="left",
    )
    component_legend.get_title().set_fontweight("normal")

    fig.subplots_adjust(
        bottom=0.04,
        top=0.95,
        left=0.05,
        right=0.97,
        wspace=0.45,
        hspace=0.30,
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
