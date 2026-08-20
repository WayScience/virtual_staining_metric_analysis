#!/usr/bin/env python
# coding: utf-8

# # 1.6.Plot variance parititoning results
# This notebook:
# 1. Reads in results obtained from 1.5.
# 2. Visualizes the full ANOVA eta2 division among covariates across all metric, degradation combinations.
# 3. Visualizes a more condensed eta2 division grouping covariates by degradation, confounders and residual for a more intuitive presentation.
# 4. Writies plots to disks.

# In[1]:


from pathlib import Path

import pandas as pd

from utils.var_partition_plot import plot_anova_variance_partition
from utils.var_partition_plot import plot_anova_radar


# In[2]:


anova_output_dir = Path(".") / "results" / "variance_partition_analysis"
if not anova_output_dir.exists():
    raise FileNotFoundError(
        f"ANOVA output directory {anova_output_dir} does not exist. "
        f"Please run 1.5.variance_parition_analysis.ipynb first."
    )

anova_results_file = anova_output_dir / "variance_partition_results.parquet"
if not anova_results_file.exists():
    raise FileNotFoundError(
        f"ANOVA results file {anova_results_file} does not exist. "
        f"Please run 1.5.variance_parition_analysis.ipynb first."
    )

variance_partition_df = pd.read_parquet(anova_results_file)

plot_dir = anova_output_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


TERM_COLORS = {
    "parameter_value": "#E69F00",
    "cell_line": "#0072B2",
    "seeding_density": "#009E73",
    "channel": "#6A3D9A",
    "cell_line:seeding_density": "#8DA0CB",
    "cell_line:channel": "#66C2A5",
    "seeding_density:channel": "#B2ABD2",
    "parameter_value:cell_line": "#D55E00",
    "parameter_value:seeding_density": "#CC79A7",
    "parameter_value:channel": "#A6761D",
    "Residual": "#9E9E9E",
}

TERM_ORDER = [
    "parameter_value",
    "cell_line",
    "seeding_density",
    "channel",
    "cell_line:seeding_density",
    "cell_line:channel",
    "seeding_density:channel",
    "parameter_value:seeding_density",
    "parameter_value:cell_line",
    "parameter_value:channel",
    "Residual",
]

TERM_LABELS = {
    "parameter_value": "Severity",
    "cell_line": "Cell line",
    "seeding_density": "Seeding density",
    "channel": "Channel",
    "cell_line:seeding_density": "Cell line x Seeding density",
    "cell_line:channel": "Cell line x Channel",
    "seeding_density:channel": "Seeding density x Channel",
    "parameter_value:cell_line": "Severity x Cell line",
    "parameter_value:seeding_density": "Severity x Seeding density",
    "parameter_value:channel": "Severity x Channel",
    "Residual": "Residual",
}

METRIC_ORDER = [
    "dists",
    "lpips",
    "foreground_ssim",
    "ssim",
    "foreground_psnr",
    "psnr",
    "mae",
]

METRIC_LABELS = {
    "dists": "DISTS",
    "lpips": "LPIPS",
    "foreground_ssim": "Foreground SSIM",
    "ssim": "SSIM",
    "foreground_psnr": "Foreground PSNR",
    "psnr": "PSNR",
    "mae": "MAE",
}


TRANSFORM_LABELS = {
    "dilate": "Dilate",
    "erode": "Erode",
    "gauss_noise": "Gaussian\nnoise",
    "gaussian_blur": "Gaussian\nblur",
    "grid_distortion": "Grid\ndistortion",
    "random_gamma": "Gamma\ncorrection",
}


# In[4]:


HATCH_GROUPS = {
    "Univariate known": (
        {
            "cell_line",
            "seeding_density",
            "channel",
        },
        "//",
    ),
    "Interaction terms": (
        {
            "cell_line:seeding_density",
            "cell_line:channel",
            "seeding_density:channel",
            "parameter_value:seeding_density",
            "parameter_value:cell_line",
            "parameter_value:channel",
        },
        "..",
    ),
}

fig, ax, plot_wide = plot_anova_variance_partition(
    variance_partition_df,
    row_cols=(
        "metric_name",
        "transform_name",
    ),
    value_col="eta2",
    term_labels=TERM_LABELS,
    term_order=TERM_ORDER,
    term_colors=TERM_COLORS,
    hatch_groups=HATCH_GROUPS,
    row_orders={
        "metric_name": METRIC_ORDER,
    },
    row_value_labels={
        "metric_name": METRIC_LABELS,
        "transform_name": {k: v.replace("\n", " ") for k, v in TRANSFORM_LABELS.items()},
    },
    output_path=plot_dir / "anova.png",
    figsize_width=12,
    row_height=0.34,
    show=True,
)


# In[5]:


RADAR_COMPONENTS = {
    "severity_independent": {
        "columns": ["parameter_value"],
        "color": "#F1A340",
        "label": "Degradation severity: independent",
        "alpha": 0.78,
    },
    "severity_interaction": {
        "columns": [
            "parameter_value:seeding_density",
            "parameter_value:cell_line",
            "parameter_value:channel",
        ],
        "color": "#D62728",
        "label": "Degradation severity: interaction",
        "alpha": 0.70,
    },
    "other_known_independent": {
        "columns": [
            "cell_line",
            "seeding_density",
            "channel",
        ],
        "color": "#3B4CC0",
        "label": "Other modeled effects: independent",
        "alpha": 0.62,
    },
    "other_known_interaction": {
        "columns": [
            "cell_line:seeding_density",
            "cell_line:channel",
            "seeding_density:channel",
        ],
        "color": "#7B8FD8",
        "label": "Other modeled effects: interaction",
        "alpha": 0.56,
    },
    "residual": {
        "columns": ["Residual"],
        "color": "#BDBDBD",
        "label": "Residual",
        "alpha": 0.50,
    },
}

_ = plot_anova_radar(
    variance_partition_df,
    component_specs=RADAR_COMPONENTS,
    metric_order=METRIC_ORDER,
    metric_labels=METRIC_LABELS,
    transform_order=list(TRANSFORM_LABELS),
    transform_labels=TRANSFORM_LABELS,
    output_path=plot_dir / "anova_radar.png",
    show=True,
    legend_adjust=0.0
)
