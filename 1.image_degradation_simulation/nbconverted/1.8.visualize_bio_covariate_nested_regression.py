#!/usr/bin/env python
# coding: utf-8

# # 1.8.Visualize confounding effects by cell line and seeding density
# This notebook:
# 1. Reads in nested regression results generated and written in 1.7.
# 2. Visualizes confounding and sensitivity against cell line and density and seeding density.
# 3. Visualizes burden across all metric and degradation types against cell line and seeding density.

# In[1]:


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.nested_regression_plot import (
    compute_burden_df,
    plot_nested_r2_multi,
    plot_burden_heatmaps,
)
from utils.validate_config import load_degradation_plot_config


# ## Pathing

# In[2]:


output_dir = Path(".") / "results" / "bio_covariate_nested_regression"
if not output_dir.exists():
    raise FileNotFoundError(f"Output directory {output_dir} does not exist. Please run the previous steps to generate the results.")

cell_line_summary_file = output_dir / "boot_nest_cell_line_summary.parquet"
seeding_density_summary_file = output_dir / "boot_nest_seeding_density_summary.parquet"

if not cell_line_summary_file.exists() or not seeding_density_summary_file.exists():
    raise FileNotFoundError(
        f"Required summary Parquet files do not exist in {output_dir}. "
        "Please run notebook 1.7 to generate the results.")

cell_line_summary = pd.read_parquet(cell_line_summary_file)
seeding_density_summary = pd.read_parquet(seeding_density_summary_file)


# In[3]:


degradation_plot_config = load_degradation_plot_config("degradation_plot_config.yaml")

METRIC_ORDER = degradation_plot_config["metrics"]["order"]
METRIC_LABELS = degradation_plot_config["metrics"]["labels"]
METRIC_PALETTE = degradation_plot_config["metrics"]["palette"]
TRANSFORM_ORDER = degradation_plot_config["transforms"]["order"]
TRANSFORM_LABELS = degradation_plot_config["transforms"]["labels"]

ABLATION_MARKERS = {
    "dilate": "o",
    "erode": "^",
    "gaussian_blur": "s",
    "gauss_noise": "D",
    "grid_distortion": "*",
    "random_gamma": "P",
}


# In[4]:


summary_df = pd.concat(
    [
        cell_line_summary.assign(partial_term="Cell line"),
        seeding_density_summary.assign(partial_term="Seeding density")
    ],
    axis=0,
    ignore_index=True
)

summary_df.head()


# In[5]:


plot_nested_r2_multi(
    summary_df,
    restricted_term="Degradation severity",
    partial_terms=["Cell line", "Seeding density"],
    sharey=False,
    metric_labels=METRIC_LABELS,
    metric_colors=METRIC_PALETTE,
    metric_order=METRIC_ORDER,
    transform_labels=TRANSFORM_LABELS,
    transform_markers=ABLATION_MARKERS,
    output_path=output_dir / "nested_r2_scatter.png",
    dpi=300,
    show=True
)


# In[6]:


cell_burden_df = compute_burden_df(
    data=cell_line_summary,
    confounder_label="Cell line",
)

density_burden_df = compute_burden_df(
    data=seeding_density_summary,
    confounder_label="Seeding density / confluence",
)

burden_df = pd.concat(
    [
        cell_burden_df,
        density_burden_df,
    ],
    ignore_index=True,
)

burden_df.head()


# In[7]:


fig, axes, burden_df = plot_burden_heatmaps(
    confounder_data={
        "Cell line": cell_line_summary,
        "Seeding density": seeding_density_summary,
    },
    metric_labels=METRIC_LABELS,
    metric_order=METRIC_ORDER,
    transform_labels={key: value.replace("\n", " ") for key, value in TRANSFORM_LABELS.items()},
    transform_order=TRANSFORM_ORDER,
    title=None,
    output_path=output_dir / "burden_heatmaps.png",
    figsize=(10, 6.25),
    dpi=300,
    show=True
)

plt.show()
