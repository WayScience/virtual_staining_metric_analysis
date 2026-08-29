#!/usr/bin/env python
# coding: utf-8

# # 1.7.Perform nested regression on metric values against cell type and density
# This notebook:
# 1. Reads in metric evaluation records generated and written in 1.4.
# 2. Merges metric evaluation records with platemap to obtain human readable metadata regarding biology.
# 3. Performs two separate nested regression analysis, one against cell type and the other against seeding density. 

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

from utils.iter_data import iter_metric_transform_frames
from utils.nested_regression import (
    BootstrapConfig,
    ColumnSpec,
    bootstrap_nested_regression,
    summarize_r2_scatter_bootstrap
)
from utils.validate_config import (
    load_yaml_config,
    require_config_directory,
)


# ## Pathing

# In[2]:


config = load_yaml_config("degradation_config.yaml")
analysis_dir = require_config_directory(config, "analysis_out_dir")

metadata_download_path = Path(".") / "metadata"
metadata_download_path.mkdir(parents=True, exist_ok=True)

platemap_dir = metadata_download_path / "platemaps"
platemap_files = sorted(path for path in platemap_dir.glob("*_platemap.csv"))
if not platemap_files:
    raise FileNotFoundError(f"No platemap CSV files found in {platemap_dir}.")
barcode_file = metadata_download_path / "Barcode_platemap_pilot_data.csv"
if not barcode_file.exists():
    raise FileNotFoundError(f"Barcode platemap CSV file not found: {barcode_file}")

metric_dir = analysis_dir / "patches" / "metrics"
if not metric_dir.exists():
    raise FileNotFoundError(
        f"Metric directory {metric_dir} does not exist. "
        "Please run [] to generate the metrics before running this notebook."
    )
metric_subdirs = [d for d in metric_dir.iterdir() if d.is_dir()]
print(f"Found {len(metric_subdirs)} metric subdirectories:")
for subdir in metric_subdirs:
    print(f"  - {subdir.name}")

output_dir = Path(".") / "results" / "bio_covariate_nested_regression"
output_dir.mkdir(parents=True, exist_ok=True)


# ## Wrangle metadata so later merge can happen smoothly

# In[3]:


barcode_df = pd.read_csv(barcode_file).rename(columns={'barcode': 'Metadata_Plate'})

platemap_df = pd.DataFrame()
for platemap in barcode_df['platemap_file'].unique():
    df = pd.read_csv(platemap_dir / f'{platemap}.csv')
    df['platemap_file'] = platemap
    platemap_df = pd.concat([platemap_df, df], ignore_index=True)
barcode_platemap_df = pd.merge(
    barcode_df, platemap_df, on='platemap_file', how='inner'
).rename(columns={'well': 'Metadata_Well'})


# In[4]:


regression_config = {
    "y": "metric_value",
    "x1": "parameter_value",
}

bootstrap_config = {
    "n_boot": 100,
    "sample_frac": 0.5,
    "replace": True,
    "standardize": False,
    "robust_cov": None,
    "min_group_size": 25,
}

colspec_density = ColumnSpec(
    group_cols=(),
    x2="seeding_density",
    x2_categorical=True,
    standardize_cols=("parameter_value", "seeding_density"),
    **regression_config,
)

colspec_cell = ColumnSpec(
    group_cols=(),
    x2="cell_line",
    x2_categorical=True,
    standardize_cols=("parameter_value",),
    **regression_config,
)

cfg = BootstrapConfig(**bootstrap_config)
bootstrap_rng = np.random.default_rng(cfg.random_state)

boot_res_density_frames: list[pd.DataFrame] = []
boot_res_cell_frames: list[pd.DataFrame] = []


# In[5]:


for subdir, transform_name, df in iter_metric_transform_frames(
    metric_subdirs
):
    metric_name = subdir.name
    print(
        f"Processing metric={metric_name}, "
        f"transform={transform_name}, "
        f"rows={len(df):,}"
    )

    metric_values = df["metric_value"]
    invalid_mask = metric_values.isna() | ~np.isfinite(metric_values)
    if invalid_mask.any():
        if transform_name != "random_gamma" and "foreground" not in metric_name:
            print(
                f"Skipping metric={metric_name}, "
                f"transform={transform_name}: "
                f"metric_value contains {invalid_mask.sum():,} invalid values."
            )
            continue

        df = df.loc[~invalid_mask].copy()

    if df.empty:
        print(
            f"Skipping metric={metric_name}, "
            f"transform={transform_name}: "
            "no valid metric values remain."
        )
        continue

    df_enrich = df.merge(
        barcode_platemap_df,
        on=["Metadata_Plate", "Metadata_Well"],
        how="inner",
        validate="many_to_one",
    )

    if df_enrich.empty:
        print(
            f"Skipping metric={metric_name}, "
            f"transform={transform_name}: "
            "no rows remained after the platemap merge."
        )

        continue

    density_result = bootstrap_nested_regression(
        df_enrich,
        colspec_density,
        cfg,
        rng=bootstrap_rng,
    )

    cell_result = bootstrap_nested_regression(
        df_enrich,
        colspec_cell,
        cfg,
        rng=bootstrap_rng,
    )

    result_keys = {
        "metric_name": metric_name,
        "transform_name": transform_name,
    }

    if not density_result.empty:
        boot_res_density_frames.append(
            density_result.assign(**result_keys)
        )

    if not cell_result.empty:
        boot_res_cell_frames.append(
            cell_result.assign(**result_keys)
        )

if not boot_res_density_frames:
    raise ValueError("No density nested regression results were produced.")
if not boot_res_cell_frames:
    raise ValueError("No cell nested regression results were produced.")

boot_res_density = pd.concat(boot_res_density_frames, ignore_index=True)
boot_res_cell = pd.concat(boot_res_cell_frames, ignore_index=True)


# In[7]:


_ = summarize_r2_scatter_bootstrap(
    boot_res_density,
    output_file=output_dir / "boot_nest_seeding_density_summary.parquet",
)

_ = summarize_r2_scatter_bootstrap(
    boot_res_cell,
    output_file=output_dir / "boot_nest_cell_line_summary.parquet",
)
