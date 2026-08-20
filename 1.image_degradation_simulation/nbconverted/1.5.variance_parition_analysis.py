#!/usr/bin/env python
# coding: utf-8

# # 1.5.Perform variance partitioning analysis (ANOVA) on metric values against degradation and biology
# This notebook:
# 1. Reads in metric evaluation records generated and written in 1.4.
# 2. Merges metric evaluation records with platemap to obtain human readable metadata regarding biology.
# 3. Performs ANOVA analysis per metric and degradation combination (over record fragment)
# 4. Collects ANOVA results summary for all metric and degradation combinations and write as output.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

from utils.validate_config import (
    load_yaml_config,
    require_config_directory,
)
from utils.iter_data import iter_metric_transform_frames
from utils.metric_anova import fit_anova, AnovaSpec


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

metric_dir = output_dir = analysis_dir / "patches" / "metrics"
if not metric_dir.exists():
    raise FileNotFoundError(
        f"Metric directory {metric_dir} does not exist. "
        "Please run [] to generate the metrics before running this notebook."
    )
metric_subdirs = [d for d in metric_dir.iterdir() if d.is_dir()]
print(f"Found {len(metric_subdirs)} metric subdirectories:")
for subdir in metric_subdirs:
    print(f"  - {subdir.name}")

output_dir = Path(".") / "results" / "variance_partition_analysis"
output_dir.mkdir(parents=True, exist_ok=True)
plot_dir = output_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


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


# ## ANOVA

# Define ANOVA configuration

# In[4]:


ANOVA_SPEC: AnovaSpec = {
    "dependent": "metric_value",
    "variables": {
        "cell_line": {"kind": "categorical"},
        "seeding_density": {"kind": "categorical"},
        "channel": {"kind": "categorical"},
        "parameter_value": {"kind": "continuous"},
    },
    "terms": (
        ("cell_line",),
        ("seeding_density",),
        ("channel",),
        ("cell_line", "seeding_density"),
        ("cell_line", "channel"),
        ("seeding_density", "channel"),
        ("parameter_value",),
        ("parameter_value", "seeding_density"),
        ("parameter_value", "cell_line"),
        ("parameter_value", "channel"),
    ),
    "anova_type": 2,
}


# Iterate over metric-degradation combinations, merge fragment record with metadata and perform ANOVA

# In[5]:


variance_results: list[pd.DataFrame] = []
# because certain tranforms (e.g. random_gamma) can result in
# infinity or NaN metric values in combination with certain metrics
# (e.g. PSNR when MSE is too small, foreground variants of PSNR and SSIM when
# the foreground mask is too small or empty), 
# we allow a small proportion of invalid values to be ignored.
# The 20% is a somewhat arbitrary threshold, but it is intended to be permissive enough to 
# allow for the expected invalid values while still being strict enough to catch unexpected issues.
max_invalid_prop = 0.2

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
        # allow some invalid values for random_gamma transform + psnr, 
        # since the max gamma correction can overbrighten image and mess with PSNR
        if transform_name == "random_gamma" and metric_name in ["psnr", "foreground_psnr"]:
            pass
        elif metric_name in ["foreground_psnr", "foreground_ssim"]:
            pass
        elif invalid_mask.mean() > max_invalid_prop:
            print(
                f"Skipping metric={metric_name}, "
                f"transform={transform_name}: "
                f"metric_value contains {invalid_mask.sum():,} invalid values "
                f"({invalid_mask.mean():.2%} > {max_invalid_prop:.2%})."
            )
            continue
        else: # explicitly print unexpected low prop invalid values
            print(
                f"Unexpected invalid values encountered in metric={metric_name}, "
                f"transform={transform_name}: "
                f"metric_value contains {invalid_mask.sum():,} invalid values "
            )

        df = df.loc[~invalid_mask].copy()

    if df.empty:
        print(
            f"Skipping metric={metric_name}, "
            f"transform={transform_name}: "
            "no valid metric values remain."
        )
        continue

    print(
        f"After filtering invalid values, "
        f"metric={metric_name}, "
        f"transform={transform_name}, "
        f"rows={len(df):,}"
    )

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

    anova_result = fit_anova(
        df_enrich,
        ANOVA_SPEC,
        metadata={
            "metric_name": metric_name,
            "transform_name": transform_name,
        }
    )

    if anova_result is not None:
        variance_results.append(anova_result)


# Write output

# In[6]:


if not variance_results:
    raise RuntimeError(
        "No valid metric/transform ANOVA results were produced."
    )

variance_partition_df = pd.concat(
    variance_results,
    ignore_index=True,
)

variance_partition_df = (
    variance_partition_df
    .sort_values(
        [
            "metric_name",
            "transform_name",
            "term",
        ]
    )
    .reset_index(drop=True)
)

variance_partition_df.to_parquet(
    output_dir / "variance_partition_results.parquet", index=False,
)

variance_partition_df.head()
