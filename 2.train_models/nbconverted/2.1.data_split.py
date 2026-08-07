#!/usr/bin/env python
# coding: utf-8

# # Create train, heldout, and evaluation splits for model training.
# Train and heldout set will exclusively contain U2-OS cells from plate 1. 
# Evaluation split will contain all other cells that is not plate U2-OS.
# The split is designed to keep all sites from a well together, preventing images from the same well from appearing in both training and heldout data. 
# 
# ## Split procedure
# 
# 1. Load processed loaddata metadata and annotate each site with plate- and well-level experimental metadata from the barcode and platemap files.
# 2. Select the **train/heldout candidate pool** using `TRAIN_CONDITION_KWARGS`. Rows outside this configured domain form the **evaluation split**. 
# Specifically this notebook predefines the train conditions to be different seeding densities of U2-OS on platemap 1
# 3. Group the candidate pool by the configured condition columns. Within each group, randomly select one complete well for the **heldout split** and assign every remaining well to the **training split**.
# 4. Write the three resulting tables to `data_split_output/`.
# 
# A fixed random seed makes the well selection reproducible as long as the input data and group ordering remain unchanged.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds

from utils.validate_config import (
    load_yaml_config,
    require_config_directory,
)


# ## Pathing
# The input loaddata files contain one row per imaging site. Barcode and platemap CSVs provide the experimental metadata used to define conditions and split groups. Only processed `*.fixed.parquet` loaddata files are included.

# In[2]:


degradation_path = Path(".").resolve().parent / "1.image_degradation_simulation"
if not degradation_path.exists():
    raise FileNotFoundError(f"Degradation simulation directory not found: {degradation_path}")

config = load_yaml_config(degradation_path / "degradation_config.yaml")

metadata_download_path = degradation_path / "metadata"
metadata_download_path.mkdir(parents=True, exist_ok=True)

loaddata_dir = metadata_download_path / "loaddatas"
loaddata_files = sorted(path for path in loaddata_dir.glob("*.fixed.parquet"))
if not loaddata_files:
    raise FileNotFoundError(f"No processed loaddata Parquet files found in {loaddata_dir}.")

platemap_dir = metadata_download_path / "platemaps"
platemap_files = sorted(path for path in platemap_dir.glob("*_platemap.csv"))
if not platemap_files:
    raise FileNotFoundError(f"No platemap CSV files found in {platemap_dir}.")
barcode_file = metadata_download_path / "Barcode_platemap_pilot_data.csv"
if not barcode_file.exists():
    raise FileNotFoundError(f"Barcode platemap CSV file not found: {barcode_file}")

local_profile_dir = require_config_directory(config, "local_profile_dir")
profile_files = sorted(local_profile_dir.glob("*_sc_normalized.parquet"))
if not profile_files:
    raise FileNotFoundError(
        f"No normalized profile Parquet files found in {local_profile_dir}."
    )

output_dir = Path(".") / "data_split_output"
output_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


CELL_COLUMN = 'cell_line'
TRAIN_CELL = 'U2-OS'
PLATEMAP_COLUMN = 'platemap_file'
TRAIN_PLATE = 'Assay_Plate1_platemap'
DENSITY_COLUMN = 'seeding_density'
TRAIN_SEEDING_DENSITIES = [1_000, 2_000, 4_000, 8_000, 12_000]

TRAIN_CONDITION_COLS = [CELL_COLUMN, PLATEMAP_COLUMN, DENSITY_COLUMN]

SITE_COLUMN = 'Metadata_Site'
WELL_COLUMN = 'Metadata_Well'
PLATE_COLUMN = 'Metadata_Plate'


# ## Build site-level experimental metadata
# The barcode table maps each acquisition plate to a platemap. The platemaps map wells to experimental conditions such as cell line and seeding density. Joining these tables creates a plate-and-well lookup that can annotate every imaging site in the loaddata.

# In[4]:


barcode_df = pd.read_csv(barcode_file).rename(columns={'barcode': 'Metadata_Plate'})

platemap_df_contents = []
for platemap in barcode_df['platemap_file'].unique():
    df = pd.read_csv(platemap_dir / f'{platemap}.csv')
    df['platemap_file'] = platemap
    platemap_df_contents.append(df)
platemap_df = pd.concat(platemap_df_contents, ignore_index=True)
barcode_platemap_df = pd.merge(barcode_df, platemap_df, on='platemap_file', how='inner').rename(columns={'well': 'Metadata_Well'})


# In[5]:


dataset = ds.dataset(loaddata_files, format="parquet")
table = dataset.to_table()
loaddata_df = table.to_pandas()


# In[6]:


loaddata_barcode_platemap_df = pd.merge(
    loaddata_df, 
    barcode_platemap_df, 
    on=['Metadata_Plate', 'Metadata_Well'], 
    how='left',
    validate='m:1'
)

print(loaddata_barcode_platemap_df.head())


# ## Splitting

# In[7]:


train_mask = (
    loaddata_barcode_platemap_df[CELL_COLUMN].eq(TRAIN_CELL)
    & loaddata_barcode_platemap_df[PLATEMAP_COLUMN].eq(TRAIN_PLATE)
    & loaddata_barcode_platemap_df[DENSITY_COLUMN].isin(TRAIN_SEEDING_DENSITIES)
)

loaddata_barcode_platemap_train_df = loaddata_barcode_platemap_df.loc[train_mask]
loaddata_barcode_platemap_eval_df = loaddata_barcode_platemap_df.loc[~train_mask]

if loaddata_barcode_platemap_train_df.empty:
    raise ValueError("No data found matching training conditions.")

print(
    f"{len(loaddata_barcode_platemap_train_df)} sites for train and heldout"
)
print(
    f"{len(loaddata_barcode_platemap_eval_df)} sites for evaluation"
)


# The candidate pool is further stratified by the condition columns in `TRAIN_CONDITION_KWARGS`. 
# One well is sampled from each condition group for heldout evaluation; all sites from that well are assigned together. 
# Sites from the remaining wells form the training split.
# 
# Using wells rather than individual sites as the split unit reduces leakage from shared well-level biology and acquisition conditions. Each condition group must contain at least two unique wells to contribute data to both splits.

# In[8]:


# Fix the random well selection for reproducible splits.
seed = 42
np.random.seed(seed)

# Stratify by the same metadata fields used to define the candidate pool.
grouped = loaddata_barcode_platemap_train_df.groupby(TRAIN_CONDITION_COLS)

heldout_list = []
train_list = []

for _, group in grouped:
    # Select one complete well per condition; never divide a well's sites across splits.
    held_out_well = [np.random.choice(group[WELL_COLUMN].unique())]
    train_wells = group[~group[WELL_COLUMN].isin(held_out_well)][WELL_COLUMN].unique()

    loaddata_held_out_df = group[group[WELL_COLUMN].isin(held_out_well)].copy()
    loaddata_train_df = group[group[WELL_COLUMN].isin(train_wells)].copy()

    condition = group[TRAIN_CONDITION_COLS].iloc[0].to_dict()
    print(f"For Condition: {condition} Heldout well: {held_out_well} Train wells: {train_wells}")

    heldout_list.append(loaddata_held_out_df)
    train_list.append(loaddata_train_df)

# Combine the condition-level partitions into the final site-level tables.
loaddata_heldout_df = pd.concat(heldout_list).reset_index(drop=True)
print(f"{loaddata_heldout_df.shape[0]} sites Heldout")
loaddata_train_df = pd.concat(train_list).reset_index(drop=True)
print(f"{loaddata_train_df.shape[0]} sites for Training")

loaddata_heldout_df.to_parquet(output_dir / 'loaddata_heldout.parquet')
loaddata_train_df.to_parquet(output_dir / 'loaddata_train.parquet')
loaddata_barcode_platemap_eval_df.to_parquet(output_dir / 'loaddata_eval.parquet')


# ## Wrangle sc features to pair with loaddata
# Helps with generating object containing crops

# In[9]:


schemas = [pl.read_parquet_schema(path) for path in profile_files]

shared_columns = set.intersection(*(set(schema) for schema in schemas))
shared_meta_columns = [
    column
    for column in schemas[0]
    if column.startswith("Metadata_") and column in shared_columns
]

if not shared_meta_columns:
    raise ValueError("Profile Parquet files have no shared Metadata_ columns.")

if "Metadata_Plate" not in shared_meta_columns:
    raise ValueError("Metadata_Plate is not shared across all profile Parquet files.")

present_plates = (
    loaddata_barcode_platemap_eval_df["Metadata_Plate"]
    .unique()
    .tolist()
)

(
    pl.scan_parquet(profile_files)
    .select(shared_meta_columns)
    .filter(pl.col("Metadata_Plate").is_in(present_plates))
    .sink_parquet(output_dir / "sc_profiles.parquet")
)


# ## Outputs
# 
# The notebook writes three parquet files to `data_split_output/`:
# 
# - `loaddata_train.parquet`: sites from non-heldout wells in the configured training domain.
# - `loaddata_heldout.parquet`: all sites from one sampled well per condition group.
# - `loaddata_eval.parquet`: sites outside the configured training domain.
# - `sc_profiles.parquet`: single cell level metadata for object location.
