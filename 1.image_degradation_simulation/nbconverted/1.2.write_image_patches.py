#!/usr/bin/env python
# coding: utf-8

# # 1.2.Generate and write image patches from loaddata, to use for later image degradation generation.
# This notebook:
# 1. Reads in the previously downloaded and processed loaddata file, as well as CellProfiler single cell level profiles.
# 2. Construct dataset & generate tiling patches, filter to keep only patches containing at least one CellProfiler object
# 3. Write normalized, patched images to disk keeping track of each image file and source metadata.

# In[1]:


from pathlib import Path
from functools import reduce
import sys
import yaml

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from virtual_stain_flow.datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize
from virtual_stain_flow.datasets.ds_engine.crop_generators import generate_tile_crops
from virtual_stain_flow.evaluation.visualization import plot_dataset_grid
from utils.crop_specs import filter_crops_with_objects
from utils.loaddata_to_index import build_dataset_inputs
from utils.patch_writing import get_crop_file_index, write_normalized_crops


# ## Pathing

# In[2]:


config_file_path = Path("degradation_config.yaml")
if not config_file_path.exists():
    raise RuntimeError(f"Config file {config_file_path} does not exist.")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

analysis_dir = config.get("analysis_out_dir", None)

if not analysis_dir:
    raise RuntimeError(
        f"Analysis output directory not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )

analysis_dir = Path(analysis_dir)

if not analysis_dir.exists():
    analysis_dir.mkdir(parents=True, exist_ok=True)

patch_out_dir = analysis_dir / "patches"

if not patch_out_dir.exists():
    patch_out_dir.mkdir(parents=True, exist_ok=True)

local_profile_dir = config.get("local_profile_dir", None)

if not local_profile_dir:
    raise RuntimeError(
        f"Local profile directory not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )

local_profile_dir = Path(local_profile_dir)

if not local_profile_dir.exists():
    raise RuntimeError(
        f"Local profile directory {local_profile_dir} does not exist."
    )

channels = config.get("channels", None)
if not channels:
    raise RuntimeError(
        f"Channels not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )

input_channel = config.get("input_channel", None)
if not input_channel:
    raise RuntimeError(
        f"Input channel not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )

metadata_download_path = Path("metadata")
if not metadata_download_path.exists():
    metadata_download_path.mkdir(parents=True, exist_ok=True)

loaddata_dir = metadata_download_path / "loaddatas"
loaddata_files = sorted(
    path for path in loaddata_dir.glob("*.fixed.csv")
)
if not loaddata_files:
    raise FileNotFoundError(f"No source loaddata CSV files found in {loaddata_dir}.")


# # Construct Dataset

# Read in downloaded and processed loaddata

# In[3]:


loaddata_df = pd.concat(pd.read_csv(file) for file in loaddata_files)
loaddata_df.reset_index(drop=True, inplace=True)
loaddata_df.head()


# Read in local profiles corresponding to the images indexed by loaddata.
# We don't need the profiles here, just the metadata columns, which faciliates:
# - merging with the loaddata to map image files to experimetnal conditions and image level identity.
# - access of the object segmentation information, suggesting where in each image exists probable cells.   

# In[4]:


profile_files = list(
        local_profile_dir.glob('*_sc_normalized.parquet')
    )
print(f"Found {len(profile_files)} parquet files in {local_profile_dir}.")
pq_schemas = [
    pq.ParquetFile(parquet).schema for parquet in profile_files
]
shared_columns = list(
    reduce(
        np.intersect1d,
        [set(schema.names) for schema in pq_schemas]
    )[0]
)

# Only read in the metadata columns to reduce memory usage and make reading faster. 
shared_meta_columns = [col for col in shared_columns if col.startswith("Metadata_")]

profiles = pd.concat(
    [
        pq.read_table(parquet, columns=shared_meta_columns).to_pandas()
        for parquet in profile_files
    ], 
    ignore_index=True
)

profiles.head()


# ## Construct & showcase patched image dataset

# In[5]:


# Raw dataset returning multi-channel full FOVs
image_file_index_metadata, pt_mapping = build_dataset_inputs(
    loaddata=loaddata_df,
    profile=profiles,
    channels=channels,
)

dataset = BaseImageDataset(
    file_index=image_file_index_metadata.loc[:, channels],
    check_exists=True,
)

dataset.input_channel_keys = [input_channel]
dataset.target_channel_keys = [chan for chan in channels if chan != input_channel]

# Generate unguided tiling patches (crops)
crop_specs = generate_tile_crops(dataset, crop_size=256)
# Filter patches to only include those containing objects
crop_specs = filter_crops_with_objects(crop_specs, pt_mapping)
# Patch dataset containing only the "relevant" crops
cropped_dataset = CropImageDataset(
    file_index=dataset.file_index,
    transforms=MaxScaleNormalize("16bit"),
    crop_specs=crop_specs,
    pil_image_mode=dataset.pil_image_mode,
    input_channel_keys=dataset.input_channel_keys,
    target_channel_keys=dataset.target_channel_keys,
)

print(f"Number of cropped images containing objects: {len(cropped_dataset)}")
_ = plot_dataset_grid(
    cropped_dataset,
    indices=[0,1,2,3,4],
    wspace=0.025,
    hspace=0.05
)


# ## Indexing the crops to ensure relationship to source FOV is retained (important for later analysis requiring stratification by experimental condition)
# Each patch (crop) manifest entry stores the positional row (`manifest_idx`) of its source multi-channel image. 
# The next cell uses that position to attach the crop coordinates, all six source file paths, and every image-level metadata field to a stable `crop_dataset_index`. 
# It also verifies the source paths against `cropped_dataset.file_index` before writing `crop_index.csv`.

# In[6]:


crop_index = get_crop_file_index(
    cropped_dataset,
    image_file_index_metadata,
)

# validate no source-file mismatches between the crop index and the original image metadata
dataset_source_files = image_file_index_metadata.loc[crop_index["source_image_row"]].reset_index(drop=True)
for channel in channels:
    if not dataset_source_files[channel].astype(str).equals(
        crop_index[channel].astype(str)
    ):
        raise ValueError(f"Source-file mismatch for channel {channel}.")
    crop_index[channel] = crop_index[channel].astype(str)

crop_index_path = patch_out_dir / "crop_index.csv"
crop_index.to_csv(crop_index_path, index=False)
print(f"Indexed {len(crop_index):,} crops in {crop_index_path}")
crop_index.head()


# ## Write normalized channel patches, still keeping track of metadata
# 
# The writer stores each normalized channel patch as a float32 TIFF under `tiffs/<plate>/<well>/site_<site>/<channel>/`. 
# Filenames contain the global patch index and patch origin, while `index.csv` records the relative TIFF path, its source image file, crop geometry, and all image-level metadata. 
# Existing TIFFs are skipped by default, so an interrupted export can be rerun safely; the complete patch index is rebuilt atomically at the end.

# In[7]:


patch_index = write_normalized_crops(
    cropped_dataset,
    crop_index,
    patch_out_dir,
    overwrite=False,
)

expected_file_count = len(cropped_dataset) * len(channels)
if len(patch_index) != expected_file_count:
    raise RuntimeError(
        f"Expected {expected_file_count:,} indexed TIFFs, found {len(patch_index):,}."
    )

patch_index.head()
