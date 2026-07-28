#!/usr/bin/env python
# coding: utf-8

# # 1.1.Process downloaded loaddata to match local image data
# This notebook:
# 1. Reads in the previously downloaded loaddata file.
# 2. Cross-compare loaddata file paths with local data downloads, correct paths with best effort. 
# 3. Write loaddata files with fixed path.

# In[1]:


from pathlib import Path, PurePosixPath
import yaml

import pandas as pd


# ## Pathing

# In[2]:


config_file_path = Path("degradation_config.yaml")
if not config_file_path.exists():
    raise RuntimeError(f"Config file {config_file_path} does not exist.")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

local_image_dir = config.get("local_image_dir", None)
if not local_image_dir:
    raise RuntimeError(
        f"Local image directory not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )
local_image_dir = Path(config["local_image_dir"]).expanduser().resolve()
if not Path(local_image_dir).exists() or not local_image_dir.is_dir():
    raise FileNotFoundError(f"Local image directory {local_image_dir} does not exist.")

channels = config.get("channels", None)
if not channels:
    raise RuntimeError(
        f"Channels not set in the config. "
        "Please configure the analysis referencing the `degradation_config.template.yaml`."
    )

metadata_download_path = Path("metadata")
if not metadata_download_path.exists():
    metadata_download_path.mkdir(parents=True, exist_ok=True)

loaddata_dir = metadata_download_path / "loaddatas"
loaddata_files = sorted(
    path for path in loaddata_dir.glob("*.csv") if not path.name.endswith(".fixed.csv")
)
if not loaddata_files:
    raise FileNotFoundError(f"No source loaddata CSV files found in {loaddata_dir}.")


# ## Fixing loaddata paths
# Assumes local image data exists as the same file structure as that being processed in pediatric cancer atlas profiling repo.  

# In[3]:


path_col_prefix = "PathName_"
file_col_prefix = "FileName_"
# original analysis data root path, shouldn't change since the loaddata are downloaded from a pinned commit
source_path_prefix = "/media/18tbdrive" 
source_path_prefix = PurePosixPath("/media/18tbdrive")

path_columns = {channel: f"{path_col_prefix}{channel}" for channel in channels}
file_columns = {channel: f"{file_col_prefix}{channel}" for channel in channels}

# Validate every input schema before writing any output, so a malformed plate fails early.
for loaddata_file in loaddata_files:
    csv_columns = set(pd.read_csv(loaddata_file, nrows=0).columns)
    missing_columns_by_channel = {
        channel: [
            column
            for column in (path_columns[channel], file_columns[channel])
                if column not in csv_columns
            ]
        for channel in channels
    }
    missing_columns_by_channel = {
        channel: columns
        for channel, columns in missing_columns_by_channel.items()
        if columns
    }
    if missing_columns_by_channel:
        missing_details = "; ".join(
            f"{channel}: {', '.join(columns)}"
            for channel, columns in missing_columns_by_channel.items()
        )
        raise KeyError(f"Missing anticipated columns in {loaddata_file.name}: {missing_details}")

# Process each loaddata CSV file to validate and fix paths and filenames.
for loaddata_file in loaddata_files:
    print(f"Processing loaddata file: {loaddata_file}")

    loaddata_df = pd.read_csv(loaddata_file)
    print(f"Loaded loaddata file with shape: {loaddata_df.shape}")

    # Collect failed path/file counts 
    failure_report = pd.DataFrame(
        0, index=pd.Index(channels, name="channel"), columns=["path", "file"], dtype=int
    )

    # Remap acquisition-machine directories to the local data root and reject missing paths.
    for channel in channels:
        path_col = path_columns[channel]
        fixed_paths = []

        for path_cell_content in loaddata_df[path_col]:
            if pd.isna(path_cell_content):
                fixed_paths.append(None)
                failure_report.at[channel, "path"] += 1
                continue

            source_path = PurePosixPath(str(path_cell_content))
            try:
                relative_path = source_path.relative_to(source_path_prefix)
            except ValueError:
                fixed_paths.append(None)
                failure_report.at[channel, "path"] += 1
                continue

            fixed_path = local_image_dir.joinpath(*relative_path.parts)
            fixed_paths.append(str(fixed_path))

        loaddata_df[path_col] = fixed_paths

    # Remove invalid directories first; only surviving rows can form meaningful TIFF paths.
    rows_before_path_drop = len(loaddata_df)
    loaddata_df = loaddata_df.dropna(subset=list(path_columns.values())).copy()
    print(f"Dropped {rows_before_path_drop - len(loaddata_df)} rows with missing paths")

    # Pair each channel's validated directory and filename to detect missing TIFF transfers.
    all_channel_files_exist = pd.Series(True, index=loaddata_df.index)
    for channel in channels:
        path_col = path_columns[channel]
        file_col = file_columns[channel]
        channel_files_exist = pd.Series(False, index=loaddata_df.index)

        for row_index, (path_cell_content, file_cell_content) in loaddata_df[
            [path_col, file_col]
        ].iterrows():
            if pd.isna(file_cell_content):
                continue

            fixed_file = Path(path_cell_content) / str(file_cell_content)
            channel_files_exist.at[row_index] = fixed_file.is_file()

        failure_report.at[channel, "file"] = int((~channel_files_exist).sum())
        all_channel_files_exist &= channel_files_exist

    # Keep only rows that are complete across every anticipated imaging channel.
    rows_before_file_drop = len(loaddata_df)
    loaddata_df = loaddata_df.loc[all_channel_files_exist].copy()
    print(f"Dropped {rows_before_file_drop - len(loaddata_df)} rows with missing TIFF files")
    print("\nFailed validation counts by channel:")
    print(failure_report.to_string())

    output_file = loaddata_file.with_name(f"{loaddata_file.stem}.fixed.csv")
    loaddata_df.to_csv(output_file, index=False)
    print(f"Saved {len(loaddata_df)} fully validated rows to {output_file}\n")

