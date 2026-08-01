#!/usr/bin/env python
# coding: utf-8

# # 1.0.Download loaddata csv from pinned pediatric atlas pilot repo commit
# This notebook downloads the CellProfiler loaddata from pinned pediatric atlas pilot repo commit.
# 
# The downloaded loaddata will serve later in this analysis directory as the image file index. 

# In[1]:


from pathlib import Path

from utils.download_csv import download_csv_files


# ## Download pathing

# In[2]:


metadata_download_path = Path("metadata")
metadata_download_path.mkdir(parents=True, exist_ok=True)


# This analysis is limited to the first two plates (24hour time point) of batch 1.

# In[3]:


source_repo = "https://github.com/WayScience/pediatric_cancer_atlas_profiling.git"
pin_commit = "6129e23e64cad5402c6d1d5ba679ff5f3eda82e4"

# Save this file directly under metadata/.
barcode = "0.download_data/metadata/platemaps/Barcode_platemap_pilot_data.csv"

# Save these files under metadata/platemaps/.
platemaps = [
    "0.download_data/metadata/platemaps/Assay_Plate1_platemap.csv",
    "0.download_data/metadata/platemaps/Assay_Plate2_platemap.csv",
]

# Save these files under metadata/loaddatas/.
loaddatas = [
    "1.illumination_correction/loaddata_csvs/Round_1_data/BR00143976_concatenated.csv",
    "1.illumination_correction/loaddata_csvs/Round_1_data/BR00143977_concatenated.csv",
]


# In[4]:


csv_downloads = {
    metadata_download_path: [barcode],
    metadata_download_path / "platemaps": platemaps,
    metadata_download_path / "loaddatas": loaddatas,
}

downloaded_files = []

for destination_dir, source_paths in csv_downloads.items():
    downloaded_files.extend(
        download_csv_files(
            source_paths=source_paths,
            destination_dir=destination_dir,
            repo_url=source_repo,
            commit=pin_commit,
        )
    )

print(f"Downloaded {len(downloaded_files)} CSV files.")

