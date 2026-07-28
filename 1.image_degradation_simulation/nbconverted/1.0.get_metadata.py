#!/usr/bin/env python
# coding: utf-8

# # 1.0.Download loaddata csv from pinned pediatric atlas pilot repo commit
# This notebook downloads the CellProfiler loaddata from pinned pediatric atlas pilot repo commit.
# 
# The downloaded loaddata will serve later in this analysis directory as the image file index. 

# In[1]:


from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


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


# Helper

# In[4]:


def download_csv_files(
    source_paths: list[str],
    destination_dir: Path,
    repo_url: str,
    commit: str,
) -> list[Path]:
    """Download CSV files from a pinned GitHub commit into one flat directory.

    Each downloaded file keeps only its source name (stem and ``.csv``
    extension), so parent directories from the repository are not recreated.
    Existing destination files are replaced.
    """
    github_prefix = "https://github.com/"
    repository_url = repo_url.removesuffix(".git")
    if not repository_url.startswith(github_prefix):
        raise ValueError(f"Expected a GitHub repository URL, got: {repo_url}")

    repository_slug = repository_url.removeprefix(github_prefix).strip("/")
    if repository_slug.count("/") != 1:
        raise ValueError(f"Could not determine owner and repository from: {repo_url}")

    relative_paths = [source_path.lstrip("/") for source_path in source_paths]
    destination_names = [Path(source_path).name for source_path in relative_paths]

    non_csv_paths = [
        source_path
        for source_path in relative_paths
        if Path(source_path).suffix.lower() != ".csv"
    ]
    if non_csv_paths:
        raise ValueError(f"All source files must be CSV files: {non_csv_paths}")

    if len(destination_names) != len(set(destination_names)):
        raise ValueError("Source paths contain duplicate filenames after flattening")

    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for source_path, destination_name in zip(relative_paths, destination_names):
        encoded_source_path = quote(source_path, safe="/")
        raw_url = (
            f"https://raw.githubusercontent.com/{repository_slug}/"
            f"{commit}/{encoded_source_path}"
        )
        destination_path = destination_dir / destination_name

        # Read the complete response before replacing any existing local file.
        with urlopen(raw_url, timeout=60) as response:
            file_contents = response.read()
        destination_path.write_bytes(file_contents)

        downloaded_paths.append(destination_path)
        print(f"Downloaded {source_path} -> {destination_path}")

    return downloaded_paths


# In[5]:


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

