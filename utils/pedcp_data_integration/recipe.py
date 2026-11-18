"""Pinned PedCP90 data ingestion recipe"""

from pathlib import PurePosixPath

# Pinned profiling repo commit for reproducibility
RECIPE_VERSION = "1"
SOURCE_COMMIT = "6129e23e64cad5402c6d1d5ba679ff5f3eda82e4"
SOURCE_REPO = "https://github.com/WayScience/pediatric_cancer_atlas_profiling.git"
SOURCE_PATH_PREFIX = PurePosixPath("/media/18tbdrive/ALSF_pilot_data")

# Data/subsetting settings, shared
FIELD_KEYS = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]  # unique FOV identifiers
PIXEL_SIZE_UM = 0.593  # helps with scale bar when visualizing images
PLATES = tuple(f"BR{number:08d}" for number in range(143976, 143982))  # only pedcp batch 1 plates
CHANNELS = ("OrigER", "OrigRNA", "OrigAGP", "OrigMito", "OrigDNA", "OrigBrightfield")
# Source/QC columns stay Orig*; corrected collection planes are named Illum*.
CORRECTED_CHANNELS = {channel: f"Illum{channel.removeprefix('Orig')}" for channel in CHANNELS}

# Degradation specific filtering and settings
DATASET_ID = "pedcp90-reference"  # naming the reference write
TIME_POINT = 24  # for degradation, only use the 24-hour time point

# Training-specific settings
TRAIN_PLATEMAP = "Assay_Plate1_platemap"
TRAIN_CELL_LINE = "U2-OS"
TRAIN_TIME_POINTS = (24, 48, 72)
TRAIN_DENSITIES = (1000, 2000, 4000, 8000, 12000)
COHORTS = ("all", "reference", "training")

# QC Settings
BLUR_METRIC = "ImageQuality_PowerLogLogSlope"
SATURATION_METRIC = "ImageQuality_PercentMaximal"
BLUR_THRESHOLD = 2.5
SATURATION_THRESHOLD = 2.0


SOURCE_FILES = {
    "Barcode_platemap_pilot_data.csv": "0.download_data/metadata/platemaps/Barcode_platemap_pilot_data.csv",
    **{
        f"platemaps/Assay_Plate{number}_platemap.csv": (
            f"0.download_data/metadata/platemaps/Assay_Plate{number}_platemap.csv"
        )
        for number in (1, 2)
    },
    **{
        f"loaddatas/{plate}_concatenated.csv": (
            f"1.illumination_correction/loaddata_csvs/Round_1_data/{plate}_concatenated.csv"
        )
        for plate in PLATES
    },
    **{
        f"whole_img_qc_output/{plate}.csv": (
            f"1.illumination_correction/img_quality_control/whole_img_qc_output/{plate}/Image.csv"
        )
        for plate in PLATES
    },
}

# Plate-specific CellProfiler illumination functions (Round 1).
ILLUMINATION_METHODS = {
    channel: "divide" if channel == "OrigBrightfield" else "subtract" for channel in CHANNELS
}
for plate in PLATES:
    for channel in CHANNELS:
        filename = f"{plate}_Illum{channel.removeprefix('Orig')}.npy"
        SOURCE_FILES[f"illumination/{plate}/{filename}"] = (
            f"1.illumination_correction/illum_directory/Round_1_data/{plate}/{filename}"
        )
