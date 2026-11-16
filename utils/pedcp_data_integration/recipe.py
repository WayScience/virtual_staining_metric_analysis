"""Pinned PedCP90 data ingestion recipe"""

from pathlib import PurePosixPath

# Pinned profiling repo commit for reproducibility
RECIPE_VERSION = "1"
SOURCE_COMMIT = "6129e23e64cad5402c6d1d5ba679ff5f3eda82e4"
SOURCE_REPO = "https://github.com/WayScience/pediatric_cancer_atlas_profiling.git"
SOURCE_PATH_PREFIX = PurePosixPath("/media/18tbdrive/ALSF_pilot_data")

# Data/subsetting settings
FIELD_KEYS = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]  # unique FOV identifiers
DATASET_ID = "pedcp90-reference"  # naming the reference write
PIXEL_SIZE_UM = 0.593  # helps with scale bar when visualizing images
PLATES = tuple(f"BR{number:08d}" for number in range(143976, 143982))  # only pedcp batch 1 plates
CHANNELS = ("OrigER", "OrigRNA", "OrigAGP", "OrigMito", "OrigDNA", "OrigBrightfield")
TIME_POINT = 24  # for degradation, only use the 24-hour time point

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
