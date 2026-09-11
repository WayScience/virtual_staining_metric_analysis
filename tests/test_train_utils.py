from pathlib import Path

import pandas as pd

from utils.train_utils import build_dataset_inputs


def test_build_dataset_inputs_without_profile_skips_point_mapping() -> None:
    loaddata = pd.DataFrame(
        {
            "PathName_input": ["/images/a", "/images/b"],
            "FileName_input": ["input.tiff", "input.tiff"],
            "PathName_target": ["/images/a", "/images/b"],
            "FileName_target": ["target.tiff", "target.tiff"],
            "Metadata_Plate": ["plate-1", "plate-1"],
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Site": [1, 2],
            "Metadata_Cells_Location_Center_X": [10, 20],
            "Metadata_Cells_Location_Center_Y": [30, 40],
        }
    )

    file_index, point_mapping = build_dataset_inputs(loaddata, "input", "target")

    expected = pd.DataFrame(
        {
            "input": [Path("/images/a/input.tiff"), Path("/images/b/input.tiff")],
            "target": [Path("/images/a/target.tiff"), Path("/images/b/target.tiff")],
            "Metadata_Plate": ["plate-1", "plate-1"],
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Site": [1, 2],
            "Metadata_Cells_Location_Center_X": [10, 20],
            "Metadata_Cells_Location_Center_Y": [30, 40],
        }
    )
    pd.testing.assert_frame_equal(file_index, expected)
    assert point_mapping is None
