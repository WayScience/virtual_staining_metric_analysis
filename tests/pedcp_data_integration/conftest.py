import os
from types import SimpleNamespace

import pytest


def pytest_collection_modifyitems(items):
    disabled = os.environ.get("PEDCP_RUN_LOCAL_TESTS") != "1"
    disabled |= os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    for item in items:
        if disabled and item.get_closest_marker("local_ingestion") is not None:
            item.add_marker(pytest.mark.skip(reason="Set PEDCP_RUN_LOCAL_TESTS=1; never Actions"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network forbidden: provide a raw cache or mock the download")

    monkeypatch.setattr("utils.pedcp_data_integration.download.urlopen", forbidden)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import tifffile

    from utils.pedcp_data_integration import recipe as r
    from utils.pedcp_data_integration.artifacts import Artifacts
    from utils.project_config import ProjectConfig

    monkeypatch.setattr(
        "utils.pedcp_data_integration.artifacts.__file__",
        str(tmp_path / "utils/pedcp_data_integration/artifacts.py"),
    )
    config = ProjectConfig(
        *(tmp_path / name for name in ("metadata", "large", "images", "outlines")),
        workers=1,
        preview_count=1,
    )
    config.image_root.mkdir()
    pixels = np.arange(1024, dtype=np.uint16).reshape(32, 32)
    for channel in r.CHANNELS:
        tifffile.imwrite(config.image_root / f"{channel}.tif", pixels)
    folder = config.outline_root / r.PLATES[0]
    folder.mkdir(parents=True)
    for kind, low, high in (("Cells", 5, 25), ("Nuclei", 10, 20)):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[low:high, low, 1] = rgb[low:high, high - 1, 1] = 255
        rgb[low, low:high, 1] = rgb[high - 1, low:high, 1] = 255
        tifffile.imwrite(folder / f"{kind}Outlines_{r.PLATES[0]}_A01_1.tiff", rgb)
    loads = pd.MultiIndex.from_product([r.PLATES, ["A01"], range(1, 11)]).to_frame(index=False)
    loads.columns = r.FIELD_KEYS
    loads["Reimaged"] = True
    qc = loads.copy()
    for channel in r.CHANNELS:
        loads[f"PathName_{channel}"] = str(r.SOURCE_PATH_PREFIX)
        loads[f"FileName_{channel}"] = f"{channel}.tif"
        for metric in (r.BLUR_METRIC, r.SATURATION_METRIC):
            qc[f"{metric}_{channel}"] = 0.0
    qc.loc[58, f"{r.BLUR_METRIC}_{r.CHANNELS[0]}"] = 100
    qc.loc[58:59, f"{r.SATURATION_METRIC}_{r.CHANNELS[0]}"] = [1000, 10]
    maps = ["Assay_Plate1_platemap", "Assay_Plate2_platemap"] * 3
    barcode = pd.DataFrame(
        zip(r.PLATES, [24, 24, 48, 48, 72, 72], maps),
        columns=["barcode", "time_point", "platemap_file"],
    )
    frames = {"Barcode_platemap_pilot_data.csv": barcode}
    for i, name in enumerate(("DAOY", "U2-OS"), 1):
        frames[f"platemaps/Assay_Plate{i}_platemap.csv"] = pd.DataFrame(
            [("A01", name, 1000 * i)], columns=["well", "cell_line", "seeding_density"]
        )
    for plate in r.PLATES:
        frames[f"loaddatas/{plate}_concatenated.csv"] = loads[loads.Metadata_Plate.eq(plate)]
        frames[f"whole_img_qc_output/{plate}.csv"] = qc[qc.Metadata_Plate.eq(plate)]
    artifacts = Artifacts(config)
    for name, frame in frames.items():
        path = artifacts.raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    for name in r.SOURCE_FILES:
        if name.endswith(".npy"):
            path = artifacts.raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.full(pixels.shape, float("Brightfield" in name), dtype=np.float32))
    return SimpleNamespace(config=config, artifacts=artifacts, loads=loads, qc=qc, pixels=pixels)
