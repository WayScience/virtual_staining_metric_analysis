"""Local-only tests: collection imports neither the workflow nor scientific libraries."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_collection_modifyitems(items):
    directory = Path(__file__).parent.resolve()
    enabled = os.environ.get("PEDCP_RUN_LOCAL_TESTS") == "1"
    ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    for item in items:
        if Path(item.path).resolve().is_relative_to(directory):
            item.add_marker(pytest.mark.local_ingestion)
            if ci or not enabled:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Local only: set PEDCP_RUN_LOCAL_TESTS=1; never Actions"
                    )
                )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network forbidden: provide a verified raw cache or mock the download")

    monkeypatch.setattr("utils.pedcp_data_integration.download.urlopen", forbidden)


@pytest.fixture
def synthetic(tmp_path):
    import numpy as np
    import pandas as pd
    import tifffile

    from utils.pedcp_data_integration.artifacts import Artifacts, write_manifest
    from utils.pedcp_data_integration.download import _source_urls
    from utils.pedcp_data_integration.recipe import (
        BLUR_METRIC,
        CHANNELS,
        PLATES,
        SATURATION_METRIC,
        SOURCE_FILES,
        SOURCE_PATH_PREFIX,
        SOURCE_REPO,
    )
    from utils.project_config import ProjectConfig

    config = ProjectConfig(
        tmp_path / "metadata",
        tmp_path / "large",
        tmp_path / "images",
        tmp_path / "outlines",
        workers=1,
        preview_count=1,
    )
    artifacts = Artifacts(config)
    config.image_root.mkdir()
    pixels = {}
    for index, channel in enumerate(CHANNELS):
        pixels[channel] = np.arange(1024, dtype=np.uint16).reshape(32, 32) * 31 + index
        tifffile.imwrite(config.image_root / f"{channel}.tif", pixels[channel])
    # Only two eligible FOVs have outlines, keeping actual storage tests tiny.
    folder = config.outline_root / PLATES[0]
    folder.mkdir(parents=True)
    for site in (1, 2):
        for kind, low, high in (("Cells", 5, 25), ("Nuclei", 10, 20)):
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            rgb[low:high, low, 1] = rgb[low:high, high - 1, 1] = 255
            rgb[low, low:high, 1] = rgb[high - 1, low:high, 1] = 255
            tifffile.imwrite(
                folder / f"{kind}Outlines_{PLATES[0]}_A01_{site}.tiff", rgb, photometric="rgb"
            )

    barcode = pd.DataFrame(
        {
            "barcode": PLATES,
            "time_point": [24, 24, 48, 48, 72, 72],
            "platemap_file": ["Assay_Plate1_platemap", "Assay_Plate2_platemap"] * 3,
        }
    )
    maps = {
        f"Assay_Plate{i}_platemap": pd.DataFrame(
            {
                "well": ["A01"],
                "cell_line": [name],
                "row": ["A"],
                "column": [1],
                "seeding_density": [1000 * i],
                "condition": ["standard"],
            }
        )
        for i, name in ((1, "DAOY"), (2, "U2-OS"))
    }
    loads = pd.DataFrame(
        [
            {
                "Metadata_Plate": plate,
                "Metadata_Well": "A01",
                "Metadata_Site": site,
                "Reimaged": True,
                **{
                    f"{prefix}_{channel}": value
                    for channel in CHANNELS
                    for prefix, value in (
                        ("PathName", str(SOURCE_PATH_PREFIX)),
                        ("FileName", f"{channel}.tif"),
                    )
                },
            }
            for plate in PLATES
            for site in range(1, 9)
        ]
    )
    qc = loads[["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Reimaged"]].copy()
    for metric in (BLUR_METRIC, SATURATION_METRIC):
        for channel in CHANNELS:
            qc[f"{metric}_{channel}"] = 0.0
    qc.loc[46, f"{BLUR_METRIC}_{CHANNELS[0]}"] = 100.0
    qc.loc[46:47, f"{SATURATION_METRIC}_{CHANNELS[0]}"] = [1000.0, 10.0]
    frames = {"Barcode_platemap_pilot_data.csv": barcode}
    frames.update({f"platemaps/{name}.csv": frame for name, frame in maps.items()})
    for plate in PLATES:
        frames[f"loaddatas/{plate}_concatenated.csv"] = loads.loc[loads.Metadata_Plate.eq(plate)]
        frames[f"whole_img_qc_output/{plate}.csv"] = qc.loc[qc.Metadata_Plate.eq(plate)]
    for name, frame in frames.items():
        path = artifacts.raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def certify():
        return write_manifest(
            artifacts.raw / "manifest.json",
            artifacts.raw,
            SOURCE_FILES,
            sources=SOURCE_FILES,
            source_urls=_source_urls(),
            source_repo=SOURCE_REPO,
        )

    certify()
    return SimpleNamespace(
        config=config,
        artifacts=artifacts,
        loads=loads,
        qc=qc,
        barcode=barcode,
        maps=maps,
        pixels=pixels,
        certify=certify,
    )


@pytest.fixture
def prepared(synthetic):
    from utils.pedcp_data_integration.metadata import prepare_metadata

    prepare_metadata(synthetic.config)
    return synthetic


@pytest.fixture
def real_config(tmp_path):
    from dataclasses import replace
    from shutil import copytree

    from utils.pedcp_data_integration.artifacts import Artifacts
    from utils.pedcp_data_integration.download import verify_raw_manifest
    from utils.project_config import load_project_config

    value = os.environ.get("PEDCP_TEST_CONFIG")
    if not value:
        pytest.skip("Real smoke: set PEDCP_TEST_CONFIG to an absolute, prepared project YAML")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        pytest.skip("PEDCP_TEST_CONFIG must name an existing absolute YAML path")
    source = load_project_config(path)
    if any(root is None or not root.is_dir() for root in (source.image_root, source.outline_root)):
        pytest.skip("Real smoke requires mounted image_root and outline_root from that YAML")
    try:
        verify_raw_manifest(Artifacts(source))
    except (OSError, ValueError, TypeError):
        pytest.skip(
            "Run the download stage for PEDCP_TEST_CONFIG first; a verified cache is required"
        )
    config = replace(
        source,
        metadata_root=tmp_path / "metadata",
        large_output_root=tmp_path / "large",
        workers=1,
        preview_count=1,
    )
    copytree(Artifacts(source).raw, Artifacts(config).raw)
    return config
