"""Local cache, tiny pixel round trips, and the optional mounted-data smoke test."""

import pytest


def test_download_cache_corruption_and_pinned_sources(synthetic, monkeypatch):
    from io import BytesIO
    from unittest.mock import Mock

    from utils.pedcp_data_integration.download import _source_urls, download_metadata
    from utils.pedcp_data_integration.recipe import SOURCE_FILES

    a = synthetic.artifacts
    cached = download_metadata(synthetic.config)  # no_network makes any cache miss fail
    assert set(cached["files"]) == set(SOURCE_FILES)
    payloads = {url: (a.raw / name).read_bytes() for name, url in _source_urls().items()}

    def response(url, **kwargs):
        payload = payloads[url]
        stream = BytesIO(payload)
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    network = Mock(side_effect=response)
    monkeypatch.setattr("utils.pedcp_data_integration.download.urlopen", network)
    next(a.raw.glob("whole_img_qc_output/*.csv")).write_bytes(b"corrupt")
    assert download_metadata(synthetic.config) == cached
    assert network.call_count == 15
    assert download_metadata(synthetic.config) == cached and network.call_count == 15
    # A valid hash list alone is insufficient: source provenance must match too.
    import json

    cached["source_repo"] = "wrong"
    (a.raw / "manifest.json").write_text(json.dumps(cached))
    download_metadata(synthetic.config)
    assert network.call_count == 30


def test_download_failure_never_certifies_partial_cache(synthetic, monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from utils.pedcp_data_integration.download import download_metadata

    a = synthetic.artifacts
    for truncated in (False, True):
        (a.raw / "manifest.json").unlink(missing_ok=True)

        def response(url, truncated=truncated, **kwargs):
            if not truncated and "img_quality_control" in url:
                raise HTTPError(url, 404, "required QC absent", {}, None)
            stream = BytesIO(b"column\nvalue\n")
            stream.headers = {"Content-Length": "100" if truncated else "13"}
            return stream

        monkeypatch.setattr("utils.pedcp_data_integration.download.urlopen", response)
        with pytest.raises(OSError):
            download_metadata(synthetic.config)
        assert not (a.raw / "manifest.json").exists()
        assert not [path for path in a.raw.rglob("*") if path.name.startswith(".")]


def test_engine_raw_pixels_resume_and_missing_outlines(prepared, tmp_path):
    import numpy as np
    import pandas as pd

    from utils.fov_ingestion.collection import MicroscopyCollection
    from utils.fov_ingestion.ingest import ingest, plan
    from utils.pedcp_data_integration.plots import save_montages, select_preview_fovs
    from utils.pedcp_data_integration.recipe import CHANNELS, DATASET_ID, PIXEL_SIZE_UM

    frame = pd.read_parquet(prepared.artifacts.export_parquet).iloc[:3]
    options = {
        "dataset_id": DATASET_ID,
        "channels": list(CHANNELS),
        "pixel_size_um": PIXEL_SIZE_UM,
    }
    with pytest.raises(ValueError, match="Missing outlines"):
        plan(frame, prepared.config.outline_root, **options)
    options["skip_missing_outlines"] = True
    output = tmp_path / "tiny"
    with pytest.warns(UserWarning, match="Skipped 1"):
        preview = ingest(frame, prepared.config.outline_root, output, dry_run=True, **options)
    assert len(preview) == 2 and not output.exists()
    with pytest.warns(UserWarning, match="Skipped 1"):
        first = ingest(frame, prepared.config.outline_root, output, **options)
    assert len(first) == 2 and not first.resumed.any()
    collection = MicroscopyCollection(output)
    assert collection.metadata().seeding_density.eq(1000).all()
    for fid in collection.fovs.fov_id:
        planes = collection.read_fov(fid, normalize=False)
        assert len(planes) == 8
        for channel in CHANNELS:
            np.testing.assert_array_equal(planes["intensity", channel], prepared.pixels[channel])
            assert planes["intensity", channel].dtype == np.uint16
        for kind, low, high in (("cells", 6, 24), ("nuclei", 11, 19)):
            expected = np.zeros((32, 32), dtype=np.uint32)
            expected[low:high, low:high] = 1
            np.testing.assert_array_equal(planes["labels", kind], expected)
            assert planes["labels", kind].dtype == np.uint32
    selected = select_preview_fovs(collection.metadata(), 1, 42)
    paths = save_montages(collection, selected, tmp_path / "montages")
    assert len(paths) == 1 and paths[0].stat().st_size > 0
    with pytest.warns(UserWarning, match="Skipped 1"):
        assert ingest(frame, prepared.config.outline_root, output, **options).resumed.all()
    with pytest.raises(ValueError, match="different configuration"):
        ingest(frame.iloc[:1], prepared.config.outline_root, output, **options)


def test_deterministic_sampling_and_headless_plots(prepared, tmp_path):
    import pandas as pd

    from utils.pedcp_data_integration.plots import plot_retention, select_preview_fovs

    frame = pd.read_parquet(prepared.artifacts.export_parquet)
    frame["fov_id"] = [f"fov_{i}" for i in range(len(frame))]
    selected = select_preview_fovs(frame, 3, 42)
    pd.testing.assert_frame_equal(selected, select_preview_fovs(frame.iloc[::-1], 3, 42))
    assert len(selected) == selected.fov_id.nunique() == 3
    assert set(selected.cell_line) == {"DAOY", "U2-OS"}
    assert select_preview_fovs(frame, 0, 42).empty
    assert len(select_preview_fovs(frame, 100, 42)) == len(frame)
    paths = plot_retention(
        pd.read_csv(prepared.artifacts.reports / "stage_counts.csv"),
        pd.read_csv(prepared.artifacts.reports / "qc_plate_counts.csv"),
        tmp_path / "plots",
    )
    assert len(paths) == 2 and all(path.stat().st_size > 0 for path in paths)


def test_pipeline_stage_interfaces_and_report_requires_ingestion(synthetic):
    import importlib
    import importlib.util

    import pandas as pd

    name = "utils.pedcp_data_integration.pipeline"
    if importlib.util.find_spec(name) is None:
        pytest.skip("Pipeline implementation pending; rerun this contract when pipeline.py exists")
    pipeline = importlib.import_module(name)
    assert isinstance(pipeline.prepare(synthetic.config), dict)
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.report(synthetic.config)
    with pytest.warns(UserWarning, match="Skipped"):
        preview = pipeline.ingest_prepared(synthetic.config, dry_run=True)
    assert isinstance(preview, pd.DataFrame) and len(preview) == 2
    assert not synthetic.artifacts.collection.exists()
    with pytest.warns(UserWarning, match="Skipped"):
        assert isinstance(pipeline.run(synthetic.config), dict)
    assert isinstance(pipeline.report(synthetic.config), dict)
    with pytest.warns(UserWarning, match="Skipped"):
        resumed = pipeline.ingest_prepared(synthetic.config)
    assert isinstance(resumed, pd.DataFrame) and resumed.resumed.all()
    manifests = list(synthetic.artifacts.reports.glob("ingest*_manifest.json"))
    assert manifests
    for path in manifests:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.report(synthetic.config)  # a completed collection alone is not certification


def test_real_data_smoke_only_in_temporary_output(real_config):
    import numpy as np
    import pandas as pd
    import tifffile

    from utils.fov_ingestion.collection import MicroscopyCollection
    from utils.fov_ingestion.ingest import ingest, plan
    from utils.pedcp_data_integration.artifacts import Artifacts
    from utils.pedcp_data_integration.download import download_metadata
    from utils.pedcp_data_integration.metadata import prepare_metadata
    from utils.pedcp_data_integration.plots import save_montages, select_preview_fovs
    from utils.pedcp_data_integration.recipe import CHANNELS, DATASET_ID, FIELD_KEYS, PIXEL_SIZE_UM

    a = Artifacts(real_config)
    download_metadata(real_config)  # verified temporary copy; network remains forbidden
    prepare_metadata(real_config)  # all six plates, never QC a smoke subset
    frame = pd.read_parquet(a.export_parquet)
    options = {
        "dataset_id": DATASET_ID,
        "channels": list(CHANNELS),
        "pixel_size_um": PIXEL_SIZE_UM,
        "skip_missing_outlines": True,
    }
    items = plan(frame, real_config.outline_root, **options)[:3]
    if not items:
        pytest.skip("No eligible FOVs with both outlines; mount the configured outline_root")
    keys = {tuple(item["metadata"][key] for key in FIELD_KEYS) for item in items}
    tiny = frame.loc[
        [tuple(row) in keys for row in frame[FIELD_KEYS].itertuples(index=False, name=None)]
    ]
    ingest(tiny, real_config.outline_root, a.collection, workers=1, **options)
    collection = MicroscopyCollection(a.collection)
    assert len(collection.fovs) == len(items) <= 3
    metadata = collection.metadata().set_index("fov_id")
    for item in items:
        for column in ("cell_line", "seeding_density", "time_point"):
            assert metadata.loc[item["fov_id"], column] == item["metadata"][column]
        planes = collection.read_fov(item["fov_id"], normalize=False)
        for channel in CHANNELS:
            source = tifffile.imread(item["paths"][channel])
            np.testing.assert_array_equal(planes["intensity", channel], source)
            assert planes["intensity", channel].dtype == source.dtype
        for kind in ("cells", "nuclei"):
            assert planes["labels", kind].shape == source.shape
            assert planes["labels", kind].dtype == np.uint32
    assert ingest(tiny, real_config.outline_root, a.collection, workers=1, **options).resumed.all()
    selected = select_preview_fovs(collection.metadata(), 1, real_config.preview_seed)
    assert all(path.stat().st_size > 0 for path in save_montages(collection, selected, a.figures))
