import os
from dataclasses import replace

import pytest


def test_cli_launcher_and_notebook_cutover(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    from unittest.mock import Mock

    from utils.pedcp_data_integration import cli, pipeline

    config = object()
    loader = Mock(return_value=config)
    stages = {
        name: Mock()
        for name in ("download_metadata", "prepare", "ingest_prepared", "report", "run")
    }
    monkeypatch.setattr(cli, "load_project_config", loader)
    for name, stage in stages.items():
        monkeypatch.setattr(pipeline, name, stage)
    for args in (["--config", "local.yaml", "prepare"], ["prepare", "--config", "local.yaml"]):
        assert cli.main(args) == 0
        loader.assert_called_with("local.yaml")
        stages["prepare"].assert_called_with(config, cohort="all")
    for command, name in (
        ("download", "download_metadata"),
        ("prepare", "prepare"),
        ("ingest", "ingest_prepared"),
        ("report", "report"),
        ("run", "run"),
    ):
        assert cli.main([command]) == 0
        options = {} if command == "download" else {"cohort": "all"}
        if command == "ingest":
            options["dry_run"] = False
        stages[name].assert_called_with(config, **options)
    assert cli.main([]) == 0
    stages["run"].assert_called_with(config, cohort="all")
    for command in ("prepare", "report", "run"):
        assert cli.main([command, "--cohort", "reference"]) == 0
        stages[command].assert_called_with(config, cohort="reference")
    assert cli.main(["ingest", "--cohort", "training", "--dry-run"]) == 0
    stages["ingest_prepared"].assert_called_with(config, dry_run=True, cohort="training")
    assert cli.main(["run", "--cohort", "training"]) == 0
    stages["run"].assert_called_with(config, cohort="training")
    assert cli.main(["download", "--cohort", "training"]) == 0
    stages["download_metadata"].assert_called_with(config)
    with pytest.raises(SystemExit) as error:
        cli.main(["prepare", "--dry-run"])
    assert error.value.code == 2
    root = Path(__file__).resolve().parents[2]
    assert not list((root / "0.data_ingestion").rglob("*.ipynb"))
    completed = subprocess.run(
        ["bash", str(root / "0.data_ingestion/run.sh"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "prepare" in completed.stdout and "ingest" in completed.stdout


def test_run_propagates_cohort(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from utils.pedcp_data_integration import pipeline

    stages = Mock()
    paths = [tmp_path / "montage.png"]
    stages.report.return_value = paths
    for name in ("download_metadata", "prepare", "ingest_prepared", "report"):
        monkeypatch.setattr(pipeline, name, getattr(stages, name))
    config = object()
    assert pipeline.run(config, cohort="training") == paths
    stages.download_metadata.assert_called_once_with(config)
    for name in ("prepare", "ingest_prepared", "report"):
        getattr(stages, name).assert_called_once_with(config, cohort="training")


def test_config_relative_paths_and_unknown_key(tmp_path, monkeypatch):
    from pathlib import Path

    from utils.pedcp_data_integration.artifacts import Artifacts
    from utils.project_config import load_project_config

    path = tmp_path / "local.yaml"
    text = """paths: {metadata_root: meta, large_output_root: large}
pedcp: {image_root: images, outline_root: null, workers: 1, preview_count: 0, preview_seed: 42}"""
    path.write_text(text)
    monkeypatch.chdir(tmp_path.parent)
    config = load_project_config(path.relative_to(tmp_path.parent))
    assert config.metadata_root == tmp_path / "meta" and config.image_root == tmp_path / "images"
    artifacts = Artifacts(config)
    root = Path(__file__).resolve().parents[2]
    assert artifacts.figures == root / "0.data_ingestion/results"
    assert artifacts.collection.is_relative_to(config.large_output_root)
    path.write_text(text.replace("workers: 1", "workers: 1, scientific_override: 2"))
    with pytest.raises(ValueError, match="exactly"):
        load_project_config(path)


def test_qc_original_pool_boundaries_and_bad_inputs(synthetic):
    import numpy as np
    from scipy.stats import zscore

    from utils.pedcp_data_integration import recipe as r
    from utils.pedcp_data_integration.file_index import file_index_qc_mask

    qc = synthetic.qc
    metrics = qc.filter(like="ImageQuality")
    limits = [2.5 if r.BLUR_METRIC in col else 2.0 for col in metrics]
    expected = ~(np.abs(zscore(metrics, ddof=0, axis=0)) > limits).any(axis=1)
    assert file_index_qc_mask(qc).dtype == bool and np.array_equal(file_index_qc_mask(qc), expected)
    assert not file_index_qc_mask(qc.drop(index=58)).loc[59]
    boundary = qc.iloc[:16].copy()
    boundary[f"{r.BLUR_METRIC}_{r.CHANNELS[0]}"] = [2.5, -2.5] + [0.5, -0.5] * 7
    boundary[f"{r.SATURATION_METRIC}_{r.CHANNELS[1]}"] = [2, -2, 2, -2] + [0] * 12
    assert file_index_qc_mask(boundary).all()
    for value in (None, float("inf"), float("nan")):
        bad = synthetic.qc.copy()
        bad.loc[0, "Metadata_Site" if value is None else bad.columns[-1]] = value
        with pytest.raises(ValueError):
            file_index_qc_mask(bad)
    with pytest.raises(ValueError, match="unique"):
        file_index_qc_mask(synthetic.qc.iloc[[0, 0]])


def test_remap_missing_and_escape(synthetic):
    from utils.pedcp_data_integration.file_index import validate_file_index_paths

    frame = synthetic.loads.iloc[:4].copy()
    frame.loc[1:3, "FileName_OrigER"] = ["missing.tif", "../outside.tif", "/outside.tif"]
    result = validate_file_index_paths(frame, synthetic.config.image_root)
    assert result.Metadata_Site.tolist() == [1]
    assert result.PathName_OrigER.iloc[0] == str(synthetic.config.image_root)
    root = synthetic.config.image_root
    outside = root.parent / "outside.tif"
    outside.touch()
    (root / "escape.tif").symlink_to(outside)
    (root / "valid.tif").symlink_to(root / "OrigER.tif")
    frame = synthetic.loads.iloc[:2].copy()
    frame["FileName_OrigER"] = ["escape.tif", "valid.tif"]
    assert validate_file_index_paths(frame, root).Metadata_Site.tolist() == [2]


def test_prepare_final_only_repeat_and_failure(synthetic):
    import pandas as pd
    import pyarrow.parquet as pq

    from utils.pedcp_data_integration.file_index import prepare_file_index

    a = synthetic.artifacts
    assert a.figures == synthetic.config.metadata_root.parent / "0.data_ingestion/results"
    paths = prepare_file_index(synthetic.config, cohort="reference")
    assert paths == {"reference": a.export_parquet}
    path = paths["reference"]
    frame = pd.read_parquet(path)
    assert path == a.export_parquet and len(frame) == 20 and frame.time_point.eq(24).all()
    assert frame.groupby("cell_line").seeding_density.first().tolist() == [1000, 2000]
    assert frame.Metadata_Site.dtype.kind == "i" and frame.Reimaged.all()
    assert {p for p in a.root.rglob("*") if p.is_file() and not p.is_relative_to(a.raw)} == {path}
    assert len(list(a.figures.glob("*.png"))) == 2
    assert pq.read_metadata(path).row_group(0).column(0).compression == "ZSTD"
    assert prepare_file_index(synthetic.config, cohort="reference") == paths
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    # Construct the legacy input independently: schema, values and order enter fingerprints.
    from utils.fov_ingestion.ingest import plan
    from utils.pedcp_data_integration import recipe as r

    legacy = synthetic.loads.iloc[:20].copy()
    for channel in r.CHANNELS:
        legacy[f"PathName_{channel}"] = str(synthetic.config.image_root)
    legacy = legacy.assign(
        cell_line=["DAOY"] * 10 + ["U2-OS"] * 10,
        seeding_density=[1000] * 10 + [2000] * 10,
        time_point=24,
    )
    pd.testing.assert_frame_equal(frame, legacy)
    assert "platemap_file" not in frame
    options = {
        "dataset_id": r.DATASET_ID,
        "channels": list(r.CHANNELS),
        "pixel_size_um": r.PIXEL_SIZE_UM,
        "image_root": synthetic.config.image_root,
        "skip_missing_outlines": True,
        "illumination": {
            (plate, channel): (
                a.raw / "illumination" / plate / f"{plate}_Illum{channel.removeprefix('Orig')}.npy",
                method,
            )
            for plate in r.PLATES[:2]
            for channel, method in r.ILLUMINATION_METHODS.items()
        },
    }
    with pytest.warns(UserWarning, match="Skipped 19"):
        expected = plan(legacy, synthetic.config.outline_root, **options)
    with pytest.warns(UserWarning, match="Skipped 19"):
        actual = plan(path, synthetic.config.outline_root, **options)
    assert actual[0]["fingerprint"] == expected[0]["fingerprint"]
    assert actual[0]["fov_id"] == expected[0]["fov_id"]
    source = next(a.raw.glob("whole_img_qc_output/*.csv"))
    pd.read_csv(source).iloc[1:].to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing QC coverage"):
        prepare_file_index(synthetic.config, cohort="reference")
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)


def test_download_cache_and_atomic_failure(synthetic, monkeypatch):
    from io import BytesIO
    from unittest.mock import Mock

    from utils.pedcp_data_integration.download import download_metadata
    from utils.pedcp_data_integration.recipe import SOURCE_COMMIT, SOURCE_FILES

    paths = download_metadata(synthetic.config)
    assert len(paths) == len(SOURCE_FILES) == 51 and synthetic.artifacts.raw.name == SOURCE_COMMIT
    target = paths[-1]
    target.write_bytes(b"")
    response = BytesIO(b"column\nvalue\n")
    response.headers = {"Content-Length": "100"}
    network = Mock(return_value=response)
    monkeypatch.setattr("utils.pedcp_data_integration.download.urlopen", network)
    with pytest.raises(OSError, match="Incomplete"):
        download_metadata(synthetic.config)
    assert target.read_bytes() == b"" and not list(target.parent.glob(".*"))
    response = BytesIO(b"column\nvalue\n")
    response.headers = {}
    network.return_value = response
    assert download_metadata(synthetic.config) == download_metadata(synthetic.config) == paths
    assert target.read_bytes() == b"column\nvalue\n" and network.call_count == 2


def test_ingest_dryrun_resume_pixels_and_mount_free_report(synthetic, monkeypatch):
    from unittest.mock import Mock

    import numpy as np

    from utils.fov_ingestion.collection import MicroscopyCollection
    from utils.pedcp_data_integration import pipeline

    config, a = synthetic.config, synthetic.artifacts
    pipeline.prepare(config, cohort="reference")
    before = set(config.large_output_root.rglob("*"))
    with pytest.warns(UserWarning, match="Skipped 19"):
        planned = pipeline.ingest_prepared(config, dry_run=True, cohort="reference")
    assert len(planned["reference"]) == 1
    assert not a.collection.exists() and before == set(config.large_output_root.rglob("*"))
    with pytest.warns(UserWarning, match="Skipped 19"):
        assert not pipeline.ingest_prepared(config, cohort="reference")["reference"].resumed.any()
    with pytest.warns(UserWarning, match="Skipped 19"):
        assert pipeline.ingest_prepared(config, cohort="reference")["reference"].resumed.all()
    collection = MicroscopyCollection(a.collection)
    planes = collection.read_fov(collection.fovs.fov_id.iloc[0], normalize=False)
    assert len(planes) == 8
    for (role, channel), plane in planes.items():
        expected = (
            synthetic.pixels.astype(np.float32)
            if role == "intensity"
            else np.zeros((32, 32), np.uint32)
        )
        if role == "labels":
            low, high = (6, 24) if channel == "cells" else (11, 19)
            expected[low:high, low:high] = 1
        np.testing.assert_array_equal(plane, expected)
        assert plane.dtype == expected.dtype
    offline = replace(config, image_root=None, outline_root=None)
    paths = pipeline.report(offline, cohort="reference")
    assert len(paths) == 1 and paths[0].stat().st_size > 0
    assert pipeline.report(offline, cohort="reference") == paths
    monkeypatch.setattr(MicroscopyCollection, "read_fov", Mock(side_effect=AssertionError))
    assert pipeline.report(replace(offline, preview_count=0), cohort="reference") == []


def test_missing_training_cohort_writes_no_requested_indexes(synthetic):
    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration.artifacts import selected_artifacts

    with pytest.raises(ValueError, match="density_1000.*no annotated"):
        pipeline.prepare(synthetic.config)
    assert not any(a.export_parquet.exists() for a in selected_artifacts(synthetic.config).values())
    assert not synthetic.artifacts.figures.exists()


def test_missing_training_timepoint_preserves_all_indexes(training_synthetic):
    import pandas as pd

    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration import recipe as r

    config, a = training_synthetic.config, training_synthetic.artifacts
    paths = pipeline.prepare(config)
    before = {path: path.read_bytes() for path in paths.values()}
    source = a.raw / "loaddatas" / f"{r.PLATES[4]}_concatenated.csv"
    frame = pd.read_csv(source)
    frame.loc[~frame.Metadata_Well.eq("B01")].to_csv(source, index=False)
    with pytest.raises(ValueError, match="density_1000.*72"):
        pipeline.prepare(config)
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_all_cohort_dryrun_writes_nothing(training_synthetic):
    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration.artifacts import selected_artifacts

    config = training_synthetic.config
    pipeline.prepare(config)
    root = config.metadata_root.parent
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*") if p.is_file()}
    with pytest.warns(UserWarning, match="Skipped 19"):
        planned = pipeline.ingest_prepared(config, dry_run=True)
    assert list(planned) == list(selected_artifacts(config))
    assert [len(frame) for frame in planned.values()] == [6, 3, 3, 3, 3, 3]
    assert {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*") if p.is_file()
    } == before
    assert not any(a.collection.exists() for a in selected_artifacts(config).values())


def test_training_selection_independent_resume_and_offline_report(training_synthetic):
    import numpy as np
    import pandas as pd

    from utils.fov_ingestion.collection import MicroscopyCollection
    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration import recipe as r
    from utils.pedcp_data_integration.artifacts import Artifacts, selected_artifacts

    config, reference = training_synthetic.config, training_synthetic.artifacts
    assert r.TRAIN_DENSITIES == (1000, 2000, 4000, 8000, 12000)
    assert r.TRAIN_TIME_POINTS == (24, 48, 72)
    outputs = selected_artifacts(config)
    names = ["reference", *(f"density_{density}" for density in r.TRAIN_DENSITIES)]
    assert list(outputs) == names
    assert list(selected_artifacts(config, "reference")) == names[:1]
    assert list(selected_artifacts(config, "training")) == names[1:]
    with pytest.raises(ValueError, match="cohort"):
        selected_artifacts(config, "unknown")
    assert reference.dataset_id == r.DATASET_ID == "pedcp90-reference"
    assert reference.export_parquet == reference.root / "loaddata_for_degradation.parquet"
    assert reference.collection == config.large_output_root / "pedcp90/reference"

    pipeline.prepare(config, cohort="reference")
    baseline = pd.read_parquet(reference.export_parquet)
    with pytest.warns(UserWarning, match="Skipped 19"):
        initial = pipeline.ingest_prepared(config, cohort="reference")["reference"]
    assert not initial.resumed.any()
    reference_spec = MicroscopyCollection(reference.collection).spec
    reference_files = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (reference.collection / "fovs").rglob("*")
        if path.is_file()
    }
    paths = pipeline.prepare(config)
    assert paths == {name: a.export_parquet for name, a in outputs.items()}
    pd.testing.assert_frame_equal(pd.read_parquet(reference.export_parquet), baseline)
    assert "platemap_file" not in baseline
    source_keys = set()
    for density in r.TRAIN_DENSITIES:
        a = Artifacts(config, density)
        assert a.name == f"density_{density}"
        assert a.dataset_id == f"pedcp90-training-density-{density}"
        assert a.export_parquet == reference.root / "training" / a.name / "loaddata.parquet"
        assert a.collection == config.large_output_root / "pedcp90/training" / a.name
        assert a.figures == reference.figures / "training" / a.name
        frame = pd.read_parquet(paths[a.name])
        assert len(frame) == 3 and set(frame.time_point) == {24, 48, 72}
        assert frame.seeding_density.eq(density).all() and frame.cell_line.eq("U2-OS").all()
        assert frame.platemap_file.eq("Assay_Plate1_platemap").all()
        assert set(frame.Metadata_Plate) == set(r.PLATES[::2])
        assert frame.Metadata_Well.str.startswith("B").all() and frame.Reimaged.all()
        keys = set(frame[r.FIELD_KEYS].itertuples(index=False, name=None))
        assert not source_keys.intersection(keys)
        source_keys.update(keys)
    assert (
        len(source_keys.intersection(baseline[r.FIELD_KEYS].itertuples(index=False, name=None)))
        == 5
    )

    written = pipeline.ingest_prepared(config, cohort="training")
    assert list(written) == names[1:] and all(not frame.resumed.any() for frame in written.values())
    with pytest.warns(UserWarning, match="Skipped 19"):
        resumed = pipeline.ingest_prepared(config)
    assert all(frame.resumed.all() for frame in resumed.values())
    pd.testing.assert_frame_equal(pd.read_parquet(reference.export_parquet), baseline)
    assert MicroscopyCollection(reference.collection).spec == reference_spec
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (reference.collection / "fovs").rglob("*")
        if path.is_file()
    } == reference_files

    reference_ids = set(MicroscopyCollection(reference.collection).fovs.fov_id)
    for a in selected_artifacts(config, "training").values():
        collection = MicroscopyCollection(a.collection)
        metadata = collection.metadata()
        assert len(metadata) == 3 and metadata.dataset_id.eq(a.dataset_id).all()
        assert metadata.platemap_file.eq("Assay_Plate1_platemap").all()
        assert not reference_ids.intersection(metadata.fov_id)
        assert "reference_collection_uri" not in collection.spec
    # Well-filtered object tables provide a nucleus-centered, spatially aligned eight-plane crop.
    collection = MicroscopyCollection(Artifacts(config, 1000).collection)
    selected = collection.metadata().query("well == 'B01' and time_point == 24")
    assert len(selected) == 1
    fov_id = selected.fov_id.iloc[0]
    objects, relationships = collection.object_tables(fov_id)
    nucleus = objects.loc[objects.compartment.eq("nuclei")].iloc[0]
    y, x = round(nucleus.centroid_y), round(nucleus.centroid_x)
    region = (slice(y - 4, y + 4), slice(x - 4, x + 4))
    planes = collection.read_fov(fov_id, normalize=False)
    assert set(planes) == {
        *(("intensity", channel) for channel in r.CHANNELS),
        ("labels", "cells"),
        ("labels", "nuclei"),
    }
    crops = {key: plane[region] for key, plane in planes.items()}
    assert all(crop.shape == (8, 8) for crop in crops.values())
    for channel in r.CHANNELS:
        np.testing.assert_array_equal(
            crops["intensity", channel], training_synthetic.pixels[region]
        )
    assert crops["labels", "nuclei"][4, 4] == nucleus.label
    assert crops["labels", "cells"][4, 4] > 0
    assert nucleus.object_id in set(relationships.nucleus_id)

    # Training outputs remain usable with reference hidden and both source mounts unavailable.
    reference.collection.rename(reference.collection.with_name("reference-hidden"))
    reference_figures = {p: p.stat().st_mtime_ns for p in reference.figures.glob("*.png")}
    offline = replace(config, image_root=None, outline_root=None)
    previews = pipeline.report(offline, cohort="training")
    assert len(previews) == 5 and all(path.stat().st_size > 0 for path in previews)
    assert {p.parent for p in previews} == {
        a.figures for a in selected_artifacts(config, "training").values()
    }
    assert {p: p.stat().st_mtime_ns for p in reference.figures.glob("*.png")} == reference_figures


def test_training_outline_attrition_warns_and_empty_density_errors(training_synthetic):
    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration import recipe as r
    from utils.pedcp_data_integration.artifacts import Artifacts

    config = training_synthetic.config
    pipeline.prepare(config, cohort="training")
    for path in (config.outline_root / r.PLATES[4]).glob("*Outlines_*_B01_1.tiff"):
        path.unlink()
    with pytest.warns(UserWarning) as warnings:
        written = pipeline.ingest_prepared(config, cohort="training")
    assert len(written["density_1000"]) == 2
    assert any("density_1000" in str(w.message) and "72" in str(w.message) for w in warnings)
    for plate in r.PLATES[::2]:
        for path in (config.outline_root / plate).glob("*Outlines_*_B01_1.tiff"):
            path.unlink()
    with (
        pytest.warns(UserWarning, match="Skipped 3"),
        pytest.raises(ValueError, match="density_1000.*(empty|no eligible)"),
    ):
        pipeline.ingest_prepared(config, dry_run=True, cohort="training")
    assert len(written) == 5 and Artifacts(config, 1000).collection.is_dir()


@pytest.mark.local_ingestion
def test_real_data_smoke(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import tifffile

    from utils.fov_ingestion.collection import MicroscopyCollection
    from utils.fov_ingestion.ingest import ingest, plan
    from utils.pedcp_data_integration import pipeline
    from utils.pedcp_data_integration import recipe as r
    from utils.pedcp_data_integration.artifacts import Artifacts
    from utils.project_config import load_project_config

    value = os.environ.get("PEDCP_TEST_CONFIG")
    if not value:
        pytest.skip("Optional mounted smoke requires PEDCP_TEST_CONFIG")
    source = load_project_config(value)
    export = Artifacts(source).export_parquet
    if not export.is_file():
        pytest.skip("Run prepare once before the real-data smoke test")
    config = replace(source, metadata_root=tmp_path / "meta", large_output_root=tmp_path / "large")
    a = Artifacts(config)
    frame = pd.read_parquet(export)
    options = {
        "dataset_id": r.DATASET_ID,
        "channels": list(r.CHANNELS),
        "pixel_size_um": r.PIXEL_SIZE_UM,
        "skip_missing_outlines": True,
    }
    candidates = []
    for start in range(0, len(frame), 10):
        candidates = plan(frame.iloc[start : start + 10], config.outline_root, **options)
        if candidates:
            break
    assert candidates, "No prepared fields with both outlines"
    item = candidates[0]
    tiny = pd.DataFrame([item["metadata"]])
    # Keep an uncorrected baseline to confirm segmentation labels remain unchanged.
    baseline = tmp_path / "uncorrected"
    ingest(tiny, config.outline_root, baseline, workers=1, **options)
    original = MicroscopyCollection(baseline).read_fov(item["fov_id"], normalize=False)
    a.export_parquet.parent.mkdir(parents=True, exist_ok=True)
    tiny.to_parquet(a.export_parquet, index=False)
    a.raw.parent.mkdir(parents=True, exist_ok=True)
    a.raw.symlink_to(Artifacts(source).raw, target_is_directory=True)
    monkeypatch.setattr(
        "utils.pedcp_data_integration.artifacts.__file__",
        str(tmp_path / "utils/pedcp_data_integration/artifacts.py"),
    )
    assert not pipeline.ingest_prepared(replace(config, workers=1), cohort="reference")[
        "reference"
    ].resumed.any()
    collection = MicroscopyCollection(a.collection)
    planes = collection.read_fov(item["fov_id"], normalize=False)
    assert len(planes) == 8
    provenance = json.loads(
        (a.collection / "fovs" / item["fov_id"] / "provenance.json").read_text()
    )
    for channel in r.CHANNELS:
        field_path = (
            a.raw
            / "illumination"
            / item["plate"]
            / f"{item['plate']}_Illum{channel.removeprefix('Orig')}.npy"
        )
        field = np.load(field_path, allow_pickle=False).astype(np.float64)
        raw = tifffile.imread(item["paths"][channel])
        method = "divide" if channel == "OrigBrightfield" else "subtract"
        assert field.shape == raw.shape and np.isfinite(field).all()
        assert method != "divide" or (field > 0).all()
        expected = raw / field if method == "divide" else raw - 65535.0 * field
        expected = np.clip(expected, 0, 65535)
        actual = planes["intensity", channel]
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.01, err_msg=channel)
        assert np.any(actual != raw), f"Correction was a no-op for {channel}"
        registry = collection.images.query("role == 'intensity' and channel == @channel")
        assert registry.illumination_method.tolist() == [method]
        assert provenance["illumination"][channel]["method"] == method
        key = f"illumination:{channel}"
        assert Path(provenance["paths"][key]).samefile(field_path)
    for compartment in ("cells", "nuclei"):
        np.testing.assert_array_equal(
            planes["labels", compartment], original["labels", compartment]
        )
    assert pipeline.ingest_prepared(replace(config, workers=1), cohort="reference")[
        "reference"
    ].resumed.all()
