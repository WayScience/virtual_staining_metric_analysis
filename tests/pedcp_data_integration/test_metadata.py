"""Small synthetic checks of the fixed recipe and typed metadata export."""

import pytest


def test_config_paths_and_validation(tmp_path, monkeypatch):
    from copy import deepcopy

    import yaml

    from utils.project_config import load_project_config

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    document = {
        "paths": {"metadata_root": "metadata", "large_output_root": "~/outputs"},
        "pedcp": {
            "image_root": "images",
            "outline_root": None,
            "workers": 1,
            "preview_count": 0,
            "preview_seed": 42,
        },
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(document))
    monkeypatch.chdir(tmp_path.parent)
    config = load_project_config(path)
    assert config.metadata_root == tmp_path / "metadata"
    assert config.large_output_root == tmp_path / "home/outputs"
    assert config.image_root == tmp_path / "images" and config.outline_root is None
    assert not config.metadata_root.exists() and not config.large_output_root.exists()
    for section, key, value in (
        ("pedcp", "workers", True),
        ("pedcp", "workers", 0),
        ("pedcp", "preview_count", -1),
        ("pedcp", "preview_seed", 2**32),
        ("paths", "metadata_root", ""),
        ("pedcp", "scientific_override", 2),
    ):
        invalid = deepcopy(document)
        invalid[section][key] = value
        path.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError):
            load_project_config(path)
    del document["pedcp"]["workers"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="exactly"):
        load_project_config(path)


def test_qc_original_population_and_reimaged(synthetic):
    from utils.pedcp_data_integration.metadata import compute_qc

    result = compute_qc(synthetic.qc)
    assert result.plate_counts.total.tolist() == [8] * 6
    assert result.fields.index[result.fields.qc_failed].tolist() == [46]
    assert result.fields.loc[46, ["blur_failed", "saturation_failed"]].all()
    assert not result.fields.loc[47, "qc_failed"]
    # Sequentially recomputing saturation after blur would wrongly exclude this field.
    sequential = compute_qc(synthetic.qc.drop(index=46))
    assert sequential.fields.loc[47, "saturation_failed"]
    assert result.counts.either.sum() == 1  # overlapping gates are not added


def test_qc_strict_thresholds_and_constants(synthetic):
    from utils.pedcp_data_integration.metadata import compute_qc
    from utils.pedcp_data_integration.recipe import BLUR_METRIC, CHANNELS, SATURATION_METRIC

    qc = synthetic.qc.iloc[:16].copy()
    assert len(compute_qc(qc).passing) == 16  # constant columns have z=0
    # Both vectors have mean=0, population variance=1, exactly representable boundaries.
    qc[f"{BLUR_METRIC}_{CHANNELS[0]}"] = [2.5, -2.5] + [0.5, -0.5] * 7
    qc[f"{SATURATION_METRIC}_{CHANNELS[1]}"] = [2, -2, 2, -2] + [0] * 12
    assert len(compute_qc(qc).passing) == 16
    qc.loc[0, f"{BLUR_METRIC}_{CHANNELS[0]}"] = 100
    qc.loc[1, f"{SATURATION_METRIC}_{CHANNELS[1]}"] = -100
    assert compute_qc(qc).fields.qc_failed.iloc[:2].all()


def test_qc_rejects_nonfinite_and_duplicate_keys(synthetic):
    import pandas as pd

    from utils.pedcp_data_integration.metadata import compute_qc
    from utils.pedcp_data_integration.recipe import BLUR_METRIC, CHANNELS

    for value in (float("nan"), float("inf"), -float("inf")):
        bad = synthetic.qc.copy()
        bad.loc[0, f"{BLUR_METRIC}_{CHANNELS[0]}"] = value
        with pytest.raises(ValueError, match="Nonfinite"):
            compute_qc(bad)
    with pytest.raises(ValueError, match="unique"):
        compute_qc(pd.concat([synthetic.qc, synthetic.qc.iloc[:1]], ignore_index=True))


def test_remap_rejects_escape_and_missing_channel(synthetic):
    from utils.pedcp_data_integration.metadata import validate_remap
    from utils.pedcp_data_integration.recipe import CHANNELS, SOURCE_PATH_PREFIX

    frame = synthetic.loads.iloc[:4].copy()
    column = f"FileName_{CHANNELS[0]}"
    frame.loc[1, column] = "../outside.tif"
    frame.loc[2, column] = "missing.tif"
    frame.loc[3, f"PathName_{CHANNELS[0]}"] = "/wrong/source"
    result = validate_remap(frame, synthetic.config.image_root)
    assert result.frame.Metadata_Site.tolist() == [1]
    assert set(result.failures.reason) == {"root_escape", "missing_tiff", "source_prefix_mismatch"}
    assert result.frame[f"PathName_{CHANNELS[0]}"].iloc[0] == str(synthetic.config.image_root)
    frame = synthetic.loads.iloc[:1].copy()
    frame[f"PathName_{CHANNELS[0]}"] = str(SOURCE_PATH_PREFIX / ".." / "outside")
    assert validate_remap(frame, synthetic.config.image_root).failures.reason.tolist() == [
        "root_escape"
    ]
    with pytest.raises(ValueError, match="missing columns"):
        validate_remap(frame.drop(columns=[column]), synthetic.config.image_root)


def test_merge_uses_plate_and_well_without_multiplication(synthetic):
    import pandas as pd

    from utils.pedcp_data_integration.metadata import curate_metadata, select_degradation_fields

    curated = curate_metadata(synthetic.barcode, synthetic.maps)
    fields = synthetic.loads.iloc[:17].copy()
    fields.loc[0, "Metadata_Well"] = "Z99"
    selected = select_degradation_fields(fields, curated)
    assert len(selected) == 15 and selected.time_point.eq(24).all()
    assert selected.groupby("Metadata_Plate").cell_line.first().tolist() == ["DAOY", "U2-OS"]
    assert selected.groupby("Metadata_Plate").size().tolist() == [7, 8]
    with pytest.raises(ValueError, match="unique"):
        select_degradation_fields(fields, pd.concat([curated, curated.iloc[:1]]))


def test_prepare_parquet_types_zstd_rerun_and_corruption(prepared):
    import pandas as pd
    import pyarrow.parquet as pq

    from utils.pedcp_data_integration.artifacts import verify_manifest
    from utils.pedcp_data_integration.metadata import prepare_metadata
    from utils.pedcp_data_integration.recipe import PLATES

    a = prepared.artifacts
    manifest = verify_manifest(a.reports / "prepare_manifest.json", a.root)
    frame = pd.read_parquet(a.export_parquet)
    assert len(frame) == 16 and set(frame.Metadata_Plate) == set(PLATES[:2])
    for column in ("Metadata_Site", "time_point", "seeding_density", "column"):
        assert pd.api.types.is_integer_dtype(frame[column])
    assert pd.api.types.is_bool_dtype(frame.Reimaged)
    assert frame.condition.eq("standard").all()
    assert not list(a.exports.glob("*.csv"))
    for path in [a.export_parquet, *a.validated.glob("*.parquet"), *a.qc_passed.glob("*.parquet")]:
        metadata = pq.ParquetFile(path).metadata
        assert all(
            metadata.row_group(i).column(j).compression == "ZSTD"
            for i in range(metadata.num_row_groups)
            for j in range(metadata.num_columns)
        )
    assert prepare_metadata(prepared.config) == manifest
    pd.testing.assert_frame_equal(pd.read_parquet(a.export_parquet), frame)
    a.export_parquet.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify_manifest(a.reports / "prepare_manifest.json", a.root)
    prepare_metadata(prepared.config)
    pd.testing.assert_frame_equal(pd.read_parquet(a.export_parquet), frame)


def test_prepare_requires_each_qc_schema_and_coverage(prepared):
    import pandas as pd

    from utils.pedcp_data_integration.metadata import prepare_metadata
    from utils.pedcp_data_integration.recipe import BLUR_METRIC, CHANNELS, PLATES

    path = prepared.artifacts.raw / f"whole_img_qc_output/{PLATES[-1]}.csv"
    original = pd.read_csv(path)
    for bad, message in (
        (original.drop(columns=[f"{BLUR_METRIC}_{CHANNELS[0]}"]), "missing columns"),
        (original.iloc[1:], "missing QC coverage"),
    ):
        bad.to_csv(path, index=False)
        prepared.certify()
        with pytest.raises(ValueError, match=message):
            prepare_metadata(prepared.config)
        assert not (prepared.artifacts.reports / "prepare_manifest.json").exists()
