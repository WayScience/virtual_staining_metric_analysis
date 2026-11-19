"""Standalone backend tests: only synthetic data under pytest's temporary directory."""

import json
import select
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from utils.fov_ingestion import ingest, load_plane, outlines_to_labels, plan
from utils.fov_ingestion.collection import MicroscopyCollection


@pytest.fixture
def source(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    outlines = tmp_path / "outlines" / "plate"
    outlines.mkdir(parents=True)
    pixels = {
        "DNA": np.arange(256, dtype=np.uint16).reshape(16, 16) * 257,
        "BF": np.full((16, 16), 1234, dtype=np.uint16),
    }
    row = {
        "Metadata_Plate": "plate",
        "Metadata_Well": "A01",
        "Metadata_Site": "01",
        "treatment": "synthetic control",
    }
    for channel, image in pixels.items():
        tifffile.imwrite(images / f"{channel}.tif", image)
        row[f"PathName_{channel}"] = "images"
        row[f"FileName_{channel}"] = f"{channel}.tif"
    masks, boundaries = {}, {}
    for kind, low, high in (("Cells", 2, 14), ("Nuclei", 5, 11)):
        boundary = np.zeros((16, 16), dtype=np.uint8)
        boundary[low:high, low] = boundary[low:high, high - 1] = 255
        boundary[low, low:high] = boundary[high - 1, low:high] = 255
        # Exercise both grayscale and green-only RGB outline input.
        stored = boundary
        if kind == "Nuclei":
            stored = np.zeros((16, 16, 3), dtype=np.uint8)
            stored[..., 1] = boundary
        tifffile.imwrite(outlines / f"{kind}Outlines_plate_A01_01.tiff", stored)
        labels = np.zeros((16, 16), dtype=np.uint32)
        labels[low + 1 : high - 1, low + 1 : high - 1] = 1
        masks[kind.lower()] = labels
        boundaries[kind.lower()] = boundary
    return SimpleNamespace(
        frame=pd.DataFrame([row]),
        options={"outline_root": outlines.parent, "image_root": tmp_path, "dataset_id": "test"},
        pixels=pixels,
        masks=masks,
        boundaries=boundaries,
        output=tmp_path / "collection",
    )


def test_outline_interiors_and_empty_masks(source):
    for name, boundary in source.boundaries.items():
        labels = outlines_to_labels(boundary)
        np.testing.assert_array_equal(labels, source.masks[name])
        assert labels.dtype == np.uint32
    np.testing.assert_array_equal(
        outlines_to_labels(np.zeros((4, 4), dtype=bool)), np.zeros((4, 4), dtype=np.uint32)
    )
    with pytest.raises(ValueError, match="2D binary"):
        outlines_to_labels(np.zeros((2, 4, 4), dtype=bool))


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_plan_resolves_metadata_files_and_dry_run(source, tmp_path, suffix):
    path = tmp_path / f"metadata.{suffix}"
    getattr(source.frame, f"to_{suffix}")(path, index=False)
    (item,) = plan(path, **source.options)
    assert item["site"] == "01"
    assert item["channels"] == ["BF", "DNA"]
    assert item["paths"]["DNA"] == str(tmp_path / "images" / "DNA.tif")
    assert item["metadata"]["treatment"] == "synthetic control"
    assert item["fov_id"] == plan(source.frame, **source.options)[0]["fov_id"]
    result = ingest(path, output=source.output, dry_run=True, **source.options)
    assert result.fov_id.tolist() == [item["fov_id"]]
    assert not source.output.exists()


def test_plan_rejects_duplicates_and_controls_missing_outlines(source):
    with pytest.raises(ValueError, match="Duplicate FOV"):
        plan(pd.concat([source.frame, source.frame]), **source.options)
    outline = source.options["outline_root"] / "plate" / "CellsOutlines_plate_A01_01.tiff"
    outline.unlink()
    with pytest.raises(ValueError, match="Missing outlines"):
        plan(source.frame, **source.options)
    with pytest.warns(UserWarning, match="Skipped 1 FOVs"):
        assert plan(source.frame, skip_missing_outlines=True, **source.options) == []
    # Skipping missing outlines must not hide missing intensity images.
    (source.options["image_root"] / "images" / "DNA.tif").unlink()
    with pytest.raises(ValueError, match="Missing intensity image"):
        plan(source.frame, skip_missing_outlines=True, **source.options)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_expanded_metadata_is_string_without_changing_json(source, tmp_path, suffix):
    frame = source.frame.assign(Metadata_Count=7, Metadata_Optional=None, Metadata_Flag=True)
    path = tmp_path / f"metadata.{suffix}"
    getattr(frame, f"to_{suffix}")(path, index=False)
    (item,) = plan(path, **source.options)
    ingest(path, output=source.output, **source.options)
    collection = MicroscopyCollection(source.output)
    row = collection.fovs.iloc[0]
    for key, value in item["metadata"].items():
        if key.startswith("Metadata_"):
            assert row[key] == str(value)
    assert json.loads(row.metadata_json) == item["metadata"]


def test_ingest_round_trip_pixels_metadata_and_objects(source):
    result = ingest(source.frame, output=source.output, pixel_size_um=0.5, **source.options)
    assert not result.resumed.any()
    collection = MicroscopyCollection(source.output)
    fov_id = result.fov_id.iloc[0]
    assert collection.metadata().treatment.tolist() == ["synthetic control"]
    assert collection.fovs.pixel_size_um.tolist() == [0.5]
    planes = collection.read_fov(fov_id)
    expected = {
        **{("intensity", name): image for name, image in source.pixels.items()},
        **{("labels", name): mask for name, mask in source.masks.items()},
    }
    assert planes.keys() == expected.keys()
    for key, image in expected.items():
        np.testing.assert_array_equal(planes[key], image)
        assert planes[key].dtype == image.dtype
    selected = collection.read_fov(fov_id, channels=["DNA"], include_masks=False, normalize=True)
    assert set(selected) == {("intensity", "DNA")}
    np.testing.assert_array_equal(
        selected[("intensity", "DNA")], source.pixels["DNA"].astype(np.float32) / 65535
    )
    for row in collection.images.itertuples(index=False):
        np.testing.assert_array_equal(
            load_plane(source.output, row.image_id), planes[row.role, row.channel]
        )
        if row.role == "intensity":
            np.testing.assert_array_equal(
                load_plane(source.output, row.image_id, normalize=True),
                source.pixels[row.channel].astype(np.float32) / 65535,
            )
        else:
            with pytest.raises(ValueError, match="must not be normalized"):
                load_plane(source.output, row.image_id, normalize=True)
    objects, relationships = collection.object_tables(fov_id)
    cells = objects.set_index("compartment").loc["cells"]
    assert (cells.y0, cells.x0, cells.y1, cells.x1, cells.area_pixels) == (3, 3, 13, 13, 100)
    assert (cells.centroid_y, cells.centroid_x) == (7.5, 7.5)
    assert not objects.touches_border.any()
    nucleus = objects.set_index("compartment").loc["nuclei"]
    (relationship,) = relationships.to_dict("records")
    assert relationship == {
        "nucleus_id": nucleus.object_id,
        "cell_id": cells.object_id,
        "overlap_pixels": 16,
        "nucleus_overlap_fraction": 1.0,
        "is_best_match": True,
    }
    with pytest.raises(KeyError):
        collection.read_fov("absent")
    with pytest.raises(ValueError, match="Channels absent"):
        collection.read_fov(fov_id, channels=["absent"])


def test_resume_and_reject_changed_inputs_or_completion_marker(source):
    ingest(source.frame, output=source.output, **source.options)
    marker = source.output / "_SUCCESS"
    original = marker.read_text()
    # A leftover lock file from an exited coordinator must not prevent resume.
    (source.output / ".ingest.lock").write_text("terminated coordinator")
    result = ingest(source.frame, output=source.output, **source.options)
    assert result.resumed.all()
    assert marker.read_text() == original
    changed = source.frame.assign(treatment="changed")
    with pytest.raises(ValueError, match="different configuration"):
        ingest(changed, output=source.output, **source.options)
    assert marker.read_text() == original
    assert (source.output / ".ingest.lock").is_file()
    marker.write_text("invalid")
    with pytest.raises(ValueError, match="completion marker"):
        MicroscopyCollection(source.output)


def test_failed_shard_is_cleaned(source):
    path = source.options["image_root"] / "images" / "BF.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="shapes differ"):
        ingest(source.frame, output=source.output, **source.options)
    assert not (source.output / "_SUCCESS").exists()
    assert (source.output / ".ingest.lock").is_file()
    assert list((source.output / "fovs").iterdir()) == []


def test_active_coordinator_excludes_others_and_killed_coordinator_can_resume(source):
    from utils.fov_ingestion.ingest import VERSION

    (item,) = plan(source.frame, **source.options)
    spec = {"version": VERSION, "dataset_id": "test",
            "items": {item["fov_id"]: item["fingerprint"]}}
    script = """
import json
import signal
import sys
from utils.fov_ingestion.writer import run_collection

def produce(fov_id):
    print('ready', flush=True)
    signal.pause()

run_collection(sys.argv[1], json.loads(sys.argv[2]), produce)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(source.output), json.dumps(spec)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 30)[0], "Coordinator did not start"
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises((BlockingIOError, FileExistsError)):
            ingest(source.frame, output=source.output, **source.options)
    finally:
        process.kill()
        process.communicate(timeout=10)
    result = ingest(source.frame, output=source.output, **source.options)
    assert not result.resumed.any()
    assert ingest(source.frame, output=source.output, **source.options).resumed.all()
