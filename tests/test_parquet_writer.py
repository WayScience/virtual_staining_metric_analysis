from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.parquet_writer import Writer


def test_constructor_schema_remains_supported(tmp_path: Path) -> None:
    schema = pa.schema([pa.field("value", pa.int64())])

    with Writer(tmp_path, schema=schema) as writer:
        writer.add_row({"value": 1})

    assert pq.read_table(tmp_path / "part-000000.parquet").to_pylist() == [
        {"value": 1}
    ]


def test_close_uses_schema_supplied_to_add_row(tmp_path: Path) -> None:
    schema = pa.schema([pa.field("value", pa.int64())])

    with Writer(tmp_path) as writer:
        writer.add_row({"value": 1}, schema=schema)

    table = pq.read_table(tmp_path / "part-000000.parquet")
    assert table.to_pylist() == [{"value": 1}]


def test_cached_schema_survives_size_triggered_flush(tmp_path: Path) -> None:
    schema = pa.schema([pa.field("value", pa.int64())])

    with Writer(tmp_path, shardsize=1) as writer:
        writer.add_row({"value": 1}, schema=schema)
        writer.add_row({"value": 2})

    assert pq.read_table(tmp_path / "part-000000.parquet").to_pylist() == [
        {"value": 1}
    ]
    assert pq.read_table(tmp_path / "part-000001.parquet").to_pylist() == [
        {"value": 2}
    ]


def test_structural_schema_change_starts_new_shard(tmp_path: Path) -> None:
    first_schema = pa.schema([pa.field("value", pa.int64())])
    second_schema = pa.schema([pa.field("label", pa.string())])

    with Writer(tmp_path, shardsize=10) as writer:
        writer.add_row({"value": 1}, schema=first_schema)
        writer.add_row({"label": "next"}, schema=second_schema)

    first = pq.read_table(tmp_path / "part-000000.parquet")
    second = pq.read_table(tmp_path / "part-000001.parquet")
    assert first.to_pylist() == [{"value": 1}]
    assert second.to_pylist() == [{"label": "next"}]


def test_metadata_only_schema_change_stays_in_same_shard(tmp_path: Path) -> None:
    first_schema = pa.schema(
        [pa.field("value", pa.int64())], metadata={b"batch": b"first"}
    )
    second_schema = pa.schema(
        [pa.field("value", pa.int64())], metadata={b"batch": b"second"}
    )

    with Writer(tmp_path, shardsize=10) as writer:
        writer.add_row({"value": 1}, schema=first_schema)
        writer.add_row({"value": 2}, schema=second_schema)

    shards = sorted(tmp_path.glob("*.parquet"))
    assert len(shards) == 1
    assert pq.read_table(shards[0]).to_pylist() == [{"value": 1}, {"value": 2}]
    assert pq.read_schema(shards[0]).metadata == {b"batch": b"first"}


def test_add_row_requires_schema_when_none_has_been_established(
    tmp_path: Path,
) -> None:
    with Writer(tmp_path) as writer:
        with pytest.raises(ValueError, match="Schema must be provided"):
            writer.add_row({"value": 1})