from pathlib import Path
from typing import Any, Self

import lance
import pyarrow as pa
import pyarrow.parquet as pq


class Writer:
    """
    Simple parquet shard writer that accumulates data rows upon add_row()
        calls and writes them to disk in shards when the row buffer reaches the
        specified maximum shard size, when the row schema changes, or when
        close() is called.
    Each shard contains rows with the same field schema. Schemas supplied to
        add_row() may vary across shards; metadata-only differences do not create
        a new shard boundary. The internal overwrite flag controls
        whether existing shards are replaced or retained. However, the overwrite
        decision is solely based on the existence of shard file name,
        rather than the content of the shard. The intended use of this
        writer is therefore purely for a pause-and-resume workflow where the
        incoming data row through add_row() is guaranteed to be consistent in terms
        of order, schema sequence, size (of content), and identical writer
        shardsize setting.
    """

    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema | None = None,
        overwrite: bool = False,
        compression: str = "zstd",
        shardsize: int = 128,
    ):
        """
        Initialize the Writer with the output directory, schema, and configuration options.

        :param output_dir: Directory where Parquet shards will be written.
        :param schema: PyArrow schema for the Parquet files.
        :param overwrite: If True, existing shards will be overwritten.
        :param compression: Compression algorithm to use for Parquet files.
        :param shardsize: Maximum number of rows to accumulate before writing a
            shard. A structural schema change may flush a smaller shard.
            Do not change this value across resuming runs, as it will affect the shard file names and overwrite behavior.
        """
        self.output_dir = output_dir
        if not output_dir.exists():
            raise ValueError(f"Output directory {output_dir} does not exist.")

        self.schema = schema
        self.overwrite = overwrite
        self.compression = compression
        self.shardsize = shardsize

        self._rows: list[dict[str, Any]] = []
        self._active_schema = schema
        self._shard_index = 0
        self._closed = False

    def _write_table(
        self,
        table: pa.Table,
        chunk_index: int,
    ) -> None:
        """
        Write a PyArrow Table to a Parquet shard.

        :param table: PyArrow Table to write.
        :param chunk_index: Index of the chunk for naming the shard file.
        """
        shard_path = self.output_dir / f"part-{chunk_index:06d}.parquet"

        if shard_path.exists() and not self.overwrite:
            return

        temp_path = shard_path.with_name(f".{shard_path.name}.tmp")
        temp_path.unlink(missing_ok=True)

        pq.write_table(
            table,
            temp_path,
            compression=self.compression,
        )
        temp_path.replace(shard_path)

    def _flush(self) -> None:
        """
        Flush the accumulated rows to a Parquet shard if there are any rows to write.
        """
        if not self._rows:
            return
        if self._active_schema is None:
            raise RuntimeError("Cannot flush buffered rows without an active schema.")

        table = pa.Table.from_pylist(
            self._rows,
            schema=self._active_schema,
        )

        self._write_table(
            table=table,
            chunk_index=self._shard_index,
        )

        self._rows.clear()
        self._shard_index += 1

    # def flush(self) -> None:
    #     """Public method for finish the current logical chunk."""
    #     if self._closed:
    #         raise RuntimeError("Cannot flush a closed writer.")
    #     self._flush()

    def add_row(self, row: dict[str, Any], schema: pa.Schema | None = None) -> None:
        """
        Main entry point for adding a row to the writer. The row is appended to the internal buffer.
        If its field schema differs from the active schema, the existing buffer is
        flushed first. Metadata-only differences do not create a shard boundary.
        The buffer is also flushed when it reaches the configured shard size.

        :param row: A dictionary representing a single row of data to be written.
        :param schema: Schema for this row. The most recently established field
            schema is reused when omitted.
        """
        if self._closed:
            raise RuntimeError("Cannot add rows to a closed Writer.")

        row_schema = schema or self._active_schema or self.schema
        if row_schema is None:
            raise ValueError(
                "Schema must be provided either at initialization or when adding a row."
            )

        if self._active_schema is None:
            self._active_schema = row_schema
        elif schema is not None and not self._active_schema.equals(
            row_schema, check_metadata=False
        ):
            self._flush()
            self._active_schema = row_schema

        self._rows.append(row)
        if len(self._rows) >= self.shardsize:
            self._flush()

    def close(self) -> None:
        """Flush the final partial shard and close the writer."""
        if self._closed:
            return

        self._flush()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed Writer.")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        if exc_type is None:
            # Normal completion: the remaining rows form the legitimate
            # final partial shard.
            self.close()
        else:
            # Exceptional completion: do not publish an incomplete shard,
            # because that would disrupt fixed-size resume alignment.
            self._rows.clear()
            self._closed = True

        # Do not suppress exceptions raised inside the with block.
        return False


class LanceWriter(Writer):
    """Append buffered Arrow tables to one Lance dataset."""

    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema | None = None,
        overwrite: bool = False,
        shardsize: int = 128,
        dataset_name: str = "data.lance",
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            schema=schema,
            overwrite=overwrite,
            shardsize=shardsize,
        )

        self.dataset_path = output_dir / dataset_name
        self._first_write = True

        if self.dataset_path.exists() and not overwrite:
            dataset = lance.dataset(self.dataset_path)

            if schema is not None and not dataset.schema.equals(
                schema,
                check_metadata=False,
            ):
                raise ValueError(
                    "Existing Lance dataset schema does not match the requested writer schema."
                )

            self._existing_fragment_count = len(dataset.get_fragments())
        else:
            self._existing_fragment_count = 0

    def _write_table(
        self,
        table: pa.Table,
        chunk_index: int,
    ) -> None:
        # Mimic the existing pause/resume behavior by skipping already
        # committed logical chunks.
        if (
            self.dataset_path.exists()
            and not self.overwrite
            and chunk_index < self._existing_fragment_count
        ):
            return

        if self._first_write and self.overwrite:
            mode = "overwrite"
        elif self.dataset_path.exists():
            mode = "append"
        else:
            mode = "create"

        lance.write_dataset(
            table,
            self.dataset_path,
            mode=mode,
            # Prevent this individual buffered table from being split based
            # on its row count under normal circumstances.
            max_rows_per_file=max(table.num_rows, 1),
            max_rows_per_group=max(table.num_rows, 1),
        )

        self._first_write = False
