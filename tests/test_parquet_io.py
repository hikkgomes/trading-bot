import pandas as pd
import pytest

from src.parquet_io import atomic_output_path, write_parquet_atomic


def test_write_parquet_atomic_replaces_complete_file(tmp_path):
    output = tmp_path / "candles.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(output, index=False)

    write_parquet_atomic(pd.DataFrame({"value": [2, 3]}), output, index=False)

    assert pd.read_parquet(output)["value"].tolist() == [2, 3]
    assert list(tmp_path.glob(".candles.parquet.*.tmp")) == []


def test_atomic_output_path_preserves_previous_file_on_failure(tmp_path):
    output = tmp_path / "artifact.parquet"
    original = b"previous-complete-artifact"
    output.write_bytes(original)

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_output_path(output) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted")

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".artifact.parquet.*.tmp")) == []


def test_atomic_output_path_rejects_symlink_destination(tmp_path):
    target = tmp_path / "target.parquet"
    target.write_bytes(b"keep")
    output = tmp_path / "linked.parquet"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        write_parquet_atomic(pd.DataFrame({"value": [1]}), output)

    assert target.read_bytes() == b"keep"


def test_partition_publication_preserves_open_readers_and_replay_availability(tmp_path):
    import pyarrow.parquet as pq

    from src.data.parquet_store import partition_files, upsert_partition_rows

    first = {"event_id": "one", "value": 1.0, "availability_time": "2026-01-01T00:00:00+00:00"}
    path = upsert_partition_rows(tmp_path, (first,), identity="event_id")
    with path.open("rb") as opened:
        replay = {**first, "availability_time": "2026-01-02T00:00:00+00:00"}
        upsert_partition_rows(
            tmp_path, (replay, {**first, "event_id": "two", "value": 2.0}), identity="event_id"
        )
        assert pq.ParquetFile(opened).read().to_pylist() == [first]
    published = pq.ParquetFile(path).read().to_pylist()
    assert len(published) == 2 and published[0] == first
    assert partition_files(tmp_path) == (path,)
    with pytest.raises(ValueError, match="conflicting partition"):
        upsert_partition_rows(tmp_path, ({**first, "value": 9.0},), identity="event_id")
    assert pq.ParquetFile(path).read().to_pylist() == published
