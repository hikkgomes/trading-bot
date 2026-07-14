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
