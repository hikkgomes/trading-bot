import json

import pytest

from src.autopilot.io import append_json_line, write_json_atomic, write_text_atomic


def test_write_text_atomic_creates_parent_and_replaces_existing_file(tmp_path):
    path = tmp_path / "nested" / "state.txt"
    write_text_atomic(path, "old")

    write_text_atomic(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_write_json_atomic_writes_sorted_pretty_json(tmp_path):
    path = tmp_path / "state.json"

    write_json_atomic(path, {"b": 2, "a": {"c": 3}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"c": 3}, "b": 2}
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')


def test_append_json_line_rejects_symlink_without_touching_target(tmp_path):
    path = tmp_path / "alerts.jsonl"
    target = tmp_path / "external_alerts.jsonl"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    path.symlink_to(target)

    with pytest.raises(ValueError, match="jsonl path must not be a symlink"):
        append_json_line(path, {"new": True})

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'
