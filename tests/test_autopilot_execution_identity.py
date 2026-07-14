from dataclasses import dataclass

import pytest

from src.autopilot import execution_identity


@dataclass
class FakeDistribution:
    name: str
    version: str

    @property
    def metadata(self):
        return {"Name": self.name}


def _source_tree(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "requirements-bot.txt").write_text("example==1.0\n", encoding="utf-8")
    return tmp_path


def test_execution_identity_changes_with_any_installed_distribution(monkeypatch, tmp_path):
    root = _source_tree(tmp_path)
    monkeypatch.setattr(
        execution_identity,
        "distributions",
        lambda: [FakeDistribution("Example_Package", "1.0")],
    )
    first = execution_identity.execution_engine_digest(root=root)

    monkeypatch.setattr(
        execution_identity,
        "distributions",
        lambda: [
            FakeDistribution("example-package", "1.0"),
            FakeDistribution("Low_Level_Network", "2.0"),
        ],
    )
    second = execution_identity.execution_engine_digest(root=root)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_execution_identity_rejects_conflicting_or_incomplete_distribution_metadata(
    monkeypatch, tmp_path
):
    root = _source_tree(tmp_path)
    monkeypatch.setattr(
        execution_identity,
        "distributions",
        lambda: [
            FakeDistribution("same_name", "1.0"),
            FakeDistribution("same-name", "2.0"),
        ],
    )
    with pytest.raises(RuntimeError, match="multiple installed versions"):
        execution_identity.execution_engine_digest(root=root)

    monkeypatch.setattr(
        execution_identity,
        "distributions",
        lambda: [FakeDistribution("", "1.0")],
    )
    with pytest.raises(RuntimeError, match="incomplete package metadata"):
        execution_identity.execution_engine_digest(root=root)
