from dataclasses import dataclass

import pytest

from src.services import execution_identity


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
    (tmp_path / "requirements-runtime.txt").write_text("example==1.0\n", encoding="utf-8")
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


def test_research_policy_identity_tracks_inference_without_binding_unrelated_services(
    tmp_path, monkeypatch
):
    from src.config import PROJECT_ROOT
    from src.research import policy_identity

    for name in policy_identity.POLICY_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
    first = policy_identity.research_policy_engine_digest(root=tmp_path)
    unrelated = tmp_path / "src/services/reporting.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("NEW_REPORT = True\n")
    assert policy_identity.research_policy_engine_digest(root=tmp_path) == first
    selection = tmp_path / "src/research/selection.py"
    selection.write_text(selection.read_text() + "\nMETHOD_REVISION = 2\n")
    assert policy_identity.research_policy_engine_digest(root=tmp_path) != first
    selection.write_bytes((PROJECT_ROOT / "src/research/selection.py").read_bytes())
    versions = {name: policy_identity.version(name) for name in policy_identity.POLICY_DEPENDENCIES}
    versions["numpy"] += ".post1"
    monkeypatch.setattr(policy_identity, "version", versions.__getitem__)
    assert policy_identity.research_policy_engine_digest(root=tmp_path) != first
