"""Immutable input-snapshot manifests for reproducible research evidence."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain._codec import canonical_hash, non_empty, timestamp


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    created_at: str
    file_hashes: tuple[str, ...]
    universe_snapshot_id: str
    feature_set_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", non_empty(self.snapshot_id, field="snapshot_id"))
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        if not self.file_hashes or any(not item.startswith("sha256:") for item in self.file_hashes):
            raise ValueError("file_hashes must contain SHA-256 content hashes")
        object.__setattr__(
            self,
            "universe_snapshot_id",
            non_empty(self.universe_snapshot_id, field="universe_snapshot_id"),
        )
        object.__setattr__(
            self,
            "feature_set_version",
            non_empty(self.feature_set_version, field="feature_set_version"),
        )

    @classmethod
    def create(
        cls,
        *,
        created_at: str,
        file_hashes: tuple[str, ...],
        universe_snapshot_id: str,
        feature_set_version: str,
    ) -> DatasetSnapshot:
        snapshot_id = canonical_hash(
            {
                "file_hashes": file_hashes,
                "universe_snapshot_id": universe_snapshot_id,
                "feature_set_version": feature_set_version,
            }
        )
        return cls(
            snapshot_id=snapshot_id,
            created_at=created_at,
            file_hashes=file_hashes,
            universe_snapshot_id=universe_snapshot_id,
            feature_set_version=feature_set_version,
        )
