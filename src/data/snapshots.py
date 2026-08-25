"""Immutable input-snapshot manifests for reproducible research evidence."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import dataset_snapshot
from src.domain._codec import canonical_hash, non_empty, timestamp, to_primitive


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
        if not self.file_hashes or any(
            not isinstance(item, str) or len(item) != 71 or not item.startswith("sha256:")
            for item in self.file_hashes
        ):
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
        expected = canonical_hash(
            {
                "file_hashes": self.file_hashes,
                "universe_snapshot_id": self.universe_snapshot_id,
                "feature_set_version": self.feature_set_version,
            }
        )
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match its immutable inputs")

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


class SqlDatasetSnapshotStore:
    """Immutable PostgreSQL manifest store for point-in-time research inputs."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, snapshot: DatasetSnapshot) -> str:
        payload = to_primitive(snapshot)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(dataset_snapshot.c.payload).where(
                    dataset_snapshot.c.id == snapshot.snapshot_id
                )
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(dataset_snapshot).values(
                        id=snapshot.snapshot_id,
                        created_at=snapshot.created_at,
                        payload=payload,
                    )
                )
            elif dict(existing) != payload:
                raise ValueError("dataset snapshot identity collision")
        return snapshot.snapshot_id

    def get(self, snapshot_id: str) -> DatasetSnapshot:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(dataset_snapshot.c.payload).where(dataset_snapshot.c.id == snapshot_id)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError(f"dataset snapshot does not exist: {snapshot_id}")
        return DatasetSnapshot(**dict(payload))
