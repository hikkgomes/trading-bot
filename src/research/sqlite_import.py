"""One-time, verified import of legacy SQLite research memory.

The importer reads SQLite in read-only mode, validates its physical and logical
integrity, writes canonical PostgreSQL research records, and optionally copies
the source to a read-only archive. It never keeps SQLite as a runtime read
path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import insert, inspect, select
from sqlalchemy.engine import Engine

from src.autopilot.experiment_memory import (
    MEMORY_FORMAT,
    SCHEMA_VERSION,
    _stored_json,
    canonical_strategy_hash,
    canonical_test_hash,
)
from src.data.database import (
    PlatformDatabase,
    experiment,
    experiment_run,
    holdout_claim,
    holdout_outcome,
    import_provenance,
    strategy_definition,
    strategy_identity,
    strategy_lineage,
    strategy_version,
    validation_result,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.domain.strategies import StrategySourceType

REQUIRED_TABLES = frozenset(
    {
        "memory_meta",
        "strategies",
        "strategy_identities",
        "lineage_edges",
        "evaluations",
    }
)


class SqliteImportError(RuntimeError):
    """The legacy source failed validation or conflicted with canonical data."""


@dataclass(frozen=True)
class ImportReport:
    source_path: str
    source_hash: str
    destination_hash: str
    counts: Mapping[str, int]
    archived_path: str | None
    provenance_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "platform.sqlite_import/v1",
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "destination_hash": self.destination_hash,
            "counts": dict(self.counts),
            "archived_path": self.archived_path,
            "provenance_id": self.provenance_id,
        }


def _decode(value: object, *, label: str) -> object:
    if isinstance(value, str):
        try:
            if value.startswith("zlib-json-v1:"):
                return _stored_json(value, label=label)
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SqliteImportError(f"invalid {label}: {exc}") from exc
    raise SqliteImportError(f"{label} must be stored as JSON text")


def _source_connection(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise SqliteImportError(f"SQLite source must be a regular non-symlink file: {path}")
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except sqlite3.DatabaseError as exc:
        raise SqliteImportError(f"cannot open SQLite source read-only: {exc}") from exc


def _validate_source(connection: sqlite3.Connection, path: Path) -> dict[str, list[sqlite3.Row]]:
    quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
    if quick != ["ok"]:
        raise SqliteImportError(f"SQLite quick_check failed for {path}: {quick[:3]}")
    foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
    if foreign_keys:
        raise SqliteImportError(f"SQLite foreign-key check failed: {foreign_keys[:3]}")
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        raise SqliteImportError(f"SQLite source is missing required tables: {missing}")
    meta = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM memory_meta")
    }
    if meta.get("schema_version") != str(SCHEMA_VERSION) or meta.get("format") != MEMORY_FORMAT:
        raise SqliteImportError("SQLite source has an unsupported experiment-memory marker")
    rows = {
        table: list(connection.execute(f"SELECT * FROM {table}"))
        for table in ("strategies", "strategy_identities", "lineage_edges", "evaluations")
    }
    strategy_hashes = {str(row["behavior_hash"]) for row in rows["strategies"]}
    for row in rows["strategies"]:
        spec = _decode(row["canonical_spec_json"], label="strategy canonical JSON")
        if not isinstance(spec, Mapping) or canonical_strategy_hash(spec) != row["behavior_hash"]:
            raise SqliteImportError(
                f"strategy hash mismatch in SQLite source: {row['behavior_hash']}"
            )
    if any(str(row["behavior_hash"]) not in strategy_hashes for row in rows["strategy_identities"]):
        raise SqliteImportError("SQLite strategy identity references an unknown strategy")
    if any(row["is_duplicate"] not in (0, 1, False, True) for row in rows["strategy_identities"]):
        raise SqliteImportError("SQLite strategy identity duplicate flags are invalid")
    if any(
        str(row["child_hash"]) not in strategy_hashes
        or str(row["parent_hash"]) not in strategy_hashes
        for row in rows["lineage_edges"]
    ):
        raise SqliteImportError("SQLite lineage references an unknown strategy")
    for row in rows["evaluations"]:
        behavior_hash = str(row["behavior_hash"])
        if behavior_hash not in strategy_hashes:
            raise SqliteImportError("SQLite evaluation references an unknown strategy")
        dataset = _decode(row["dataset_json"], label="evaluation dataset JSON")
        window = _decode(row["window_json"], label="evaluation window JSON")
        protocol = _decode(row["protocol_json"], label="evaluation protocol JSON")
        if not all(isinstance(item, Mapping) for item in (dataset, window, protocol)):
            raise SqliteImportError(f"evaluation context is not an object: {row['evaluation_key']}")
        expected_key = canonical_test_hash(
            behavior_hash,
            dataset=dataset,
            window=window,
            protocol=protocol,
            phase=str(row["phase"]),
        )
        if expected_key != row["evaluation_key"]:
            raise SqliteImportError(f"evaluation hash mismatch: {row['evaluation_key']}")
    return rows


def _source_type(value: str) -> StrategySourceType:
    normalised = value.strip().lower().replace("-", "_")
    aliases = {
        "generated": StrategySourceType.GENERATED_DSL,
        "dsl": StrategySourceType.GENERATED_DSL,
        "mutation": StrategySourceType.MUTATION,
        "crossover": StrategySourceType.CROSSOVER,
        "agent": StrategySourceType.AGENT_GENERATED_PYTHON,
        "python": StrategySourceType.AGENT_GENERATED_PYTHON,
    }
    return aliases.get(normalised, StrategySourceType.GENERATED_DSL)


def _timestamp(value: object) -> str:
    return str(value or datetime.now(UTC).replace(microsecond=0).isoformat())


def _insert_immutable(connection, table, values: dict[str, object]) -> None:
    identity = values["id"]
    existing = connection.execute(select(table).where(table.c.id == identity)).mappings().first()
    if existing is None:
        connection.execute(insert(table).values(**values))
        return
    for key, value in values.items():
        if existing[key] != value:
            raise SqliteImportError(
                f"canonical import identity collision in {table.name}: {identity}"
            )


def _definition_payload(strategy_row: sqlite3.Row) -> dict[str, object]:
    metadata = _decode(strategy_row["metadata_json"], label="strategy metadata JSON")
    if not isinstance(metadata, Mapping):
        metadata = {}
    product = str(strategy_row["product"] or metadata.get("product") or "unknown")
    return {
        "identity": str(strategy_row["primary_strategy_id"]),
        "version": "sqlite-import-v1",
        "family": str(strategy_row["generation_method"] or "legacy"),
        "product": product,
        "universe": {},
        "data_requirements": {},
        "feature_graph": {},
        "signal_model": {
            "legacy_behavior_hash": str(strategy_row["behavior_hash"]),
            "spec": _decode(strategy_row["canonical_spec_json"], label="strategy canonical JSON"),
        },
        "position_model": {},
        "execution_preferences": {},
        "risk_policy": {},
        "validation_policy": {},
        "source_type": _source_type(str(strategy_row["generation_method"] or "generated")).value,
        "source_hash": str(strategy_row["behavior_hash"]),
        "metadata": {"legacy_import": True, **dict(metadata)},
    }


def _destination_payload(
    strategy_rows: Iterable[sqlite3.Row],
    identity_rows: Iterable[sqlite3.Row],
    lineage_rows: Iterable[sqlite3.Row],
    evaluation_rows: Iterable[sqlite3.Row],
) -> dict[str, object]:
    return {
        "strategies": sorted(str(row["behavior_hash"]) for row in strategy_rows),
        "identities": sorted(str(row["strategy_id"]) for row in identity_rows),
        "lineage": sorted(f"{row['child_hash']}:{row['parent_hash']}" for row in lineage_rows),
        "evaluations": sorted(str(row["evaluation_key"]) for row in evaluation_rows),
    }


def archive_sqlite_source(source: Path, destination: Path) -> Path:
    """Copy a verified source to a read-only archive without deleting it."""

    if source.is_symlink() or not source.is_file():
        raise SqliteImportError(f"SQLite source must be a regular file: {source}")
    if destination.exists() or destination.is_symlink():
        raise SqliteImportError(f"archive destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if (
        hashlib.sha256(destination.read_bytes()).hexdigest()
        != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        destination.unlink(missing_ok=True)
        raise SqliteImportError("archived SQLite source hash does not match source")
    os.chmod(destination, 0o444)
    return destination


def _experiment_id(behavior_hash: str) -> str:
    """Keep imported experiment IDs within the canonical key length."""

    return canonical_hash({"legacy_import": "experiment", "behavior_hash": behavior_hash})


def _strategy_version_id(strategy_row: sqlite3.Row) -> str:
    return f"{strategy_row['primary_strategy_id']}:sqlite-import-v1"


def import_sqlite_memory(
    source: Path,
    database: PlatformDatabase | Engine,
    *,
    archive_to: Path | None = None,
) -> ImportReport:
    """Validate and import the complete legacy SQLite research memory."""

    source = Path(source)
    if source.is_symlink():
        raise SqliteImportError(f"SQLite source must not be a symlink: {source}")
    source = source.resolve()
    source_hash = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    connection = _source_connection(source)
    try:
        rows = _validate_source(connection, source)
    finally:
        connection.close()

    engine = database.engine if isinstance(database, PlatformDatabase) else database
    if isinstance(database, PlatformDatabase):
        if database.is_postgresql:
            database.migrate()
        else:
            inspector = inspect(engine)
            if "experiment" not in inspector.get_table_names():
                database.create_schema()

    strategy_rows = rows["strategies"]
    identity_rows = rows["strategy_identities"]
    lineage_rows = rows["lineage_edges"]
    evaluation_rows = rows["evaluations"]
    destination_payload = _destination_payload(
        strategy_rows, identity_rows, lineage_rows, evaluation_rows
    )
    destination_hash = canonical_hash(destination_payload)
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    archived_path = None
    if archive_to is not None:
        archived_path = str(archive_sqlite_source(source, archive_to))
    imported_stages: set[tuple[str, str]] = set()
    with engine.begin() as target:
        for row in strategy_rows:
            behavior_hash = str(row["behavior_hash"])
            definition = _definition_payload(row)
            _insert_immutable(
                target,
                strategy_definition,
                {
                    "id": behavior_hash,
                    "identity": str(row["primary_strategy_id"]),
                    "product_id": str(definition["product"]),
                    "source_type": str(definition["source_type"]),
                    "source_hash": behavior_hash,
                    "definition": definition,
                },
            )
            _insert_immutable(
                target,
                strategy_version,
                {
                    "id": _strategy_version_id(row),
                    "definition_id": behavior_hash,
                    "version": "sqlite-import-v1",
                    "created_at": _timestamp(row["created_at"]),
                    "payload": {"legacy_behavior_hash": behavior_hash},
                },
            )
            experiment_id = _experiment_id(behavior_hash)
            evals_for_strategy = [
                item for item in evaluation_rows if str(item["behavior_hash"]) == behavior_hash
            ]
            snapshots: set[str] = set()
            for item in evals_for_strategy:
                dataset = _decode(item["dataset_json"], label="evaluation dataset JSON")
                if not isinstance(dataset, Mapping):
                    raise SqliteImportError(
                        f"evaluation dataset is not an object: {item['evaluation_key']}"
                    )
                snapshots.add(str(dataset.get("snapshot_id")))
            snapshot_ids = sorted(snapshots)
            _insert_immutable(
                target,
                experiment,
                {
                    "id": experiment_id,
                    "strategy_version_id": _strategy_version_id(row),
                    "provider": "legacy_sqlite_import",
                    "state": "queued",
                    "submitted_at": _timestamp(row["created_at"]),
                    "dataset_snapshot_hashes": snapshot_ids or [behavior_hash],
                    "metadata": {"legacy_behavior_hash": behavior_hash},
                },
            )

        for row in identity_rows:
            _insert_immutable(
                target,
                strategy_identity,
                {
                    "id": str(row["strategy_id"]),
                    "behavior_hash": str(row["behavior_hash"]),
                    "submitted_spec": _decode(
                        row["submitted_spec_json"], label="strategy submitted JSON"
                    ),
                    "generation_method": str(row["generation_method"]),
                    "metadata": _decode(
                        row["metadata_json"], label="strategy identity metadata JSON"
                    ),
                    "parent_hashes": _decode(
                        row["parent_hashes_json"], label="strategy parent hashes JSON"
                    ),
                    "is_duplicate": bool(row["is_duplicate"]),
                    "created_at": _timestamp(row["created_at"]),
                },
            )

        for row in lineage_rows:
            child = str(row["child_hash"])
            parent = str(row["parent_hash"])
            _insert_immutable(
                target,
                strategy_lineage,
                {
                    "id": f"sqlite-import:{child}:{parent}",
                    "created_at": _timestamp(row["created_at"]),
                    "payload": {
                        "child_hash": child,
                        "parent_hash": parent,
                        "parent_ordinal": int(row["parent_ordinal"]),
                        "generation_method": str(row["generation_method"]),
                        "legacy_import": True,
                    },
                },
            )

        for row in evaluation_rows:
            evaluation_key = str(row["evaluation_key"])
            behavior_hash = str(row["behavior_hash"])
            strategy_row = next(
                item for item in strategy_rows if str(item["behavior_hash"]) == behavior_hash
            )
            strategy_version_id = _strategy_version_id(strategy_row)
            phase = str(row["phase"])
            run_payload = {
                "legacy_import": True,
                "evaluation_key": evaluation_key,
                "behavior_hash": behavior_hash,
                "phase": phase,
                "dataset": _decode(row["dataset_json"], label="evaluation dataset JSON"),
                "window": _decode(row["window_json"], label="evaluation window JSON"),
                "protocol": _decode(row["protocol_json"], label="evaluation protocol JSON"),
                "status": str(row["status"]),
                "outcome": row["outcome"],
                "rejection_reasons": _decode(
                    row["rejection_reasons_json"], label="evaluation rejection JSON"
                ),
                "metrics": _decode(row["metrics_json"], label="evaluation metrics JSON"),
                "details": _decode(row["details_json"], label="evaluation details JSON"),
            }
            _insert_immutable(
                target,
                experiment_run,
                {
                    "id": evaluation_key,
                    "created_at": _timestamp(row["claimed_at"]),
                    "payload": run_payload,
                },
            )
            accepted = str(row["outcome"] or "").lower() in {"accept", "accepted", "pass", "passed"}
            reasons = run_payload["rejection_reasons"]
            reason_code = str(reasons[0]) if isinstance(reasons, list) and reasons else None
            _insert_immutable(
                target,
                validation_result,
                {
                    "id": evaluation_key,
                    "experiment_id": _experiment_id(behavior_hash),
                    "state": "forward_paper" if accepted else f"{phase}_rejected",
                    "accepted": accepted,
                    "reason_code": reason_code,
                    "evidence": run_payload,
                },
            )
            stage_name = {
                "holdout": "protected",
                "final_holdout": "protected",
                "final": "protected",
            }.get(phase, phase)
            stage_identity = (_experiment_id(behavior_hash), stage_name)
            if stage_identity not in imported_stages:
                imported_stages.add(stage_identity)
                stage_accepted = accepted
                stage_reason = reason_code or (
                    None if stage_accepted else "legacy_outcome_not_accepted"
                )
                stage_payload = {
                    "legacy_import": True,
                    "evaluation_key": evaluation_key,
                    "phase": phase,
                    "source_hash": behavior_hash,
                }
                _insert_immutable(
                    target,
                    validation_stage,
                    {
                        "id": canonical_hash(
                            {
                                "experiment_id": stage_identity[0],
                                "stage": stage_name,
                                "source_run_id": evaluation_key,
                                "payload": stage_payload,
                            }
                        ),
                        "experiment_id": stage_identity[0],
                        "stage": stage_name,
                        "source_run_id": evaluation_key,
                        "evaluated_at": _timestamp(row["completed_at"] or row["claimed_at"]),
                        "state": "accepted" if stage_accepted else "rejected",
                        "accepted": stage_accepted,
                        "reason_code": stage_reason,
                        "evidence_hash": canonical_hash(run_payload),
                        "payload": stage_payload,
                    },
                )
            if phase in {"holdout", "final_holdout", "final"}:
                claim_id = f"sqlite-import:holdout:{evaluation_key}"
                claim_payload = {
                    "legacy_import": True,
                    "strategy_version_id": strategy_version_id,
                    "evaluation_key": evaluation_key,
                    "behavior_hash": behavior_hash,
                    "data_snapshot_id": run_payload["dataset"].get("snapshot_id"),
                    "protocol_hash": str(row["protocol_hash"]),
                    "claimed_at": _timestamp(row["claimed_at"]),
                }
                _insert_immutable(
                    target,
                    holdout_claim,
                    {
                        "id": claim_id,
                        "created_at": _timestamp(row["claimed_at"]),
                        "payload": claim_payload,
                    },
                )
                outcome_payload = {
                    "legacy_import": True,
                    "evaluation_key": evaluation_key,
                    "outcome": row["outcome"],
                    "metrics": run_payload["metrics"],
                    "details": run_payload["details"],
                }
                outcome_hash = canonical_hash(outcome_payload)
                _insert_immutable(
                    target,
                    holdout_outcome,
                    {
                        "id": f"sqlite-import:holdout-outcome:{evaluation_key}",
                        "holdout_claim_id": claim_id,
                        "evaluated_at": _timestamp(row["completed_at"] or row["claimed_at"]),
                        "accepted": accepted,
                        "outcome_hash": outcome_hash,
                        "payload": outcome_payload,
                    },
                )

        provenance_id = f"sqlite-import:{source_hash}"
        provenance_payload = {
            "schema": "platform.sqlite_import/v1",
            "source_counts": {
                "strategies": len(strategy_rows),
                "strategy_identities": len(identity_rows),
                "lineage_edges": len(lineage_rows),
                "evaluations": len(evaluation_rows),
            },
            "source_tables": sorted(REQUIRED_TABLES),
        }
        _insert_immutable(
            target,
            import_provenance,
            {
                "id": provenance_id,
                "source_path": str(source),
                "source_hash": source_hash,
                "destination_hash": destination_hash,
                "imported_at": now,
                "archived_path": archived_path,
                "payload": provenance_payload,
            },
        )

    imported_strategy_ids = [str(row["behavior_hash"]) for row in strategy_rows]
    imported_identity_ids = [str(row["strategy_id"]) for row in identity_rows]
    imported_lineage_ids = [
        f"sqlite-import:{row['child_hash']}:{row['parent_hash']}" for row in lineage_rows
    ]
    imported_evaluation_ids = [str(row["evaluation_key"]) for row in evaluation_rows]
    with engine.connect() as target:
        strategy_count = len(
            target.execute(
                select(strategy_definition.c.id).where(
                    strategy_definition.c.id.in_(imported_strategy_ids)
                )
            ).all()
        )
        identity_count = len(
            target.execute(
                select(strategy_identity.c.id).where(
                    strategy_identity.c.id.in_(imported_identity_ids)
                )
            ).all()
        )
        lineage_payloads = [
            row[0]
            for row in target.execute(
                select(strategy_lineage.c.payload).where(
                    strategy_lineage.c.id.in_(imported_lineage_ids)
                )
            ).all()
        ]
        run_payloads = [
            row[0]
            for row in target.execute(
                select(experiment_run.c.payload).where(
                    experiment_run.c.id.in_(imported_evaluation_ids)
                )
            ).all()
        ]
    actual_payload = {
        "strategies": sorted(imported_strategy_ids),
        "identities": sorted(imported_identity_ids),
        "lineage": sorted(
            f"{item['child_hash']}:{item['parent_hash']}"
            for item in lineage_payloads
            if isinstance(item, Mapping)
        ),
        "evaluations": sorted(
            str(item["evaluation_key"])
            for item in run_payloads
            if isinstance(item, Mapping) and "evaluation_key" in item
        ),
    }
    actual_counts = {
        "strategies": strategy_count,
        "strategy_identities": identity_count,
        "lineage_edges": len(lineage_payloads),
        "evaluations": len(run_payloads),
    }
    expected_counts = {
        "strategies": len(strategy_rows),
        "strategy_identities": len(identity_rows),
        "lineage_edges": len(lineage_rows),
        "evaluations": len(evaluation_rows),
    }
    if actual_counts != expected_counts or canonical_hash(actual_payload) != destination_hash:
        raise SqliteImportError(
            "SQLite import verification failed: "
            f"expected counts/hash {expected_counts}/{destination_hash}, "
            f"got {actual_counts}/{canonical_hash(actual_payload)}"
        )
    counts = {
        "strategies": len(strategy_rows),
        "strategy_identities": len(identity_rows),
        "lineage_edges": len(lineage_rows),
        "evaluations": len(evaluation_rows),
    }
    return ImportReport(
        source_path=str(source),
        source_hash=source_hash,
        destination_hash=destination_hash,
        counts=counts,
        archived_path=archived_path,
        provenance_id=f"sqlite-import:{source_hash}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import legacy SQLite research memory.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    database = PlatformDatabase(args.database_url)
    report = import_sqlite_memory(args.source, database, archive_to=args.archive)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
