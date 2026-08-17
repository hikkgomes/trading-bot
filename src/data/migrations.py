"""Versioned PostgreSQL schema migrations for the platform control plane.

The service schema is deliberately small and explicit. SQLAlchemy metadata is
used to describe the domain tables, but PostgreSQL production initialisation
goes through this module and records every applied migration.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import insert, inspect, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.schema import CreateIndex, CreateTable

from src.data.database import CORE_TABLE_NAMES, metadata, schema_migration

MIGRATIONS = (
    "001_platform_schema",
    "002_canonical_evidence_constraints",
)

_APPEND_ONLY_TABLES = (
    "universe",
    "universe_snapshot",
    "universe_member",
    "dataset_snapshot",
    "feature_set",
    "feature_manifest",
    "strategy_definition",
    "strategy_version",
    "strategy_identity",
    "strategy_lineage",
    "experiment_run",
    "experiment_metric",
    "validation_result",
    "validation_stage",
    "holdout_claim",
    "model_artifact",
    "holdout_outcome",
    "forward_evidence",
    "forward_paper_observation",
    "strategy_artefact",
    "strategy_approval",
    "production_preflight",
    "import_provenance",
    "agent_action",
    "agent_proposal",
    "agent_patch",
    "agent_review",
    "agent_disposition",
    "alpha_forecast",
    "target_position",
    "risk_snapshot",
    "risk_decision",
    "promotion_event",
    "promotion_policy",
)


_MIGRATION_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _migration_hash(version: str) -> str:
    path = _MIGRATION_DIR / f"{version}.sql"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"migration file is missing: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _create_postgresql_schema(connection: Connection) -> None:
    """Create all declared tables without using ``metadata.create_all``."""

    for table in metadata.sorted_tables:
        connection.execute(CreateTable(table, if_not_exists=True))
    for table in metadata.sorted_tables:
        for index in table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))


def _create_sqlite_schema(connection: Connection) -> None:
    # SQLite is retained for isolated tests only. Its schema is still created
    # through the same table declaration so those tests exercise the full
    # repository surface.
    metadata.create_all(connection)


def _install_append_only_guards(connection: Connection) -> None:
    dialect = connection.dialect.name
    if dialect == "postgresql":
        connection.execute(
            text(
                "ALTER TABLE active_strategy_assignment "
                "DROP CONSTRAINT IF EXISTS uq_active_strategy_assignment_product_active"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_strategy_assignment_product_active "
                "ON active_strategy_assignment (product_id) WHERE active"
            )
        )
        connection.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION trading_platform_reject_mutation()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RAISE EXCEPTION 'append-only evidence table cannot be mutated: %', TG_TABLE_NAME;
                END;
                $$;
                """
            )
        )
        for table_name in _APPEND_ONLY_TABLES:
            connection.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF to_regclass('public.{table_name}') IS NOT NULL THEN
                            EXECUTE format(
                                'DROP TRIGGER IF EXISTS %I ON %I',
                                '{table_name}_append_only',
                                '{table_name}'
                            );
                            EXECUTE format(
                                'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I '
                                'FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation()',
                                '{table_name}_append_only',
                                '{table_name}'
                            );
                        END IF;
                    END
                    $$;
                    """
                )
            )
        return

    if dialect == "sqlite":
        for table_name in _APPEND_ONLY_TABLES:
            connection.execute(
                text(
                    f"CREATE TRIGGER IF NOT EXISTS {table_name}_append_only_update "
                    f"BEFORE UPDATE ON {table_name} BEGIN "
                    "SELECT RAISE(ABORT, 'append-only evidence table cannot be updated'); END"
                )
            )
            connection.execute(
                text(
                    f"CREATE TRIGGER IF NOT EXISTS {table_name}_append_only_delete "
                    f"BEFORE DELETE ON {table_name} BEGIN "
                    "SELECT RAISE(ABORT, 'append-only evidence table cannot be deleted'); END"
                )
            )


def _ensure_migration_table(connection: Connection) -> None:
    if connection.dialect.name == "postgresql":
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migration (
                    version VARCHAR(80) PRIMARY KEY,
                    applied_at VARCHAR(40) NOT NULL,
                    content_hash VARCHAR(80) NOT NULL UNIQUE
                )
                """
            )
        )
    else:
        schema_migration.create(connection, checkfirst=True)


def _applied(connection: Connection) -> set[str]:
    return {
        str(version) for version in connection.execute(select(schema_migration.c.version)).scalars()
    }


def apply_migrations(engine: Engine, *, target: str | None = None) -> tuple[str, ...]:
    """Apply pending migrations and return the versions applied in this call."""

    if target is not None and target not in MIGRATIONS:
        raise ValueError(f"unknown database migration target: {target}")
    applied_now: list[str] = []
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext('trading_platform_schema'))")
            )
        _ensure_migration_table(connection)
        applied = _applied(connection)
        for version in MIGRATIONS:
            if version in applied:
                if version == target:
                    break
                continue
            if connection.dialect.name == "postgresql":
                _create_postgresql_schema(connection)
            else:
                _create_sqlite_schema(connection)
            if version == "002_canonical_evidence_constraints":
                _install_append_only_guards(connection)
            connection.execute(
                insert(schema_migration).values(
                    version=version,
                    applied_at=_utc_now(),
                    content_hash=_migration_hash(version),
                )
            )
            applied_now.append(version)
            if version == target:
                break
        if target is None:
            _install_append_only_guards(connection)
    return tuple(applied_now)


def assert_migrated(engine: Engine) -> None:
    """Verify that the complete production schema is present."""

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing = sorted(CORE_TABLE_NAMES - tables)
    if missing:
        raise RuntimeError(f"database migrations are incomplete; missing tables: {missing}")
    with engine.connect() as connection:
        rows = {
            str(row["version"]): str(row["content_hash"])
            for row in connection.execute(select(schema_migration)).mappings()
        }
    missing_versions = sorted(set(MIGRATIONS) - set(rows))
    if missing_versions:
        raise RuntimeError(
            f"database migrations are incomplete; missing versions: {missing_versions}"
        )
    mismatched = {
        version: rows[version]
        for version in MIGRATIONS
        if rows[version] != _migration_hash(version)
    }
    if mismatched:
        raise RuntimeError(f"database migration content hashes do not match: {mismatched}")
