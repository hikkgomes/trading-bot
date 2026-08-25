from __future__ import annotations

from src.data.database import PlatformDatabase, schema_migration


def test_platform_migrations_record_immutable_revision_hashes(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.migrate()
    database.assert_migrated()
    with database.engine.connect() as connection:
        rows = connection.execute(schema_migration.select()).mappings().all()
    assert all(row["revision_hash"].startswith("sha256:") for row in rows)
