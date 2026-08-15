"""Read-only DuckDB queries over partitioned Parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DuckDBHistoricalQuery:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _safe_pattern(self, relative_pattern: str) -> str:
        if not relative_pattern or Path(relative_pattern).is_absolute():
            raise ValueError("relative_pattern must be a relative path")
        pattern_path = (self.root / relative_pattern).resolve()
        if pattern_path != self.root and self.root not in pattern_path.parents:
            raise ValueError("relative_pattern escapes the data root")
        return str(pattern_path)

    def query_arrow(
        self,
        *,
        relative_pattern: str,
        columns: tuple[str, ...] = (),
        where_sql: str = "",
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for historical Parquet queries") from exc
        selected = ", ".join(self._quote_identifier(name) for name in columns) or "*"
        predicate = f" WHERE {where_sql}" if where_sql.strip() else ""
        sql = f"SELECT {selected} FROM read_parquet(?, hive_partitioning = true){predicate}"
        connection = duckdb.connect(":memory:")
        try:
            return connection.execute(
                sql,
                (self._safe_pattern(relative_pattern), *parameters),
            ).fetch_arrow_table()
        finally:
            connection.close()

    def query_frame(self, **kwargs: Any) -> Any:
        return self.query_arrow(**kwargs).to_pandas()

    @staticmethod
    def _quote_identifier(value: str) -> str:
        if not value or "\x00" in value:
            raise ValueError("column names must be non-empty identifiers")
        return '"' + value.replace('"', '""') + '"'
