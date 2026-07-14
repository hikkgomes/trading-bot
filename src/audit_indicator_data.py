import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.config import INDICATOR_DATA_DIR, PROJECT_ROOT

DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "indicator_data_audit.json"


def _column_null_ratio(parquet_file: pq.ParquetFile, column_index: int) -> float | None:
    null_count = 0
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        column = parquet_file.metadata.row_group(row_group_index).column(column_index)
        stats = column.statistics
        if stats is None or stats.null_count is None:
            return None
        null_count += stats.null_count
    return null_count / parquet_file.metadata.num_rows


def audit_file(path: Path, mostly_null_threshold: float) -> dict[str, object]:
    parquet_file = pq.ParquetFile(path)
    names = parquet_file.schema.names
    timestamp_index = names.index("timestamp") if "timestamp" in names else None
    timestamp_stats = None
    if timestamp_index is not None:
        mins = []
        maxs = []
        timestamp_nulls = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            column = parquet_file.metadata.row_group(row_group_index).column(timestamp_index)
            stats = column.statistics
            if stats is not None:
                mins.append(stats.min)
                maxs.append(stats.max)
                timestamp_nulls += stats.null_count or 0
        timestamp_stats = {
            "min": str(mins[0]) if mins else None,
            "max": str(maxs[-1]) if maxs else None,
            "null_count": timestamp_nulls,
        }

    mostly_null_columns: list[dict[str, object]] = []
    for column_index, column_name in enumerate(names):
        ratio = _column_null_ratio(parquet_file, column_index)
        if ratio is not None and ratio >= mostly_null_threshold:
            mostly_null_columns.append({"column": column_name, "null_ratio": round(ratio, 6)})

    return {
        "file": str(path),
        "rows": parquet_file.metadata.num_rows,
        "columns": parquet_file.metadata.num_columns,
        "row_groups": parquet_file.metadata.num_row_groups,
        "first_columns": names[:25],
        "last_columns": names[-10:],
        "timestamp": timestamp_stats,
        "mostly_null_threshold": mostly_null_threshold,
        "mostly_null_columns": mostly_null_columns,
    }


def run(
    indicator_dir: Path = INDICATOR_DATA_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    mostly_null_threshold: float = 0.95,
) -> None:
    reports = [
        audit_file(path, mostly_null_threshold) for path in sorted(indicator_dir.glob("*.parquet"))
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit large indicator parquet files.")
    parser.add_argument("--indicator-dir", type=Path, default=INDICATOR_DATA_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--mostly-null-threshold", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.indicator_dir, args.report_path, args.mostly_null_threshold)


if __name__ == "__main__":
    main()
