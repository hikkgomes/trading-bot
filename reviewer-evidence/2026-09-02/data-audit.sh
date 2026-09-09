#!/bin/sh

set +e
cd /home/alfred/trading-bot || exit 1

echo '=== DATA FILE INVENTORY ==='
for root in \
    data/raw \
    data/bars \
    data/features \
    data/research \
    data/artefacts \
    data/reports \
    runtime/research \
    runtime/candidates \
    runtime/events
do
    if [ -d "$root" ]; then
        files=$(find "$root" -type f | wc -l)
        bytes=$(find "$root" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
        newest=$(find "$root" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        printf '%s files=%s bytes=%s newest=%s\n' \
            "$root" "$files" "$bytes" "${newest:-none}"
    else
        printf '%s MISSING\n' "$root"
    fi
done

echo '=== RAW BINANCE PARTITIONS ==='
find data/raw -type f -name '*.parquet' -printf '%p\n' 2>/dev/null |
    awk -F/ 'NF >= 5 {print $4 "/" $5}' |
    sort | uniq -c | sort -k2

echo '=== PARQUET SAMPLE METADATA ==='
if [ -x .venv-runtime/bin/python ]; then
    .venv-runtime/bin/python -c '
from pathlib import Path
import sys

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    print(f"pyarrow unavailable: {exc}")
    raise SystemExit(0)

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    try:
        metadata = pq.ParquetFile(path).metadata
        print(
            f"{path}\trows={metadata.num_rows}"
            f"\trow_groups={metadata.num_row_groups}"
            f"\tbytes={path.stat().st_size}"
        )
    except Exception as exc:
        print(f"{path}\tERROR={type(exc).__name__}: {exc}")
' $(find data -type f -name '*.parquet' -print 2>/dev/null | head -20)
fi
