# OptiPlex reviewer evidence

Generated on 2 September 2026 from the deployed checkout on `192.168.1.41`.

## Contents

- `postgresql-audit.sql` is the read-only PostgreSQL funnel audit.
- `postgresql-audit-results.txt` contains its results from `trading_platform`.
- `legacy-experiment-memory.sql` is the read-only legacy SQLite audit.
- `legacy-experiment-memory-results.txt` contains results from all four remaining legacy databases.
- `system-audit.sh` and `system-audit-results.txt` contain non-secret host, systemd, journal-summary, permission-metadata, and process evidence.
- `data-audit.sh` and `data-audit-results.txt` contain filesystem counts, sizes, raw-data partitions, and metadata for 20 representative Parquet files.
- `quality-audit.sh` and `quality-audit-results.txt` contain deployed static checks and focused-test results.

## Scope and limits

- No API keys, API secrets, passwords, database URLs, or environment-file contents are included.
- The database queries are read-only.
- The Parquet evidence is aggregated. The script samples 20 files because the server contains hundreds of thousands of small partitions.
- The report and derived-research directories were already removed before this audit. Their absence is recorded by the data inventory.
- Historical `reporting` jobs and worker rows still exist in PostgreSQL as evidence of the old runtime. The current source no longer assigns or schedules the filesystem report worker.
- The server focused tests did not complete because the process entered uninterruptible kernel I/O wait. The same focused suite passed locally: 22 passed, 1 skipped.

## Reproduction

The three shell scripts can be run from the repository checkout on the OptiPlex. They do not read credential values. The SQL files can be passed to PostgreSQL or SQLite using the database paths shown in the corresponding results file.
