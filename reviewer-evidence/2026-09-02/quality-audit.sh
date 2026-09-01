#!/bin/sh

set +e
cd /home/alfred/trading-bot || exit 1

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv-research/bin/python ]; then
    PY=.venv-research/bin/python
else
    echo 'No repository Python environment found.'
    exit 1
fi

echo '=== CYCLOMATIC COMPLEXITY ==='
"$PY" -m ruff check . --select C901

echo '=== COMPLETE RUFF CHECK ==='
"$PY" -m ruff check .

echo '=== FORMAT CHECK ==='
"$PY" -m ruff format --check .

echo '=== FOCUSED TESTS ==='
"$PY" -m pytest -q \
    tests/test_research_evidence_integrity.py \
    tests/test_research_job_authority.py \
    tests/test_platform_live_authority.py \
    tests/test_platform_testnet_rehearsal.py \
    tests/test_platform_testnet_rehearsal_integration.py
