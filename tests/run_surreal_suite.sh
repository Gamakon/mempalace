#!/usr/bin/env bash
# mp-ick: Run the full pytest suite with the Surreal backend forced as default.
#
# Purpose
# -------
# We track how close the Surreal backend is to being a drop-in replacement
# for Chroma by running every test — not just the Surreal-specific ones —
# with ``MEMPALACE_BACKEND=surreal`` in the environment. Any failure tells
# us one of three things:
#
#   1. The test itself hardcodes Chroma (fixture bug — see mp-om7 umbrella).
#   2. The Surreal backend diverges from Chroma in a user-visible way
#      (real bug — file a new bd ticket, see mp-mge, mp-1y1 examples).
#   3. The Surreal server isn't up / network flake.
#
# Baseline from the first run (mp-ick):
#   - 1231 passed, 18 failed
#   - 16 fixture-level Chroma hardcodes (mp-om7 umbrella)
#   -  1 real divergence: tool_status vs missing palace (mp-mge)
#   -  1 cross-test state leak / palace isolation bug (mp-1y1)
#
# Usage
# -----
#   bash tests/run_surreal_suite.sh
#
# Requires a running local SurrealDB at 127.0.0.1:8000 (see
# docs/surrealdb-local.md). Writes the full log to
# /tmp/mp-ick-surreal-suite.log for categorisation.

set -u

LOG=/tmp/mp-ick-surreal-suite.log

echo "[mp-ick] forcing MEMPALACE_BACKEND=surreal for the full suite"
echo "[mp-ick] log: ${LOG}"

MEMPALACE_BACKEND=surreal \
    python -m pytest tests/ -v --ignore=tests/benchmarks --tb=short 2>&1 | tee "${LOG}"

status=${PIPESTATUS[0]}

echo
echo "[mp-ick] pytest exit status: ${status}"
echo "[mp-ick] summary:"
grep -E "^(FAILED|=+ .* (passed|failed) .*=+)" "${LOG}" | tail -40

exit "${status}"
