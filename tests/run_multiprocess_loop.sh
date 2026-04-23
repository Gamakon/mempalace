#!/usr/bin/env bash
# mp-33y Task 5e: 10-iteration flake detector for the Surreal multiprocess suite.
#
# Runs ``pytest tests/test_surreal_multiprocess.py -x -v`` ten times in a
# row. Zero flakes required to declare mp-33y done. First failure halts
# the loop with a non-zero exit status; a final line reports pass count.
#
# Usage:
#   bash tests/run_multiprocess_loop.sh
#
# Requires a running local SurrealDB at 127.0.0.1:8000 (see
# docs/surrealdb-local.md); iterations where Surreal is unreachable
# will skip rather than fail, but that is not what we want — check the
# server is up before invoking this.

set -u  # do NOT set -e; we handle non-zero exits ourselves so the loop
        # always reports a final tally.

iters=10
passes=0
first_failure=""

for i in $(seq 1 $iters); do
    echo "=== mp-33y loop iteration ${i}/${iters} ==="
    if python -m pytest tests/test_surreal_multiprocess.py -x -v; then
        passes=$((passes + 1))
    else
        first_failure="${i}"
        echo "=== iteration ${i} FAILED ==="
        break
    fi
done

echo ""
echo "=== mp-33y 10-iter loop result: ${passes}/${iters} passed ==="
if [ -n "${first_failure}" ]; then
    echo "=== first failure on iteration ${first_failure} ==="
    exit 1
fi
exit 0
