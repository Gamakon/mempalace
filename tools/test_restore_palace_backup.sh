#!/usr/bin/env bash
# test_restore_palace_backup.sh
#
# Self-test for tools/restore_palace_backup.sh.
#
# WHY WE NEVER USE $HOME IN TESTS:
#   An earlier version of this self-test round-tripped against ~/.mempalace
#   and at one point moved the real palace to ~/.mempalace.test-victim without
#   restoring it. We got lucky and noticed. Never again.
#
#   This test operates ENTIRELY inside /tmp sandboxes. It must never read,
#   write, move, rename, or even stat anything under $HOME/.mempalace. If you
#   are tempted to touch the real palace to "make the test more realistic",
#   stop — the restore script has its own integration coverage via the
#   home-palace guard rail, and this test is intentionally narrow.
#
# What this test verifies:
#   1. The restore script copies a synthetic "palace" backup to a fresh target.
#   2. The restored tree is byte-identical to the source (diff -r).
#
# Exit 0 on PASS, non-zero on FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTORE_SCRIPT="$SCRIPT_DIR/restore_palace_backup.sh"

if [[ ! -x "$RESTORE_SCRIPT" ]]; then
  echo "FAIL: restore script not found or not executable at $RESTORE_SCRIPT" >&2
  exit 1
fi

# Generate unique sandbox paths. Use mktemp -d for safety.
SRC_DIR="$(mktemp -d "/tmp/fake-palace-XXXXXX")"
DST_DIR="$(mktemp -d "/tmp/restore-target-XXXXXX")"
# mktemp creates DST_DIR; remove it so the restore script creates a fresh one.
rmdir "$DST_DIR"

# Safety assertion: neither sandbox path may resolve under $HOME.
HOME_ABS="$(python3 -c 'import os; print(os.path.abspath(os.path.expanduser("~")))')"
for p in "$SRC_DIR" "$DST_DIR"; do
  case "$p" in
    "$HOME_ABS"*|"$HOME"*)
      echo "FAIL: sandbox path $p resolves under \$HOME — refusing to run" >&2
      exit 1
      ;;
  esac
done

cleanup() {
  # Extra paranoia: never rm -rf anything that looks like it could be a real palace.
  for p in "$SRC_DIR" "$DST_DIR"; do
    case "$p" in
      /tmp/fake-palace-*|/tmp/restore-target-*)
        rm -rf "$p" 2>/dev/null || true
        ;;
      *)
        echo "cleanup: refusing to remove suspicious path: $p" >&2
        ;;
    esac
  done
}
trap cleanup EXIT

# Build a fake palace in SRC_DIR with the marker files the restore script
# looks for, plus some nested content so diff -r has something to compare.
touch "$SRC_DIR/config.json"
touch "$SRC_DIR/chroma.sqlite3"
touch "$SRC_DIR/knowledge_graph.sqlite3"
touch "$SRC_DIR/mempalace.yaml"
mkdir -p "$SRC_DIR/palace/wings/people"
printf 'hello palace\n' >"$SRC_DIR/palace/wings/people/alice.txt"
printf '{"version":"test"}\n' >"$SRC_DIR/config.json"
# A binary-ish payload to catch any accidental text conversion.
printf '\x00\x01\x02\x03binary-marker\xff\xfe' >"$SRC_DIR/chroma.sqlite3"

echo "Sandbox source : $SRC_DIR"
echo "Sandbox target : $DST_DIR"
echo

# Run the restore against the sandbox target. --yes is fine here because the
# target is NOT the home palace (the guard rail only triggers for that).
"$RESTORE_SCRIPT" "$SRC_DIR" --target "$DST_DIR" --yes

# Verify byte-for-byte identical.
if diff -r "$SRC_DIR" "$DST_DIR" >/dev/null; then
  echo
  echo "PASS: restored tree matches source byte-for-byte"
  exit 0
else
  echo >&2
  echo "FAIL: restored tree differs from source" >&2
  diff -r "$SRC_DIR" "$DST_DIR" >&2 || true
  exit 1
fi
