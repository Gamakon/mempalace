#!/usr/bin/env bash
# restore_palace_backup.sh
#
# Restore a MemPalace backup snapshot back to ~/.mempalace/.
# Created for bd issue mp-8jp (pre-SurrealDB-migration safety net).
#
# Usage:
#   tools/restore_palace_backup.sh <backup-path> [--dry-run] [--target <dir>] [--yes]
#
# Safety:
#   - Validates that <backup-path> looks like a palace (has chroma.sqlite3
#     or a palace/ subdir or knowledge_graph.sqlite3).
#   - Refuses to overwrite a non-empty target unless the user confirms.
#   - Uses cp -cRp on APFS (clonefile: byte-identical, preserves perms/times,
#     zero extra disk space). Falls back to cp -Rp on non-APFS filesystems.
#   - --dry-run prints what would happen without touching anything.

set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage: restore_palace_backup.sh <backup-path> [options]

Restore a MemPalace backup snapshot to ~/.mempalace/.

Arguments:
  <backup-path>      Path to a timestamped backup directory
                     (e.g. ~/.mempalace-backup-2026-04-23T22-05-00Z)

Options:
  --dry-run          Show what would happen; do not copy anything.
  --target <dir>     Restore target (default: ~/.mempalace).
  --yes              Skip interactive confirmation (scripted use).
  -h, --help         Show this help.
USAGE
}

BACKUP=""
TARGET="${HOME}/.mempalace"
DRY_RUN=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) print_usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; print_usage; exit 2 ;;
    *)
      if [[ -z "$BACKUP" ]]; then
        BACKUP="$1"
      else
        echo "Unexpected positional arg: $1" >&2; exit 2
      fi
      shift ;;
  esac
done

if [[ -z "$BACKUP" ]]; then
  echo "Error: backup path is required." >&2
  print_usage
  exit 2
fi

if [[ ! -d "$BACKUP" ]]; then
  echo "Error: backup path does not exist or is not a directory: $BACKUP" >&2
  exit 1
fi

# Validate it looks like a palace
looks_like_palace=0
for marker in \
  "chroma.sqlite3" \
  "palace/chroma.sqlite3" \
  "palace" \
  "knowledge_graph.sqlite3" \
  "mempalace.yaml" \
  "config.json"
do
  if [[ -e "$BACKUP/$marker" ]]; then
    looks_like_palace=1
    break
  fi
done

if [[ $looks_like_palace -eq 0 ]]; then
  echo "Error: $BACKUP does not look like a MemPalace backup" >&2
  echo "  (missing chroma.sqlite3, palace/, knowledge_graph.sqlite3," \
       "mempalace.yaml, or config.json)" >&2
  exit 1
fi

# Detect APFS for clonefile fast-path
cp_flags=("-Rp")
fs_type="$(df -T apfs "$BACKUP" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
if [[ "$(uname)" == "Darwin" ]]; then
  # On macOS cp supports -c (clonefile) on APFS; it is harmless to try and
  # fall back if the filesystem doesn't support it.
  cp_flags=("-cRp")
fi

echo "Backup source : $BACKUP"
echo "Restore target: $TARGET"
echo "cp flags      : ${cp_flags[*]}"
echo "Dry run       : $([[ $DRY_RUN -eq 1 ]] && echo yes || echo no)"
echo

# Summarize backup contents
file_count="$(find "$BACKUP" -type f | wc -l | tr -d ' ')"
echo "Backup file count: $file_count"

# Resolve target to an absolute path for the home-palace guard comparison.
# realpath isn't portable on macOS without coreutils, so use a small fallback.
resolve_path() {
  local p="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$p"
  else
    # Best-effort: expand ~ and make absolute
    case "$p" in
      "~") echo "$HOME" ;;
      "~/"*) echo "$HOME/${p#~/}" ;;
      /*) echo "$p" ;;
      *) echo "$PWD/$p" ;;
    esac
  fi
}

TARGET_ABS="$(resolve_path "$TARGET")"
HOME_PALACE_ABS="$(resolve_path "$HOME/.mempalace")"
IS_HOME_PALACE=0
if [[ "$TARGET_ABS" == "$HOME_PALACE_ABS" ]]; then
  IS_HOME_PALACE=1
fi

if [[ -e "$TARGET" ]]; then
  if [[ -n "$(ls -A "$TARGET" 2>/dev/null || true)" ]]; then
    echo
    echo "WARNING: target $TARGET already exists and is non-empty."
    echo "This will overlay the backup contents into it (cp -Rp)."
    echo "Files in the backup will overwrite files of the same name."
    echo "Files in the target that are NOT in the backup will be left alone."
  fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "[dry-run] Would run: cp ${cp_flags[*]} \"$BACKUP/.\" \"$TARGET/\""
  echo "[dry-run] No changes made."
  exit 0
fi

# Guard rail: if target is the real home palace AND it is non-empty, require
# the user to TYPE the exact path (not just 'y'). Refuses with --yes because
# an unattended script should never blindly overwrite the real palace.
# This is a belt-and-braces defense against the near-miss where a self-test
# accidentally ran against ~/.mempalace.
if [[ $IS_HOME_PALACE -eq 1 && -e "$TARGET" && -n "$(ls -A "$TARGET" 2>/dev/null || true)" ]]; then
  if [[ $ASSUME_YES -eq 1 ]]; then
    echo >&2
    echo "REFUSING: --yes cannot be used to overwrite the real home palace ($TARGET_ABS)." >&2
    echo "Rerun without --yes and type the full path to confirm." >&2
    exit 3
  fi
  echo
  echo "DANGER: target is your real home palace ($TARGET_ABS) and it is non-empty."
  echo "To confirm, type the full target path exactly:"
  echo "  expected: $TARGET_ABS"
  read -r -p "> " typed_path
  if [[ "$typed_path" != "$TARGET_ABS" ]]; then
    echo "Path did not match. Aborted." >&2
    exit 1
  fi
elif [[ $ASSUME_YES -ne 1 ]]; then
  echo
  read -r -p "Proceed with restore? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) : ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

mkdir -p "$TARGET"
# Copy the *contents* of the backup into the target so we don't nest the
# backup dir inside it. The trailing /. is intentional.
cp "${cp_flags[@]}" "$BACKUP/." "$TARGET/"

echo "Restore complete: $TARGET"
