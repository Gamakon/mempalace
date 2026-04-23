# SurrealDB — Local Dev Setup

For the ChromaDB -> SurrealDB backend refactor. Local-only.

**Install** (v3.0.4 at time of writing, lands in `/opt/homebrew/bin/`):

```bash
brew install surrealdb/tap/surreal
```

**Data dir** — backend `surrealkv` (native embedded KV):

```bash
mkdir -p ~/.mempalace/surreal
```

**Start (background):**

```bash
nohup surreal start --user root --pass root --bind 127.0.0.1:8000 \
  surrealkv:///Users/andrewmorgan/.mempalace/surreal/surreal.db \
  > ~/.mempalace/surreal/server.log 2>&1 &
```

**Stop:** `pkill -f "surreal start"`

**Connect**

- Endpoint: `http://127.0.0.1:8000` (HTTP) / `ws://127.0.0.1:8000` (WS)
- Credentials: `root` / `root` (dev only — do not reuse)
- Pick NS/DB at connect time, e.g. `--namespace mempalace --database mempalace`

Smoke test:

```bash
echo "INFO FOR DB;" | surreal sql --endpoint http://127.0.0.1:8000 \
  --user root --pass root --namespace test --database test --pretty
curl -s http://127.0.0.1:8000/version   # -> surrealdb-3.0.4
```

## Before migrating your palace

**Always take a snapshot first.** The migration reads from `~/.mempalace/`
but accidents happen — a failed run that corrupts Chroma state, a bad
`--allow-merge`, or just the classic "I wanted to diff before and after."

Use the repo helper — on APFS it uses `cp -c` (clonefile): near-instant,
byte-identical, preserves permissions/timestamps, and consumes no extra
disk space until files diverge.

```bash
# 1. Snapshot the live palace (timestamped, non-destructive).
ISO=$(date -u +%Y-%m-%dT%H-%M-%SZ)
cp -cRp ~/.mempalace ~/.mempalace-backup-"$ISO"

# 2. (optional) dry-run a restore to preview.
tools/restore_palace_backup.sh ~/.mempalace-backup-"$ISO" --dry-run

# 3. If migration goes sideways, restore from the snapshot.
tools/restore_palace_backup.sh ~/.mempalace-backup-"$ISO"
```

The restore script (`tools/restore_palace_backup.sh`) validates that the
backup path looks like a palace (has `chroma.sqlite3`, a `palace/` subdir,
or `knowledge_graph.sqlite3`) before touching anything, and always prompts
for confirmation unless `--yes` is passed.

## Migrating a Chroma palace into Surreal

```bash
mempalace migrate-to-surreal --source ~/.mempalace
```

**Target DB naming (mp-2v9).** When `--target-db` is omitted, the target
SurrealDB name is derived as `<palace-basename>_<sha8>`, where `sha8` is
the first 8 hex chars of `sha256(abspath(source_palace))`. This means
two palaces at different paths but with the same basename (e.g.
`/home/alice/mem` and `/home/bob/mem`) resolve to **different** target
DBs — there is no silent merge by default.

If you pass `--target-db` explicitly and the target DB already contains
drawers or closets, the migration **refuses to run** and exits with
code 5. To deliberately merge into an existing Surreal DB, add
`--allow-merge`:

```bash
# Default: safe — derived name includes a path-hash slug.
mempalace migrate-to-surreal --source /home/alice/mem
mempalace migrate-to-surreal --source /home/bob/mem   # no collision

# Explicit override into a fresh target: proceeds silently.
mempalace migrate-to-surreal --source ~/.mempalace --target-db my_palace

# Explicit override into a populated target: refused.
mempalace migrate-to-surreal --source ~/.mempalace --target-db my_palace
#   REFUSING to migrate: target DB 'my_palace' already contains N row(s)...

# Opt in to the merge explicitly.
mempalace migrate-to-surreal --source ~/.mempalace --target-db my_palace --allow-merge
```
