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
