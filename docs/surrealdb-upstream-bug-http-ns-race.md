# HTTP transport: cross-session NS/DB routing race under concurrent multi-process clients

## Summary

Under concurrent write load from multiple independent processes (each with its own
`surrealdb` Python SDK client connecting over HTTP), rows written to one
`(namespace, database)` pair intermittently become visible when querying a
different `(namespace, database)` pair on the same server. Switching the client
transport from HTTP to WebSocket eliminates the leak. We suspect the HTTP
transport's per-request `Surreal-NS` / `Surreal-DB` header routing has a race
window in the session resolver when multiple distinct peers target distinct
`(ns, db)` pairs at the same moment.

## Environment

- SurrealDB: **3.0.4** (darwin-universal, Homebrew formula)
- `surrealdb` Python SDK: **1.0.8**, default blocking-HTTP backend
- macOS arm64 (Apple Silicon)
- Single-node server, in-memory / on-disk `surrealkv` store at `~/.mempalace/surreal`
- Server start:
  ```
  surreal start --user root --pass root --bind 127.0.0.1:8000 \
      surrealkv://$HOME/.mempalace/surreal
  ```

## Steps to reproduce

A minimal standalone repro (no third-party dependencies beyond `surrealdb`) is
included at [`surrealdb-upstream-repro.py`](./surrealdb-upstream-repro.py).

```
python docs/surrealdb-upstream-repro.py
```

What it does:

1. Generates three distinct `(namespace, database)` pairs, each with a unique
   id-prefix (`palace0`, `palace1`, `palace2`).
2. Spawns three **independent** writer processes (via `multiprocessing.spawn`).
   Each process creates its own `Surreal(URL)` client, signs in, runs
   `DEFINE NAMESPACE / USE NS / DEFINE DATABASE`, calls `conn.use(ns, db)`,
   and writes 40 `drawer:<prefix>-<i>` rows.
3. Spawns three independent reader processes. Each reads back its own
   `(ns, db)` and reports the ids it saw.
4. Checks each reader's id set for rows whose id-prefix belongs to a
   **different** palace.

Observations across 10 runs on our local environment:

| Transport | Runs with at least one foreign id visible |
|---|---|
| `http://127.0.0.1:8000` | 8 / 10 |
| `ws://127.0.0.1:8000`   | 0 / 10 |

Typical HTTP failure: reader for `(ns_B, db_Y)` returns rows tagged
`palace0-*` even though those were only ever written via `(ns_A, db_X)`.

## Expected behaviour

Rows written through one `(namespace, database)` pair must never be returned
when querying a different `(namespace, database)` pair, regardless of how
many concurrent clients are talking to the same server.

## Actual behaviour

Under the HTTP transport, concurrent writers to distinct `(ns, db)` pairs
cause cross-session row leakage: a reader for `(ns_B, db_Y)` intermittently
observes rows that were only written via `(ns_A, db_X)`. The leak is
**not deterministic** — run count matters — but is reliably reproducible
within ~10 iterations on a stock macOS install.

Our hypothesis is that the HTTP transport threads `Surreal-NS` /
`Surreal-DB` request headers through a session-resolution step that has a
race window when multiple peers hit the server near-simultaneously with
different target `(ns, db)` values: one peer's query executes against
another peer's bound session. We have not instrumented the server to
confirm, but the behaviour (and the fact that the WebSocket transport,
which binds `(ns, db)` at session setup rather than per-request, is
immune) is consistent with that model.

## Workaround

Use the WebSocket transport (`ws://host:port`) instead of HTTP. On the
WebSocket wire, `USE NS / DB` is bound to the underlying connection's
session, so there is no per-request header route to race on. We have
switched our production default to `ws://` and the leak is gone
(10/10 clean runs on the same repro).

## Related issues we found while investigating

The following open issues share an error class (concurrent multi-session
edge cases around NS/DB resolution or catalog DDL) but do not appear to
describe this exact failure mode:

- [#6681](https://github.com/surrealdb/surrealdb/issues/6681)
- [#7009](https://github.com/surrealdb/surrealdb/issues/7009)
- [#7071](https://github.com/surrealdb/surrealdb/issues/7071)
- [#7072](https://github.com/surrealdb/surrealdb/issues/7072)

Filing this separately because the leak surface is cross-namespace row
visibility rather than a transaction conflict or DDL race — a correctness
problem rather than a liveness one.
