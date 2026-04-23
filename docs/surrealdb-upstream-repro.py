"""Standalone repro: HTTP cross-session NS/DB routing race (SurrealDB 3.0.4).

Three writer processes, each with its own `surrealdb` SDK client connecting
over HTTP, concurrently write rows to three distinct ``(namespace, database)``
pairs. Three reader processes then query each ``(namespace, database)`` back.

On SurrealDB 3.0.4 + Python SDK 1.0.8 over ``http://127.0.0.1:8000``, rows
written via one ``(ns, db)`` occasionally surface when reading a different
``(ns, db)``. Switching the URL to ``ws://127.0.0.1:8000`` eliminates the
leak 10/10 runs.

Prereq: run SurrealDB locally with::

    surreal start --user root --pass root --bind 127.0.0.1:8000 \\
        surrealkv://$HOME/.mempalace/surreal

Then::

    python docs/surrealdb-upstream-repro.py

The script exits 0 either way and prints ``LEAK OBSERVED`` or
``no leak observed`` at the end. Flip ``URL`` to ``ws://127.0.0.1:8000``
to confirm the workaround.
"""

from __future__ import annotations

import multiprocessing as mp
import uuid

URL = "http://127.0.0.1:8000"  # swap to "ws://127.0.0.1:8000" for the workaround
USER = "root"
PASS = "root"
N_ROWS = 40
N_PALACES = 3


def writer(ns: str, db: str, prefix: str, q: "mp.Queue") -> None:
    from surrealdb import Surreal

    conn = Surreal(URL)
    conn.signin({"username": USER, "password": PASS})
    conn.query(f"DEFINE NAMESPACE IF NOT EXISTS {ns}; USE NS {ns}; DEFINE DATABASE IF NOT EXISTS {db};")
    conn.use(ns, db)
    conn.query("DEFINE TABLE IF NOT EXISTS drawer SCHEMALESS;")
    for i in range(N_ROWS):
        conn.query(
            "CREATE type::thing('drawer', $id) CONTENT { doc: $doc };",
            {"id": f"{prefix}-{i}", "doc": f"{prefix} doc {i}"},
        )
    q.put({"prefix": prefix, "ok": True})


def reader(ns: str, db: str, prefix: str, foreign_prefixes: list[str], q: "mp.Queue") -> None:
    from surrealdb import Surreal

    conn = Surreal(URL)
    conn.signin({"username": USER, "password": PASS})
    conn.use(ns, db)
    rows = conn.query("SELECT id FROM drawer;") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "result" in rows[0]:
        rows = rows[0]["result"] or []
    ids = [str(r.get("id")) for r in rows if isinstance(r, dict)]
    foreign = [i for i in ids if any(f"{fp}-" in i for fp in foreign_prefixes)]
    q.put({"prefix": prefix, "ns": ns, "db": db, "ids": ids, "foreign": foreign})


def main() -> None:
    ctx = mp.get_context("spawn")
    palaces = [
        (f"ns_{uuid.uuid4().hex[:10]}", f"db_{uuid.uuid4().hex[:10]}", f"palace{i}")
        for i in range(N_PALACES)
    ]

    q: "mp.Queue" = ctx.Queue()
    writers = [ctx.Process(target=writer, args=(ns, db, p, q)) for ns, db, p in palaces]
    for w in writers:
        w.start()
    for w in writers:
        w.join(timeout=60)

    q2: "mp.Queue" = ctx.Queue()
    readers = []
    for ns, db, p in palaces:
        foreign = [fp for _, _, fp in palaces if fp != p]
        readers.append(ctx.Process(target=reader, args=(ns, db, p, foreign, q2)))
    for r in readers:
        r.start()
    for r in readers:
        r.join(timeout=60)

    results = []
    while not q2.empty():
        results.append(q2.get_nowait())

    leaks = [r for r in results if r.get("foreign")]
    print(f"URL={URL}")
    for r in results:
        print(f"  palace={r['prefix']} ns={r['ns']} rows={len(r['ids'])} foreign={r['foreign']}")
    if leaks:
        print(f"LEAK OBSERVED: {len(leaks)}/{N_PALACES} palaces saw foreign ids")
    else:
        print("no leak observed")


if __name__ == "__main__":
    main()
