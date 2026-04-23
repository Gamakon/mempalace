#!/usr/bin/env python3
"""Two-session concurrent MCP smoke test against the live Surreal palace (mp-rta).

Proves the original problem is solved: two concurrent Claude Code sessions
= two MCP server processes. Under ChromaDB, the second process failed the
moment it tried to acquire the SQLite file lock. Under SurrealDB,
both processes should complete every read+write operation without contention.

Target palace
-------------

This script points both children at the REAL migrated palace in
``ns=mempalace``, ``db=palace_2a6be670`` (156,811 drawers + 61 closets).
Mapping is done via :class:`PalaceRef(namespace="palace_2a6be670")` — the
same override used by :mod:`mempalace.verify_migration` — rather than the
mcp_server default (which hashes ``_config.palace_path`` into a fresh,
empty DB).

Safety
------

All writes land in test-only wings (``_mprta_test_A`` / ``_mprta_test_B``)
and test-only KG subjects (``_mprta_test_subj_*``). The parent process
tears every test drawer down via ``tool_delete_drawer`` and invalidates
every test triple via ``tool_kg_invalidate`` at the end. Any cleanup
failures are reported, not silently swallowed.

The real user drawers/triples are never touched.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config — the real palace the MCP server maps to on this machine.
# ---------------------------------------------------------------------------

PALACE_PATH = os.environ.get(
    "MP_RTA_PALACE_PATH", os.path.expanduser("~/.mempalace/palace")
)
SURREAL_NAMESPACE = os.environ.get("MP_RTA_SURREAL_NS", "mempalace")
# Default: the migrated 156k-drawer palace. Override with MP_RTA_SURREAL_DB
# to run against a smaller target — a large HNSW index can block writes for
# minutes while Surreal rebuilds it, which is scaling behaviour, not a
# concurrency bug. The point of this proof is to exercise two MCP
# processes against the same Surreal DB, not to measure HNSW build throughput.
SURREAL_DB = os.environ.get("MP_RTA_SURREAL_DB", "palace_2a6be670")
KG_DB = os.environ.get("MP_RTA_KG_DB", "kg")

TEST_WING_A = "mprta-test-a"
TEST_WING_B = "mprta-test-b"

SEARCH_QUERIES = [
    "python",
    "memory palace",
    "surrealdb",
    "chromadb",
    "drawer",
    "embedding",
    "knowledge graph",
    "entity",
    "family",
    "project",
    "code review",
    "test",
    "config",
    "hooks",
    "search",
    "wing",
    "room",
    "identity",
    "creative",
    "technical",
]

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------


def _surreal_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Child worker — MUST be module-level for spawn pickling.
# ---------------------------------------------------------------------------


def _bind_mcp_to_migrated_palace() -> None:
    """Override the mcp_server's cached PalaceRef AND ``palace.palace_ref_for``
    so both children target the migrated 156k-drawer palace
    (``palace_2a6be670``) rather than the hash-of-palace_path DB that the
    default derivation would produce (``mcp_2a6be67083f59677`` — empty).

    Two distinct routing paths need overriding:

    1. **Write tools** (``tool_add_drawer`` / ``tool_delete_drawer`` /
       ``tool_kg_*``): resolve via
       ``mcp_server._get_surreal_backend`` -> cached
       ``_surreal_palace_ref``.  We replace that cached PalaceRef with one
       whose ``namespace`` equals the migrated DB name — ``namespace`` on
       :class:`PalaceRef` takes priority over ``id`` in
       :func:`mempalace.backends.surreal._safe_db_name`.

    2. **Search tool** (``tool_search``): routes through
       :func:`mempalace.searcher.search_memories` ->
       :func:`mempalace.palace.get_collection` ->
       :func:`mempalace.palace.palace_ref_for`, which hashes ``palace_path``
       unconditionally and ignores our cache override.  Monkey-patch
       ``palace.palace_ref_for`` to return the migrated ref.

    KG writes (``tool_kg_add``) land in the separate ``mempalace:kg`` DB
    which ``KnowledgeGraphSurreal`` hits directly with hard-coded defaults;
    no override needed there.
    """
    from mempalace import mcp_server, palace as _palace_mod
    from mempalace.backends.base import PalaceRef

    migrated_ref = PalaceRef(
        id=SURREAL_DB,
        local_path=PALACE_PATH,
        namespace=SURREAL_DB,
    )

    # (1) Pre-seed the write-path cache so `_get_surreal_backend` returns
    # the migrated ref without running its sha256 derivation.
    mcp_server._surreal_palace_ref = migrated_ref
    mcp_server._surreal_palace_ref_path = PALACE_PATH
    mcp_server._collection_cache = None
    mcp_server._metadata_cache = None
    mcp_server._metadata_cache_time = 0

    # (2) Redirect the read-path derivation. The signature must match the
    # original (``palace_path: str -> PalaceRef``) because internal callers
    # pass the palace path unconditionally.
    def _patched_palace_ref_for(palace_path: str) -> PalaceRef:
        return PalaceRef(
            id=SURREAL_DB,
            local_path=palace_path,
            namespace=SURREAL_DB,
        )

    _palace_mod.palace_ref_for = _patched_palace_ref_for


def _child(
    *,
    label: str,
    wing: str,
    barrier_evt: Any,
    queue: Any,
    n_searches: int,
    n_adds: int,
    n_kg: int,
    n_status: int,
) -> None:
    """Run the operation mix inside its own process + own MCP server module."""
    # Force surreal before importing mcp_server.
    os.environ["MEMPALACE_BACKEND"] = "surreal"
    os.environ["MEMPALACE_PALACE_PATH"] = PALACE_PATH
    os.environ.setdefault("MEMPALACE_SURREAL_URL", "ws://127.0.0.1:8000")
    os.environ.setdefault("MEMPALACE_SURREAL_USER", "root")
    os.environ.setdefault("MEMPALACE_SURREAL_PASS", "root")
    os.environ.setdefault("MEMPALACE_SURREAL_NS", SURREAL_NAMESPACE)

    results = {
        "label": label,
        "wing": wing,
        "successes": 0,
        "errors": [],
        "drawer_ids_added": [],
        "kg_triples_added": [],  # list of (subject, predicate, object)
        "elapsed": 0.0,
    }

    try:
        from mempalace import mcp_server

        _bind_mcp_to_migrated_palace()

        # Wait for the barrier so both children launch their op mix
        # at (near) the same instant.
        barrier_evt.wait(timeout=30)
        t0 = time.time()

        # --- 20 x tool_search ---
        for i in range(n_searches):
            q = SEARCH_QUERIES[i % len(SEARCH_QUERIES)]
            try:
                res = mcp_server.tool_search(query=q, limit=3)
                if isinstance(res, dict) and "error" in res and "results" not in res:
                    results["errors"].append(f"search[{i}] q={q!r}: {res['error']}")
                else:
                    results["successes"] += 1
            except Exception as e:
                results["errors"].append(
                    f"search[{i}] q={q!r}: {e!r}\n{traceback.format_exc()}"
                )

        # --- 10 x tool_add_drawer ---
        run_id = uuid.uuid4().hex[:10]
        for i in range(n_adds):
            room = f"mprta_room_{i % 3}"
            content = (
                f"mp-rta concurrent proof drawer from {label} run={run_id} "
                f"idx={i} uuid={uuid.uuid4().hex}"
            )
            try:
                res = mcp_server.tool_add_drawer(
                    wing=wing, room=room, content=content, added_by=f"mprta_{label}"
                )
                if res.get("success"):
                    results["successes"] += 1
                    did = res.get("drawer_id")
                    if did and res.get("reason") != "already_exists":
                        results["drawer_ids_added"].append(did)
                else:
                    results["errors"].append(
                        f"add_drawer[{i}]: {res.get('error', res)!r}"
                    )
            except Exception as e:
                results["errors"].append(
                    f"add_drawer[{i}]: {e!r}\n{traceback.format_exc()}"
                )

        # --- 10 x tool_kg_add ---
        for i in range(n_kg):
            subj = f"_mprta_test_subj_{label}_{run_id}_{i}"
            pred = "mprta_proof_relates_to"
            obj = f"_mprta_test_obj_{label}_{run_id}_{i}"
            try:
                res = mcp_server.tool_kg_add(subject=subj, predicate=pred, object=obj)
                if res.get("success"):
                    results["successes"] += 1
                    results["kg_triples_added"].append((subj, pred, obj))
                else:
                    results["errors"].append(f"kg_add[{i}]: {res.get('error', res)!r}")
            except Exception as e:
                results["errors"].append(f"kg_add[{i}]: {e!r}\n{traceback.format_exc()}")

        # --- 10 x tool_status ---
        for i in range(n_status):
            try:
                res = mcp_server.tool_status()
                if isinstance(res, dict) and "error" in res:
                    results["errors"].append(f"status[{i}]: {res['error']}")
                else:
                    results["successes"] += 1
            except Exception as e:
                results["errors"].append(f"status[{i}]: {e!r}\n{traceback.format_exc()}")

        results["elapsed"] = time.time() - t0
    except Exception as e:
        results["errors"].append(f"FATAL: {e!r}\n{traceback.format_exc()}")
    finally:
        try:
            queue.put(results)
        except Exception:
            # best-effort — parent has a timeout
            pass


# ---------------------------------------------------------------------------
# Parent orchestration + verification
# ---------------------------------------------------------------------------


def _count_drawers_in_wing(backend, palace_ref, wing: str) -> int:
    col = backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=False
    )
    # Use backend.query via col.get with a where filter — safer than raw SQL.
    total = 0
    offset = 0
    while True:
        batch = col.get(include=["metadatas"], where={"wing": wing}, limit=1000, offset=offset)
        ids = batch.get("ids") or []
        if not ids:
            break
        total += len(ids)
        offset += len(ids)
        if len(ids) < 1000:
            break
    return total


def _kg_triple_count(kg) -> int:
    stats = kg.stats()
    return int(stats.get("total_triples") or stats.get("triples") or 0)


def _cleanup(
    *,
    mcp_server,
    all_added_drawer_ids: list[str],
    all_added_triples: list[tuple[str, str, str]],
) -> list[str]:
    """Best-effort teardown. Returns list of residual errors."""
    residue: list[str] = []
    for did in all_added_drawer_ids:
        try:
            res = mcp_server.tool_delete_drawer(drawer_id=did)
            if not res.get("success"):
                residue.append(f"delete_drawer {did}: {res.get('error')}")
        except Exception as e:
            residue.append(f"delete_drawer {did}: {e!r}")
    for subj, pred, obj in all_added_triples:
        try:
            res = mcp_server.tool_kg_invalidate(subject=subj, predicate=pred, object=obj)
            if not res.get("success"):
                residue.append(f"kg_invalidate {subj!r}/{pred!r}/{obj!r}: {res.get('error')}")
        except Exception as e:
            residue.append(f"kg_invalidate {subj!r}/{pred!r}/{obj!r}: {e!r}")
    return residue


def main() -> int:
    if not _surreal_reachable():
        print("FAIL: local SurrealDB not reachable on 127.0.0.1:8000", file=sys.stderr)
        return 2

    # Import backend for pre/post observations. This is the PARENT
    # process — it participates in the observation path but NOT in
    # the concurrent write path (the children are the writers).
    from mempalace.backends.base import PalaceRef
    from mempalace.backends.surreal import SurrealBackend
    from mempalace.kg_surreal import KnowledgeGraphSurreal

    parent_backend = SurrealBackend(
        url=os.environ.get("MEMPALACE_SURREAL_URL", "ws://127.0.0.1:8000"),
        username=os.environ.get("MEMPALACE_SURREAL_USER", "root"),
        password=os.environ.get("MEMPALACE_SURREAL_PASS", "root"),
        namespace=SURREAL_NAMESPACE,
    )
    palace_ref = PalaceRef(id=SURREAL_DB, local_path=PALACE_PATH, namespace=SURREAL_DB)

    # Pre-state
    pre_col = parent_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=False
    )
    pre_drawer_total = pre_col.count()
    pre_wing_a = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_A)
    pre_wing_b = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_B)

    parent_kg = KnowledgeGraphSurreal(
        url=os.environ.get("MEMPALACE_SURREAL_URL", "ws://127.0.0.1:8000"),
        user=os.environ.get("MEMPALACE_SURREAL_USER", "root"),
        password=os.environ.get("MEMPALACE_SURREAL_PASS", "root"),
    )
    pre_kg_total = _kg_triple_count(parent_kg)

    print(f"[pre] total drawers in palace_2a6be670: {pre_drawer_total}")
    print(f"[pre] drawers in {TEST_WING_A}: {pre_wing_a}  {TEST_WING_B}: {pre_wing_b}")
    print(f"[pre] KG triples: {pre_kg_total}")

    # Spawn two children via multiprocessing.spawn — matches production
    # (MCP server processes do not share Python state).
    ctx = mp.get_context("spawn")
    q_a: mp.Queue = ctx.Queue()
    q_b: mp.Queue = ctx.Queue()
    barrier = ctx.Event()

    common = dict(n_searches=20, n_adds=10, n_kg=10, n_status=10)
    p_a = ctx.Process(
        target=_child,
        kwargs={"label": "A", "wing": TEST_WING_A, "barrier_evt": barrier, "queue": q_a, **common},
    )
    p_b = ctx.Process(
        target=_child,
        kwargs={"label": "B", "wing": TEST_WING_B, "barrier_evt": barrier, "queue": q_b, **common},
    )

    p_a.start()
    p_b.start()

    # Give both children a moment to import their modules and reach
    # ``barrier_evt.wait`` before firing the starting gun.
    time.sleep(2.5)
    t_launch = time.time()
    barrier.set()

    p_a.join(timeout=180)
    p_b.join(timeout=180)
    elapsed_total = time.time() - t_launch

    if p_a.is_alive():
        p_a.terminate()
        print("FAIL: child A did not exit within 180s", file=sys.stderr)
    if p_b.is_alive():
        p_b.terminate()
        print("FAIL: child B did not exit within 180s", file=sys.stderr)

    try:
        res_a = q_a.get(timeout=5)
    except Exception:
        res_a = {"label": "A", "successes": 0, "errors": ["no result from queue"]}
    try:
        res_b = q_b.get(timeout=5)
    except Exception:
        res_b = {"label": "B", "successes": 0, "errors": ["no result from queue"]}

    # Post-state
    post_col = parent_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=False
    )
    post_drawer_total = post_col.count()
    post_wing_a = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_A)
    post_wing_b = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_B)
    post_kg_total = _kg_triple_count(parent_kg)

    print()
    print("=== RESULT ===")
    print(f"Total wall time (post-barrier): {elapsed_total:.2f}s")
    for r in (res_a, res_b):
        print(
            f"[child {r.get('label')}] successes={r.get('successes')}/50"
            f"  elapsed={r.get('elapsed', 0):.2f}s"
            f"  added_drawers={len(r.get('drawer_ids_added', []))}"
            f"  added_triples={len(r.get('kg_triples_added', []))}"
            f"  errors={len(r.get('errors', []))}"
        )
        for err in r.get("errors", []):
            print(f"  ! {err}")

    delta_drawers = post_drawer_total - pre_drawer_total
    delta_kg = post_kg_total - pre_kg_total
    print()
    print(f"[post] total drawers: {post_drawer_total}  (Δ = {delta_drawers:+d})")
    print(
        f"[post] drawers in {TEST_WING_A}: {post_wing_a} (Δ {post_wing_a - pre_wing_a:+d});"
        f" {TEST_WING_B}: {post_wing_b} (Δ {post_wing_b - pre_wing_b:+d})"
    )
    print(f"[post] KG triples: {post_kg_total} (Δ {delta_kg:+d})")

    # Assemble the cleanup list BEFORE declaring pass/fail so we always
    # try to clean up even when the assertions trip.
    from mempalace import mcp_server as parent_mcp

    os.environ["MEMPALACE_BACKEND"] = "surreal"
    os.environ["MEMPALACE_PALACE_PATH"] = PALACE_PATH
    # Rebind parent's mcp_server to the migrated palace so cleanup tools
    # hit the right DB.
    parent_mcp._surreal_palace_ref = PalaceRef(
        id=SURREAL_DB, local_path=PALACE_PATH, namespace=SURREAL_DB
    )
    parent_mcp._surreal_palace_ref_path = PALACE_PATH
    parent_mcp._collection_cache = None

    all_ids = list(res_a.get("drawer_ids_added", [])) + list(res_b.get("drawer_ids_added", []))
    all_triples = list(res_a.get("kg_triples_added", [])) + list(
        res_b.get("kg_triples_added", [])
    )
    residue = _cleanup(
        mcp_server=parent_mcp,
        all_added_drawer_ids=all_ids,
        all_added_triples=all_triples,
    )

    # Re-check wing counts after cleanup to make sure we actually
    # dropped our test writes.
    final_wing_a = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_A)
    final_wing_b = _count_drawers_in_wing(parent_backend, palace_ref, TEST_WING_B)
    print()
    print(
        f"[cleanup] final drawers in {TEST_WING_A}: {final_wing_a} "
        f"(expected {pre_wing_a});  {TEST_WING_B}: {final_wing_b} "
        f"(expected {pre_wing_b})"
    )
    if residue:
        print("[cleanup] RESIDUE:")
        for r in residue:
            print(f"  ! {r}")
    else:
        print("[cleanup] clean")

    # ---- assertions ----
    passed = True
    reasons: list[str] = []

    if res_a.get("successes", 0) != 50:
        passed = False
        reasons.append(f"child A successes={res_a.get('successes')} != 50")
    if res_b.get("successes", 0) != 50:
        passed = False
        reasons.append(f"child B successes={res_b.get('successes')} != 50")

    # Lock-error scan — the exact failure mode we're proving away.
    lock_patterns = (
        "database is locked",
        "Transaction conflict",
        "file is locked",
        "sqlite3.OperationalError",
        "could not acquire",
    )
    for r in (res_a, res_b):
        for err in r.get("errors", []):
            for pat in lock_patterns:
                if pat.lower() in err.lower():
                    passed = False
                    reasons.append(f"child {r.get('label')} saw lock-like error: {err[:200]}")
                    break

    if delta_drawers != 20:
        passed = False
        reasons.append(f"delta drawer count = {delta_drawers}, expected 20")

    if delta_kg < 20:
        passed = False
        reasons.append(f"delta KG triple count = {delta_kg}, expected >= 20")

    if final_wing_a != pre_wing_a or final_wing_b != pre_wing_b:
        passed = False
        reasons.append(
            f"cleanup did not restore test wings: "
            f"{TEST_WING_A} {pre_wing_a}->{final_wing_a}, "
            f"{TEST_WING_B} {pre_wing_b}->{final_wing_b}"
        )

    if residue:
        passed = False
        reasons.append(f"{len(residue)} cleanup errors")

    print()
    if passed:
        print("PASS: two concurrent MCP processes completed 100/100 ops without lock errors.")
        return 0
    else:
        print("FAIL:")
        for r in reasons:
            print(f"  - {r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
