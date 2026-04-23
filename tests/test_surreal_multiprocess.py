"""Real multi-process concurrency tests for the SurrealDB backend (mp-d89).

These tests are the PROOF POINT for the whole Surreal refactor. The
original problem that killed the MCP server in earlier sessions was
"two Claude Code sessions -> two processes -> ChromaDB locks." If these
tests pass against Surreal, we have solved it.

Unlike :mod:`tests.test_surreal_backend` — which uses ``threading`` for
its concurrency checks — this module uses :mod:`multiprocessing` and
spawns **real child processes**. Each child creates its own
:class:`SurrealBackend` instance (and therefore its own HTTP connection
to ``127.0.0.1:8000``) so there is no shared Python state between
writers. That is the precise shape of the production failure mode.

Each test uses a unique Surreal namespace (and a unique palace id within
it) to isolate. Child processes return a status dict via a
``multiprocessing.Queue`` so the parent can assert both on the per-child
outcome and on the post-exit state of the database.

If the local Surreal instance is not running the whole module is
skipped, matching :mod:`tests.test_surreal_backend`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
import uuid

import pytest

from mempalace.backends import (
    DimensionMismatchError,
    PalaceRef,
)


# ---------------------------------------------------------------------------
# Skip gate — same pattern as test_surreal_backend.py.
# ---------------------------------------------------------------------------


def _surreal_reachable() -> bool:
    host, port = "127.0.0.1", 8000
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _surreal_reachable(),
    reason="local SurrealDB not running on 127.0.0.1:8000 (see docs/surrealdb-local.md)",
)


# ---------------------------------------------------------------------------
# Child-process workers.
#
# These MUST be module-level functions so :mod:`multiprocessing` can
# pickle them. Closures or lambdas will raise ``PicklingError`` on the
# ``spawn`` start method (which is the default on macOS, matching the
# CI environment this project targets).
# ---------------------------------------------------------------------------


def _worker_add_drawers(
    *,
    namespace: str,
    palace_id: str,
    id_prefix: str,
    count: int,
    queue: "mp.Queue",
) -> None:
    """Child: open an independent backend and add ``count`` drawers.

    Each child uses id_ext values like ``{id_prefix}-{i}`` so two
    workers writing to the same palace never collide on an id. The
    parent process then asserts both worker's ids are present.
    """
    try:
        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            docs = [f"{id_prefix} doc {i}" for i in range(count)]
            ids = [f"{id_prefix}-{i}" for i in range(count)]
            # No embeddings on purpose — this test is about write
            # durability, not about the HNSW path.
            col.add(documents=docs, ids=ids)
            queue.put({"ok": True, "prefix": id_prefix, "count": count})
        finally:
            backend.close()
    except Exception as e:  # surface the whole traceback to the parent
        import traceback

        queue.put({"ok": False, "prefix": id_prefix, "error": f"{e!r}\n{traceback.format_exc()}"})


def _worker_update_metadata_key(
    *,
    namespace: str,
    palace_id: str,
    row_id: str,
    key: str,
    value: str,
    queue: "mp.Queue",
) -> None:
    """Child: open an independent backend and add/patch one metadata key.

    The row is seeded by the parent before the children run. Each child
    issues an ``update(metadatas=[{key: value}])`` targeting the same
    id. With atomic server-side MERGE (mp-93e) both keys survive.
    """
    try:
        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            col.update(ids=[row_id], metadatas=[{key: value}])
            queue.put({"ok": True, "key": key})
        finally:
            backend.close()
    except Exception as e:
        import traceback

        queue.put({"ok": False, "key": key, "error": f"{e!r}\n{traceback.format_exc()}"})


def _worker_first_vector_write(
    *,
    namespace: str,
    palace_id: str,
    tag: str,
    dim: int,
    queue: "mp.Queue",
) -> None:
    """Child: race to be the first writer to lock ``embedding_dim``.

    Two children call this with different dims. Exactly one must
    succeed; the loser must observe the winner's dim and raise
    :class:`DimensionMismatchError` (mp-hlo).
    """
    try:
        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            col.add(
                documents=[f"doc-{tag}"],
                ids=[f"id-{tag}"],
                embeddings=[[0.1] * dim],
            )
            queue.put({"ok": True, "tag": tag, "dim": dim})
        finally:
            backend.close()
    except DimensionMismatchError as e:
        queue.put(
            {
                "ok": False,
                "tag": tag,
                "dim": dim,
                "error_type": "DimensionMismatchError",
                "error": str(e),
            }
        )
    except Exception as e:
        import traceback

        queue.put(
            {
                "ok": False,
                "tag": tag,
                "dim": dim,
                "error_type": type(e).__name__,
                "error": f"{e!r}\n{traceback.format_exc()}",
            }
        )


def _worker_writer_burst(
    *,
    namespace: str,
    palace_id: str,
    id_prefix: str,
    count: int,
    queue: "mp.Queue",
) -> None:
    """Child: steadily adds rows while the other child reads.

    Used by the search-vs-write interference test. We stagger each
    write with a tiny yield so the reader gets real "concurrent"
    conditions rather than a batch-at-once write followed by quiet
    reads.
    """
    try:
        import time

        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            for i in range(count):
                col.add(
                    documents=[f"{id_prefix} interleaved write {i}"],
                    ids=[f"{id_prefix}-{i}"],
                    embeddings=[[float(i % 7) / 7.0, 0.1, 0.2]],
                )
                # Yield so the reader actually interleaves. Not a
                # correctness requirement — just shape.
                time.sleep(0.005)
            queue.put({"ok": True, "count": count})
        finally:
            backend.close()
    except Exception as e:
        import traceback

        queue.put({"ok": False, "error": f"{e!r}\n{traceback.format_exc()}"})


def _worker_reader_burst(
    *,
    namespace: str,
    palace_id: str,
    iterations: int,
    queue: "mp.Queue",
) -> None:
    """Child: repeatedly queries while the writer inserts.

    Asserts no query raises and every result has a consistent typed
    shape. Partial reads (rows with missing fields) would blow up the
    ``_split_rows`` path — this worker is the net that catches those.
    """
    try:
        import time

        from mempalace.backends.surreal import SurrealBackend
        from mempalace.backends import QueryResult

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            # The reader must tolerate the HNSW index not yet existing
            # on the first iterations — ``create=True`` is safe and
            # idempotent.
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            crashes: list[str] = []
            observed_counts: list[int] = []
            for _ in range(iterations):
                try:
                    r = col.query(
                        query_embeddings=[[1.0, 0.0, 0.0]],
                        n_results=5,
                    )
                    # Type check — partial reads would land here.
                    if not isinstance(r, QueryResult):
                        crashes.append(f"unexpected type {type(r).__name__}")
                        continue
                    if len(r.ids) != 1:
                        crashes.append(f"outer dim {len(r.ids)} != 1")
                        continue
                    observed_counts.append(len(r.ids[0]))
                except Exception as e:
                    crashes.append(f"{type(e).__name__}: {e}")
                time.sleep(0.003)
            queue.put(
                {
                    "ok": not crashes,
                    "crashes": crashes,
                    "observed_counts": observed_counts,
                }
            )
        finally:
            backend.close()
    except Exception as e:
        import traceback

        queue.put({"ok": False, "crashes": [f"{e!r}\n{traceback.format_exc()}"]})


def _worker_bootstrap_loop(
    *,
    namespace: str,
    palace_id: str,
    queue: "mp.Queue",
) -> None:
    """Child that runs get_collection(create=True) in a tight loop.

    Used by the SIGKILL chaos test (mp-33y Task 5c). The parent kills
    this child at a random moment so some iterations are guaranteed to
    be cut off mid-bootstrap. The queue is mainly a liveness signal —
    if the child ever posts before being killed, we know it reached
    steady state.
    """
    try:
        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            # Announce readiness BEFORE the first bootstrap so the parent
            # knows the child is alive. Then loop — we expect to be
            # SIGKILLed mid-iteration.
            queue.put({"phase": "alive"})
            while True:
                backend.get_collection(
                    palace=palace, collection_name="mempalace_drawers", create=True
                )
        finally:
            backend.close()
    except Exception as e:  # pragma: no cover - child is killed
        import traceback

        try:
            queue.put({"phase": "error", "error": f"{e!r}\n{traceback.format_exc()}"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _spawn_ctx() -> "mp.context.BaseContext":
    """Use the ``spawn`` start method.

    macOS defaults to ``spawn`` already, but ``fork`` is still available
    and would inherit parent-process state (including any SDK globals),
    which is precisely what we are trying to avoid. Forcing ``spawn``
    keeps the test semantics identical on Linux CI too.
    """
    return mp.get_context("spawn")


def _run_children(targets: list[dict]) -> list[dict]:
    """Spawn all children in parallel, join, and return their result dicts.

    Each entry in ``targets`` is ``{"target": fn, "kwargs": {...}}``.
    A per-test ``Queue`` is created here so the worker does not share a
    queue across tests.
    """
    ctx = _spawn_ctx()
    q: "mp.Queue" = ctx.Queue()
    procs = []
    for t in targets:
        kwargs = dict(t["kwargs"])
        kwargs["queue"] = q
        p = ctx.Process(target=t["target"], kwargs=kwargs)
        p.start()
        procs.append(p)
    for p in procs:
        # Generous timeout — Surreal+HNSW bootstrap can take a few
        # seconds cold. If a child hangs past this we've hit a real
        # bug and want the test to surface it loudly.
        p.join(timeout=60)
        assert not p.is_alive(), f"child {p.pid} did not exit within 60s"

    results: list[dict] = []
    while not q.empty():
        results.append(q.get_nowait())
    return results


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_namespace():
    """Yield a unique Surreal namespace; drop it after the test.

    Each test gets its own namespace so parallel runs of this module do
    not collide. We clean up via a parent-process connection after the
    children have exited.

    Uniqueness hardening (mp-85q): the namespace suffix combines a full
    uuid4 hex with a nanosecond timestamp so two tests — even on the same
    machine, same clock-second, and colliding uuid prefixes (astronomically
    unlikely but the original ``[:12]`` truncation narrowed the space) —
    can never reuse the same name. Surreal 3.0.4's ``REMOVE NAMESPACE``
    returns before the storage engine has necessarily reclaimed every key,
    so recycling a name before the server has flushed would let a stale
    row appear in what looks like a fresh palace.
    """
    ns = f"mp_mp_{uuid.uuid4().hex}_{time.time_ns()}"
    yield ns
    _drop_namespace_and_verify(ns)


def _drop_namespace_and_verify(ns: str, *, poll_deadline_s: float = 5.0) -> None:
    """Remove ``ns`` and poll ``INFO FOR KV`` until it is really gone.

    mp-85q fixture hardening. ``REMOVE NAMESPACE`` returns almost
    immediately on Surreal 3.0.4 but the underlying LSM / WAL flush is
    asynchronous, so a subsequent test that happens to pick a colliding
    db-name can race the tail of the previous test's cleanup. We poll
    ``INFO FOR KV`` after issuing REMOVE so that, by the time the next
    test starts, the server has actually acknowledged the drop.

    Cleanup failures must not mask the test result, so any exception
    from the REMOVE itself is swallowed — but the post-REMOVE poll is
    best-effort and gives up quietly after ``poll_deadline_s`` rather
    than blocking indefinitely.
    """
    from mempalace.backends.surreal import SurrealBackend

    try:
        backend = SurrealBackend(namespace=ns)
    except Exception:
        return
    try:
        try:
            conn = backend._connect("cleanup_dummy")
            conn.query(f"REMOVE NAMESPACE IF EXISTS {ns};")
        except Exception:
            # REMOVE itself failed — best-effort, nothing else we can do.
            return

        # Poll INFO FOR KV to confirm the namespace is truly gone before
        # the next test starts. This guards against Surreal's async
        # reclaim briefly leaving stale keys visible under the same
        # (ns, db) pair a later test might reuse.
        deadline = time.monotonic() + poll_deadline_s
        while time.monotonic() < deadline:
            try:
                info = conn.query("INFO FOR KV")
            except Exception:
                break
            if isinstance(info, list):
                info = info[0] if info else {}
            namespaces = info.get("namespaces", {}) if isinstance(info, dict) else {}
            if ns not in namespaces:
                return
            time.sleep(0.05)
    finally:
        backend.close()


@pytest.fixture()
def palace_id() -> str:
    return f"palace_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# 1. Two processes, concurrent distinct adds — no lost writes.
# ---------------------------------------------------------------------------


def test_two_processes_concurrent_adds_no_loss(isolated_namespace, palace_id):
    """Both children add 50 distinct drawers; parent asserts all 100 land."""
    n_each = 50
    children = [
        {
            "target": _worker_add_drawers,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "id_prefix": "procA",
                "count": n_each,
            },
        },
        {
            "target": _worker_add_drawers,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "id_prefix": "procB",
                "count": n_each,
            },
        },
    ]
    results = _run_children(children)
    assert len(results) == 2, f"expected 2 child results, got {results}"
    for r in results:
        assert r["ok"], f"child {r.get('prefix')!r} failed: {r.get('error')}"

    # Parent-side verification via a fresh backend. Proves the writes
    # are durable on the server, not just cached in the child.
    from mempalace.backends.surreal import SurrealBackend

    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        total = col.count()
        assert total == 2 * n_each, f"expected {2 * n_each} rows, got {total}"
        # Spot-check that both prefix sets survived intact.
        got = col.get()
        ids_set = set(got.ids)
        expected_a = {f"procA-{i}" for i in range(n_each)}
        expected_b = {f"procB-{i}" for i in range(n_each)}
        missing_a = expected_a - ids_set
        missing_b = expected_b - ids_set
        assert not missing_a, f"lost procA ids: {sorted(missing_a)[:5]}..."
        assert not missing_b, f"lost procB ids: {sorted(missing_b)[:5]}..."
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 2. Two processes, concurrent MERGE updates on the same id.
# ---------------------------------------------------------------------------


def test_two_processes_concurrent_updates_merge_correctly(isolated_namespace, palace_id):
    """Same row, disjoint keys from two processes; both keys survive."""
    from mempalace.backends.surreal import SurrealBackend

    # Seed the row from the parent so both children only ever update.
    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        col.add(documents=["seed"], ids=["shared-row"], metadatas=[{"seed": True}])
    finally:
        backend.close()

    children = [
        {
            "target": _worker_update_metadata_key,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "row_id": "shared-row",
                "key": "from_a",
                "value": "alpha",
            },
        },
        {
            "target": _worker_update_metadata_key,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "row_id": "shared-row",
                "key": "from_b",
                "value": "bravo",
            },
        },
    ]
    results = _run_children(children)
    assert len(results) == 2, f"expected 2 child results, got {results}"
    for r in results:
        assert r["ok"], f"child for key {r.get('key')!r} failed: {r.get('error')}"

    # Parent-side read: both keys must be present AND the seed key
    # must still survive (no full-object overwrite).
    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        got = col.get(ids=["shared-row"])
        assert got.metadatas, f"row disappeared: {got!r}"
        meta = got.metadatas[0]
        assert meta.get("from_a") == "alpha", f"lost process A's key: {meta!r}"
        assert meta.get("from_b") == "bravo", f"lost process B's key: {meta!r}"
        assert meta.get("seed") is True, f"lost seed key: {meta!r}"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 3. Two processes racing the embedding_dim lock (mp-hlo).
# ---------------------------------------------------------------------------


def test_two_processes_dim_lock_race(isolated_namespace, palace_id):
    """Fresh palace; one proc writes dim-384, the other dim-768. One must fail."""
    children = [
        {
            "target": _worker_first_vector_write,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "tag": "a",
                "dim": 384,
            },
        },
        {
            "target": _worker_first_vector_write,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "tag": "b",
                "dim": 768,
            },
        },
    ]
    results = _run_children(children)
    assert len(results) == 2, f"expected 2 child results, got {results}"

    successes = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]
    assert len(successes) == 1, (
        f"exactly one child should succeed; got successes={successes} failures={failures}"
    )
    assert len(failures) == 1, (
        f"exactly one child should fail with DimensionMismatchError; "
        f"got successes={successes} failures={failures}"
    )
    failed = failures[0]
    assert failed.get("error_type") == "DimensionMismatchError", (
        f"loser raised wrong exception type: {failed!r}"
    )

    # Confirm the winner's dim is durably locked: a fresh parent-side
    # write at the loser's dim must still be rejected.
    winner_dim = successes[0]["dim"]
    loser_dim = 768 if winner_dim == 384 else 384
    from mempalace.backends.surreal import SurrealBackend

    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        locked = col._expected_embedding_dim()
        assert locked == winner_dim, f"locked dim {locked} != winner {winner_dim}"
        with pytest.raises(DimensionMismatchError):
            col.add(
                documents=["late"],
                ids=["late"],
                embeddings=[[0.2] * loser_dim],
            )
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 4. Search vs. write — no interference, consistent reads.
# ---------------------------------------------------------------------------


def test_two_processes_concurrent_search_no_interference(isolated_namespace, palace_id):
    """One writer, one reader — reads must never crash, shapes stay typed."""
    # Seed with one vector so the HNSW index exists before the reader
    # starts. Without this the reader's first query races the index
    # DDL — legitimate, but tested elsewhere (mp-hlo). This test is
    # about steady-state read/write interleaving.
    from mempalace.backends.surreal import SurrealBackend

    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        col.add(
            documents=["seed"],
            ids=["seed"],
            embeddings=[[0.1, 0.2, 0.3]],
        )
    finally:
        backend.close()

    children = [
        {
            "target": _worker_writer_burst,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "id_prefix": "live",
                "count": 40,
            },
        },
        {
            "target": _worker_reader_burst,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "iterations": 40,
            },
        },
    ]
    results = _run_children(children)
    assert len(results) == 2, f"expected 2 child results, got {results}"

    # Writer must have completed cleanly.
    writer_r = next((r for r in results if r.get("count") is not None), None)
    reader_r = next((r for r in results if r.get("crashes") is not None), None)
    assert writer_r is not None and reader_r is not None, f"results={results}"
    assert writer_r["ok"], f"writer failed: {writer_r}"

    # Reader must have issued every query without an exception.
    assert reader_r["ok"], f"reader observed crashes: {reader_r['crashes']}"
    # And it must have actually observed rows at some point (otherwise
    # the test would pass trivially even against a broken reader).
    assert any(c > 0 for c in reader_r["observed_counts"]), (
        f"reader never saw any rows across {len(reader_r['observed_counts'])} "
        f"queries; counts={reader_r['observed_counts'][:10]}..."
    )

    # Final state: all writer rows landed.
    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        # 40 live rows + 1 seed. Assert by id-set rather than raw count so
        # if the backend (or Surreal itself) ever materialises a phantom
        # duplicate the failure message names the culprit.
        got = col.get()
        expected_ids = {"seed"} | {f"live-{i}" for i in range(40)}
        missing = expected_ids - set(got.ids)
        extra = set(got.ids) - expected_ids
        assert not missing, f"missing rows after concurrent run: {sorted(missing)}"
        assert not extra, f"unexpected rows after concurrent run: {sorted(extra)}"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 4b. Three palaces in three namespaces, concurrent writes + reads — no
#     cross-palace row leakage (mp-85q regression).
#
# Opens THREE palaces (distinct namespaces), writes N drawers to each from
# three separate writer processes, then reads each palace from a fourth
# process and asserts that no palace sees any row written to the other two.
#
# This is the explicit regression test for mp-85q: under SurrealDB 3.0.4's
# HTTP wire, concurrent writes from three processes to three distinct
# (ns, db) pairs caused rows to bleed across namespaces — proven to be a
# server-side bug in the HTTP handler's session-resolution. The fix was
# to switch the backend default to WebSocket (``ws://``) which binds NS/DB
# at session level and is not subject to the race. This test guards that
# fix: a regression (env override back to HTTP, or a future SDK change
# that re-enters the HTTP path) will show up as foreign ids leaking
# between palaces, caught by the per-palace id-set assertion below.
# ---------------------------------------------------------------------------


def _worker_reader_check_palace(
    *,
    namespace: str,
    palace_id: str,
    expected_ids: list[str],
    foreign_ids: list[str],
    queue: "mp.Queue",
) -> None:
    """Child: open one palace and return the id-set it sees.

    The parent then asserts the id-set matches only ``expected_ids`` and
    contains none of ``foreign_ids`` (ids written to OTHER palaces in the
    same test run). ``foreign_ids`` is passed purely so the child can
    emit a richer failure message if it trips the guard.
    """
    try:
        from mempalace.backends.surreal import SurrealBackend

        backend = SurrealBackend(namespace=namespace)
        try:
            palace = PalaceRef(id=palace_id)
            col = backend.get_collection(
                palace=palace, collection_name="mempalace_drawers", create=True
            )
            got = col.get()
            observed = set(got.ids)
            queue.put(
                {
                    "ok": True,
                    "namespace": namespace,
                    "palace_id": palace_id,
                    "observed": sorted(observed),
                    "expected_missing": sorted(set(expected_ids) - observed),
                    "foreign_contamination": sorted(observed & set(foreign_ids)),
                }
            )
        finally:
            backend.close()
    except Exception as e:
        import traceback

        queue.put(
            {
                "ok": False,
                "namespace": namespace,
                "palace_id": palace_id,
                "error": f"{e!r}\n{traceback.format_exc()}",
            }
        )


def test_three_palaces_no_cross_namespace_leakage():
    """Three concurrent writers, three distinct palaces — each stays pure.

    Regression for mp-85q: if the backend ever routes back through the
    HTTP wire (via ``MEMPALACE_SURREAL_URL`` override, for instance),
    one palace will observe ids written to another. We don't just
    assert row counts — we assert the exact id-set each palace sees,
    so any single cross-namespace row surfaces a clear failure.
    """
    n_each = 40

    palaces: list[tuple[str, str, str]] = []
    for i in range(3):
        ns = f"mp_mp_{uuid.uuid4().hex}_{time.time_ns()}_{i}"
        pid = f"palace_{uuid.uuid4().hex[:10]}"
        prefix = f"palace{i}"
        palaces.append((ns, pid, prefix))

    # Writers: one child per palace, each writes ``n_each`` drawers with
    # a palace-specific id prefix.
    writer_targets = [
        {
            "target": _worker_add_drawers,
            "kwargs": {
                "namespace": ns,
                "palace_id": pid,
                "id_prefix": prefix,
                "count": n_each,
            },
        }
        for (ns, pid, prefix) in palaces
    ]

    try:
        writer_results = _run_children(writer_targets)
        assert len(writer_results) == 3, f"expected 3 writer results, got {writer_results}"
        for r in writer_results:
            assert r["ok"], f"writer {r.get('prefix')!r} failed: {r.get('error')}"

        # Readers: one child per palace. Each reader must see ONLY the
        # ids written to its own palace, never the other two palaces'
        # ids.
        per_palace_expected = {
            prefix: [f"{prefix}-{i}" for i in range(n_each)] for (_ns, _pid, prefix) in palaces
        }
        reader_targets = []
        for ns, pid, prefix in palaces:
            foreign_prefixes = [p for (_n, _p, p) in palaces if p != prefix]
            foreign_ids = [fid for fp in foreign_prefixes for fid in per_palace_expected[fp]]
            reader_targets.append(
                {
                    "target": _worker_reader_check_palace,
                    "kwargs": {
                        "namespace": ns,
                        "palace_id": pid,
                        "expected_ids": per_palace_expected[prefix],
                        "foreign_ids": foreign_ids,
                    },
                }
            )
        reader_results = _run_children(reader_targets)
        assert len(reader_results) == 3, f"expected 3 reader results, got {reader_results}"

        for r in reader_results:
            assert r["ok"], (
                f"reader for palace {r.get('palace_id')!r} "
                f"in ns {r.get('namespace')!r} failed: {r.get('error')}"
            )
            # No foreign contamination: if this trips, mp-85q has regressed.
            assert not r["foreign_contamination"], (
                f"palace {r['palace_id']!r} in ns {r['namespace']!r} "
                f"saw foreign ids {r['foreign_contamination']!r} — "
                "cross-namespace leakage (mp-85q)"
            )
            # All expected rows present.
            assert not r["expected_missing"], (
                f"palace {r['palace_id']!r} in ns {r['namespace']!r} "
                f"missing expected ids {r['expected_missing']!r}"
            )
    finally:
        # Drop every namespace we created, even if the assertions
        # above fail — we must never leave debris behind.
        for ns, _pid, _prefix in palaces:
            _drop_namespace_and_verify(ns)


# ---------------------------------------------------------------------------
# 5. Four processes, 100 drawers each — hard stress (mp-33y Task 5b).
# ---------------------------------------------------------------------------


def test_four_processes_stress_no_loss(isolated_namespace, palace_id):
    """Four children each add 100 drawers to a fresh palace. All 400 must land."""
    n_each = 100
    n_procs = 4
    children = [
        {
            "target": _worker_add_drawers,
            "kwargs": {
                "namespace": isolated_namespace,
                "palace_id": palace_id,
                "id_prefix": f"p{pid}",
                "count": n_each,
            },
        }
        for pid in range(n_procs)
    ]
    results = _run_children(children)
    assert len(results) == n_procs, f"expected {n_procs} child results, got {results}"
    for r in results:
        assert r["ok"], f"child {r.get('prefix')!r} failed: {r.get('error')}"

    from mempalace.backends.surreal import SurrealBackend

    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        total = col.count()
        assert total == n_procs * n_each, f"expected {n_procs * n_each} rows, got {total}"
        got = col.get()
        ids_set = set(got.ids)
        for pid in range(n_procs):
            expected = {f"p{pid}-{i}" for i in range(n_each)}
            missing = expected - ids_set
            assert not missing, f"lost p{pid} ids: {sorted(missing)[:5]}..."
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 6. Chaos: SIGKILL a bootstrapping peer, next peer must recover (mp-33y 5c).
# ---------------------------------------------------------------------------


def test_sigkilled_bootstrap_peer_recovers(isolated_namespace, palace_id, chaos_seed):
    """Kill child A mid-bootstrap; child B must still converge cleanly.

    Exercises the partial-bootstrap recovery path in ``get_collection``
    (mp-33y Task 4): if the SIGKILL lands after the DEFINE TABLE calls
    but before ``palace_meta:main`` is UPSERTed, child B must re-run
    bootstrap rather than treating the tables-exist observation as
    "ready".

    Chaos replay (mp-ou9): every timing decision in this test derives
    from a seeded ``random.Random`` instance. If the test flakes on CI,
    re-run with ``pytest ... --chaos-seed N`` (N is printed in the
    failure message and at session start) to reproduce the exact timing.
    """
    import random as _random
    import signal
    import time

    # Local Random instance — never touch the global random state so
    # other tests stay independent of this seed.
    rng = _random.Random(chaos_seed)
    sigkill_delay = rng.uniform(0.0, 0.2)
    # Include the seed and computed delay in a banner printed to stdout
    # so CI logs carry the replay recipe on both success and failure.
    print(
        f"[chaos-seed] test_sigkilled_bootstrap_peer_recovers "
        f"seed={chaos_seed} sigkill_delay={sigkill_delay:.6f}s"
    )

    ctx = _spawn_ctx()
    q_a: "mp.Queue" = ctx.Queue()
    child_a = ctx.Process(
        target=_worker_bootstrap_loop,
        kwargs={
            "namespace": isolated_namespace,
            "palace_id": palace_id,
            "queue": q_a,
        },
    )
    child_a.start()

    # Let the child start but kill it at a random moment within the first
    # 200ms. Some runs will land the SIGKILL mid-bootstrap, some after the
    # first successful bootstrap loop iteration — both are valid chaos.
    time.sleep(sigkill_delay)
    try:
        os.kill(child_a.pid, signal.SIGKILL)
    except ProcessLookupError:  # pragma: no cover - child exited already
        pass
    child_a.join(timeout=5)
    assert not child_a.is_alive(), (
        f"child {child_a.pid} did not die from SIGKILL "
        f"(replay with --chaos-seed {chaos_seed}, "
        f"sigkill_delay={sigkill_delay:.6f}s)"
    )

    # Child B: open the same palace, add one drawer, read it back. This
    # MUST succeed, regardless of what state child A left the catalog in.
    q_b: "mp.Queue" = ctx.Queue()
    child_b = ctx.Process(
        target=_worker_add_drawers,
        kwargs={
            "namespace": isolated_namespace,
            "palace_id": palace_id,
            "id_prefix": "survivor",
            "count": 1,
            "queue": q_b,
        },
    )
    child_b.start()
    child_b.join(timeout=60)
    assert not child_b.is_alive(), (
        f"child {child_b.pid} did not exit within 60s "
        f"(replay with --chaos-seed {chaos_seed}, "
        f"sigkill_delay={sigkill_delay:.6f}s)"
    )

    results: list[dict] = []
    while not q_b.empty():
        results.append(q_b.get_nowait())
    assert len(results) == 1, (
        f"expected 1 child-B result, got {results} "
        f"(replay with --chaos-seed {chaos_seed}, "
        f"sigkill_delay={sigkill_delay:.6f}s)"
    )
    assert results[0]["ok"], (
        f"child B failed after SIGKILL of A: {results[0]!r} "
        f"(replay with --chaos-seed {chaos_seed}, "
        f"sigkill_delay={sigkill_delay:.6f}s)"
    )

    # Parent-side read-back: the one drawer child B wrote must be visible
    # AND the palace must be fully consistent (palace_meta:main row exists,
    # tables exist, no orphan state).
    from mempalace.backends.surreal import SurrealBackend

    backend = SurrealBackend(namespace=isolated_namespace)
    try:
        palace = PalaceRef(id=palace_id)
        col = backend.get_collection(
            palace=palace, collection_name="mempalace_drawers", create=True
        )
        got = col.get(ids=["survivor-0"])
        assert got.ids == ["survivor-0"], (
            f"survivor row missing or wrong: {got!r} "
            f"(replay with --chaos-seed {chaos_seed}, "
            f"sigkill_delay={sigkill_delay:.6f}s)"
        )
        # palace_meta:main must exist — that's the signal the bootstrap
        # re-run actually completed, not that we just got lucky.
        db_name = list(backend._conns.keys())[0]
        assert backend._palace_meta_present(backend._conns[db_name]), (
            f"palace_meta:main is missing after SIGKILL recovery "
            f"(replay with --chaos-seed {chaos_seed}, "
            f"sigkill_delay={sigkill_delay:.6f}s)"
        )
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Environment note: suppress any stray namespace leaks from interrupted
# test runs by clearing the env var the fixture sets on the parent.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _scrub_env_var():
    """Ensure MEMPALACE_SURREAL_NS from a previous aborted run does not
    bleed into child-process behaviour here."""
    saved = os.environ.pop("MEMPALACE_SURREAL_NS", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["MEMPALACE_SURREAL_NS"] = saved
