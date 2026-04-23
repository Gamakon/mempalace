"""test_mcp_server_surreal.py — MCP server <-> Surreal backend smoke tests (mp-sw5).

Runs the MCP server's tool handlers with ``MEMPALACE_BACKEND=surreal`` so
the module wires up :class:`SurrealBackend` + :class:`KnowledgeGraphSurreal`
instead of the Chroma + SQLite defaults.

The module is already imported at test-collection time (default path), so
these tests swap ``_config`` / ``_kg`` in-place and reset the collection
cache between tests — matching the pattern used in ``test_mcp_server.py``
but pointing at Surreal. The whole file is skipped if the local Surreal
server is not reachable.
"""

from __future__ import annotations

import socket
import uuid

import pytest


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


@pytest.fixture()
def surreal_mcp(monkeypatch):
    """Swap the MCP server module to use the Surreal backend + KG.

    * ``_config`` reports ``backend == "surreal"`` so ``_get_collection``
      takes the Surreal branch.
    * ``_kg`` is a fresh :class:`KnowledgeGraphSurreal` bound to a unique
      namespace/database so the triples written here never leak into or
      collide with ``~/.mempalace`` or other test modules.
    * The Surreal backend instance itself is rebuilt against a unique
      namespace so the drawer tables are isolated too.
    * All caches (``_collection_cache``, ``_surreal_backend``,
      ``_surreal_palace_ref``) are cleared before the test and restored
      afterwards so test order does not matter.
    """
    from mempalace import mcp_server
    from mempalace.backends.surreal import SurrealBackend
    from mempalace.kg_surreal import KnowledgeGraphSurreal

    run_id = uuid.uuid4().hex[:10]
    ns_drawer = f"mp_mcp_{run_id}"
    db_kg = f"mp_mcp_kg_{run_id}"
    palace_id = f"palace_{run_id}"

    # Fake config whose public surface matches MempalaceConfig for the
    # fields mcp_server actually reads. ``collection_name`` matches the
    # default so the Surreal backend's ``_TABLE_ALIASES`` routes it to
    # the ``drawer`` table.
    class _SurrealCfg:
        backend = "surreal"
        palace_path = f"/tmp/surreal_palace_{run_id}"  # unused by surreal backend
        collection_name = "mempalace_drawers"

    monkeypatch.setattr(mcp_server, "_config", _SurrealCfg())

    kg = KnowledgeGraphSurreal(namespace="test", database=db_kg)
    # Wipe any leftover state from prior runs in this DB — test isolation.
    kg._db.query("REMOVE TABLE IF EXISTS triple")
    kg._db.query("REMOVE TABLE IF EXISTS entity")
    kg._ensure_schema()
    monkeypatch.setattr(mcp_server, "_kg", kg)

    # Swap in a fresh Surreal backend targeting a unique namespace so
    # drawer tables don't collide with other tests.
    backend = SurrealBackend(namespace=ns_drawer)
    from mempalace.backends.base import PalaceRef

    palace_ref = PalaceRef(id=palace_id, local_path=None)

    # Reset caches so the module actually uses our patched objects.
    monkeypatch.setattr(mcp_server, "_collection_cache", None)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None)
    monkeypatch.setattr(mcp_server, "_metadata_cache_time", 0)
    monkeypatch.setattr(mcp_server, "_surreal_backend", backend)
    monkeypatch.setattr(mcp_server, "_surreal_palace_ref", palace_ref)

    yield mcp_server

    # Tear down both the drawer namespace and the KG tables so reruns start
    # clean and nothing leaks server-side.
    try:
        conn = backend._connect("cleanup_dummy")
        conn.query(f"REMOVE NAMESPACE IF EXISTS {ns_drawer};")
    except Exception:
        pass
    try:
        backend.close()
    except Exception:
        pass
    try:
        kg.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_env_var_selects_surreal_backend(monkeypatch):
    """Setting ``MEMPALACE_BACKEND=surreal`` flips ``_config.backend``.

    This is the sanity check that the surreal path is reachable without
    any test-only monkeypatching — just the env var.
    """
    monkeypatch.setenv("MEMPALACE_BACKEND", "surreal")
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig()
    assert cfg.backend == "surreal"


def test_mcp_server_wires_surreal_kg_when_backend_is_surreal(monkeypatch):
    """``_init_kg`` returns ``KnowledgeGraphSurreal`` when backend is surreal."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "surreal")
    from mempalace import mcp_server
    from mempalace.kg_surreal import KnowledgeGraphSurreal

    kg = mcp_server._init_kg()
    try:
        assert isinstance(kg, KnowledgeGraphSurreal)
    finally:
        kg.close()


# ---------------------------------------------------------------------------
# add_drawer -> search
# ---------------------------------------------------------------------------


class TestSurrealDrawerSearch:
    def test_add_drawer_then_search_finds_it(self, surreal_mcp):
        add = surreal_mcp.tool_add_drawer(
            wing="project",
            room="backend",
            content=(
                "Authentication uses JWT tokens stored in HttpOnly cookies. "
                "Refresh tokens rotate every 24 hours."
            ),
        )
        assert add["success"] is True, add
        drawer_id = add["drawer_id"]

        # Search by a topic the drawer covers — vector search should return it.
        result = surreal_mcp.tool_search(query="jwt authentication tokens", limit=5)
        assert "error" not in result, result
        hit_ids_or_texts = [h.get("text", "") for h in result.get("hits", [])]
        assert any("JWT" in t or "jwt" in t.lower() for t in hit_ids_or_texts), result
        assert drawer_id.startswith("drawer_project_backend_")

    def test_search_empty_palace_returns_no_hits(self, surreal_mcp):
        # Bootstrap an empty collection first so `get_collection(create=False)`
        # in the search path can see the drawer table.
        surreal_mcp._get_collection(create=True)
        result = surreal_mcp.tool_search(query="anything at all", limit=3)
        # Either a well-formed empty result or the standard no-palace shape
        # is acceptable — what we MUST NOT see is a Python traceback /
        # an unexpected error surface.
        assert "error" not in result or result["error"] == "No palace found"
        if "hits" in result:
            assert result["hits"] == []


# ---------------------------------------------------------------------------
# kg_add -> kg_query
# ---------------------------------------------------------------------------


class TestSurrealKGTools:
    def test_kg_add_then_kg_query_outgoing(self, surreal_mcp):
        add = surreal_mcp.tool_kg_add(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
            valid_from="2024-01-01",
        )
        assert add["success"] is True
        assert "Alice" in add["fact"]

        q = surreal_mcp.tool_kg_query(entity="Alice", direction="outgoing")
        assert q["entity"] == "Alice"
        assert q["count"] >= 1
        preds = [f["predicate"] for f in q["facts"]]
        assert "works_at" in preds

    def test_kg_invalidate_closes_fact(self, surreal_mcp):
        surreal_mcp.tool_kg_add(subject="Bob", predicate="lives_in", object="Paris")
        # Before invalidate → Bob has one current fact.
        before = surreal_mcp.tool_kg_query(entity="Bob", direction="outgoing")
        assert any(f.get("current") for f in before["facts"])

        result = surreal_mcp.tool_kg_invalidate(
            subject="Bob", predicate="lives_in", object="Paris", ended="2026-01-01"
        )
        assert result["success"] is True

        after = surreal_mcp.tool_kg_query(entity="Bob", direction="outgoing")
        # The triple is still returned, but `current` is now False because
        # valid_to was stamped.
        assert all(not f.get("current") for f in after["facts"])

    def test_kg_stats_reflects_surreal_state(self, surreal_mcp):
        surreal_mcp.tool_kg_add(subject="Carol", predicate="knows", object="Dan")
        surreal_mcp.tool_kg_add(subject="Carol", predicate="knows", object="Eve")
        stats = surreal_mcp.tool_kg_stats()
        # Carol + Dan + Eve → 3 entities; 2 triples.
        assert stats["entities"] >= 3
        assert stats["triples"] >= 2
        assert "knows" in stats["relationship_types"]


# ---------------------------------------------------------------------------
# diary_write -> diary_read
# ---------------------------------------------------------------------------


class TestSurrealDiaryTools:
    def test_diary_write_then_read_round_trip(self, surreal_mcp):
        entry_text = "SESSION:2026-04-23|built.surreal.mcp.wiring|*warm*|★★★★"
        w = surreal_mcp.tool_diary_write(
            agent_name="testagent",
            entry=entry_text,
            topic="mp-sw5",
        )
        assert w["success"] is True, w

        r = surreal_mcp.tool_diary_read(agent_name="testagent", last_n=5)
        assert "error" not in r, r
        assert r["agent"] == "testagent"
        assert r["total"] >= 1
        texts = [e["content"] for e in r["entries"]]
        assert entry_text in texts

    def test_diary_read_empty_agent(self, surreal_mcp):
        # Bootstrap collection without writing any diary entries for this agent.
        surreal_mcp._get_collection(create=True)
        r = surreal_mcp.tool_diary_read(agent_name="nobody", last_n=5)
        # Either an empty-entries payload or the no-palace shape.
        if "entries" in r:
            assert r["entries"] == []
        else:
            assert "error" in r or "message" in r
