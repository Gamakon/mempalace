"""test_mcp_server_surreal.py — MCP server <-> Surreal backend smoke tests (mp-sw5, mp-jkx).

Runs the MCP server's tool handlers with ``MEMPALACE_BACKEND=surreal`` so
the module wires up :class:`SurrealBackend` + :class:`KnowledgeGraphSurreal`
instead of the Chroma + SQLite defaults.

The module is already imported at test-collection time (default path), so
these tests swap ``_config`` / ``_kg`` in-place and reset the collection
cache between tests — matching the pattern used in ``test_mcp_server.py``
but pointing at Surreal. The whole file is skipped if the local Surreal
server is not reachable.

mp-jkx extends mp-sw5's initial coverage to exercise *every* MCP tool
handler end-to-end, proving a Claude Code session using
``MEMPALACE_BACKEND=surreal`` sees the same behaviour as against Chroma.
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
def surreal_mcp(monkeypatch, tmp_path):
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
    * ``palace_graph._TUNNEL_FILE`` is redirected to ``tmp_path`` so
      tunnel tests never touch the real ``~/.mempalace/tunnels.json``.
    * ``palace_graph`` graph cache is invalidated so a stale graph from
      a prior test can't leak across.
    """
    from mempalace import mcp_server, palace_graph
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

    # Redirect the on-disk tunnel store so tunnel tool tests never
    # touch the real ``~/.mempalace/tunnels.json`` — and start empty.
    tunnel_file = tmp_path / "tunnels.json"
    monkeypatch.setattr(palace_graph, "_TUNNEL_FILE", str(tunnel_file))

    # Flush the module-level graph cache so a stale (nodes, edges) tuple
    # from a previous test can't leak into this one.
    palace_graph.invalidate_graph_cache()

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
    # Invalidate again on the way out so the next module that runs
    # doesn't see our in-memory Surreal-derived graph.
    palace_graph.invalidate_graph_cache()


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


# ---------------------------------------------------------------------------
# mp-jkx: exhaustive smoke test coverage for every remaining MCP tool
# ---------------------------------------------------------------------------


def _seed_drawers(mcp):
    """Populate a small multi-wing/room palace so graph/listing tools
    have something real to walk."""
    ids = []
    for wing, room, content in (
        ("project", "backend", "Postgres migration for user auth service."),
        ("project", "frontend", "React dashboard wiring for user auth."),
        ("notes", "backend", "Redis cache layer in front of Postgres."),
        ("notes", "planning", "Q2 roadmap — ship auth, then billing."),
    ):
        r = mcp.tool_add_drawer(wing=wing, room=room, content=content)
        assert r["success"] is True, r
        ids.append(r["drawer_id"])
    return ids


class TestSurrealStatusAndCatalog:
    """Read-only catalog tools — status / list_wings / list_rooms /
    get_taxonomy / get_aaak_spec. Shape-only assertions plus non-error."""

    def test_status_empty_palace(self, surreal_mcp):
        # mp-mge: tool_status now probes for a bootstrapped palace on
        # Surreal the same way it checks for chroma.sqlite3 on Chroma.
        # To exercise the "bootstrapped but empty" path we force-bootstrap
        # the collection first, then call status.
        surreal_mcp._get_collection(create=True)
        r = surreal_mcp.tool_status()
        assert "error" not in r, r
        assert "total_drawers" in r
        assert r["total_drawers"] == 0
        assert isinstance(r["wings"], dict)
        assert isinstance(r["rooms"], dict)

    def test_status_no_palace_returns_error(self, surreal_mcp):
        # mp-mge: an un-bootstrapped Surreal palace must return the same
        # "No palace found" error shape that Chroma does. Before the fix
        # this hardcoded db_exists=True and silently returned an empty
        # success, hiding the fact that init was never run.
        r = surreal_mcp.tool_status()
        assert r.get("error") == "No palace found", r
        assert "hint" in r

    def test_status_after_writes(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_status()
        assert r["total_drawers"] >= 4
        assert set(r["wings"].keys()) >= {"project", "notes"}
        assert "backend" in r["rooms"]
        # The protocol + AAAK spec must be embedded on every status call
        # so a cold Claude session learns the dialect on wake-up.
        assert "protocol" in r
        assert "aaak_dialect" in r

    def test_list_wings_after_writes(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_list_wings()
        assert "error" not in r, r
        wings = r["wings"]
        assert wings.get("project", 0) >= 2
        assert wings.get("notes", 0) >= 2

    def test_list_rooms_all(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_list_rooms()
        assert "error" not in r, r
        assert r["wing"] == "all"
        assert set(r["rooms"].keys()) >= {"backend", "frontend", "planning"}

    def test_list_rooms_filtered(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_list_rooms(wing="project")
        assert "error" not in r, r
        assert r["wing"] == "project"
        # Only the two project rooms should appear.
        assert set(r["rooms"].keys()) == {"backend", "frontend"}

    def test_get_taxonomy(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_get_taxonomy()
        assert "error" not in r, r
        tax = r["taxonomy"]
        assert tax["project"]["backend"] >= 1
        assert tax["notes"]["planning"] >= 1

    def test_get_aaak_spec(self, surreal_mcp):
        r = surreal_mcp.tool_get_aaak_spec()
        assert "aaak_spec" in r
        # Spec must reference its core primitives so the AI can decode.
        assert "ENTITIES" in r["aaak_spec"]
        assert "EMOTIONS" in r["aaak_spec"]


class TestSurrealDrawerCRUD:
    """add → get → update → list → delete round trips on the drawer table."""

    def test_get_drawer_round_trip(self, surreal_mcp):
        add = surreal_mcp.tool_add_drawer(
            wing="project", room="api", content="GET /v1/users returns 200."
        )
        assert add["success"] is True
        got = surreal_mcp.tool_get_drawer(drawer_id=add["drawer_id"])
        assert "error" not in got, got
        assert got["content"] == "GET /v1/users returns 200."
        assert got["wing"] == "project"
        assert got["room"] == "api"

    def test_get_drawer_missing_id(self, surreal_mcp):
        # Bootstrap an empty palace so the path reaches "not found" not "no palace".
        surreal_mcp._get_collection(create=True)
        got = surreal_mcp.tool_get_drawer(drawer_id="drawer_nope_nope_deadbeef")
        assert "error" in got

    def test_list_drawers_paginates(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_list_drawers(limit=2, offset=0)
        assert "error" not in r, r
        assert r["count"] == 2
        assert r["limit"] == 2
        assert r["offset"] == 0
        r2 = surreal_mcp.tool_list_drawers(limit=2, offset=2)
        assert r2["count"] >= 1
        # No overlap between pages.
        ids1 = {d["drawer_id"] for d in r["drawers"]}
        ids2 = {d["drawer_id"] for d in r2["drawers"]}
        assert ids1.isdisjoint(ids2)

    def test_list_drawers_wing_room_filter(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_list_drawers(wing="project", room="backend")
        assert "error" not in r, r
        assert r["count"] >= 1
        for d in r["drawers"]:
            assert d["wing"] == "project"
            assert d["room"] == "backend"

    def test_update_drawer_content_and_metadata(self, surreal_mcp):
        add = surreal_mcp.tool_add_drawer(wing="project", room="old_room", content="original")
        drawer_id = add["drawer_id"]
        upd = surreal_mcp.tool_update_drawer(
            drawer_id=drawer_id,
            content="updated content v2",
            room="new_room",
        )
        assert upd["success"] is True, upd
        assert upd["room"] == "new_room"
        got = surreal_mcp.tool_get_drawer(drawer_id=drawer_id)
        assert got["content"] == "updated content v2"
        assert got["room"] == "new_room"

    def test_update_drawer_missing_id(self, surreal_mcp):
        surreal_mcp._get_collection(create=True)
        r = surreal_mcp.tool_update_drawer(drawer_id="drawer_nonexistent_xyz", content="no")
        assert r["success"] is False
        assert "not found" in r["error"].lower()

    def test_update_drawer_noop(self, surreal_mcp):
        add = surreal_mcp.tool_add_drawer(wing="project", room="api", content="noop check")
        r = surreal_mcp.tool_update_drawer(drawer_id=add["drawer_id"])
        # All-None -> short-circuits to a no-op success.
        assert r["success"] is True
        assert r.get("noop") is True

    def test_delete_drawer_round_trip(self, surreal_mcp):
        add = surreal_mcp.tool_add_drawer(
            wing="project", room="disposable", content="this will die"
        )
        did = add["drawer_id"]
        rm = surreal_mcp.tool_delete_drawer(drawer_id=did)
        assert rm["success"] is True, rm
        # Second delete is a miss.
        miss = surreal_mcp.tool_delete_drawer(drawer_id=did)
        assert miss["success"] is False
        assert "not found" in miss["error"].lower()

    def test_check_duplicate_detects_near_identical(self, surreal_mcp):
        surreal_mcp.tool_add_drawer(
            wing="project",
            room="api",
            content="Authentication uses JWT tokens in HttpOnly cookies.",
        )
        r = surreal_mcp.tool_check_duplicate(
            content="Authentication uses JWT tokens in HttpOnly cookies.",
            threshold=0.9,
        )
        assert "error" not in r, r
        assert r["is_duplicate"] is True
        assert r["matches"], r

    def test_check_duplicate_no_match(self, surreal_mcp):
        surreal_mcp._get_collection(create=True)
        r = surreal_mcp.tool_check_duplicate(
            content="Totally unrelated content about volcanoes.",
            threshold=0.99,
        )
        assert "error" not in r, r
        assert r["is_duplicate"] is False


# ---------------------------------------------------------------------------
# Knowledge-graph timeline (kg_add / kg_query / kg_invalidate / kg_stats
# already covered above).
# ---------------------------------------------------------------------------


class TestSurrealKGTimeline:
    def test_kg_timeline_for_entity(self, surreal_mcp):
        surreal_mcp.tool_kg_add(
            subject="Frank", predicate="joined", object="Acme", valid_from="2024-01-01"
        )
        surreal_mcp.tool_kg_add(
            subject="Frank", predicate="promoted_to", object="VP", valid_from="2025-06-01"
        )
        r = surreal_mcp.tool_kg_timeline(entity="Frank")
        assert "error" not in r, r
        assert r["entity"] == "Frank"
        assert r["count"] >= 2
        preds = {f.get("predicate") for f in r["timeline"]}
        assert {"joined", "promoted_to"} <= preds

    def test_kg_timeline_all_entities(self, surreal_mcp):
        surreal_mcp.tool_kg_add(subject="Gina", predicate="knows", object="Henry")
        r = surreal_mcp.tool_kg_timeline()
        assert r["entity"] == "all"
        assert r["count"] >= 1


# ---------------------------------------------------------------------------
# Palace graph / tunnels
# ---------------------------------------------------------------------------


class TestSurrealGraphReadTools:
    """traverse / find_tunnels / graph_stats iterate palace metadata —
    they must walk the Surreal collection's paged get() exactly like Chroma."""

    def test_graph_stats_after_writes(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_graph_stats()
        assert "error" not in r, r
        assert "total_rooms" in r
        assert r["total_rooms"] >= 1
        # "backend" is the shared room across project+notes → a real tunnel.
        assert r["tunnel_rooms"] >= 1

    def test_find_tunnels_discovers_shared_room(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_find_tunnels()
        # shared "backend" room across project + notes should show up.
        rooms = [t.get("room") for t in r]
        assert "backend" in rooms, r

    def test_find_tunnels_filtered_by_wings(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_find_tunnels(wing_a="project", wing_b="notes")
        rooms = [t.get("room") for t in r]
        assert "backend" in rooms, r

    def test_traverse_known_room(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_traverse_graph(start_room="backend", max_hops=2)
        # Happy-path traverse returns a list of hop dicts; error path returns
        # a dict with "error". The room exists so we expect the list.
        assert isinstance(r, list), r
        assert any(entry.get("room") == "backend" for entry in r)

    def test_traverse_unknown_room_returns_error(self, surreal_mcp):
        _seed_drawers(surreal_mcp)
        r = surreal_mcp.tool_traverse_graph(start_room="nonexistent-room")
        assert isinstance(r, dict)
        assert "error" in r


class TestSurrealExplicitTunnels:
    """Explicit tunnels are file-backed (tunnels.json) — backend-agnostic,
    but we still want to confirm the full create/find/follow/delete path
    from the MCP layer under the Surreal config."""

    def test_create_list_follow_delete(self, surreal_mcp):
        # Seed the endpoints so follow_tunnels has drawer IDs to fetch.
        a = surreal_mcp.tool_add_drawer(
            wing="wing_api", room="auth", content="API auth endpoint spec."
        )
        b = surreal_mcp.tool_add_drawer(wing="wing_db", room="users", content="Users table schema.")

        created = surreal_mcp.tool_create_tunnel(
            source_wing="wing_api",
            source_room="auth",
            target_wing="wing_db",
            target_room="users",
            label="auth reads users",
            source_drawer_id=a["drawer_id"],
            target_drawer_id=b["drawer_id"],
        )
        assert "error" not in created, created
        assert created["label"] == "auth reads users"
        tunnel_id = created["id"]

        # list_tunnels (unfiltered + filtered by wing).
        all_t = surreal_mcp.tool_list_tunnels()
        assert any(t["id"] == tunnel_id for t in all_t), all_t
        api_t = surreal_mcp.tool_list_tunnels(wing="wing_api")
        assert any(t["id"] == tunnel_id for t in api_t), api_t

        # follow_tunnels from the source endpoint.
        follow = surreal_mcp.tool_follow_tunnels(wing="wing_api", room="auth")
        assert isinstance(follow, list)
        assert any(
            c["connected_wing"] == "wing_db" and c["connected_room"] == "users" for c in follow
        ), follow

        # delete_tunnel cleans up.
        rm = surreal_mcp.tool_delete_tunnel(tunnel_id=tunnel_id)
        assert rm == {"deleted": tunnel_id}
        after = surreal_mcp.tool_list_tunnels()
        assert all(t["id"] != tunnel_id for t in after)

    def test_delete_tunnel_validates_input(self, surreal_mcp):
        r = surreal_mcp.tool_delete_tunnel(tunnel_id="")
        assert "error" in r


# ---------------------------------------------------------------------------
# Hook / settings / reconnect / memories_filed_away
# ---------------------------------------------------------------------------


class TestSurrealHookAndSettingsTools:
    def test_hook_settings_read_only(self, surreal_mcp):
        # Calling with no args must just report the current state without error.
        r = surreal_mcp.tool_hook_settings()
        assert r["success"] is True, r
        assert "settings" in r
        assert "silent_save" in r["settings"]
        assert "desktop_toast" in r["settings"]

    def test_hook_settings_updates(self, surreal_mcp, tmp_path, monkeypatch):
        # Redirect the config dir so we never mutate the real user config.
        monkeypatch.setenv("HOME", str(tmp_path))
        r = surreal_mcp.tool_hook_settings(silent_save=True, desktop_toast=False)
        assert r["success"] is True
        assert r["settings"]["silent_save"] is True
        assert r["settings"]["desktop_toast"] is False
        assert "updated" in r

    def test_memories_filed_away_no_checkpoint(self, surreal_mcp, tmp_path, monkeypatch):
        # Point HOME at a clean tmp dir so the "last_checkpoint" file does
        # not exist → the tool returns its "quiet" shape.
        monkeypatch.setenv("HOME", str(tmp_path))
        r = surreal_mcp.tool_memories_filed_away()
        assert r["status"] == "quiet"
        assert r["count"] == 0
        assert r["timestamp"] is None

    def test_memories_filed_away_with_checkpoint(self, surreal_mcp, tmp_path, monkeypatch):
        import json as _json

        state_dir = tmp_path / ".mempalace" / "hook_state"
        state_dir.mkdir(parents=True)
        ack = state_dir / "last_checkpoint"
        ack.write_text(_json.dumps({"msgs": 7, "ts": "2026-04-23T10:00:00"}))
        monkeypatch.setenv("HOME", str(tmp_path))
        r = surreal_mcp.tool_memories_filed_away()
        assert r["status"] == "ok"
        assert r["count"] == 7
        # File is consumed on successful read.
        assert not ack.exists()

    def test_reconnect_returns_success(self, surreal_mcp):
        # Populate so count() returns something meaningful.
        surreal_mcp.tool_add_drawer(
            wing="project", room="api", content="reconnect smoke test content"
        )
        r = surreal_mcp.tool_reconnect()
        assert r["success"] is True, r
        assert r["drawers"] >= 1
