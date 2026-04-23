"""
test_kg_surreal.py — SurrealDB KG port tests (mp-4yf).

Runs against the live local SurrealDB (see ``docs/surrealdb-local.md``).
Each test gets a clean ``test`` / ``mp_kg_test`` namespace+database so
state never leaks between tests or between test runs.

If the local Surreal server is not reachable on 127.0.0.1:8000 the whole
module is skipped — this keeps CI happy on machines without Surreal
installed while still exercising the real DB locally.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest


SURREAL_URL = "http://127.0.0.1:8000"


def _surreal_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{SURREAL_URL}/version", timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _surreal_reachable(),
    reason="Local SurrealDB not reachable at 127.0.0.1:8000 (see docs/surrealdb-local.md)",
)


@pytest.fixture
def kg():
    """Fresh KG instance with a wiped test DB.

    We wipe both tables before the test (not after) so that if a test
    fails you can poke at the state in the Surreal CLI.
    """
    from mempalace.kg_surreal import KnowledgeGraphSurreal

    instance = KnowledgeGraphSurreal(namespace="test", database="mp_kg_test")
    # Tear down whatever the previous run left behind.
    instance._db.query("REMOVE TABLE IF EXISTS triple")
    instance._db.query("REMOVE TABLE IF EXISTS entity")
    instance._ensure_schema()
    yield instance
    instance.close()


@pytest.fixture
def seeded_kg(kg):
    """Same shape as ``tests/conftest.py::seeded_kg`` for the SQLite KG.

    Pattern:
        Alice parent_of Max            (open)
        Alice works_at Acme Corp       2020-01-01 -> 2024-06-01
        Alice works_at NewCo           2024-06-01 -> open
        Max   does swimming            2025-01-01 -> open
        Max   does chess               2025-10-01 -> open
    """
    kg.add_triple("Alice", "parent_of", "Max")
    kg.add_triple(
        "Alice",
        "works_at",
        "Acme Corp",
        valid_from="2020-01-01",
        valid_to="2024-06-01",
    )
    kg.add_triple("Alice", "works_at", "NewCo", valid_from="2024-06-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "does", "chess", valid_from="2025-10-01")
    return kg


# ── Entity ops ─────────────────────────────────────────────────────────


class TestEntity:
    def test_add_entity_returns_slug(self, kg):
        assert kg.add_entity("Alice") == "alice"

    def test_add_entity_slug_matches_sqlite(self, kg):
        # mp-1jb: Surreal slug must be bit-identical to the SQLite KG's
        # ``_entity_id`` so a palace migrated between backends keeps the
        # same entity IDs (and all existing triples resolve).
        assert kg.add_entity("Dr. Chen") == "dr._chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", entity_type="person")
        kg.add_entity("Alice", entity_type="engineer")
        stats = kg.stats()
        assert stats["entities"] == 1


class TestSlugParityWithSQLite:
    """mp-1jb: Surreal ``_entity_id`` must be byte-for-byte identical to
    the SQLite KG's ``_entity_id`` so ingesting the same names into either
    backend yields the same entity IDs. Without this, a palace re-ingested
    into the SurrealDB backend creates new entity rows that don't collide
    with the existing SQLite IDs, and every existing triple points at a
    ghost.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "Dr. Chen",
            "J.R.R. Tolkien",
            "O'Brien",
            "C3-PO",
            "Alice",
            "Acme Corp",
            "Dr. O'Malley",
            "   ",  # pathological but must not explode
            "",
        ],
    )
    def test_surreal_slug_matches_sqlite(self, name):
        from mempalace.kg_surreal import KnowledgeGraphSurreal
        from mempalace.knowledge_graph import KnowledgeGraph

        # Use the unbound methods directly — no DB connection needed.
        # ``KnowledgeGraph._entity_id`` is an instance method, but it
        # doesn't touch ``self``; call it via the class to avoid opening
        # a SQLite file just for a string op.
        sqlite_slug = KnowledgeGraph._entity_id(None, name)  # type: ignore[arg-type]
        surreal_slug = KnowledgeGraphSurreal._entity_id(name)
        assert surreal_slug == sqlite_slug, (
            f"Slug divergence for {name!r}: surreal={surreal_slug!r} sqlite={sqlite_slug!r}"
        )

    def test_known_outputs(self):
        """Explicit expectations (hardcoded from SQLite's normaliser) so a
        change to either implementation fails loudly with the exact shape
        we want, not just a self-consistency check."""
        from mempalace.kg_surreal import KnowledgeGraphSurreal

        cases = {
            "Dr. Chen": "dr._chen",
            "J.R.R. Tolkien": "j.r.r._tolkien",
            "O'Brien": "obrien",
            "C3-PO": "c3-po",
            "Alice": "alice",
            "Acme Corp": "acme_corp",
            "Dr. O'Malley": "dr._omalley",
        }
        for name, expected in cases.items():
            assert KnowledgeGraphSurreal._entity_id(name) == expected, name


# ── Triple ops ─────────────────────────────────────────────────────────


class TestAddTriple:
    def test_add_triple_auto_creates_entities(self, kg):
        tid = kg.add_triple("Alice", "knows", "Bob")
        assert tid.startswith("triple:")
        assert kg.stats()["entities"] == 2

    def test_add_triple_returns_surreal_id(self, kg):
        tid = kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
        # Surreal returns ``triple:<random>``; format is opaque but stable.
        assert tid.startswith("triple:")

    def test_duplicate_open_triple_returns_existing_id(self, kg):
        a = kg.add_triple("Alice", "knows", "Bob")
        b = kg.add_triple("Alice", "knows", "Bob")
        assert a == b
        assert kg.stats()["triples"] == 1

    def test_predicate_is_normalised(self, kg):
        kg.add_triple("Alice", "Works At", "Acme")
        rows = kg.query_relationship("works_at")
        assert len(rows) == 1

    def test_add_triple_persists_temporal_fields(self, kg):
        kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            valid_from="2020-01-01",
            valid_to="2024-06-01",
            confidence=0.9,
        )
        rows = kg.query_relationship("works_at")
        assert rows[0]["valid_from"] == "2020-01-01"
        assert rows[0]["valid_to"] == "2024-06-01"
        assert rows[0]["current"] is False

    def test_add_triple_records_extracted_at(self, kg):
        """mp-2um: every new triple must carry an ``extracted_at`` stamp
        for provenance parity with the SQLite KG
        (``knowledge_graph.py:88``).

        The schema doc promised this via a ``DEFAULT time::now()`` field
        definition, but this module uses SCHEMALESS tables so that DEFAULT
        never fires. We now set ``extracted_at = time::now()`` explicitly
        on the RELATE so the stamp lands regardless of whether the schema
        is tightened later (mp-6gu).
        """
        from datetime import datetime, timedelta, timezone

        before = datetime.now(timezone.utc) - timedelta(seconds=5)
        kg.add_triple("Alice", "knows", "Bob")
        after = datetime.now(timezone.utc) + timedelta(seconds=5)

        # Read the raw triple row back — ``extracted_at`` is not surfaced
        # via the high-level query helpers (yet), so we hit the table
        # directly. The SDK decodes SurrealDB ``datetime`` values as
        # ``datetime.datetime`` with timezone info.
        rows = kg._db.query("SELECT extracted_at FROM triple")
        assert rows, "expected one triple row"
        stamp = rows[0].get("extracted_at")
        assert stamp is not None, f"extracted_at not set on triple: {rows[0]!r}"
        assert isinstance(stamp, datetime), (
            f"extracted_at should be a datetime, got {type(stamp).__name__}"
        )
        # Stamped "now" — must fall inside the observation window.
        assert before <= stamp <= after, f"extracted_at {stamp} outside window [{before}, {after}]"


# ── Queries ────────────────────────────────────────────────────────────


class TestQueryEntity:
    def test_outgoing(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "parent_of" in predicates
        assert "works_at" in predicates
        # No incoming triples sneaking in.
        assert all(r["direction"] == "outgoing" for r in results)

    def test_incoming(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="incoming")
        assert any(r["subject"] == "Alice" and r["predicate"] == "parent_of" for r in results)
        assert all(r["direction"] == "incoming" for r in results)

    def test_both_directions(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="both")
        directions = {r["direction"] for r in results}
        assert directions == {"outgoing", "incoming"}

    def test_as_of_filters_expired(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2023-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "Acme Corp" in employers
        assert "NewCo" not in employers

    def test_as_of_shows_current(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2025-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "NewCo" in employers
        assert "Acme Corp" not in employers

    def test_invalid_direction_raises(self, seeded_kg):
        with pytest.raises(ValueError):
            seeded_kg.query_entity("Alice", direction="sideways")

    def test_query_unknown_entity_returns_empty(self, seeded_kg):
        assert seeded_kg.query_entity("Nobody") == []


class TestQueryRelationship:
    def test_finds_all_for_predicate(self, seeded_kg):
        rows = seeded_kg.query_relationship("does")
        objects = {r["object"] for r in rows}
        assert objects == {"swimming", "chess"}

    def test_respects_as_of(self, seeded_kg):
        # Only swimming was valid on 2025-06-01 (chess starts 2025-10-01).
        rows = seeded_kg.query_relationship("does", as_of="2025-06-01")
        assert {r["object"] for r in rows} == {"swimming"}

    def test_unknown_predicate_returns_empty(self, seeded_kg):
        assert seeded_kg.query_relationship("never_seen") == []


class TestListTriples:
    def test_returns_all(self, seeded_kg):
        rows = seeded_kg.list_triples()
        # 5 seeded triples.
        assert len(rows) == 5

    def test_limit_clamps(self, seeded_kg):
        rows = seeded_kg.list_triples(limit=2)
        assert len(rows) == 2


# ── Stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_empty(self, kg):
        stats = kg.stats()
        assert stats["entities"] == 0
        assert stats["triples"] == 0
        assert stats["current_facts"] == 0
        assert stats["expired_facts"] == 0
        assert stats["relationship_types"] == []

    def test_seeded_counts(self, seeded_kg):
        stats = seeded_kg.stats()
        assert stats["triples"] == 5
        # 4 open + 1 closed (Acme Corp).
        assert stats["current_facts"] == 4
        assert stats["expired_facts"] == 1
        assert set(stats["relationship_types"]) == {"parent_of", "works_at", "does"}


# ── Timeline (mp-84n) ──────────────────────────────────────────────────


class TestTimeline:
    def test_timeline_global_ascending(self, seeded_kg):
        """Oldest-first ordering on valid_from."""
        tl = seeded_kg.timeline()
        dated = [t["valid_from"] for t in tl if t["valid_from"] is not None]
        assert dated == sorted(dated)

    def test_timeline_descending(self, seeded_kg):
        """``order='desc'`` flips the chronological direction."""
        tl = seeded_kg.timeline(order="desc")
        dated = [t["valid_from"] for t in tl if t["valid_from"] is not None]
        assert dated == sorted(dated, reverse=True)

    def test_timeline_invalid_order_raises(self, kg):
        with pytest.raises(ValueError):
            kg.timeline(order="sideways")

    def test_timeline_valid_from_none_sorts_last(self, kg):
        """Triples with ``valid_from IS NONE`` always come last (NULLS LAST
        parity with the SQLite KG), regardless of asc/desc."""
        # First entry has no valid_from — this is the fallback case.
        kg.add_triple("Alice", "knows", "Bob")  # valid_from=None
        kg.add_triple("Carol", "knows", "Dave", valid_from="2020-01-01")
        kg.add_triple("Erin", "knows", "Frank", valid_from="2025-01-01")

        asc = kg.timeline(order="asc")
        assert asc[-1]["valid_from"] is None  # Alice/Bob pushed to end
        assert asc[0]["valid_from"] == "2020-01-01"

        desc = kg.timeline(order="desc")
        assert desc[-1]["valid_from"] is None  # still last in desc
        assert desc[0]["valid_from"] == "2025-01-01"

    def test_timeline_filters_by_entity(self, seeded_kg):
        tl = seeded_kg.timeline("Max")
        touched = {t["subject"] for t in tl} | {t["object"] for t in tl}
        assert "Max" in touched
        # Alice-only triples like "Alice works_at Acme Corp" should be
        # excluded unless Max is one of the endpoints.
        assert not any(t["subject"] != "Max" and t["object"] != "Max" for t in tl), tl

    def test_timeline_respects_limit(self, kg):
        for i in range(10):
            kg.add_triple("hub", "connects_to", f"spoke_{i}", valid_from=f"2025-01-{i + 1:02d}")
        tl = kg.timeline(limit=3)
        assert len(tl) == 3


# ── Invalidate (mp-84n) ────────────────────────────────────────────────


class TestInvalidate:
    """Surface must match ``KnowledgeGraph.invalidate(sub, pred, obj, ended=...)``
    exactly — see mp-s2k. ``mcp_server.tool_kg_invalidate`` calls the SPO
    form regardless of backend.
    """

    def test_invalidate_sets_valid_to(self, kg):
        kg.add_triple("Alice", "works_at", "Acme", valid_from="2020-01-01")
        changed = kg.invalidate("Alice", "works_at", "Acme", ended="2024-06-01")
        assert changed is True

        rows = kg.query_relationship("works_at")
        assert len(rows) == 1
        assert rows[0]["valid_to"] == "2024-06-01"
        assert rows[0]["current"] is False

    def test_invalidate_defaults_ended_to_today(self, kg):
        from datetime import date

        kg.add_triple("Alice", "works_at", "Acme")
        assert kg.invalidate("Alice", "works_at", "Acme") is True

        rows = kg.query_relationship("works_at")
        assert rows[0]["valid_to"] == date.today().isoformat()

    def test_invalidate_is_idempotent(self, kg):
        """Second call with the same SPO after closure is a no-op."""
        kg.add_triple("Alice", "works_at", "Acme")
        assert kg.invalidate("Alice", "works_at", "Acme", ended="2024-06-01") is True
        # Second call should not overwrite the earlier valid_to or create
        # a second closure event.
        assert kg.invalidate("Alice", "works_at", "Acme", ended="2099-12-31") is False

        rows = kg.query_relationship("works_at")
        assert rows[0]["valid_to"] == "2024-06-01"

    def test_invalidate_unknown_triple_returns_false(self, kg):
        """No matching open triple -> falsy, no error (SQLite parity)."""
        assert kg.invalidate("Ghost", "works_at", "Nowhere") is False

    def test_invalidate_mismatched_spo_returns_false(self, kg):
        """Partial SPO mismatch doesn't close anything, returns falsy."""
        kg.add_triple("Alice", "works_at", "Acme")
        # Wrong predicate — open triple stays open.
        assert kg.invalidate("Alice", "knows", "Acme") is False
        rows = kg.query_relationship("works_at")
        assert rows[0]["valid_to"] is None

    def test_invalidate_normalises_predicate(self, kg):
        """Predicate normalisation matches add_triple so callers can pass
        either ``"Works At"`` or ``"works_at"``."""
        kg.add_triple("Alice", "works_at", "Acme")
        assert kg.invalidate("Alice", "Works At", "Acme", ended="2024-06-01") is True

    def test_invalidated_triple_allows_re_add(self, kg):
        """After closing an open triple, re-adding it creates a new open
        row (matches SQLite ``test_invalidated_triple_allows_re_add``)."""
        a = kg.add_triple("Alice", "works_at", "Acme")
        kg.invalidate("Alice", "works_at", "Acme", ended="2024-06-01")
        b = kg.add_triple("Alice", "works_at", "Acme")
        assert a != b
        assert kg.stats()["triples"] == 2
        assert kg.stats()["current_facts"] == 1


# ── seed_from_entity_facts (mp-84n) ────────────────────────────────────


class TestSeedFromEntityFacts:
    def test_seed_person_with_partner(self, kg):
        kg.seed_from_entity_facts(
            {
                "alice": {
                    "full_name": "Alice Smith",
                    "type": "person",
                    "gender": "female",
                    "partner": "bob",
                    "relationship": "husband",
                }
            }
        )
        results = kg.query_entity("Alice Smith", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "married_to" in predicates
        assert "is_partner_of" in predicates

    def test_seed_child(self, kg):
        kg.seed_from_entity_facts(
            {
                "max": {
                    "full_name": "Max",
                    "type": "person",
                    "birthday": "2015-04-01",
                    "parent": "alice",
                    "relationship": "daughter",
                }
            }
        )
        results = kg.query_entity("Max", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "child_of" in predicates
        assert "is_child_of" in predicates

    def test_seed_interests(self, kg):
        kg.seed_from_entity_facts(
            {
                "max": {
                    "full_name": "Max",
                    "type": "person",
                    "interests": ["swimming", "chess"],
                }
            }
        )
        results = kg.query_entity("Max", direction="outgoing")
        loves = {r["object"] for r in results if r["predicate"] == "loves"}
        assert loves == {"Swimming", "Chess"}

    def test_seed_minimal_facts_creates_entity(self, kg):
        kg.seed_from_entity_facts({"bob": {"full_name": "Bob"}})
        assert kg.stats()["entities"] >= 1

    def test_seed_empty_is_noop(self, kg):
        kg.seed_from_entity_facts({})
        assert kg.stats()["entities"] == 0
        assert kg.stats()["triples"] == 0
