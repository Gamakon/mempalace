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

    def test_add_entity_slug_normalises_punctuation(self, kg):
        # "Dr. Chen" -> lowercased, space->_, dot stripped as unsafe char.
        # Note: diverges from SQLite KG which keeps the literal ".".
        slug = kg.add_entity("Dr. Chen")
        assert slug == "dr_chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", entity_type="person")
        kg.add_entity("Alice", entity_type="engineer")
        stats = kg.stats()
        assert stats["entities"] == 1


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


# ── Deferred operations ────────────────────────────────────────────────


class TestDeferredToMp84n:
    """Temporal invalidate + full timeline land in mp-84n; make sure the
    stubs speak up instead of silently returning stale answers."""

    def test_invalidate_raises(self, kg):
        with pytest.raises(NotImplementedError, match="mp-84n"):
            kg.invalidate("Alice", "works_at", "Acme")

    def test_timeline_raises(self, kg):
        with pytest.raises(NotImplementedError, match="mp-84n"):
            kg.timeline()

    def test_seed_from_entity_facts_raises(self, kg):
        with pytest.raises(NotImplementedError, match="mp-84n"):
            kg.seed_from_entity_facts({})
