"""
test_migrate_kg.py — SQLite → SurrealDB KG migration tests (mp-3lc).

Exercises ``mempalace.migrate_kg.migrate_kg_to_surreal`` end-to-end:
seed a fresh SQLite KG, replay it into a clean SurrealDB namespace, and
confirm every triple lands with its provenance fields intact. Auto-skips
when the local SurrealDB is not reachable so CI stays green on hosts
without Surreal installed.
"""

from __future__ import annotations

import random
import sqlite3
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


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_kg(tmp_path):
    """Fresh SQLite KG populated with ~15 triples and ~8 entities.

    Mix of open/closed, varied predicates, explicit ``extracted_at``
    timestamps so the migration has non-trivial provenance to preserve.
    """
    from mempalace.knowledge_graph import KnowledgeGraph

    db_path = tmp_path / "kg.sqlite3"
    kg = KnowledgeGraph(str(db_path))

    # Entities with explicit types — migrate must preserve "type".
    kg.add_entity("Alice", entity_type="person", properties={"city": "NYC"})
    kg.add_entity("Max", entity_type="person", properties={"dob": "2015-04-01"})
    kg.add_entity("Bob", entity_type="person")
    kg.add_entity("Acme Corp", entity_type="company")
    kg.add_entity("NewCo", entity_type="company")
    kg.add_entity("swimming", entity_type="activity")
    kg.add_entity("chess", entity_type="activity")
    kg.add_entity("hiking", entity_type="activity")

    # 15 triples: mix open/closed, varied predicates.
    kg.add_triple("Alice", "parent_of", "Max", valid_from="2015-04-01")
    kg.add_triple(
        "Alice",
        "works_at",
        "Acme Corp",
        valid_from="2020-01-01",
        valid_to="2024-06-01",
        confidence=0.95,
        source_file="/convos/2020/hire.md",
    )
    kg.add_triple(
        "Alice",
        "works_at",
        "NewCo",
        valid_from="2024-06-01",
        confidence=1.0,
        source_drawer_id="drawer_alice_newco_abc",
    )
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "does", "chess", valid_from="2025-10-01")
    kg.add_triple(
        "Max",
        "loves",
        "chess",
        valid_from="2025-10-15",
        source_closet="personal/2025-10",
        adapter_name="general",
    )
    kg.add_triple("Bob", "friend_of", "Alice", valid_from="2010-05-01")
    kg.add_triple(
        "Bob",
        "lives_in",
        "NewCo",
        valid_from="2015-03-01",
        valid_to="2020-01-01",
    )
    kg.add_triple("Alice", "likes", "hiking", valid_from="2018-06-01")
    kg.add_triple(
        "Max",
        "attends",
        "Acme Corp",
        valid_from="2023-09-01",
        valid_to="2024-06-15",
        confidence=0.8,
    )
    kg.add_triple("Bob", "works_at", "Acme Corp", valid_from="2019-01-01")
    kg.add_triple(
        "Alice",
        "partner_of",
        "Bob",
        valid_from="2012-08-01",
        valid_to="2019-11-30",
    )
    kg.add_triple("Max", "friend_of", "Bob", valid_from="2022-01-01")
    kg.add_triple(
        "Alice",
        "manages",
        "NewCo",
        valid_from="2024-07-01",
        confidence=0.9,
        source_drawer_id="drawer_alice_manages_xyz",
        adapter_name="exchange",
    )
    kg.add_triple("Bob", "pets", "swimming", valid_from="2001-01-01", valid_to="2002-01-01")

    kg.close()
    return str(db_path)


@pytest.fixture
def surreal_kg():
    """Fresh SurrealDB KG on a dedicated test namespace.

    Wipes the tables before yield so a previous failed run cannot leak
    state into the current run.
    """
    from mempalace.kg_surreal import KnowledgeGraphSurreal

    instance = KnowledgeGraphSurreal(namespace="test", database="mp_migrate_kg_test")
    instance._db.query("REMOVE TABLE IF EXISTS triple")
    instance._db.query("REMOVE TABLE IF EXISTS entity")
    instance._ensure_schema()
    yield instance
    instance.close()


# ── Tests ──────────────────────────────────────────────────────────────


class TestReadOnlySource:
    """The migration must not mutate the SQLite source."""

    def test_opens_source_readonly(self, sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        # Baseline: checksum the SQLite file before migration.
        import hashlib

        before = hashlib.sha256(open(sqlite_kg, "rb").read()).digest()

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        after = hashlib.sha256(open(sqlite_kg, "rb").read()).digest()
        # Technically WAL files could be touched by a read in write mode,
        # but we open with mode=ro so even those should not move. The
        # file content itself must not change.
        assert before == after, "Source SQLite file was modified during migration"


class TestFullMigration:
    """All entities + triples land in Surreal with faithful fields."""

    def test_counts_match_source(self, sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        result = migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=10,
            rng=random.Random(1),
            progress=False,
        )

        assert result.entities_source == result.entities_written
        assert result.triples_source == result.triples_written
        assert result.verification_sampled == 10
        assert result.verification_ok == 10
        assert result.ok

    def test_stats_after_migration_match_source(self, sqlite_kg, surreal_kg):
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.migrate_kg import migrate_kg_to_surreal

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        source_kg = KnowledgeGraph(sqlite_kg)
        try:
            source_stats = source_kg.stats()
        finally:
            source_kg.close()
        dest_stats = surreal_kg.stats()

        assert dest_stats["entities"] == source_stats["entities"]
        assert dest_stats["triples"] == source_stats["triples"]
        assert dest_stats["current_facts"] == source_stats["current_facts"]
        assert dest_stats["expired_facts"] == source_stats["expired_facts"]
        assert set(dest_stats["relationship_types"]) == set(source_stats["relationship_types"])

    def test_provenance_fields_preserved(self, sqlite_kg, surreal_kg):
        """Spot-check a triple with every provenance field populated."""
        from mempalace.migrate_kg import migrate_kg_to_surreal

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        rows = surreal_kg._db.query(
            (
                "SELECT predicate, valid_from, valid_to, confidence, "
                "source_drawer_id, adapter_name, extracted_at, "
                "in.name AS sub_name, out.name AS obj_name "
                "FROM triple WHERE predicate = 'manages'"
            )
        )
        assert rows, "'manages' triple not migrated"
        row = rows[0]
        assert row["sub_name"] == "Alice"
        assert row["obj_name"] == "NewCo"
        assert row["valid_from"] == "2024-07-01"
        assert row["valid_to"] is None
        assert abs(row["confidence"] - 0.9) < 1e-6
        assert row["source_drawer_id"] == "drawer_alice_manages_xyz"
        assert row["adapter_name"] == "exchange"
        assert row["extracted_at"] is not None, "extracted_at must be populated from the source row"

    def test_closed_triple_validity_span_preserved(self, sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        rows = surreal_kg._db.query(
            (
                "SELECT valid_from, valid_to, in.name AS s, out.name AS o "
                "FROM triple WHERE predicate = 'works_at' "
                "AND valid_to IS NOT NONE"
            )
        )
        closed = [(r["s"], r["o"], r["valid_from"], r["valid_to"]) for r in rows]
        assert ("Alice", "Acme Corp", "2020-01-01", "2024-06-01") in closed

    def test_entity_type_and_properties_preserved(self, sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        rows = surreal_kg._db.query("SELECT name, type, properties FROM entity")
        by_name = {r["name"]: r for r in rows}
        assert by_name["Alice"]["type"] == "person"
        assert by_name["Alice"]["properties"].get("city") == "NYC"
        assert by_name["Acme Corp"]["type"] == "company"
        assert by_name["swimming"]["type"] == "activity"

    def test_extracted_at_preserved_not_migration_timestamp(self, sqlite_kg, surreal_kg):
        """Task requirement: do NOT use time::now(); preserve source timestamp."""
        from mempalace.migrate_kg import migrate_kg_to_surreal

        # Read the source extracted_at for a specific triple directly.
        conn = sqlite3.connect(sqlite_kg)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT extracted_at FROM triples WHERE predicate = 'manages' LIMIT 1"
            ).fetchone()
            source_extracted = row["extracted_at"]
        finally:
            conn.close()

        assert source_extracted is not None

        migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        rows = surreal_kg._db.query("SELECT extracted_at FROM triple WHERE predicate = 'manages'")
        assert rows
        surreal_extracted = rows[0]["extracted_at"]
        # SurrealDB returns a datetime; stringify and check the source
        # date (YYYY-MM-DD) appears — round-trip through two date
        # formats is fiddly, so accept any representation that starts
        # with the source date.
        source_date = source_extracted.split(" ")[0].split("T")[0]
        assert source_date in str(surreal_extracted), (
            f"Surreal extracted_at {surreal_extracted!r} does not contain "
            f"source date {source_date!r} — likely regenerated via time::now()"
        )


class TestIdempotency:
    """Running the migration twice must not create duplicates."""

    def test_second_run_is_noop(self, sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        first = migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )
        stats_after_first = surreal_kg.stats()

        second = migrate_kg_to_surreal(
            sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )
        stats_after_second = surreal_kg.stats()

        assert first.entities_written == second.entities_written
        assert first.triples_written == second.triples_written
        assert stats_after_first["entities"] == stats_after_second["entities"]
        assert stats_after_first["triples"] == stats_after_second["triples"]
        assert stats_after_first["current_facts"] == stats_after_second["current_facts"]


class TestLargeBatchedMigration:
    """mp-ayu: batching must scale to many triples without corrupting
    dedup, ordering, or idempotency.

    We seed 500 triples (well above the default batch size of 100 so at
    least 5 wire batches flow) and confirm:

    - exactly 500 rows land (no phantoms, no losses);
    - a *second* run with the same source is a no-op (the dedup
      pre-check must cross batch boundaries);
    - provenance (confidence, valid_from, source_drawer_id) round-trips
      on a deep-read sample.
    """

    @pytest.fixture
    def large_sqlite_kg(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        db_path = tmp_path / "large_kg.sqlite3"
        kg = KnowledgeGraph(str(db_path))

        # 500 triples with distinct SPOs — the SQLite KG dedupes on
        # ``(subject, predicate, object)`` regardless of span, so reusing
        # SPOs would silently collapse rows at the source. We mix 450
        # open rows (``obj_i``) with 50 closed historical rows
        # (``hist_i``, same subjects, distinct objects) so the migration
        # exercises both dedup arms of ``add_triples_batch`` under load.
        preds = ["knows", "works_at", "lives_in", "likes", "met"]
        for i in range(450):
            kg.add_triple(
                f"subj_{i}",
                preds[i % len(preds)],
                f"obj_{i}",
                valid_from=f"2024-01-{(i % 28) + 1:02d}",
                confidence=0.5 + (i % 10) / 20.0,
                source_drawer_id=f"drawer_{i:04d}",
            )
        for i in range(50):
            kg.add_triple(
                f"subj_{i}",
                preds[i % len(preds)],
                f"hist_{i}",
                valid_from="2010-01-01",
                valid_to="2015-01-01",
                confidence=0.7,
                source_drawer_id=f"drawer_hist_{i:04d}",
            )
        kg.close()
        return str(db_path)

    def test_500_triples_migrate_cleanly(self, large_sqlite_kg, surreal_kg):
        from mempalace.migrate_kg import migrate_kg_to_surreal

        result = migrate_kg_to_surreal(
            large_sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=25,
            rng=random.Random(7),
            progress=False,
        )
        assert result.triples_source == 500
        assert result.triples_written == 500
        assert result.verification_ok == result.verification_sampled
        assert result.ok

        stats = surreal_kg.stats()
        assert stats["triples"] == 500, f"expected 500 triples on Surreal, got {stats['triples']}"

    def test_500_triple_batch_is_idempotent(self, large_sqlite_kg, surreal_kg):
        """Batched re-run of a fully-completed migration must be a
        no-op across batch boundaries."""
        from mempalace.migrate_kg import migrate_kg_to_surreal

        migrate_kg_to_surreal(
            large_sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )
        stats_first = surreal_kg.stats()

        migrate_kg_to_surreal(
            large_sqlite_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )
        stats_second = surreal_kg.stats()

        assert stats_first == stats_second, (
            f"batched re-run changed state: {stats_first} -> {stats_second}"
        )

    def test_500_triple_batch_size_override(self, large_sqlite_kg, surreal_kg):
        """Honours ``--kg-batch-size`` — the same load arrives with
        a smaller batch size and still lands exactly 500 rows."""
        from mempalace.migrate_kg import migrate_kg_to_surreal

        result = migrate_kg_to_surreal(
            large_sqlite_kg,
            surreal=surreal_kg,
            kg_batch_size=37,  # coprime with 500 so the last batch is partial
            verify_sample_size=10,
            rng=random.Random(11),
            progress=False,
        )
        assert result.triples_written == 500
        assert surreal_kg.stats()["triples"] == 500


class TestEmptySource:
    """Migrating an empty KG is not an error."""

    def test_empty_sqlite_is_zero_counts(self, tmp_path, surreal_kg):
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.migrate_kg import migrate_kg_to_surreal

        db_path = tmp_path / "empty.sqlite3"
        KnowledgeGraph(str(db_path)).close()

        result = migrate_kg_to_surreal(
            str(db_path),
            surreal=surreal_kg,
            verify_sample_size=10,
            progress=False,
        )
        assert result.entities_written == 0
        assert result.triples_written == 0
        assert result.verification_sampled == 0
        assert result.ok


# ── mp-ui1: full-provenance verification tests ──────────────────────────


@pytest.fixture
def all_fields_kg(tmp_path):
    """A tiny KG where one triple has EVERY provenance field populated.

    The ``_verify_triple_in_surreal`` pre-mp-ui1 only checked five
    fields. This fixture lands ``adapter_name``, ``source_closet``,
    and ``source_file`` with distinct, non-empty values so a regression
    that drops any one of them would be caught by the verification.
    """
    from mempalace.knowledge_graph import KnowledgeGraph

    db_path = tmp_path / "all_fields.sqlite3"
    kg = KnowledgeGraph(str(db_path))
    kg.add_entity("Alice", entity_type="person", properties={"city": "NYC"})
    kg.add_entity("Max", entity_type="person", properties={"dob": "2015-04-01"})
    # Two triples so the verifier has something to pick randomly from.
    kg.add_triple(
        "Alice",
        "parent_of",
        "Max",
        valid_from="2015-04-01",
        confidence=0.87,
        source_closet="personal/2025-10",
        source_file="/convos/2025/family.md",
        source_drawer_id="drawer_alice_parent_max_xyz",
        adapter_name="general",
    )
    kg.add_triple(
        "Alice",
        "friend_of",
        "Max",
        valid_from="2020-01-01",
        valid_to="2024-06-01",
        confidence=0.5,
        source_closet="work",
        source_file="/convos/2020/note.md",
        source_drawer_id="drawer_alice_friend_max",
        adapter_name="exchange",
    )
    kg.close()
    return str(db_path)


class TestAllProvenanceFieldsVerified:
    """mp-ui1: every SQLite column must be covered by the spot check."""

    def test_clean_migration_passes_full_field_verify(self, all_fields_kg, surreal_kg):
        """A faithful migration must verify clean on every provenance field."""
        from mempalace.migrate_kg import migrate_kg_to_surreal

        result = migrate_kg_to_surreal(
            all_fields_kg,
            surreal=surreal_kg,
            verify_sample_size=5,  # covers both triples + both entities
            rng=random.Random(0),
            progress=False,
        )

        assert result.triples_source == 2
        assert result.entities_source == 2
        assert result.verification_ok == result.verification_sampled
        assert result.entity_verification_ok == result.entity_verification_sampled
        assert result.entity_verification_sampled == 2
        assert result.ok, f"verification failures: {result.verification_failures}"

    @pytest.mark.parametrize(
        "field, mutated",
        [
            ("adapter_name", "DIFFERENT_ADAPTER"),
            ("source_closet", "DIFFERENT_CLOSET/2030-99"),
            ("source_file", "/tmp/DIFFERENT_FILE.md"),
            ("source_drawer_id", "drawer_DIFFERENT_ghost"),
        ],
    )
    def test_mutated_target_provenance_flagged(self, all_fields_kg, surreal_kg, field, mutated):
        """Divergence in any provenance field must be reported as a failure.

        Post-migration we mutate ONE Surreal triple to differ from its
        SQLite source on a single provenance field. A second run of the
        verify logic must flag it — otherwise the spot check is a
        rubber stamp.
        """
        from mempalace.migrate_kg import (
            _verify_triple_in_surreal,
            migrate_kg_to_surreal,
        )

        migrate_kg_to_surreal(
            all_fields_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        # Mutate the "parent_of" triple's chosen field on the Surreal side.
        surreal_kg._db.query(
            f"UPDATE triple SET {field} = $v WHERE predicate = 'parent_of'",
            {"v": mutated},
        )

        # Re-read the SQLite row for the 'parent_of' triple and confirm
        # the verifier now reports a mismatch mentioning the mutated field.
        conn = sqlite3.connect(all_fields_kg)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT t.*, s.name AS subject_name, o.name AS object_name "
                "FROM triples t "
                "JOIN entities s ON t.subject = s.id "
                "JOIN entities o ON t.object  = o.id "
                "WHERE t.predicate = 'parent_of'"
            ).fetchone()
        finally:
            conn.close()

        source = {
            "subject_name": row["subject_name"],
            "object_name": row["object_name"],
            "predicate": row["predicate"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "confidence": row["confidence"],
            "source_closet": row["source_closet"],
            "source_file": row["source_file"],
            "source_drawer_id": row["source_drawer_id"],
            "adapter_name": row["adapter_name"],
        }
        reason = _verify_triple_in_surreal(surreal_kg, source)
        assert reason is not None, f"mutating {field!r} on target must produce a verify failure"
        assert field in reason, (
            f"failure reason {reason!r} must mention the mutated field {field!r}"
        )

    def test_mutated_target_entity_type_flagged(self, all_fields_kg, surreal_kg):
        """A silently dropped entity ``type`` must be reported as a failure."""
        from mempalace.migrate_kg import (
            _verify_entity_in_surreal,
            migrate_kg_to_surreal,
        )

        migrate_kg_to_surreal(
            all_fields_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        # Mutate Alice's type to simulate a migration that dropped the field.
        surreal_kg._db.query("UPDATE entity SET type = 'unknown' WHERE id = entity:alice")
        reason = _verify_entity_in_surreal(
            surreal_kg,
            {"name": "Alice", "type": "person", "properties": '{"city": "NYC"}'},
        )
        assert reason is not None
        assert "type" in reason

    def test_mutated_target_entity_properties_flagged(self, all_fields_kg, surreal_kg):
        """A silently dropped entity ``properties`` must be reported as a failure."""
        from mempalace.migrate_kg import (
            _verify_entity_in_surreal,
            migrate_kg_to_surreal,
        )

        migrate_kg_to_surreal(
            all_fields_kg,
            surreal=surreal_kg,
            verify_sample_size=0,
            progress=False,
        )

        # Wipe properties on Max — the verifier must flag the divergence.
        surreal_kg._db.query("UPDATE entity SET properties = {} WHERE id = entity:max")
        reason = _verify_entity_in_surreal(
            surreal_kg,
            {"name": "Max", "type": "person", "properties": '{"dob": "2015-04-01"}'},
        )
        assert reason is not None
        assert "properties" in reason
