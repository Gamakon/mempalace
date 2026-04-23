"""
migrate_kg.py — Copy the SQLite Knowledge Graph into SurrealDB (mp-3lc).

Reads entities + triples straight out of the SQLite KG (`knowledge_graph.py`)
read-only and replays them through :class:`mempalace.kg_surreal.KnowledgeGraphSurreal`
so the Surreal backend ends up with the same graph state. The SQLite file is
never modified.

Coordination with mp-ciw
------------------------
mp-ciw is landing the drawers+embeddings migration as a separate
``mempalace migrate-to-surreal`` command. At the time this module was
written mp-ciw had not yet merged, so KG migration ships as its own
subcommand — ``mempalace migrate-kg-to-surreal`` — and its own module.
Unifying surfaces is explicitly a later task: the plumbing is in a
dedicated module so unification is a rename plus an ``--include-kg``
option on the unified command, not a rewrite.

Why not reuse ``mempalace migrate``?
------------------------------------
``mempalace/migrate.py`` is the existing ChromaDB-version recovery tool
(it reads Chroma's internal SQLite schema, not ours). Overloading that
name would confuse users — there is no backend switch happening in
``migrate.py``, just a Chroma-to-Chroma upgrade.

Design rules (from the task brief)
----------------------------------
* **Read-only SQLite access.** Opens the source DB via
  ``sqlite3.connect(..., uri=True)`` with ``?mode=ro`` so an accidental
  write is physically impossible.
* **Provenance preserved.** ``extracted_at`` from the SQLite row is
  passed through to :meth:`KnowledgeGraphSurreal.add_triple` — the
  Surreal edge keeps the original extraction timestamp instead of
  ``time::now()`` at migration time.
* **Idempotent.** Re-running the migration must not create duplicate
  triples. ``add_triple`` dedupes open triples by
  ``(subject, predicate, object)`` already; mp-3lc extended it to dedupe
  closed triples on the full ``(subject, predicate, object, valid_from,
  valid_to)`` key. Entity UPSERT is naturally idempotent.
* **Entity order.** Entities are written first so triples that reference
  them never see a missing endpoint — though ``add_triple`` UPSERTs the
  endpoints defensively anyway. This keeps behaviour correct even if the
  SQLite ``entities`` table is missing a row its own triples reference.
* **Verification.** After migration we sample 10 random triples from
  SQLite and confirm they exist in Surreal with matching core fields.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .kg_surreal import KnowledgeGraphSurreal


DEFAULT_BATCH_SIZE = 500


# ── Source reader ──────────────────────────────────────────────────────


def _open_readonly(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` read-only via URI so we cannot mutate the source.

    Using ``mode=ro`` means any write attempt raises ``SQLITE_READONLY``
    at the SQL layer — stronger than just "we promise not to write".
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _iter_entities(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Yield every row from the ``entities`` table as a plain dict."""
    cur = conn.execute("SELECT id, name, type, properties FROM entities ORDER BY id")
    for row in cur:
        yield {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"] or "unknown",
            "properties": row["properties"] or "{}",
        }


def _iter_triples(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Yield every row from the ``triples`` table joined with entity names.

    We need the entity *display names* (not slugs) because
    ``KnowledgeGraphSurreal.add_triple`` re-normalises names via its own
    ``_entity_id``; passing it the slug works (idempotent) but passing
    the original display name keeps the entity's ``name`` field in the
    Surreal ``entity`` record consistent with what the SQLite KG held.
    """
    cur = conn.execute(
        """
        SELECT
            t.id                AS id,
            t.subject           AS subject_slug,
            t.predicate         AS predicate,
            t.object            AS object_slug,
            t.valid_from        AS valid_from,
            t.valid_to          AS valid_to,
            t.confidence        AS confidence,
            t.source_closet     AS source_closet,
            t.source_file       AS source_file,
            t.source_drawer_id  AS source_drawer_id,
            t.adapter_name      AS adapter_name,
            t.extracted_at      AS extracted_at,
            s.name              AS subject_name,
            o.name              AS object_name
        FROM triples t
        LEFT JOIN entities s ON t.subject = s.id
        LEFT JOIN entities o ON t.object  = o.id
        ORDER BY t.id
        """
    )
    for row in cur:
        # Fall back to the slug if the entity row is missing — keeps the
        # migration total-at-all-costs. SQLite KG's FK is declared but
        # not enforced, so dangling triples do occur in the wild.
        yield {
            "id": row["id"],
            "subject_slug": row["subject_slug"],
            "object_slug": row["object_slug"],
            "subject_name": row["subject_name"] or row["subject_slug"],
            "object_name": row["object_name"] or row["object_slug"],
            "predicate": row["predicate"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
            "source_closet": row["source_closet"],
            "source_file": row["source_file"],
            "source_drawer_id": row["source_drawer_id"],
            "adapter_name": row["adapter_name"],
            "extracted_at": row["extracted_at"],
        }


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


# ── Migration ──────────────────────────────────────────────────────────


@dataclass
class MigrationResult:
    """Summary of a single ``migrate_kg_to_surreal`` run."""

    entities_source: int
    entities_written: int
    triples_source: int
    triples_written: int
    verification_sampled: int
    verification_ok: int
    verification_failures: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return (
            self.entities_written == self.entities_source
            and self.triples_written == self.triples_source
            and self.verification_ok == self.verification_sampled
        )


def migrate_kg_to_surreal(
    sqlite_path: str,
    *,
    surreal: Optional[KnowledgeGraphSurreal] = None,
    url: str = "ws://127.0.0.1:8000",
    user: str = "root",
    password: str = "root",
    namespace: str = "mempalace",
    database: str = "kg",
    batch_size: int = DEFAULT_BATCH_SIZE,
    verify_sample_size: int = 10,
    rng: Optional[random.Random] = None,
    progress: bool = True,
) -> MigrationResult:
    """Copy a SQLite KG at ``sqlite_path`` into SurrealDB.

    Parameters
    ----------
    sqlite_path:
        Path to the SQLite KG file (``~/.mempalace/knowledge_graph.sqlite3``
        by default). Opened read-only.
    surreal:
        Optional pre-connected :class:`KnowledgeGraphSurreal`. When
        ``None`` we build one from ``url/user/password/namespace/database``
        and close it before returning. Tests pass a live instance so the
        fixture can tear the DB down cleanly.
    batch_size:
        Progress-print granularity. Does not affect Surreal write
        batching — ``add_triple`` is one network round-trip per triple
        regardless, matching the SQLite KG's per-triple commit model.
    verify_sample_size:
        How many random triples to re-read from Surreal post-migration
        and check against SQLite. ``0`` disables verification.
    rng:
        Optional deterministic RNG for the verification sample. CI uses
        this so failures are reproducible.
    progress:
        When ``True`` prints ``"  [n/total] …"`` lines as it goes. Tests
        pass ``False`` to keep output clean.

    Returns
    -------
    :class:`MigrationResult` with per-stage counts and verification
    status. Caller checks ``.ok`` for a one-liner success test.
    """
    rng = rng or random.Random()
    owns_surreal = surreal is None
    if surreal is None:
        surreal = KnowledgeGraphSurreal(
            url=url,
            user=user,
            password=password,
            namespace=namespace,
            database=database,
        )

    try:
        conn = _open_readonly(sqlite_path)
        try:
            entities_total = _count(conn, "entities")
            triples_total = _count(conn, "triples")

            if progress:
                print(f"  KG source: {entities_total} entities, {triples_total} triples")

            # ── Entities ───────────────────────────────────────────────
            entities_written = 0
            for i, ent in enumerate(_iter_entities(conn), start=1):
                # Parse properties JSON lazily — most rows are "{}".
                import json

                try:
                    props = json.loads(ent["properties"]) if ent["properties"] else {}
                except (ValueError, TypeError):
                    # Corrupt/invalid JSON: migrate the row with empty
                    # properties rather than abort the whole job.
                    props = {}

                surreal.add_entity(
                    ent["name"],
                    entity_type=ent["type"],
                    properties=props,
                )
                entities_written += 1
                if progress and (i % batch_size == 0 or i == entities_total):
                    print(f"  entities: {i}/{entities_total}")

            # ── Triples ────────────────────────────────────────────────
            triples_written = 0
            for i, tri in enumerate(_iter_triples(conn), start=1):
                surreal.add_triple(
                    tri["subject_name"],
                    tri["predicate"],
                    tri["object_name"],
                    valid_from=tri["valid_from"],
                    valid_to=tri["valid_to"],
                    confidence=tri["confidence"],
                    source_closet=tri["source_closet"],
                    source_file=tri["source_file"],
                    source_drawer_id=tri["source_drawer_id"],
                    adapter_name=tri["adapter_name"],
                    extracted_at=_normalize_extracted_at(tri["extracted_at"]),
                )
                triples_written += 1
                if progress and (i % batch_size == 0 or i == triples_total):
                    print(f"  triples:  {i}/{triples_total}")

            # ── Verification ───────────────────────────────────────────
            sample_size = min(verify_sample_size, triples_total)
            failures: list[dict[str, Any]] = []
            ok = 0
            if sample_size > 0:
                # Re-read from SQLite — source of truth for the check.
                all_triples = list(_iter_triples(conn))
                sample = rng.sample(all_triples, sample_size)
                for tri in sample:
                    if _verify_triple_in_surreal(surreal, tri):
                        ok += 1
                    else:
                        failures.append(
                            {
                                "subject": tri["subject_name"],
                                "predicate": tri["predicate"],
                                "object": tri["object_name"],
                                "valid_from": tri["valid_from"],
                                "valid_to": tri["valid_to"],
                            }
                        )

                if progress:
                    print(f"  verification: {ok}/{sample_size} triples round-tripped")
                    for f in failures:
                        print(f"    MISS: {f}")

            return MigrationResult(
                entities_source=entities_total,
                entities_written=entities_written,
                triples_source=triples_total,
                triples_written=triples_written,
                verification_sampled=sample_size,
                verification_ok=ok,
                verification_failures=failures,
            )
        finally:
            conn.close()
    finally:
        if owns_surreal:
            surreal.close()


def _normalize_extracted_at(raw: Optional[str]) -> Optional[str]:
    """Coerce SQLite's ``CURRENT_TIMESTAMP`` format to ISO 8601 for Surreal.

    SQLite stores ``"YYYY-MM-DD HH:MM:SS"`` (space separator) when
    ``CURRENT_TIMESTAMP`` fires. SurrealDB's ``<datetime>`` cast wants
    RFC 3339 / ISO 8601 (``T`` separator, trailing ``Z`` for UTC). We do
    the cheap substitution here so every row flows through
    :meth:`KnowledgeGraphSurreal.add_triple` unchanged downstream.
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if "T" not in value and " " in value:
        value = value.replace(" ", "T", 1)
    if value.endswith("Z") or "+" in value or value.count("-") > 2:
        return value
    # Assume UTC — matches SQLite's CURRENT_TIMESTAMP semantics.
    return value + "Z"


def _verify_triple_in_surreal(surreal: KnowledgeGraphSurreal, source: dict[str, Any]) -> bool:
    """Confirm ``source`` exists in Surreal with matching core fields.

    Checks predicate, valid_from/to, confidence, source_drawer_id. We
    deliberately do *not* check ``extracted_at`` or triple ``id`` because
    both are naturally regenerated by the Surreal side and the task
    brief explicitly accepts new triple IDs.
    """
    pred = source["predicate"]
    sub_rec = surreal._entity_record(source["subject_name"])
    obj_rec = surreal._entity_record(source["object_name"])

    rows = surreal._db.query(
        (
            "SELECT predicate, valid_from, valid_to, confidence, "
            "source_drawer_id "
            "FROM triple WHERE in = $sub AND out = $obj "
            "AND predicate = $pred"
        ),
        {"sub": sub_rec, "obj": obj_rec, "pred": pred},
    )
    if not rows:
        return False
    for row in rows:
        if (
            row.get("valid_from") == source["valid_from"]
            and row.get("valid_to") == source["valid_to"]
            and _approx_eq(row.get("confidence"), source["confidence"])
            and row.get("source_drawer_id") == source["source_drawer_id"]
        ):
            return True
    return False


def _approx_eq(a: Any, b: Any, tol: float = 1e-6) -> bool:
    """Floats round-trip through JSON; compare with a small tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b
