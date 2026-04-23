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
  SQLite and confirm they exist in Surreal with every provenance field
  matching (predicate, valid_from, valid_to, confidence, source_closet,
  source_file, source_drawer_id, adapter_name). We also sample 10
  entities and confirm ``type`` and ``properties`` round-trip intact.
  Only ``extracted_at`` and generated record ids are excluded — both
  cross format boundaries (see ``_verify_triple_in_surreal``).
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .kg_surreal import KnowledgeGraphSurreal


DEFAULT_BATCH_SIZE = 500
# Surreal write batch — how many triples we group into one multi-statement
# RELATE round-trip (mp-ayu). 100 keeps the outgoing WebSocket frame well
# under typical limits while collapsing the per-triple RTT that dominated
# 10k+-triple migrations. Override per-run via the CLI ``--kg-batch-size``.
DEFAULT_KG_BATCH_SIZE = 100


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
    entity_verification_sampled: int = 0
    entity_verification_ok: int = 0
    entity_verification_failures: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Dataclass can't default a mutable ``list`` via ``= []`` — use
        # ``None`` sentinel and fix up here. Keeps the dataclass body
        # simple without a ``field(default_factory=list)`` import.
        if self.entity_verification_failures is None:
            self.entity_verification_failures = []

    @property
    def ok(self) -> bool:
        return (
            self.entities_written == self.entities_source
            and self.triples_written == self.triples_source
            and self.verification_ok == self.verification_sampled
            and self.entity_verification_ok == self.entity_verification_sampled
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
    kg_batch_size: int = DEFAULT_KG_BATCH_SIZE,
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
        Progress-print granularity. Independent of ``kg_batch_size``:
        entity rows still print one line per ``batch_size`` upserts,
        and triple progress prints one line per Surreal write batch.
    kg_batch_size:
        How many triples to bundle into a single
        :meth:`KnowledgeGraphSurreal.add_triples_batch` call — that is
        the on-the-wire batch size (mp-ayu). 100 is the default and
        keeps round-trips in the sub-millisecond-per-triple range even
        at 10k+ triples. Raise it for faster bulk loads if the Surreal
        frame size holds; lower it to cap memory on very wide triples.
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
            # mp-ayu: batch N triples per Surreal round-trip.
            # ``add_triples_batch`` collapses entity ensure + dedup check
            # + N RELATEs into 3 queries total (vs the old 3N+). Progress
            # prints per batch, not per triple, so the output stays sane
            # at 10k+ scale.
            triples_written = 0
            batch: list[dict[str, Any]] = []

            def _flush() -> None:
                nonlocal triples_written
                if not batch:
                    return
                surreal.add_triples_batch(batch)
                triples_written += len(batch)
                if progress:
                    print(f"  triples:  {triples_written}/{triples_total}")
                batch.clear()

            for tri in _iter_triples(conn):
                batch.append(
                    {
                        "subject": tri["subject_name"],
                        "predicate": tri["predicate"],
                        "obj": tri["object_name"],
                        "valid_from": tri["valid_from"],
                        "valid_to": tri["valid_to"],
                        "confidence": tri["confidence"],
                        "source_closet": tri["source_closet"],
                        "source_file": tri["source_file"],
                        "source_drawer_id": tri["source_drawer_id"],
                        "adapter_name": tri["adapter_name"],
                        "extracted_at": _normalize_extracted_at(tri["extracted_at"]),
                    }
                )
                if len(batch) >= kg_batch_size:
                    _flush()
            _flush()

            # ── Verification ───────────────────────────────────────────
            # Every SQLite column that exists on the source must be
            # checked on the target — otherwise the spot check is a
            # rubber stamp. We audit:
            #   triples: predicate, valid_from, valid_to, confidence,
            #            source_closet, source_file, source_drawer_id,
            #            adapter_name. (extracted_at + id are excluded
            #            by design; see ``_verify_triple_in_surreal``.)
            #   entities: type, properties (the only non-provenance,
            #             non-trivial columns on the entities table —
            #             created_at is SQLite-generated, id is the
            #             slug, name is already checked by the triple
            #             endpoint match.)
            sample_size = min(verify_sample_size, triples_total)
            failures: list[dict[str, Any]] = []
            ok = 0
            if sample_size > 0:
                # Re-read from SQLite — source of truth for the check.
                all_triples = list(_iter_triples(conn))
                sample = rng.sample(all_triples, sample_size)
                for tri in sample:
                    mismatch = _verify_triple_in_surreal(surreal, tri)
                    if mismatch is None:
                        ok += 1
                    else:
                        failures.append(
                            {
                                "subject": tri["subject_name"],
                                "predicate": tri["predicate"],
                                "object": tri["object_name"],
                                "valid_from": tri["valid_from"],
                                "valid_to": tri["valid_to"],
                                "reason": mismatch,
                            }
                        )

                if progress:
                    print(f"  verification: {ok}/{sample_size} triples round-tripped")
                    for f in failures:
                        print(f"    MISS: {f}")

            # Entity verification — sample the same number of entities
            # (capped by source count) and confirm type + properties
            # round-trip. Without this step a migration that silently
            # dropped every entity's ``type``/``properties`` would still
            # pass the old spot check.
            ent_sample_size = min(verify_sample_size, entities_total)
            ent_failures: list[dict[str, Any]] = []
            ent_ok = 0
            if ent_sample_size > 0:
                all_entities = list(_iter_entities(conn))
                ent_sample = rng.sample(all_entities, ent_sample_size)
                for ent in ent_sample:
                    mismatch = _verify_entity_in_surreal(surreal, ent)
                    if mismatch is None:
                        ent_ok += 1
                    else:
                        ent_failures.append(
                            {
                                "name": ent["name"],
                                "reason": mismatch,
                            }
                        )

                if progress:
                    print(f"  verification: {ent_ok}/{ent_sample_size} entities round-tripped")
                    for f in ent_failures:
                        print(f"    MISS: {f}")

            return MigrationResult(
                entities_source=entities_total,
                entities_written=entities_written,
                triples_source=triples_total,
                triples_written=triples_written,
                verification_sampled=sample_size,
                verification_ok=ok,
                verification_failures=failures,
                entity_verification_sampled=ent_sample_size,
                entity_verification_ok=ent_ok,
                entity_verification_failures=ent_failures,
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


def _verify_triple_in_surreal(
    surreal: KnowledgeGraphSurreal, source: dict[str, Any]
) -> Optional[str]:
    """Confirm ``source`` exists in Surreal with matching core fields.

    Returns
    -------
    ``None`` when every checked field matches; otherwise a short string
    describing the first mismatch (for failure reporting).

    Checks EVERY SQLite triple column that the migration passes through:
    ``predicate``, ``valid_from``, ``valid_to``, ``confidence``,
    ``source_closet``, ``source_file``, ``source_drawer_id``,
    ``adapter_name``.

    ``extracted_at`` and triple ``id`` are intentionally excluded:

    * ``id`` is regenerated by the Surreal RELATE — the migration brief
      accepts a fresh id.
    * ``extracted_at`` crosses a format boundary (SQLite
      ``"YYYY-MM-DD HH:MM:SS"`` -> Surreal ``datetime``). Bit-exact
      equality is impossible; the dedicated audit
      (``verify_migration.py``) handles it with a date-prefix rule.

    Any other SQLite column must be checked here — otherwise a silently
    dropped field would round-trip as "OK" and the spot check becomes a
    rubber stamp.
    """
    pred_norm = surreal._normalize_predicate(source["predicate"])
    sub_rec = surreal._entity_record(source["subject_name"])
    obj_rec = surreal._entity_record(source["object_name"])

    rows = surreal._db.query(
        (
            "SELECT predicate, valid_from, valid_to, confidence, "
            "source_closet, source_file, source_drawer_id, adapter_name "
            "FROM triple WHERE in = $sub AND out = $obj "
            "AND predicate = $pred"
        ),
        {"sub": sub_rec, "obj": obj_rec, "pred": pred_norm},
    )
    if not rows:
        return "missing: no triple with matching (subject, predicate, object)"

    # Candidates for this (sub, pred, obj) — the right one matches
    # valid_from + valid_to first, then every provenance field.
    last_reason = "no candidate matched valid_from/valid_to"
    for row in rows:
        if row.get("valid_from") != source["valid_from"]:
            last_reason = f"valid_from: src={source['valid_from']!r} dst={row.get('valid_from')!r}"
            continue
        if row.get("valid_to") != source["valid_to"]:
            last_reason = f"valid_to: src={source['valid_to']!r} dst={row.get('valid_to')!r}"
            continue
        # Provenance must match field-for-field on the located row.
        mismatch = _check_triple_provenance(row, source)
        if mismatch is None:
            return None
        last_reason = mismatch
    return last_reason


def _check_triple_provenance(row: dict[str, Any], source: dict[str, Any]) -> Optional[str]:
    """Compare every provenance field between a Surreal row and SQLite source.

    Returns the first mismatch as a short string, or ``None`` on full
    equality. ``confidence`` is compared with a float tolerance; all
    other fields are compared with ``==`` after treating ``""``/``None``
    as equivalent (SurrealDB returns ``None`` for unset optional fields
    and SQLite may store ``NULL`` or empty string depending on the
    write path).
    """
    if not _approx_eq(row.get("confidence"), source["confidence"]):
        return f"confidence: src={source['confidence']!r} dst={row.get('confidence')!r}"
    for field_name in (
        "source_closet",
        "source_file",
        "source_drawer_id",
        "adapter_name",
    ):
        src_val = source.get(field_name)
        dst_val = row.get(field_name)
        # Treat ``None`` and empty string as equivalent — Surreal drops
        # NULL-valued optional fields from query results, and SQLite can
        # legitimately store either. No other falsy coercion: ``0`` and
        # ``False`` are never valid values for these string columns.
        if (src_val or None) != (dst_val or None):
            return f"{field_name}: src={src_val!r} dst={dst_val!r}"
    return None


def _verify_entity_in_surreal(
    surreal: KnowledgeGraphSurreal, source: dict[str, Any]
) -> Optional[str]:
    """Confirm an entity row round-tripped with type + properties intact.

    SQLite's ``entities`` table has these columns:

    * ``id``         — slug; derived from ``name`` the same way on both
                       sides. Checked implicitly by looking up the
                       Surreal record by slug.
    * ``name``       — already verified whenever a triple's endpoint
                       matches on either side; also checked here on the
                       Surreal record itself for defence in depth.
    * ``type``       — must round-trip. Default ``"unknown"`` is still a
                       value and must land on the target.
    * ``properties`` — JSON blob; parsed here and compared structurally.
    * ``created_at`` — SQLite-generated timestamp. Intentionally
                       excluded (same reason as triples' ``extracted_at``
                       — format boundary, and the migration brief does
                       not require it preserved).

    Returns ``None`` on full match, otherwise a short mismatch string.
    """
    import json

    rows = surreal._db.query(
        "SELECT name, type, properties FROM entity WHERE id = $rec",
        {"rec": surreal._entity_record(source["name"])},
    )
    if not rows:
        return f"missing: no entity for name {source['name']!r}"
    row = rows[0]

    if row.get("name") != source["name"]:
        return f"name: src={source['name']!r} dst={row.get('name')!r}"

    # ``type`` default is "unknown" on the SQLite side. Surreal receives
    # that value verbatim through ``add_entity`` — must round-trip.
    src_type = source.get("type") or "unknown"
    dst_type = row.get("type")
    if dst_type != src_type:
        return f"type: src={src_type!r} dst={dst_type!r}"

    # Properties: SQLite stores JSON text; Surreal stores a native
    # object. Normalise both to a dict and compare structurally.
    src_props_raw = source.get("properties") or "{}"
    try:
        src_props = json.loads(src_props_raw) if isinstance(src_props_raw, str) else src_props_raw
    except (ValueError, TypeError):
        src_props = {}
    dst_props = row.get("properties") or {}
    if src_props != dst_props:
        return f"properties: src={src_props!r} dst={dst_props!r}"
    return None


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
