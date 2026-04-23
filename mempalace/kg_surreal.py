"""
kg_surreal.py — SurrealDB port of the temporal knowledge graph.
================================================================

Parallel implementation of ``knowledge_graph.KnowledgeGraph`` backed by
SurrealDB RELATE edges instead of SQLite join tables. This module is
additive — the SQLite KG stays the source of truth for now; the backend
switch happens in a later task (see mp-6gu for the overall migration
plan).

Scope (mp-4yf):
    - ``add_entity``           (parity with SQLite KG)
    - ``add_triple``           via ``RELATE $sub->triple->$obj SET ...``
    - ``query_entity``         outgoing/incoming/both, with ``as_of`` filter
    - ``query_relationship``   all triples of a given predicate
    - ``list_triples``         basic listing helper (no SQLite equivalent)
    - ``stats``                counts + distinct predicates

Out of scope for this task (stubbed — raise ``NotImplementedError`` pointing
at mp-84n):
    - ``timeline``
    - ``invalidate`` (write-side temporal close)
    - ``seed_from_entity_facts``

Connection model:
    A fresh ``KnowledgeGraphSurreal`` signs in to the local SurrealDB
    instance (see ``docs/surrealdb-local.md``) and ``USE`` the namespace +
    database passed to ``__init__``. One HTTP connection per instance; the
    underlying client is not inherently thread-safe, so callers that share
    an instance across threads must serialise themselves (mirrors the
    ``threading.Lock`` discipline of the SQLite KG; we rely on the caller
    for now since the blocking HTTP client has no obvious contention story
    documented).

Schema notes:
    ``docs/surrealdb-schema.md`` §2 defines ``triple`` as SCHEMAFULL with
    ``valid_from``/``valid_to`` typed ``option<datetime>``. The SQLite KG
    however accepts arbitrary ISO date strings (``"2015-04-01"``). To
    preserve bit-exact caller semantics for mp-4yf we define the ``entity``
    and ``triple`` tables as SCHEMALESS here and store the date fields as
    strings. The schema-tightening pass is mp-6gu. This divergence is
    called out in the task report.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from surrealdb import RecordID, Surreal


DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_USER = "root"
DEFAULT_PASS = "root"
DEFAULT_NS = "mempalace"
DEFAULT_DB = "kg"

# SurrealDB record ids have restrictive character rules when not quoted. We
# match the SQLite ``_entity_id`` normalisation as closely as possible and
# then strip anything outside ``[a-z0-9_]`` so the raw slug is always a
# valid bare record id. Display name is preserved on the entity record.
_SLUG_SAFE_RE = re.compile(r"[^a-z0-9_]+")


class KnowledgeGraphSurreal:
    """SurrealDB-backed temporal knowledge graph (mp-4yf scope).

    Parameters
    ----------
    url, user, password:
        SurrealDB endpoint + root credentials. Defaults match
        ``docs/surrealdb-local.md``.
    namespace, database:
        Target ``NS`` / ``DB``. Callers pick these; tests use a dedicated
        ``test`` / ``mp_kg_test`` pair so they never collide with real data.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        namespace: str = DEFAULT_NS,
        database: str = DEFAULT_DB,
    ) -> None:
        self.url = url
        self.namespace = namespace
        self.database = database
        self._db = Surreal(url)
        self._db.signin({"username": user, "password": password})
        self._db.use(namespace, database)
        self._ensure_schema()

    # ── Internal helpers ───────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Idempotently create the ``entity`` + ``triple`` tables.

        SCHEMALESS for mp-4yf so caller date strings flow through
        unchanged (see module docstring). mp-6gu tightens this.
        """
        self._db.query("DEFINE TABLE IF NOT EXISTS entity SCHEMALESS")
        self._db.query(
            "DEFINE TABLE IF NOT EXISTS triple TYPE RELATION FROM entity TO entity SCHEMALESS"
        )

    @staticmethod
    def _entity_id(name: str) -> str:
        """Normalise a display name to a Surreal-safe slug.

        Mirrors ``KnowledgeGraph._entity_id`` (lowercase + underscored)
        with an extra sanitisation pass so the result is a valid bare
        record id for SurrealDB.
        """
        slug = name.lower().replace(" ", "_").replace("'", "")
        slug = _SLUG_SAFE_RE.sub("", slug)
        return slug or "unknown"

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        return predicate.lower().replace(" ", "_")

    def _entity_record(self, name: str) -> RecordID:
        return RecordID("entity", self._entity_id(name))

    def close(self) -> None:
        """Close the underlying SurrealDB connection."""
        try:
            self._db.close()
        except Exception:
            # Best-effort: the HTTP client has no persistent socket to
            # close cleanly; swallow to mirror SQLite KG's idempotent
            # ``close``.
            pass

    # ── Write operations ───────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        properties: Optional[dict] = None,
    ) -> str:
        """Upsert an entity node. Returns the slug (matches SQLite KG)."""
        eid = self._entity_id(name)
        rec = RecordID("entity", eid)
        self._db.query(
            "UPSERT $rec SET name = $name, type = $type, properties = $props",
            {
                "rec": rec,
                "name": name,
                "type": entity_type,
                "props": properties or {},
            },
        )
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        source_closet: Optional[str] = None,
        source_file: Optional[str] = None,
        source_drawer_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ) -> str:
        """Add a ``subject -> predicate -> object`` edge.

        Returns the Surreal record id of the triple (e.g.
        ``"triple:abc123"``) — caller treats it as an opaque string, same
        contract as the SQLite KG's ``t_<sub>_<pred>_<obj>_<hash>`` id.

        Dedupe rule mirrors SQLite: if an *open* triple (``valid_to IS
        NONE``) with the same ``(subject, predicate, object)`` already
        exists, return its id without creating a new edge.
        """
        sub_rec = self._entity_record(subject)
        obj_rec = self._entity_record(obj)
        pred = self._normalize_predicate(predicate)

        # Auto-create the endpoints if missing. UPSERT keeps it idempotent
        # and cheap — any existing entity keeps its type/properties.
        self._db.query(
            "UPSERT $rec SET name = $name",
            {"rec": sub_rec, "name": subject},
        )
        self._db.query(
            "UPSERT $rec SET name = $name",
            {"rec": obj_rec, "name": obj},
        )

        existing = self._db.query(
            (
                "SELECT id FROM triple WHERE in = $sub AND out = $obj "
                "AND predicate = $pred AND valid_to IS NONE"
            ),
            {"sub": sub_rec, "obj": obj_rec, "pred": pred},
        )
        if existing:
            return str(existing[0]["id"])

        created = self._db.query(
            (
                "RELATE $sub->triple->$obj SET "
                "predicate = $pred, "
                "valid_from = $valid_from, "
                "valid_to = $valid_to, "
                "confidence = $confidence, "
                "source_closet = $source_closet, "
                "source_file = $source_file, "
                "source_drawer_id = $source_drawer_id, "
                "adapter_name = $adapter_name"
            ),
            {
                "sub": sub_rec,
                "obj": obj_rec,
                "pred": pred,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": confidence,
                "source_closet": source_closet,
                "source_file": source_file,
                "source_drawer_id": source_drawer_id,
                "adapter_name": adapter_name,
            },
        )
        if not created:
            raise RuntimeError(f"RELATE returned no record for {subject}->{pred}->{obj}")
        return str(created[0]["id"])

    def invalidate(self, *_args, **_kwargs) -> None:
        """Temporal close of an open triple. Deferred to mp-84n."""
        raise NotImplementedError(
            "Temporal invalidate is out of scope for mp-4yf; tracked in mp-84n."
        )

    # ── Query operations ───────────────────────────────────────────────

    @staticmethod
    def _as_of_clause(prefix: str = "") -> str:
        """Build the standard temporal-validity filter.

        ``prefix`` lets callers prefix fields for alias disambiguation if
        they ever need it; left empty today since each of our queries
        only references one triple alias.
        """
        vf = f"{prefix}valid_from" if prefix else "valid_from"
        vt = f"{prefix}valid_to" if prefix else "valid_to"
        return f"({vf} IS NONE OR {vf} <= $as_of) " f"AND ({vt} IS NONE OR {vt} >= $as_of)"

    def query_entity(
        self,
        name: str,
        as_of: Optional[str] = None,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Return all triples touching ``name``.

        ``direction`` is one of ``"outgoing"``, ``"incoming"``, ``"both"``.
        ``as_of`` is an ISO date string; when provided, only triples whose
        ``(valid_from, valid_to)`` span contains it are returned.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"direction must be outgoing|incoming|both, got {direction!r}")

        rec = self._entity_record(name)
        params: dict[str, Any] = {"rec": rec}
        if as_of is not None:
            params["as_of"] = as_of

        results: list[dict[str, Any]] = []

        if direction in ("outgoing", "both"):
            q = (
                "SELECT id, predicate, valid_from, valid_to, confidence, "
                "source_closet, source_file, source_drawer_id, adapter_name, "
                "in.name AS sub_name, out.name AS obj_name "
                "FROM triple WHERE in = $rec"
            )
            if as_of is not None:
                q += " AND " + self._as_of_clause()
            rows = self._db.query(q, params)
            for row in rows or []:
                results.append(self._row_to_result(row, "outgoing", name, row["obj_name"]))

        if direction in ("incoming", "both"):
            q = (
                "SELECT id, predicate, valid_from, valid_to, confidence, "
                "source_closet, source_file, source_drawer_id, adapter_name, "
                "in.name AS sub_name, out.name AS obj_name "
                "FROM triple WHERE out = $rec"
            )
            if as_of is not None:
                q += " AND " + self._as_of_clause()
            rows = self._db.query(q, params)
            for row in rows or []:
                results.append(self._row_to_result(row, "incoming", row["sub_name"], name))

        return results

    def query_relationship(
        self,
        predicate: str,
        as_of: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return every triple whose predicate matches ``predicate``."""
        pred = self._normalize_predicate(predicate)
        params: dict[str, Any] = {"pred": pred}
        q = (
            "SELECT id, predicate, valid_from, valid_to, confidence, "
            "in.name AS sub_name, out.name AS obj_name "
            "FROM triple WHERE predicate = $pred"
        )
        if as_of is not None:
            q += " AND " + self._as_of_clause()
            params["as_of"] = as_of
        rows = self._db.query(q, params) or []
        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r.get("valid_from"),
                "valid_to": r.get("valid_to"),
                "current": r.get("valid_to") is None,
            }
            for r in rows
        ]

    def list_triples(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return up to ``limit`` triples (chronological tie-break).

        Deliberately simpler than ``timeline`` — mp-84n owns the full
        timeline semantics (NULLS LAST ordering, entity filtering, etc.).
        """
        rows = (
            self._db.query(
                (
                    "SELECT id, predicate, valid_from, valid_to, "
                    "in.name AS sub_name, out.name AS obj_name "
                    "FROM triple ORDER BY valid_from ASC LIMIT $limit"
                ),
                {"limit": limit},
            )
            or []
        )
        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r.get("valid_from"),
                "valid_to": r.get("valid_to"),
                "current": r.get("valid_to") is None,
            }
            for r in rows
        ]

    def timeline(self, *_args, **_kwargs) -> list[dict[str, Any]]:
        """Full chronological traversal. Deferred to mp-84n."""
        raise NotImplementedError("timeline() is out of scope for mp-4yf; tracked in mp-84n.")

    def seed_from_entity_facts(self, *_args, **_kwargs) -> None:
        """Bootstrap from fact_checker.ENTITY_FACTS. Deferred to mp-84n."""
        raise NotImplementedError(
            "seed_from_entity_facts() is out of scope for mp-4yf; tracked in mp-84n."
        )

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        entities = self._scalar_count("SELECT count() FROM entity GROUP ALL")
        triples = self._scalar_count("SELECT count() FROM triple GROUP ALL")
        current = self._scalar_count("SELECT count() FROM triple WHERE valid_to IS NONE GROUP ALL")
        predicates_rows = (
            self._db.query("SELECT array::distinct(predicate) AS ps FROM triple GROUP ALL") or []
        )
        predicates = sorted(predicates_rows[0]["ps"]) if predicates_rows else []
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": triples - current,
            "relationship_types": predicates,
        }

    def _scalar_count(self, query: str) -> int:
        rows = self._db.query(query) or []
        if not rows:
            return 0
        row = rows[0]
        # ``SELECT count()`` with ``GROUP ALL`` emits ``{"count": N}``.
        return int(row.get("count", 0))

    # ── Row shaping ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_result(
        row: dict[str, Any],
        direction: str,
        subject: str,
        obj: str,
    ) -> dict[str, Any]:
        return {
            "direction": direction,
            "subject": subject,
            "predicate": row["predicate"],
            "object": obj,
            "valid_from": row.get("valid_from"),
            "valid_to": row.get("valid_to"),
            "confidence": row.get("confidence"),
            "source_closet": row.get("source_closet"),
            "current": row.get("valid_to") is None,
        }
