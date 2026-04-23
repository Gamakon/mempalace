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

Scope (mp-84n, this module):
    - ``timeline``             entity-filtered/global chronological scan
    - ``invalidate``           close the open triple matching a
                               ``(subject, predicate, object)`` — surface
                               restored to SQLite parity in mp-s2k so the
                               MCP server can call either backend uniformly
    - ``seed_from_entity_facts`` bootstrap from ``fact_checker.ENTITY_FACTS``

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

Slug strategy (mp-1jb):
    Entity slugs *exactly* match the SQLite KG's ``_entity_id`` — i.e.
    ``name.lower().replace(" ", "_").replace("'", "")`` and nothing else.
    ``"Dr. Chen"`` therefore becomes ``"dr._chen"`` in both backends, so a
    palace migrated from SQLite to SurrealDB keeps its entity IDs and all
    existing triples continue to resolve.

    SurrealDB only allows ``[A-Za-z0-9_]`` in *bare* record ids, but IDs
    containing other chars (``.``, ``-``, digits leading, etc.) are legal
    once wrapped in ``⟨…⟩``. The ``surrealdb`` Python client handles that
    wrapping automatically when a ``RecordID`` is passed as a bound
    parameter (``$rec``) — verified against SurrealDB 3.0.4 — so callers
    never have to think about escaping. Every query below uses parameter
    binding; none interpolate slugs into query strings.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from surrealdb import RecordID, Surreal


# mp-85q: WS over HTTP for the same reason as ``backends/surreal.py``'s
# default — SurrealDB 3.0.4's HTTP path has a concurrent cross-session
# NS/DB routing bug that the WebSocket wire avoids.
DEFAULT_URL = "ws://127.0.0.1:8000"
DEFAULT_USER = "root"
DEFAULT_PASS = "root"
DEFAULT_NS = "mempalace"
DEFAULT_DB = "kg"


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
        """Normalise a display name to a slug.

        Bit-identical to ``KnowledgeGraph._entity_id`` — lowercase, spaces
        to underscores, strip apostrophes. No extra sanitisation: IDs that
        contain characters outside ``[a-z0-9_]`` (e.g. ``"dr._chen"``,
        ``"c3-po"``) are legal SurrealDB record IDs when passed via a
        bound ``RecordID`` parameter, which is how every query in this
        module references them. See module docstring ("Slug strategy")
        for the cross-backend parity rationale.
        """
        return name.lower().replace(" ", "_").replace("'", "")

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
        extracted_at: Optional[str] = None,
    ) -> str:
        """Add a ``subject -> predicate -> object`` edge.

        Returns the Surreal record id of the triple (e.g.
        ``"triple:abc123"``) — caller treats it as an opaque string, same
        contract as the SQLite KG's ``t_<sub>_<pred>_<obj>_<hash>`` id.

        Dedupe rule mirrors SQLite: if an *open* triple (``valid_to IS
        NONE``) with the same ``(subject, predicate, object)`` already
        exists, return its id without creating a new edge.

        ``extracted_at`` (mp-3lc) is optional. When ``None`` the edge
        stamps ``time::now()`` (default behaviour — preserves mp-2um).
        Callers that need to preserve provenance (the SQLite→Surreal KG
        migration is the motivating case) pass the source row's original
        timestamp as an ISO 8601 string; it is stored as a Surreal
        ``datetime`` via ``<datetime>`` casting.
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

        # Dedupe rule: an open triple (valid_to IS NONE) matches on
        # (sub, pred, obj). A closed triple additionally matches on the
        # full (valid_from, valid_to) span so re-running a migration of
        # historical facts does not create duplicates. The SQLite KG
        # itself only dedupes the open case; we strengthen the Surreal
        # path for idempotent migrations (mp-3lc).
        if valid_to is None:
            existing = self._db.query(
                (
                    "SELECT id FROM triple WHERE in = $sub AND out = $obj "
                    "AND predicate = $pred AND valid_to IS NONE"
                ),
                {"sub": sub_rec, "obj": obj_rec, "pred": pred},
            )
        else:
            existing = self._db.query(
                (
                    "SELECT id FROM triple WHERE in = $sub AND out = $obj "
                    "AND predicate = $pred AND valid_from = $valid_from "
                    "AND valid_to = $valid_to"
                ),
                {
                    "sub": sub_rec,
                    "obj": obj_rec,
                    "pred": pred,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                },
            )
        if existing:
            return str(existing[0]["id"])

        # mp-2um: the schema doc declares
        # ``extracted_at ON triple TYPE datetime VALUE $value OR time::now()
        # DEFAULT time::now()`` — but this port uses SCHEMALESS tables, so
        # the DEFAULT never fires and triples had no provenance timestamp,
        # regressing parity with the SQLite KG (knowledge_graph.py:88).
        # Set ``extracted_at`` explicitly on the RELATE so every new edge
        # carries a timestamp regardless of whether we later tighten the
        # schema (mp-6gu). mp-3lc: if the caller supplied an ISO string
        # we cast it to a Surreal ``datetime`` literal so it survives
        # across the schema-tightening pass without a data rewrite.
        if extracted_at is None:
            extracted_clause = "extracted_at = time::now()"
            extracted_param: dict[str, Any] = {}
        else:
            extracted_clause = "extracted_at = <datetime> $extracted_at"
            extracted_param = {"extracted_at": extracted_at}

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
                "adapter_name = $adapter_name, "
                f"{extracted_clause}"
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
                **extracted_param,
            },
        )
        if not created:
            raise RuntimeError(f"RELATE returned no record for {subject}->{pred}->{obj}")
        return str(created[0]["id"])

    def add_triples_batch(self, triples: list[dict[str, Any]]) -> list[str]:
        """Add a batch of triples in a minimal number of network round-trips.

        Semantically identical to calling :meth:`add_triple` for each
        dict in ``triples`` — same dedup rule, same provenance handling,
        same return contract — but collapses the work into three queries
        per batch:

        1. One multi-statement ``UPSERT`` that ensures every referenced
           entity exists (de-duplicated by slug across the batch).
        2. One ``SELECT`` that pre-fetches the subset of the batch's
           ``(sub, pred, obj[, valid_from, valid_to])`` keys already
           present in the graph, so we skip writes that would otherwise
           be deduped by :meth:`add_triple`.
        3. One multi-statement ``RELATE`` that writes every non-duplicate
           triple. Surreal returns each statement's result separately, so
           we can hand back a list of triple record ids aligned to the
           input order (duplicates get the existing id just like the
           single-triple path).

        This is the scaling fix for mp-ayu: per-triple RTT dominates a
        10k+-triple migration, and three round-trips per batch — instead
        of ``N × 3+`` — is what unlocks it.

        Parameters
        ----------
        triples:
            List of triple dicts. Each dict supports the same keys as
            :meth:`add_triple`'s parameters:
            ``subject``, ``predicate``, ``obj``, ``valid_from``,
            ``valid_to``, ``confidence``, ``source_closet``,
            ``source_file``, ``source_drawer_id``, ``adapter_name``,
            ``extracted_at``. Missing optional keys default to the same
            values :meth:`add_triple` uses.

        Returns
        -------
        list[str]
            One id per input triple, in order. Dedup hits return the
            existing triple's id (matching :meth:`add_triple`).

        Notes
        -----
        * **Idempotent.** Calling twice with the same batch produces the
          same final graph state — the pre-check ensures no duplicate
          open triples and no duplicate closed triples.
        * **All-or-nothing writes.** A batch-level failure propagates
          the ``RuntimeError`` from SurrealDB; caller retries the whole
          batch. UPSERT + the dedup pre-check make retries safe.
        * **Intra-batch dedup.** If the same ``(sub, pred, obj)`` appears
          multiple times with ``valid_to is None`` inside one batch, only
          the first is written; subsequent entries share the same id.
          This mirrors how per-triple calls would behave if the first
          one landed before the second ran.
        """
        if not triples:
            return []

        # ── Stage 1: resolve slugs + normalise fields up front ────────
        # Each item becomes a dict with the Surreal-native types (record
        # ids, normalised predicate) so the later stages can focus on
        # SurrealQL assembly without re-doing the per-item work.
        prepared: list[dict[str, Any]] = []
        entities: dict[str, str] = {}  # slug -> display name (first wins)
        for t in triples:
            subject = t["subject"]
            obj = t["obj"]
            pred = self._normalize_predicate(t["predicate"])
            sub_rec = self._entity_record(subject)
            obj_rec = self._entity_record(obj)
            # First display name for a given slug wins — the single-triple
            # path is "last write wins" via UPSERT, but in a batch we do
            # want determinism, and first-wins matches the typical mining
            # order (entities are created before triples, so the name on
            # the first triple matches the entity row already in place).
            entities.setdefault(sub_rec.id, subject)
            entities.setdefault(obj_rec.id, obj)
            prepared.append(
                {
                    "subject": subject,
                    "obj": obj,
                    "pred": pred,
                    "sub_rec": sub_rec,
                    "obj_rec": obj_rec,
                    "valid_from": t.get("valid_from"),
                    "valid_to": t.get("valid_to"),
                    "confidence": t.get("confidence", 1.0),
                    "source_closet": t.get("source_closet"),
                    "source_file": t.get("source_file"),
                    "source_drawer_id": t.get("source_drawer_id"),
                    "adapter_name": t.get("adapter_name"),
                    "extracted_at": t.get("extracted_at"),
                }
            )

        # ── Stage 2: ensure all referenced entities in one query ──────
        # One UPSERT statement per unique slug, concatenated with ``;``.
        # SurrealDB's wire protocol executes every statement in order.
        # We use ``query_raw`` to avoid any surprise about which
        # statement's result the sync helper returns — we don't need the
        # results anyway, we just need the side effect.
        ent_parts: list[str] = []
        ent_params: dict[str, Any] = {}
        for i, (slug, name) in enumerate(entities.items()):
            ent_parts.append(f"UPSERT $rec_{i} SET name = $name_{i}")
            ent_params[f"rec_{i}"] = RecordID("entity", slug)
            ent_params[f"name_{i}"] = name
        self._db.query_raw("; ".join(ent_parts) + ";", ent_params)

        # ── Stage 3: pre-fetch existing triples that would dedup ──────
        # Two scoped SELECTs (one for the open half, one for the closed
        # half) bounded to the entity/predicate sets we actually touch,
        # so the DB never scans the full ``triple`` table.
        sub_recs_all = list({p["sub_rec"].id: p["sub_rec"] for p in prepared}.values())
        obj_recs_all = list({p["obj_rec"].id: p["obj_rec"] for p in prepared}.values())
        preds_all = list({p["pred"] for p in prepared})

        open_keys: dict[tuple[str, str, str], str] = {}
        closed_keys: dict[tuple[str, str, str, Optional[str], Optional[str]], str] = {}

        has_open = any(p["valid_to"] is None for p in prepared)
        has_closed = any(p["valid_to"] is not None for p in prepared)

        if has_open:
            rows = self._db.query(
                (
                    "SELECT id, in AS s, out AS o, predicate AS p "
                    "FROM triple WHERE valid_to IS NONE "
                    "AND in IN $subs AND out IN $objs AND predicate IN $preds"
                ),
                {"subs": sub_recs_all, "objs": obj_recs_all, "preds": preds_all},
            ) or []
            for r in rows:
                key = (r["s"].id, r["p"], r["o"].id)
                # Keep the first id seen; the open dedup key is unique by
                # definition (SQLite enforces it, and the Surreal path
                # enforces it via this very batch check).
                open_keys.setdefault(key, str(r["id"]))

        if has_closed:
            rows = self._db.query(
                (
                    "SELECT id, in AS s, out AS o, predicate AS p, "
                    "valid_from AS vf, valid_to AS vt "
                    "FROM triple WHERE valid_to IS NOT NONE "
                    "AND in IN $subs AND out IN $objs AND predicate IN $preds"
                ),
                {"subs": sub_recs_all, "objs": obj_recs_all, "preds": preds_all},
            ) or []
            for r in rows:
                key = (
                    r["s"].id,
                    r["p"],
                    r["o"].id,
                    r.get("vf"),
                    r.get("vt"),
                )
                closed_keys.setdefault(key, str(r["id"]))

        # ── Stage 4: build the RELATE batch, skipping dedup hits ──────
        # ``ids`` is the output aligned to the input order. We fill it
        # as we go: dedup hits get the pre-fetched id immediately, new
        # writes leave a placeholder that's filled after the RELATE
        # query returns.
        ids: list[Optional[str]] = [None] * len(prepared)
        relate_parts: list[str] = []
        relate_params: dict[str, Any] = {}
        # Position of each RELATE statement -> index in ``ids`` to fill.
        relate_slots: list[int] = []
        # Within this batch, collapse same-key open writes so they share
        # one id (matches the per-call "first-wins, later calls dedup"
        # behaviour of add_triple).
        pending_open: dict[tuple[str, str, str], int] = {}

        for i, p in enumerate(prepared):
            sub_slug = p["sub_rec"].id
            obj_slug = p["obj_rec"].id
            pred = p["pred"]
            if p["valid_to"] is None:
                key = (sub_slug, pred, obj_slug)
                # Existing in DB -> dedup.
                if key in open_keys:
                    ids[i] = open_keys[key]
                    continue
                # Earlier in this batch -> share slot.
                if key in pending_open:
                    # Mark as alias; filled after the write completes.
                    ids[i] = f"__ALIAS__{pending_open[key]}"
                    continue
                pending_open[key] = i
            else:
                key_c = (sub_slug, pred, obj_slug, p["valid_from"], p["valid_to"])
                if key_c in closed_keys:
                    ids[i] = closed_keys[key_c]
                    continue

            # Build the RELATE statement for this triple.
            n = len(relate_slots)
            relate_slots.append(i)
            if p["extracted_at"] is None:
                extracted_clause = "extracted_at = time::now()"
            else:
                extracted_clause = f"extracted_at = <datetime> $extracted_at_{n}"
                relate_params[f"extracted_at_{n}"] = p["extracted_at"]
            relate_parts.append(
                f"RELATE $sub_{n}->triple->$obj_{n} SET "
                f"predicate = $pred_{n}, "
                f"valid_from = $valid_from_{n}, "
                f"valid_to = $valid_to_{n}, "
                f"confidence = $confidence_{n}, "
                f"source_closet = $source_closet_{n}, "
                f"source_file = $source_file_{n}, "
                f"source_drawer_id = $source_drawer_id_{n}, "
                f"adapter_name = $adapter_name_{n}, "
                f"{extracted_clause}"
            )
            relate_params[f"sub_{n}"] = p["sub_rec"]
            relate_params[f"obj_{n}"] = p["obj_rec"]
            relate_params[f"pred_{n}"] = pred
            relate_params[f"valid_from_{n}"] = p["valid_from"]
            relate_params[f"valid_to_{n}"] = p["valid_to"]
            relate_params[f"confidence_{n}"] = p["confidence"]
            relate_params[f"source_closet_{n}"] = p["source_closet"]
            relate_params[f"source_file_{n}"] = p["source_file"]
            relate_params[f"source_drawer_id_{n}"] = p["source_drawer_id"]
            relate_params[f"adapter_name_{n}"] = p["adapter_name"]

        # ── Stage 5: execute the RELATE batch (if anything to write) ──
        if relate_parts:
            raw = self._db.query_raw("; ".join(relate_parts) + ";", relate_params)
            stmts = raw.get("result", []) if isinstance(raw, dict) else []
            if len(stmts) != len(relate_parts):
                raise RuntimeError(
                    f"RELATE batch returned {len(stmts)} statement results, "
                    f"expected {len(relate_parts)}"
                )
            for pos, stmt in enumerate(stmts):
                status = stmt.get("status")
                if status != "OK":
                    # A single bad statement poisons the batch — raise so
                    # the caller can retry (UPSERT + dedup pre-check make
                    # the retry idempotent).
                    raise RuntimeError(
                        f"RELATE statement {pos} failed: {stmt.get('result')!r}"
                    )
                rows = stmt.get("result") or []
                if not rows:
                    raise RuntimeError(
                        f"RELATE statement {pos} returned no record "
                        f"(triple index {relate_slots[pos]})"
                    )
                ids[relate_slots[pos]] = str(rows[0]["id"])

        # ── Stage 6: resolve intra-batch alias slots ─────────────────
        for i, v in enumerate(ids):
            if isinstance(v, str) and v.startswith("__ALIAS__"):
                src = int(v[len("__ALIAS__") :])
                ids[i] = ids[src]

        # Every slot must be filled.
        out: list[str] = []
        for i, v in enumerate(ids):
            if v is None:
                raise RuntimeError(f"add_triples_batch: triple at index {i} got no id")
            out.append(v)
        return out

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: Optional[str] = None,
    ) -> bool:
        """Close the open triple matching ``(subject, predicate, object)``.

        Parameters
        ----------
        subject, predicate, obj:
            The triplet identifying the open fact to close. Normalised the
            same way as on the write path (see :meth:`add_triple`) so the
            exact display-name casing used at insert time is not required.
        ended:
            ISO date/datetime string to stamp on the triple's ``valid_to``
            field. Defaults to today's ISO date, matching the SQLite KG.

        Returns
        -------
        ``True`` if an open triple was closed, ``False`` if no matching
        open triple existed (idempotent: a second call with the same SPO
        is a no-op and never overwrites a prior ``valid_to``).

        Notes
        -----
        Signature matches ``KnowledgeGraph.invalidate`` exactly — the MCP
        server (``mcp_server.py::tool_kg_invalidate``) calls
        ``_kg.invalidate(subject, predicate, object, ended=ended)`` without
        caring which backend is underneath. Earlier mp-84n drafts exposed
        an id-based surface; mp-s2k restores contract parity. The SQLite
        KG returns ``None`` from this method, but we return a bool here
        (``True``/``False``) so callers get a cheap idempotency signal —
        strictly more informative than the SQLite return, which is a
        forward-compatible superset.
        """
        sub_rec = self._entity_record(subject)
        obj_rec = self._entity_record(obj)
        pred = self._normalize_predicate(predicate)
        stamp = ended if ended is not None else date.today().isoformat()
        updated = self._db.query(
            (
                "UPDATE triple SET valid_to = $valid_to "
                "WHERE in = $sub AND out = $obj "
                "AND predicate = $pred AND valid_to IS NONE"
            ),
            {
                "sub": sub_rec,
                "obj": obj_rec,
                "pred": pred,
                "valid_to": stamp,
            },
        )
        # SurrealDB returns the updated rows (empty list if nothing matched
        # the ``valid_to IS NONE`` predicate — that's the idempotency path
        # and also the "unknown triple" path).
        return bool(updated)

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
        return f"({vf} IS NONE OR {vf} <= $as_of) AND ({vt} IS NONE OR {vt} >= $as_of)"

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

    def timeline(
        self,
        entity_name: Optional[str] = None,
        limit: int = 100,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Return triples ordered chronologically by ``valid_from``.

        Parameters
        ----------
        entity_name:
            When set, restricts results to triples where the entity appears
            as either subject or object. When ``None``, returns a global
            timeline (parity with ``KnowledgeGraph.timeline(None)``).
        limit:
            Maximum number of rows. Matches SQLite's hard cap of 100 by
            default; caller can raise or lower.
        order:
            ``"asc"`` (default, oldest first) or ``"desc"`` (newest first).
            Triples whose ``valid_from`` is NONE always sort *last*
            regardless of direction — mirrors the SQLite ``NULLS LAST``
            clause so callers comparing timelines across backends see the
            same shape.

        Notes
        -----
        SurrealQL 3.0.4 does not support ``ORDER BY ... NULLS LAST``, so we
        synthesise a leading ``null_sort`` column (``0`` when ``valid_from``
        is set, ``1`` when it's NONE) and order on that first. The effect
        is identical and the extra field is stripped before return.
        """
        order_norm = order.lower()
        if order_norm not in ("asc", "desc"):
            raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
        direction_clause = "ASC" if order_norm == "asc" else "DESC"

        select = (
            "SELECT id, predicate, valid_from, valid_to, "
            "in.name AS sub_name, out.name AS obj_name, "
            "IF valid_from IS NONE THEN 1 ELSE 0 END AS null_sort "
            "FROM triple"
        )
        params: dict[str, Any] = {"limit": limit}
        if entity_name is not None:
            select += " WHERE in = $rec OR out = $rec"
            params["rec"] = self._entity_record(entity_name)
        select += f" ORDER BY null_sort ASC, valid_from {direction_clause} LIMIT $limit"

        rows = self._db.query(select, params) or []
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

    def seed_from_entity_facts(self, entity_facts: dict[str, dict[str, Any]]) -> None:
        """Bootstrap the graph from ``fact_checker.ENTITY_FACTS``.

        Mirrors :meth:`KnowledgeGraph.seed_from_entity_facts` exactly — same
        predicate names, same capitalisation rules, same ``valid_from``
        defaults. Pure batch wrapper over :meth:`add_entity` /
        :meth:`add_triple`; de-dupe is the usual open-triple rule on the
        write path.
        """
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(
                name,
                etype,
                {
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )

            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name,
                    "child_of",
                    parent.capitalize(),
                    valid_from=facts.get("birthday"),
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of", facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of", facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of", facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(), valid_from="2025-01-01")

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
