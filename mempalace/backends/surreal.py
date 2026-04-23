"""SurrealDB-backed MemPalace storage backend (RFC 001; mp-6xi).

This module implements **drawer CRUD only** against a local SurrealDB
server. Hybrid / vector search is deliberately out of scope and is tracked
separately in issue mp-j19; the `query()` method raises
``NotImplementedError`` pointing at that issue.

Layout
------

One Surreal **namespace + database per palace**, following
``docs/surrealdb-schema.md``::

    NS mempalace DB <palace_id>

Tables handled by this backend right now: ``drawer`` (and ``closet``,
treated as the same shape). Entity / knowledge-graph tables are scoped to
a separate issue (mp-4yf) and are not defined here.

Mapping from RFC 001 collection semantics to Surreal rows:

* ``ids[i]``          -> ``drawer.id_ext`` (string, UNIQUE) and also the
                         Surreal record id ``drawer:<safe_id>``.
* ``documents[i]``    -> ``drawer.document``.
* ``embeddings[i]``   -> ``drawer.embedding`` (optional).
* ``metadatas[i]``    -> ``drawer.metadata`` (flexible object). Opaque to
                         the backend, matching Chroma's behavior.

Connection lifecycle is intentionally simple: one ``BlockingHttp``
connection per palace, cached on the backend instance. No pool, no retry
loop — mp-6xi is the thinnest layer that satisfies the base contract.
"""

from __future__ import annotations

import logging
import os
import re
from threading import Lock
from typing import Any, Optional

from .base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("MEMPALACE_SURREAL_URL", "http://127.0.0.1:8000")
DEFAULT_USER = os.environ.get("MEMPALACE_SURREAL_USER", "root")
DEFAULT_PASS = os.environ.get("MEMPALACE_SURREAL_PASS", "root")
DEFAULT_NAMESPACE = os.environ.get("MEMPALACE_SURREAL_NS", "mempalace")

_SCHEMA_VERSION = 1
_DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
_DEFAULT_EMBEDDING_DIM = 384
_DEFAULT_HNSW_SPACE = "cosine"

# Drawer-like tables this backend knows how to service. Chroma's
# ``mempalace_drawers`` / ``mempalace_closets`` names map onto these.
_TABLE_ALIASES = {
    "drawer": "drawer",
    "drawers": "drawer",
    "mempalace_drawers": "drawer",
    "closet": "closet",
    "closets": "closet",
    "mempalace_closets": "closet",
}

_REQUIRED_OPERATORS = frozenset({"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains"})
_OPTIONAL_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_SUPPORTED_OPERATORS = _REQUIRED_OPERATORS | _OPTIONAL_OPERATORS


def _validate_where(where: Optional[dict]) -> None:
    """Scan a where-clause for unknown operators.

    Matches ``mempalace.backends.chroma._validate_where``. Silent dropping
    of unknown operators is forbidden by RFC 001 §1.4.
    """
    if not where:
        return
    stack: list[Any] = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k.startswith("$") and k not in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(f"operator {k!r} not supported by surreal backend")
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


_DB_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _raise_if_sdk_error(result: Any, context: str) -> None:
    """Translate surrealdb-python's error-as-string returns into exceptions.

    The blocking-HTTP connection in ``surrealdb`` 1.0.x returns certain
    errors — most notably ``create`` on a duplicate id, and schema-validation
    failures — as plain strings instead of raising. Treat a string response
    where a dict (single record) or list (set of records) was expected as
    an error.
    """
    if isinstance(result, str):
        raise RuntimeError(f"surreal backend {context}: {result}")


def _safe_close(conn) -> None:
    """Close a Surreal connection, tolerating SDK variants that no-op close.

    The blocking-HTTP backend in surrealdb 1.0.x raises ``NotImplementedError``
    from ``close()`` — we treat that as "nothing to do" rather than noise.
    Real I/O errors are logged but not raised so the backend's shutdown path
    stays clean.
    """
    try:
        conn.close()
    except NotImplementedError:
        pass
    except Exception:
        logger.exception("error closing surreal connection")


def _safe_db_name(palace_ref: PalaceRef) -> str:
    """Derive a Surreal database name from a :class:`PalaceRef`.

    Priority: ``palace.namespace`` (explicit caller choice) -> sanitized
    ``palace.id``. The result is restricted to ``[A-Za-z0-9_]`` so it
    lands without needing backtick-escaping in SurrealQL ``USE DB``.
    """
    raw = palace_ref.namespace or palace_ref.id
    cleaned = _DB_NAME_SAFE_RE.sub("_", raw).strip("_")
    return cleaned or "palace"


def _normalize_collection_name(name: str) -> str:
    """Map a caller-supplied collection name onto a Surreal table name.

    Only drawer-shaped tables are understood at this stage (mp-6xi).
    ``mempalace_drawers`` / ``drawers`` / ``drawer`` all resolve to the
    ``drawer`` table; same for closets. Unknown names raise ``ValueError``
    rather than silently creating a new table.
    """
    key = name.lower()
    if key in _TABLE_ALIASES:
        return _TABLE_ALIASES[key]
    raise ValueError(
        f"surreal backend does not know collection {name!r}; "
        f"expected one of {sorted(_TABLE_ALIASES)}"
    )


# ---------------------------------------------------------------------------
# Bootstrap DDL — applied idempotently on first create.
# ---------------------------------------------------------------------------


def _bootstrap_ddl() -> str:
    """Return the schema DDL applied on first ``create=True``.

    This is a narrower slice of ``docs/surrealdb-schema.md``: drawer + closet
    payload + ``palace_meta:main``. Wing/room/entity/triple are deferred
    to the graph-layer task (mp-4yf).

    Note on schema-doc divergence: the doc uses
    ``FLEXIBLE TYPE option<object>`` for ``metadata`` — that parses as a
    syntax error in SurrealDB 3.0.4, which requires ``TYPE ... FLEXIBLE``.
    See the report on mp-6xi; the doc needs a follow-up fix.
    """
    return f"""
    DEFINE TABLE IF NOT EXISTS palace_meta SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS schema_version  ON palace_meta TYPE int;
    DEFINE FIELD IF NOT EXISTS embedder_name   ON palace_meta TYPE string;
    DEFINE FIELD IF NOT EXISTS embedding_dim   ON palace_meta TYPE int;
    DEFINE FIELD IF NOT EXISTS hnsw_space      ON palace_meta TYPE string DEFAULT 'cosine';
    DEFINE FIELD IF NOT EXISTS created_at      ON palace_meta TYPE datetime DEFAULT time::now();

    DEFINE TABLE IF NOT EXISTS drawer SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS id_ext     ON drawer TYPE string;
    DEFINE FIELD IF NOT EXISTS document   ON drawer TYPE string ASSERT $value != NONE;
    DEFINE FIELD IF NOT EXISTS embedding  ON drawer TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS metadata   ON drawer TYPE option<object> FLEXIBLE;
    -- VALUE clause keeps filed_at populated on UPSERT (CONTENT) calls,
    -- which otherwise clear fields not in the new payload and then hit
    -- the `TYPE datetime` check. See mp-6xi report.
    DEFINE FIELD IF NOT EXISTS filed_at   ON drawer TYPE datetime
        VALUE $value OR time::now() DEFAULT time::now();
    DEFINE INDEX IF NOT EXISTS drawer_id_ext ON drawer FIELDS id_ext UNIQUE;

    DEFINE TABLE IF NOT EXISTS closet SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS id_ext     ON closet TYPE string;
    DEFINE FIELD IF NOT EXISTS document   ON closet TYPE string ASSERT $value != NONE;
    DEFINE FIELD IF NOT EXISTS embedding  ON closet TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS metadata   ON closet TYPE option<object> FLEXIBLE;
    DEFINE FIELD IF NOT EXISTS filed_at   ON closet TYPE datetime
        VALUE $value OR time::now() DEFAULT time::now();
    DEFINE INDEX IF NOT EXISTS closet_id_ext ON closet FIELDS id_ext UNIQUE;

    UPSERT palace_meta:main SET
        schema_version = {_SCHEMA_VERSION},
        embedder_name  = '{_DEFAULT_EMBEDDER}',
        embedding_dim  = {_DEFAULT_EMBEDDING_DIM},
        hnsw_space     = '{_DEFAULT_HNSW_SPACE}';
    """


# ---------------------------------------------------------------------------
# Collection adapter
# ---------------------------------------------------------------------------


def _validate_writes(
    *,
    documents: list[str],
    ids: list[str],
    metadatas: Optional[list[dict]] = None,
    embeddings: Optional[list[list[float]]] = None,
) -> None:
    """Shared length / type checks for ``add`` and ``upsert``."""
    if not isinstance(documents, list) or not isinstance(ids, list):
        raise ValueError("documents and ids must be lists")
    if len(documents) != len(ids):
        raise ValueError(f"documents length {len(documents)} does not match ids length {len(ids)}")
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError(f"metadatas length {len(metadatas)} does not match ids length {len(ids)}")
    if embeddings is not None and len(embeddings) != len(ids):
        raise ValueError(
            f"embeddings length {len(embeddings)} does not match ids length {len(ids)}"
        )


class SurrealCollection(BaseCollection):
    """Thin adapter translating SurrealDB results into typed RFC 001 results.

    Each instance is bound to a single table (``drawer`` or ``closet``) inside
    a single Surreal database (= palace). The underlying Surreal connection
    is owned by the parent :class:`SurrealBackend`; callers MUST NOT close it
    via the collection.
    """

    def __init__(self, db, table: str):
        from surrealdb import RecordID  # late import keeps module load fast

        self._db = db
        self._table = table
        self._RecordID = RecordID

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rid(self, ext_id: str):
        """Build a ``RecordID`` for this table from a caller id."""
        return self._RecordID(self._table, ext_id)

    def _record_payload(
        self,
        *,
        ext_id: str,
        document: str,
        metadata: Optional[dict],
        embedding: Optional[list[float]],
    ) -> dict:
        payload: dict[str, Any] = {
            "id_ext": ext_id,
            "document": document,
            "metadata": dict(metadata) if metadata else {},
        }
        if embedding is not None:
            payload["embedding"] = list(embedding)
        return payload

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_writes(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        for i, ext_id in enumerate(ids):
            payload = self._record_payload(
                ext_id=ext_id,
                document=documents[i],
                metadata=metadatas[i] if metadatas is not None else None,
                embedding=embeddings[i] if embeddings is not None else None,
            )
            # The blocking-HTTP SDK returns error strings from `create` on
            # duplicate-id instead of raising. Detect + raise so ``add``
            # matches Chroma's "fail on duplicate" contract.
            result = self._db.create(self._rid(ext_id), payload)
            _raise_if_sdk_error(result, f"add id={ext_id!r}")

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_writes(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        for i, ext_id in enumerate(ids):
            payload = self._record_payload(
                ext_id=ext_id,
                document=documents[i],
                metadata=metadatas[i] if metadatas is not None else None,
                embedding=embeddings[i] if embeddings is not None else None,
            )
            result = self._db.upsert(self._rid(ext_id), payload)
            _raise_if_sdk_error(result, f"upsert id={ext_id!r}")

    def update(
        self,
        *,
        ids,
        documents=None,
        metadatas=None,
        embeddings=None,
    ):
        """Atomic per-id merge update.

        Surreal's ``MERGE`` preserves fields that were not passed, so this is
        cheaper and race-free vs. the base ``update`` (get + merge + upsert).
        """
        if documents is None and metadatas is None and embeddings is None:
            raise ValueError("update requires at least one of documents, metadatas, embeddings")
        n = len(ids)
        for label, value in (
            ("documents", documents),
            ("metadatas", metadatas),
            ("embeddings", embeddings),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{label} length {len(value)} does not match ids length {n}")

        for i, ext_id in enumerate(ids):
            patch: dict[str, Any] = {}
            if documents is not None:
                patch["document"] = documents[i]
            if metadatas is not None:
                # Merge into the stored metadata dict, matching the base-class
                # semantics (keys not passed survive).
                existing = self._db.select(self._rid(ext_id))
                current_meta: dict = {}
                if existing:
                    row = existing[0] if isinstance(existing, list) else existing
                    current_meta = dict(row.get("metadata") or {})
                current_meta.update(metadatas[i] or {})
                patch["metadata"] = current_meta
            if embeddings is not None:
                patch["embedding"] = list(embeddings[i])
            if patch:
                self._db.merge(self._rid(ext_id), patch)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
    ) -> QueryResult:
        """Semantic / hybrid search — NOT implemented in mp-6xi.

        Tracked in mp-j19. This method intentionally raises to keep the
        interface honest: the base-class ``query`` is abstract, so we must
        implement it, but silently returning empty results would violate
        the RFC 001 spec.
        """
        raise NotImplementedError(
            "SurrealCollection.query (vector / hybrid search) is not implemented yet; "
            "see issue mp-j19"
        )

    def get(
        self,
        *,
        ids=None,
        where=None,
        where_document=None,
        limit=None,
        offset=None,
        include=None,
    ) -> GetResult:
        _validate_where(where)
        _validate_where(where_document)
        spec = _IncludeSpec.resolve(include, default_distances=False)

        select_fields = ["id_ext"]
        if spec.documents:
            select_fields.append("document")
        if spec.metadatas:
            select_fields.append("metadata")
        if spec.embeddings:
            select_fields.append("embedding")

        where_parts: list[str] = []
        bindings: dict[str, Any] = {}

        if ids is not None:
            bindings["ids"] = list(ids)
            where_parts.append("id_ext INSIDE $ids")

        if where:
            clause, extra_vars = _translate_where(where, key_prefix="m")
            if clause:
                where_parts.append(clause)
                bindings.update(extra_vars)

        if where_document:
            clause, extra_vars = _translate_where_document(where_document, key_prefix="d")
            if clause:
                where_parts.append(clause)
                bindings.update(extra_vars)

        query = f"SELECT {', '.join(select_fields)} FROM {self._table}"
        if where_parts:
            query += " WHERE " + " AND ".join(f"({p})" for p in where_parts)
        # Stable ordering so ``offset`` is meaningful across calls.
        query += " ORDER BY id_ext"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        if offset is not None:
            query += f" START {int(offset)}"

        rows = self._db.query(query, bindings) or []
        if not isinstance(rows, list):
            rows = [rows]

        out_ids: list[str] = []
        out_docs: list[str] = []
        out_metas: list[dict] = []
        out_embeds: list[list[float]] = []
        for row in rows:
            out_ids.append(row.get("id_ext", ""))
            if spec.documents:
                out_docs.append(row.get("document") or "")
            if spec.metadatas:
                out_metas.append(dict(row.get("metadata") or {}))
            if spec.embeddings:
                emb = row.get("embedding")
                out_embeds.append(list(emb) if emb is not None else [])

        return GetResult(
            ids=out_ids,
            documents=out_docs if spec.documents else [],
            metadatas=out_metas if spec.metadatas else [],
            embeddings=out_embeds if spec.embeddings else None,
        )

    def delete(self, *, ids=None, where=None):
        _validate_where(where)
        if ids is None and not where:
            # Chroma treats this as a no-op rather than a full-table nuke.
            return
        if ids is not None and not where:
            # Fast path: ``DELETE ... WHERE id_ext INSIDE $ids``.
            self._db.query(
                f"DELETE {self._table} WHERE id_ext INSIDE $ids",
                {"ids": list(ids)},
            )
            return

        # Where-clause (optionally combined with ids).
        clauses: list[str] = []
        bindings: dict[str, Any] = {}
        if ids is not None:
            clauses.append("id_ext INSIDE $ids")
            bindings["ids"] = list(ids)
        if where:
            clause, extra_vars = _translate_where(where, key_prefix="m")
            if clause:
                clauses.append(clause)
                bindings.update(extra_vars)
        query = f"DELETE {self._table}"
        if clauses:
            query += " WHERE " + " AND ".join(f"({c})" for c in clauses)
        self._db.query(query, bindings)

    def count(self) -> int:
        rows = self._db.query(f"SELECT count() FROM {self._table} GROUP ALL") or []
        if not rows:
            return 0
        first = rows[0] if isinstance(rows, list) else rows
        if isinstance(first, dict):
            return int(first.get("count") or 0)
        return 0

    def close(self) -> None:
        # Connection lifecycle is owned by the backend.
        return None

    def health(self) -> HealthStatus:
        try:
            self._db.query(f"INFO FOR TABLE {self._table}")
            return HealthStatus.healthy()
        except Exception as e:  # pragma: no cover - defensive
            return HealthStatus.unhealthy(str(e))


# ---------------------------------------------------------------------------
# where-clause translation (opaque metadata filtering)
# ---------------------------------------------------------------------------


_OP_SURREAL = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _translate_where(where: dict, *, key_prefix: str) -> tuple[str, dict[str, Any]]:
    """Translate an RFC 001 ``where=`` dict into SurrealQL + bindings.

    Returns ``("" , {})`` for empty input. Unknown operators have already
    been rejected by :func:`_validate_where`; this translator only handles
    the supported set.

    Metadata fields land under ``metadata.<key>`` (opaque object). ``$and``
    / ``$or`` recurse; leaf scalars map to ``$eq``.
    """
    counter: dict[str, int] = {"n": 0}

    def bind(value: Any, vars: dict[str, Any]) -> str:
        counter["n"] += 1
        name = f"{key_prefix}{counter['n']}"
        vars[name] = value
        return f"${name}"

    def field_path(key: str) -> str:
        # Non-dollar keys are metadata fields — live under the metadata object.
        return f"metadata.`{key}`"

    def translate_node(node: dict, vars: dict[str, Any]) -> str:
        if not node:
            return ""
        parts: list[str] = []
        for key, value in node.items():
            if key == "$and":
                if not isinstance(value, list) or not value:
                    raise UnsupportedFilterError("$and requires a non-empty list")
                sub = [translate_node(v, vars) for v in value]
                sub = [s for s in sub if s]
                if sub:
                    parts.append("(" + " AND ".join(f"({s})" for s in sub) + ")")
            elif key == "$or":
                if not isinstance(value, list) or not value:
                    raise UnsupportedFilterError("$or requires a non-empty list")
                sub = [translate_node(v, vars) for v in value]
                sub = [s for s in sub if s]
                if sub:
                    parts.append("(" + " OR ".join(f"({s})" for s in sub) + ")")
            elif key.startswith("$"):
                # Top-level operator without a field — illegal.
                raise UnsupportedFilterError(
                    f"operator {key!r} requires a field context in where-clause"
                )
            else:
                # Leaf: field -> scalar or field -> {op: value}
                fp = field_path(key)
                if isinstance(value, dict):
                    for op, val in value.items():
                        if op in _OP_SURREAL:
                            parts.append(f"{fp} {_OP_SURREAL[op]} {bind(val, vars)}")
                        elif op == "$in":
                            parts.append(f"{fp} INSIDE {bind(list(val), vars)}")
                        elif op == "$nin":
                            parts.append(f"{fp} NOT INSIDE {bind(list(val), vars)}")
                        elif op == "$contains":
                            # Chroma applies $contains to the document, but it
                            # is accepted on any field — fall back to
                            # string::contains.
                            parts.append(f"string::contains({fp}, {bind(val, vars)})")
                        else:
                            raise UnsupportedFilterError(
                                f"operator {op!r} not supported by surreal backend"
                            )
                else:
                    parts.append(f"{fp} = {bind(value, vars)}")
        if not parts:
            return ""
        return " AND ".join(f"({p})" for p in parts)

    vars: dict[str, Any] = {}
    clause = translate_node(where, vars)
    return clause, vars


def _translate_where_document(
    where_document: dict, *, key_prefix: str
) -> tuple[str, dict[str, Any]]:
    """Translate Chroma's ``where_document={'$contains': '<substring>'}``.

    Surreal does have a full-text BM25 operator, but RFC 001 conformance
    requires exact substring semantics for ``$contains``. We therefore use
    ``string::contains`` unconditionally.
    """
    if not where_document:
        return "", {}
    if not isinstance(where_document, dict):
        raise UnsupportedFilterError("where_document must be a dict")

    vars: dict[str, Any] = {}
    parts: list[str] = []
    counter = 0
    for op, val in where_document.items():
        if op == "$contains":
            counter += 1
            name = f"{key_prefix}{counter}"
            vars[name] = val
            parts.append(f"string::contains(document, ${name})")
        elif op in {"$and", "$or"}:
            # Not exercised by core today; conservative refusal beats a
            # silent no-op.
            raise UnsupportedFilterError(
                f"{op!r} on where_document is not implemented in the surreal backend"
            )
        else:
            raise UnsupportedFilterError(
                f"where_document operator {op!r} not supported by surreal backend"
            )
    return " AND ".join(parts), vars


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SurrealBackend(BaseBackend):
    """SurrealDB backend — one namespace + database per palace (RFC 001 §2).

    Connection strategy (mp-6xi): one blocking HTTP connection per palace,
    cached on the backend. Future work (mp-j19 onward) may introduce
    WebSocket or pooled connections, but for CRUD the HTTP path is the
    simplest thing that satisfies the contract.
    """

    name = "surreal"
    capabilities = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            # Vector / hybrid search advertised only after mp-j19 lands.
        }
    )

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        username: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        namespace: str = DEFAULT_NAMESPACE,
    ):
        self._url = url
        self._username = username
        self._password = password
        self._namespace = namespace
        # palace_id -> Surreal handle
        self._conns: dict[str, Any] = {}
        # palace_id -> set[table_name] of tables we've bootstrapped
        self._bootstrapped: dict[str, set[str]] = {}
        self._closed = False
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self, db_name: str):
        """Open or reuse a Surreal connection bound to ``db_name``."""
        if self._closed:
            from .base import BackendClosedError

            raise BackendClosedError("SurrealBackend has been closed")

        with self._lock:
            conn = self._conns.get(db_name)
            if conn is not None:
                return conn

            from surrealdb import Surreal  # late import

            conn = Surreal(self._url)
            conn.signin({"username": self._username, "password": self._password})
            conn.use(self._namespace, db_name)
            self._conns[db_name] = conn
            return conn

    def _ensure_bootstrap(self, db_name: str, conn) -> None:
        """Apply schema DDL once per (palace, process). Idempotent on Surreal."""
        with self._lock:
            tables = self._bootstrapped.setdefault(db_name, set())
            if "drawer" in tables and "closet" in tables:
                return
        conn.query(_bootstrap_ddl())
        with self._lock:
            self._bootstrapped[db_name].update({"drawer", "closet"})

    # ------------------------------------------------------------------
    # BaseBackend surface
    # ------------------------------------------------------------------

    def get_collection(
        self,
        *,
        palace: PalaceRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> SurrealCollection:
        if not isinstance(palace, PalaceRef):
            raise TypeError("palace= must be a PalaceRef instance")
        table = _normalize_collection_name(collection_name)
        db_name = _safe_db_name(palace)

        conn = self._connect(db_name)

        # Detect "does the palace exist?" via a cheap INFO query.
        info = conn.query("INFO FOR DB") or {}
        if isinstance(info, list):
            info = info[0] if info else {}
        existing_tables = set((info or {}).get("tables") or {})

        palace_present = "drawer" in existing_tables or "closet" in existing_tables
        if not create and not palace_present:
            raise PalaceNotFoundError(f"surreal palace {db_name!r} has no drawer/closet tables")

        if create:
            self._ensure_bootstrap(db_name, conn)

        return SurrealCollection(conn, table)

    def close_palace(self, palace: PalaceRef) -> None:
        if not isinstance(palace, PalaceRef):
            return
        db_name = _safe_db_name(palace)
        with self._lock:
            conn = self._conns.pop(db_name, None)
            self._bootstrapped.pop(db_name, None)
        if conn is not None:
            _safe_close(conn)

    def close(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
            self._bootstrapped.clear()
            self._closed = True
        for conn in conns:
            _safe_close(conn)

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        if palace is None:
            return HealthStatus.healthy()
        try:
            conn = self._connect(_safe_db_name(palace))
            conn.query("INFO FOR DB")
            return HealthStatus.healthy()
        except Exception as e:
            return HealthStatus.unhealthy(str(e))

    @classmethod
    def detect(cls, path: str) -> bool:
        # Surreal palaces do not leave an on-disk marker in the palace
        # directory (they live inside a shared surrealkv store). Auto-detect
        # is therefore always False — explicit config / env / kwarg wins.
        return False
