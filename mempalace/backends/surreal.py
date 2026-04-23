"""SurrealDB-backed MemPalace storage backend (RFC 001; mp-6xi + mp-j19).

mp-6xi landed drawer CRUD; mp-j19 added hybrid search:

* **HNSW** vector KNN via ``<|K,COSINE|>`` for both ``query_embeddings``
  and ``query_texts`` (the latter embed-then-search for parity with Chroma).
* **FULLTEXT BM25** via ``DEFINE INDEX ... FULLTEXT ANALYZER ... BM25`` for
  explicit full-text queries triggered by
  ``where_document={'$search': '<tokens>'}``.

Hybrid re-ranking stays in :mod:`mempalace.searcher`: the backend exposes
the two paths cleanly, the orchestration layer combines them. See the
docstring on :meth:`SurrealCollection.query` for the rationale.

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
    BackendError,
    BaseBackend,
    BaseCollection,
    DimensionMismatchError,
    GetResult,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)


class HnswIndexCreationError(BackendError):
    """Raised when the lazy HNSW index DDL fails on first vector write.

    A silent failure here means every subsequent vector query returns zero
    rows for the life of the process — which is the opposite of the 100%
    recall design principle. We raise and leave ``_hnsw_checked`` unset so
    the next call retries (mp-o7v).
    """


class DuplicateIdError(BackendError):
    """Raised when ``add()`` is called with an id that already exists.

    Matches Chroma's fail-on-duplicate contract. Detection is driven by
    the SurrealDB wire protocol's per-statement ``status`` / ``kind``
    fields (see :meth:`SurrealCollection._create_record`), not by
    string-matching the error message.
    """


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
# ``$search`` is the Surreal-specific BM25 hook on ``where_document``; gating
# the BM25 path behind an explicit operator keeps ``$contains`` semantics
# identical to Chroma (exact substring), per RFC 001 §1.4.
_SEARCH_OPERATORS = frozenset({"$search"})
_SUPPORTED_OPERATORS = _REQUIRED_OPERATORS | _OPTIONAL_OPERATORS | _SEARCH_OPERATORS


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


def _raise_on_statement_error(
    raw_response: Any, context: str, *, duplicate_id: Optional[str] = None
) -> Any:
    """Inspect a ``query_raw`` response and raise on per-statement errors.

    SurrealDB's RPC protocol wraps every statement in a result envelope
    of the form::

        {"result": [{"status": "OK"|"ERR", "result": <value>, ...}, ...]}

    The Python SDK (1.0.8) only surfaces top-level transport errors via
    ``check_response_for_error`` — a per-statement ``status == "ERR"`` is
    left for the caller to discover, and the high-level ``create()`` /
    ``upsert()`` helpers pass the error string through as the return
    value. That made duplicate-id detection fragile: the previous
    implementation string-sniffed the return of ``create()`` and treated
    any ``str`` result as an error, which would break silently if the SDK
    ever returned a legitimate string or changed its error format.

    This helper uses the protocol-level ``status`` + ``kind`` fields
    instead. It returns the unwrapped ``result`` on success; on error it
    raises :class:`DuplicateIdError` for ``kind == "AlreadyExists"`` (or
    unique-index violations that carry the same semantics) and
    :class:`BackendError` otherwise.

    The ``duplicate_id`` parameter lets callers tag the raised
    ``DuplicateIdError`` with the caller's id for a clearer message.
    """
    if not isinstance(raw_response, dict):
        raise BackendError(f"surreal backend {context}: unexpected response {raw_response!r}")
    results = raw_response.get("result")
    if not isinstance(results, list) or not results:
        raise BackendError(f"surreal backend {context}: no result in response")
    stmt = results[0]
    if not isinstance(stmt, dict):
        raise BackendError(f"surreal backend {context}: malformed statement result")
    status = stmt.get("status")
    if status == "OK":
        return stmt.get("result")
    # Any non-OK status is an error. Duplicate / unique-violation kinds
    # map to DuplicateIdError so callers can catch them specifically.
    kind = stmt.get("kind") or ""
    message = stmt.get("result") if isinstance(stmt.get("result"), str) else str(stmt)
    # ``kind`` is the stable protocol-level signal (AlreadyExists for a
    # direct record-id collision, IndexExists for a UNIQUE-index hit).
    if kind in ("AlreadyExists", "IndexExists"):
        raise DuplicateIdError(
            f"surreal backend {context}: duplicate id {duplicate_id!r}: {message}"
            if duplicate_id is not None
            else f"surreal backend {context}: {message}"
        )
    raise BackendError(f"surreal backend {context}: {message}")


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
# Default text -> embedding function (parity with ChromaBackend)
# ---------------------------------------------------------------------------
#
# ChromaBackend relies on chromadb's ``DefaultEmbeddingFunction`` to convert
# ``query_texts`` into vectors before hitting HNSW. To match parity from
# ``searcher.py`` — which only ever passes ``query_texts`` — the Surreal
# backend borrows the same embedder when the caller hasn't supplied
# ``query_embeddings`` directly. Loaded lazily so a pure ``query_embeddings``
# workload never pays the import cost.

_default_text_embedder = None


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of query strings with the ChromaDB default embedder.

    Parity note: this is the same model (all-MiniLM-L6-v2) that
    :class:`ChromaBackend` uses when a caller passes ``query_texts=``. We
    reuse it rather than taking a fresh sentence-transformers dependency so
    the vector space is identical across backends — important when the same
    palace has been embedded once (at ingest) and is then queried by text.
    """
    global _default_text_embedder
    if _default_text_embedder is None:
        from chromadb.utils import embedding_functions  # late import

        _default_text_embedder = embedding_functions.DefaultEmbeddingFunction()
    # chromadb returns numpy arrays; normalize to plain lists for SurrealQL
    # binding which only knows about Python scalars.
    raw = _default_text_embedder(texts)
    return [[float(v) for v in row] for row in raw]


# ---------------------------------------------------------------------------
# Bootstrap DDL — applied idempotently on first create.
# ---------------------------------------------------------------------------


def _bootstrap_ddl(embedding_dim: Optional[int] = None) -> str:
    """Return the schema DDL applied on first ``create=True``.

    This is a narrower slice of ``docs/surrealdb-schema.md``: drawer + closet
    payload + ``palace_meta:main`` + FULLTEXT (BM25) indexes used by hybrid
    search (mp-j19). Wing/room/entity/triple remain deferred to the graph-
    layer task (mp-4yf).

    The HNSW vector index is **not** emitted here. SurrealDB's HNSW locks a
    fixed dimension at DEFINE time and silently drops records whose vector
    shape does not match, so we defer index creation to the first embedding
    write (see :meth:`SurrealCollection._ensure_hnsw_index`). That matches
    Chroma's "first write locks the dim" semantics — and keeps bootstrap
    cheap when a caller plans to use BM25-only search.

    Note on schema-doc divergence (mp-j19): the doc says
    ``DEFINE INDEX ... SEARCH ANALYZER ... BM25``. SurrealDB 3.0.4 rejects
    the ``SEARCH`` keyword here — the current syntax is
    ``DEFINE INDEX ... FULLTEXT ANALYZER ... BM25``. We emit the 3.0.4 form.

    Earlier note on schema-doc (mp-6xi): the doc uses
    ``FLEXIBLE TYPE option<object>`` for ``metadata`` — that parses as a
    syntax error in SurrealDB 3.0.4, which requires ``TYPE ... FLEXIBLE``.
    """
    # mp-hlo: only initialise ``embedding_dim`` when the caller has
    # explicitly declared one. A caller-agnostic palace leaves this as
    # NONE so the first real write can lock the dim atomically via the
    # conditional UPDATE in :meth:`SurrealCollection._ensure_hnsw_index`.
    # Pre-seeding the field (the old behaviour) meant the "WHERE
    # embedding_dim IS NONE" guard always saw a stale 384 and silently
    # no-op'd, which defeats the race guard.
    if embedding_dim is not None:
        dim_upsert = (
            "UPSERT palace_meta:main SET "
            f"schema_version = {_SCHEMA_VERSION}, "
            f"embedder_name = '{_DEFAULT_EMBEDDER}', "
            f"embedding_dim = {int(embedding_dim)}, "
            f"hnsw_space = '{_DEFAULT_HNSW_SPACE}';"
        )
    else:
        dim_upsert = (
            "UPSERT palace_meta:main SET "
            f"schema_version = {_SCHEMA_VERSION}, "
            f"embedder_name = '{_DEFAULT_EMBEDDER}', "
            f"hnsw_space = '{_DEFAULT_HNSW_SPACE}';"
        )
    return f"""
    DEFINE TABLE IF NOT EXISTS palace_meta SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS schema_version  ON palace_meta TYPE int;
    DEFINE FIELD IF NOT EXISTS embedder_name   ON palace_meta TYPE string;
    DEFINE FIELD IF NOT EXISTS embedding_dim   ON palace_meta TYPE option<int>;
    DEFINE FIELD IF NOT EXISTS hnsw_space      ON palace_meta TYPE string DEFAULT 'cosine';
    DEFINE FIELD IF NOT EXISTS created_at      ON palace_meta TYPE datetime DEFAULT time::now();

    DEFINE ANALYZER IF NOT EXISTS mp_text
        TOKENIZERS blank, class
        FILTERS lowercase, ascii, snowball(english);

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
    DEFINE INDEX IF NOT EXISTS drawer_ft     ON drawer FIELDS document
        FULLTEXT ANALYZER mp_text BM25(1.2, 0.75) HIGHLIGHTS;

    DEFINE TABLE IF NOT EXISTS closet SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS id_ext     ON closet TYPE string;
    DEFINE FIELD IF NOT EXISTS document   ON closet TYPE string ASSERT $value != NONE;
    DEFINE FIELD IF NOT EXISTS embedding  ON closet TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS metadata   ON closet TYPE option<object> FLEXIBLE;
    DEFINE FIELD IF NOT EXISTS filed_at   ON closet TYPE datetime
        VALUE $value OR time::now() DEFAULT time::now();
    DEFINE INDEX IF NOT EXISTS closet_id_ext ON closet FIELDS id_ext UNIQUE;
    DEFINE INDEX IF NOT EXISTS closet_ft     ON closet FIELDS document
        FULLTEXT ANALYZER mp_text BM25(1.2, 0.75) HIGHLIGHTS;

    {dim_upsert}
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
        # Cached per-process flag: has the HNSW index been checked/created on
        # this collection? The check is lazy because SurrealDB's HNSW locks a
        # fixed dim at DEFINE time, so we can only build the index once we
        # have an actual embedding in hand.
        self._hnsw_checked = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rid(self, ext_id: str):
        """Build a ``RecordID`` for this table from a caller id."""
        return self._RecordID(self._table, ext_id)

    def _ensure_hnsw_index(self, observed_dim: int) -> None:
        """Create the HNSW index on first embedding write (mp-j19, mp-o7v, mp-hlo).

        SurrealDB 3.0.4 DEFINE INDEX with HNSW bakes the dimension into the
        index and silently drops mismatched writes, so we defer creation
        until we see a real vector. On subsequent calls, the index already
        exists and the ``IF NOT EXISTS`` guard makes this a no-op.

        The dim is locked into ``palace_meta:main.embedding_dim`` using a
        conditional SET (only writes when the field is still ``NONE``) so
        two racing writers cannot install conflicting dims (mp-hlo). After
        the conditional write we SELECT the stored dim back: if it differs
        from our observation, the loser raises
        :class:`DimensionMismatchError`.

        DDL failure raises :class:`HnswIndexCreationError` and leaves
        ``_hnsw_checked`` unset so the next call retries — silent swallow
        here (the old behaviour) violates the 100% recall principle by
        turning every subsequent vector query into a zero-row return
        (mp-o7v).
        """
        if self._hnsw_checked:
            return
        index_name = f"{self._table}_vec"
        observed = int(observed_dim)

        # Race-free dim lock (mp-hlo): only set embedding_dim when it is
        # still NONE. Concurrent writers with different dims will observe
        # the winner's value on read-back and raise DimensionMismatchError.
        try:
            self._db.query(
                "UPDATE palace_meta:main SET embedding_dim = $d WHERE embedding_dim IS NONE;",
                {"d": observed},
            )
            rows = self._db.query("SELECT embedding_dim FROM palace_meta:main;")
        except Exception as e:
            raise HnswIndexCreationError(
                f"failed to lock embedding_dim for {self._table!r}: {e}"
            ) from e

        stored_dim: Optional[int] = None
        if rows:
            row = rows[0] if isinstance(rows, list) else rows
            if isinstance(row, dict) and isinstance(row.get("embedding_dim"), int):
                stored_dim = row["embedding_dim"]
        if stored_dim is not None and stored_dim != observed:
            # Do NOT set _hnsw_checked — the caller's write is invalid but
            # the process is free to retry with the correct dim.
            raise DimensionMismatchError(
                f"palace embedding_dim is {stored_dim} but write uses dim {observed}"
            )

        # Define the HNSW index itself. IF NOT EXISTS makes this idempotent
        # across concurrent first-writers; the dim baked in by the winner
        # wins, and because we already agreed on `observed == stored_dim`
        # both processes pass through cleanly.
        ddl = (
            f"DEFINE INDEX IF NOT EXISTS {index_name} ON {self._table} "
            f"FIELDS embedding HNSW DIMENSION {observed} "
            f"DIST COSINE M 16 EFC 150;"
        )
        try:
            self._db.query(ddl)
        except Exception as e:
            # Silent swallow here means every vector query returns zero
            # rows for the life of this process. Raise instead so the
            # caller can retry or fail fast (mp-o7v).
            raise HnswIndexCreationError(f"failed to create HNSW index {index_name!r}: {e}") from e

        self._hnsw_checked = True

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

    def _enforce_write_dims(self, embeddings: Optional[list[list[float]]]) -> Optional[int]:
        """Reject writes whose embedding length does not match the locked dim.

        Chroma raises ``DimensionMismatchError`` when a wrong-dim vector is
        added after the collection has seen its first embedding; Surreal's
        HNSW silently drops the row, which is the opposite of the 100%
        recall contract. Check *before* any write lands in Surreal (mp-s58).

        Returns the dim of the first non-empty embedding in ``embeddings``,
        so callers can chain into :meth:`_ensure_hnsw_index`. Returns
        ``None`` when the payload has no vectors at all.
        """
        if not embeddings:
            return None
        first_dim: Optional[int] = None
        for i, emb in enumerate(embeddings):
            if emb is None:
                continue
            # Surreal stores lists; reject ragged payloads eagerly.
            if not isinstance(emb, (list, tuple)):
                raise DimensionMismatchError(
                    f"embedding {i} is not a list/tuple (got {type(emb).__name__})"
                )
            if not emb:
                continue
            if first_dim is None:
                first_dim = len(emb)
            elif len(emb) != first_dim:
                # Ragged within a single call — reject before hitting DB.
                raise DimensionMismatchError(
                    f"embedding {i} has length {len(emb)} but batch dim is {first_dim}"
                )
        if first_dim is None:
            return None
        expected = self._expected_embedding_dim()
        if expected is not None and first_dim != expected:
            raise DimensionMismatchError(
                f"embedding dim {first_dim} does not match palace embedding_dim {expected}"
            )
        return first_dim

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_writes(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        # Reject dim-mismatched writes BEFORE they hit Surreal (mp-s58) and
        # lazy-create the HNSW index using the observed dim (mp-j19).
        observed = self._enforce_write_dims(embeddings)
        if observed is not None:
            self._ensure_hnsw_index(observed)
        for i, ext_id in enumerate(ids):
            payload = self._record_payload(
                ext_id=ext_id,
                document=documents[i],
                metadata=metadatas[i] if metadatas is not None else None,
                embedding=embeddings[i] if embeddings is not None else None,
            )
            # mp-bac: bypass the SDK's high-level ``create`` helper — it
            # swallows per-statement errors into its return value (as a
            # plain string) and the previous "treat-any-string-as-error"
            # sniff broke silently whenever the SDK surface changed. Drive
            # ``CREATE ... CONTENT $c`` through ``query_raw`` instead and
            # inspect the wire-level ``status`` / ``kind`` fields, which
            # are the protocol's stable error signal.
            rid = self._rid(ext_id)
            raw = self._db.query_raw(
                "CREATE $rec CONTENT $c",
                {"rec": rid, "c": payload},
            )
            _raise_on_statement_error(raw, f"add id={ext_id!r}", duplicate_id=ext_id)

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_writes(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        observed = self._enforce_write_dims(embeddings)
        if observed is not None:
            self._ensure_hnsw_index(observed)
        for i, ext_id in enumerate(ids):
            payload = self._record_payload(
                ext_id=ext_id,
                document=documents[i],
                metadata=metadatas[i] if metadatas is not None else None,
                embedding=embeddings[i] if embeddings is not None else None,
            )
            # mp-bac: same protocol-level check as ``add`` — upsert can
            # still fail with schema-validation errors that the SDK
            # would otherwise surface as a return-value string.
            rid = self._rid(ext_id)
            raw = self._db.query_raw(
                "UPSERT $rec CONTENT $c",
                {"rec": rid, "c": payload},
            )
            _raise_on_statement_error(raw, f"upsert id={ext_id!r}")

    # Max retries for the optimistic concurrency loop in :meth:`update`
    # (mp-93e). SurrealDB 3.0.4 uses MVCC without compare-and-swap; two
    # concurrent updates to the same record can race with last-writer-
    # wins. Retrying a server-side MERGE after a lost-write detection
    # closes the gap without needing a row-level lock the engine does
    # not expose.
    _UPDATE_MAX_RETRIES = 8

    def update(
        self,
        *,
        ids,
        documents=None,
        metadatas=None,
        embeddings=None,
    ):
        """Atomic per-id merge update (mp-93e).

        The merge lands in a single server-side statement —
        ``UPDATE $rid MERGE { metadata: $patch, ... }`` — so the
        client never round-trips a SELECT-then-MERGE (which the old
        implementation did, and which could lose an entire writer's
        keys under contention).

        SurrealDB 3.0.4's storage engine is optimistic-MVCC with no
        CAS primitive, so two concurrent MERGEs can still race and
        one writer's keys may disappear on the first attempt. To
        guarantee the merge is effectively atomic at the user-visible
        level we add a read-after-write verification and retry loop
        bounded by ``_UPDATE_MAX_RETRIES``. A retry is only issued
        when a key the caller just wrote is missing from the stored
        metadata — a true lost-update — so unrelated concurrent writes
        that happen to overlap on a different key set never retry.
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

        # Pre-validate embedding dims the same way add/upsert do (mp-s58),
        # so partial updates cannot slip a wrong-dim vector past the lock.
        if embeddings is not None:
            self._enforce_write_dims(embeddings)

        for i, ext_id in enumerate(ids):
            meta_patch = dict(metadatas[i] or {}) if metadatas is not None else None
            doc = documents[i] if documents is not None else None
            emb = list(embeddings[i]) if embeddings is not None else None
            self._update_one(ext_id, doc, meta_patch, emb)

    def _update_one(
        self,
        ext_id: str,
        document: Optional[str],
        metadata_patch: Optional[dict],
        embedding: Optional[list[float]],
    ) -> None:
        """Apply one MERGE-style update, retrying on MVCC lost-updates (mp-93e).

        The merge payload is a single ``{metadata, document, embedding}``
        dict passed via ``UPDATE $rid MERGE $m``. For metadata specifically,
        SurrealDB's ``MERGE`` operator deep-merges caller keys into the
        stored object — so the statement is itself atomic — but since the
        engine cannot CAS against a version, a concurrent MERGE on the
        same record can still overwrite us. We detect that by re-reading
        and re-issuing until the caller's keys are visible (bounded by
        :attr:`_UPDATE_MAX_RETRIES`).
        """
        merge_payload: dict[str, Any] = {}
        if document is not None:
            merge_payload["document"] = document
        if metadata_patch is not None:
            merge_payload["metadata"] = metadata_patch
        if embedding is not None:
            merge_payload["embedding"] = embedding
        if not merge_payload:
            return

        rid = self._rid(ext_id)
        patch_keys = set(metadata_patch.keys()) if metadata_patch else set()

        for attempt in range(self._UPDATE_MAX_RETRIES):
            # Single server-side MERGE — no client-side SELECT-then-MERGE
            # round trip. Chroma's UPDATE call is analogous: one shot to
            # the storage engine with the caller's patch.
            self._db.query(
                "UPDATE $rid MERGE $m;",
                {"rid": rid, "m": merge_payload},
            )
            if not patch_keys:
                # No metadata merge semantics to verify — single statement
                # is sufficient for document / embedding writes.
                return

            # Verify the metadata merge landed. A concurrent writer's
            # MERGE can race ours (MVCC, no CAS) and we may find our keys
            # missing even though the statement returned cleanly.
            existing = self._db.select(rid)
            if existing is None:
                return
            row = existing[0] if isinstance(existing, list) else existing
            stored_meta = dict(row.get("metadata") or {}) if isinstance(row, dict) else {}
            all_present = all(
                stored_meta.get(k) == metadata_patch[k]  # type: ignore[index]
                for k in patch_keys
            )
            if all_present:
                return
            # Lost update detected — loop and retry the MERGE. Because
            # MERGE is additive, retrying is safe even if the other
            # writer's keys are now interleaved with ours.
            logger.debug(
                "mp-93e retry %d for id=%r: keys %r not visible after MERGE",
                attempt + 1,
                ext_id,
                [k for k in patch_keys if stored_meta.get(k) != metadata_patch[k]],  # type: ignore[index]
            )
        # Exhausted retries — surface a clear error so the caller sees
        # the race rather than a silent lost-write.
        raise BackendError(
            f"update for id={ext_id!r} could not durably land metadata keys "
            f"{sorted(patch_keys)} after {self._UPDATE_MAX_RETRIES} retries"
        )

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
        """Semantic / hybrid search (mp-j19).

        Dispatch rules:

        * Exactly one of ``query_texts`` / ``query_embeddings`` MUST be
          provided — mirrors :class:`ChromaCollection.query` and the base
          contract.
        * ``query_embeddings`` → HNSW KNN via
          ``WHERE embedding <|n,COSINE|> $vec``. Distances are cosine.
        * ``query_texts`` → embed with the ChromaDB default embedder (the
          same model Chroma uses), then HNSW KNN. This preserves parity
          with the existing ``searcher.py`` orchestration layer, which only
          ever passes ``query_texts``.
        * ``where_document={"$search": "<tokens>"}`` promotes the call to a
          BM25-only path against the FULLTEXT index. Scores are converted
          into pseudo-distances (``1 / (1 + score)``) so the existing
          ``searcher.py`` ``similarity = 1 - distance`` math keeps working.
          ``{"$contains": "<substr>"}`` keeps exact substring semantics via
          ``string::contains``.

        Hybrid approach (mp-j19): the backend exposes BM25 + HNSW as two
        independent paths and lets ``searcher.py`` continue to own the
        cross-path re-ranking. SurrealQL *could* combine them in one query
        (``... WHERE document @0@ $q OR embedding <|k,COSINE|> $vec``) but
        merging result sets from two indexes inside one ``SELECT`` gives
        non-comparable scores (BM25 vs cosine distance) and eliminates the
        rank-based closet boost that ``search_memories`` depends on. Keeping
        the two paths separate matches Chroma's behaviour exactly and
        satisfies the "return shape must match what searcher.py expects"
        constraint without modifying it.
        """
        _validate_where(where)
        _validate_where(where_document)

        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("query requires exactly one of query_texts or query_embeddings")
        chosen = query_texts if query_texts is not None else query_embeddings
        if not isinstance(chosen, list) or not chosen:
            raise ValueError("query input must be a non-empty list")

        spec = _IncludeSpec.resolve(include, default_distances=True)

        # BM25 path: caller explicitly asked for full-text. We support this
        # only for query_texts (embedding-as-search-token makes no sense).
        if (
            where_document is not None
            and isinstance(where_document, dict)
            and "$search" in where_document
        ):
            if query_texts is None:
                raise UnsupportedFilterError(
                    "where_document={'$search': ...} requires query_texts, not query_embeddings"
                )
            search_term = where_document["$search"]
            # Any other keys become AND'd where filters.
            extra_wd = {k: v for k, v in where_document.items() if k != "$search"}
            # mp-15y: callers pass the explicit BM25 phrase via
            # ``$search`` and use ``query_texts`` to shape the outer
            # dimension. Chroma runs one independent search per
            # ``query_texts`` element; to match that, treat the value of
            # ``$search`` as a template. A single string applies to every
            # outer query (legacy behaviour). A list must match
            # ``len(query_texts)`` so each outer row gets its own
            # BM25 ranking rather than a broadcasted copy.
            if isinstance(search_term, list):
                if len(search_term) != len(query_texts):
                    raise UnsupportedFilterError(
                        "where_document={'$search': [...]} length "
                        f"{len(search_term)} must match query_texts length "
                        f"{len(query_texts)} (one search term per query)"
                    )
                terms = [str(t) for t in search_term]
            else:
                terms = [str(search_term)] * len(query_texts)
            return self._bm25_query_multi(
                query_texts=query_texts,
                search_terms=terms,
                n_results=n_results,
                where=where,
                where_document=extra_wd or None,
                spec=spec,
            )

        # Vector path (default): HNSW KNN over embeddings.
        if query_embeddings is not None:
            vectors = [list(v) for v in query_embeddings]
        else:
            vectors = _embed_texts(list(query_texts))

        expected_dim = self._expected_embedding_dim()
        for i, vec in enumerate(vectors):
            if expected_dim is not None and len(vec) != expected_dim:
                raise DimensionMismatchError(
                    f"query vector {i} has length {len(vec)} "
                    f"but palace embedding_dim is {expected_dim}"
                )

        out_ids: list[list[str]] = []
        out_docs: list[list[str]] = []
        out_metas: list[list[dict]] = []
        out_dists: list[list[float]] = []
        out_embeds: list[list[list[float]]] = []

        for vec in vectors:
            rows = self._hnsw_knn(
                vec=vec,
                n_results=n_results,
                where=where,
                where_document=where_document,
                spec=spec,
            )
            ids_i, docs_i, metas_i, dists_i, embs_i = self._split_rows(rows, spec)
            out_ids.append(ids_i)
            out_docs.append(docs_i)
            out_metas.append(metas_i)
            out_dists.append(dists_i)
            out_embeds.append(embs_i)

        return QueryResult(
            ids=out_ids,
            documents=out_docs if spec.documents else [[] for _ in vectors],
            metadatas=out_metas if spec.metadatas else [[] for _ in vectors],
            distances=out_dists if spec.distances else [[] for _ in vectors],
            embeddings=out_embeds if spec.embeddings else None,
        )

    # ------------------------------------------------------------------
    # Query helpers (mp-j19)
    # ------------------------------------------------------------------

    def _expected_embedding_dim(self) -> Optional[int]:
        """Read ``palace_meta:main.embedding_dim`` if present.

        Returns ``None`` on any failure — the HNSW index itself will reject
        wrong-dim vectors (returning zero hits) so missing meta is a soft
        failure, not a hard one.
        """
        try:
            rows = self._db.query("SELECT embedding_dim FROM palace_meta:main;")
        except Exception:  # pragma: no cover - defensive
            return None
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        if isinstance(row, dict) and isinstance(row.get("embedding_dim"), int):
            return row["embedding_dim"]
        return None

    def _build_filter_clause(
        self,
        *,
        where: Optional[dict],
        where_document: Optional[dict],
    ) -> tuple[str, dict[str, Any]]:
        """Combine ``where`` and ``where_document`` into a single WHERE fragment.

        Returns ``(clause_without_leading_WHERE, bindings)``. Empty string if
        no filters apply.
        """
        parts: list[str] = []
        bindings: dict[str, Any] = {}
        if where:
            clause, extra = _translate_where(where, key_prefix="m")
            if clause:
                parts.append(clause)
                bindings.update(extra)
        if where_document:
            clause, extra = _translate_where_document(where_document, key_prefix="d")
            if clause:
                parts.append(clause)
                bindings.update(extra)
        if not parts:
            return "", bindings
        return " AND ".join(f"({p})" for p in parts), bindings

    def _hnsw_knn(
        self,
        *,
        vec: list[float],
        n_results: int,
        where: Optional[dict],
        where_document: Optional[dict],
        spec: _IncludeSpec,
    ) -> list[dict]:
        """Run a single KNN lookup via HNSW.

        The ``<|K,COSINE|>`` operator does the index walk; we wrap it in a
        plain ``SELECT`` so we can co-project any ``where=`` /
        ``where_document=`` predicates. Distance comes out of
        ``vector::distance::knn()``.
        """
        select_fields = ["id_ext", "vector::distance::knn() AS _distance"]
        if spec.documents:
            select_fields.append("document")
        if spec.metadatas:
            select_fields.append("metadata")
        if spec.embeddings:
            select_fields.append("embedding")

        filter_clause, bindings = self._build_filter_clause(
            where=where, where_document=where_document
        )
        bindings["vec"] = list(vec)
        n = int(n_results)

        q = (
            f"SELECT {', '.join(select_fields)} FROM {self._table} "
            f"WHERE embedding <|{n},COSINE|> $vec"
        )
        if filter_clause:
            q += f" AND {filter_clause}"
        # ORDER BY distance ASC so best matches land first; LIMIT belongs
        # here too because the KNN operator caps candidates at K but the
        # combined filter may yield fewer, which is fine.
        q += " ORDER BY _distance LIMIT $k"
        bindings["k"] = n

        try:
            rows = self._db.query(q, bindings)
        except Exception as e:
            logger.warning("HNSW query failed, returning empty: %s", e)
            return []
        if rows is None:
            return []
        if not isinstance(rows, list):
            rows = [rows]
        return [r for r in rows if isinstance(r, dict)]

    def _bm25_query_multi(
        self,
        *,
        query_texts: list[str],
        search_terms: list[str],
        n_results: int,
        where: Optional[dict],
        where_document: Optional[dict],
        spec: _IncludeSpec,
    ) -> QueryResult:
        """Run one independent BM25 query per element of ``query_texts`` (mp-15y).

        The old implementation ran a single BM25 query and broadcast the
        same row set across every outer row, which meant a caller passing
        ``query_texts=["foo", "bar"]`` saw identical rankings for both — a
        correctness bug when the two queries have different best matches.
        Chroma's behaviour is one independent ranking per outer element;
        this helper matches that by looping.

        Each ``search_terms[i]`` is the literal phrase passed to
        ``document @0@ $term`` for query ``i``. BM25 score ->
        pseudo-distance ``1 / (1 + score)`` so ``searcher.py``'s
        ``similarity = 1 - distance`` convention keeps producing sensible
        numbers (higher score -> lower pseudo-distance -> higher sim).
        """
        assert len(search_terms) == len(query_texts)

        select_fields = [
            "id_ext",
            "search::score(0) AS _score",
        ]
        if spec.documents:
            select_fields.append("document")
        if spec.metadatas:
            select_fields.append("metadata")
        if spec.embeddings:
            select_fields.append("embedding")

        def _score_to_dist(row: dict) -> float:
            s = row.get("_score") or 0.0
            return 1.0 / (1.0 + float(s))

        out_ids: list[list[str]] = []
        out_docs: list[list[str]] = []
        out_metas: list[list[dict]] = []
        out_dists: list[list[float]] = []
        out_embeds: list[list[list[float]]] = []

        for term in search_terms:
            filter_clause, bindings = self._build_filter_clause(
                where=where, where_document=where_document
            )
            bindings["term"] = term
            bindings["k"] = int(n_results)

            q = f"SELECT {', '.join(select_fields)} FROM {self._table} WHERE document @0@ $term"
            if filter_clause:
                q += f" AND {filter_clause}"
            q += " ORDER BY _score DESC LIMIT $k"

            try:
                rows = self._db.query(q, bindings)
            except Exception as e:
                logger.warning("BM25 query failed, returning empty: %s", e)
                rows = []
            if rows is None:
                rows = []
            if not isinstance(rows, list):
                rows = [rows]
            rows = [r for r in rows if isinstance(r, dict)]

            ids_i: list[str] = []
            docs_i: list[str] = []
            metas_i: list[dict] = []
            dists_i: list[float] = []
            embs_i: list[list[float]] = []
            for row in rows:
                ids_i.append(row.get("id_ext") or "")
                if spec.documents:
                    docs_i.append(row.get("document") or "")
                if spec.metadatas:
                    metas_i.append(dict(row.get("metadata") or {}))
                if spec.distances:
                    dists_i.append(_score_to_dist(row))
                if spec.embeddings:
                    emb = row.get("embedding")
                    embs_i.append(list(emb) if emb is not None else [])
            out_ids.append(ids_i)
            out_docs.append(docs_i)
            out_metas.append(metas_i)
            out_dists.append(dists_i)
            out_embeds.append(embs_i)

        n = len(query_texts)
        return QueryResult(
            ids=out_ids,
            documents=out_docs if spec.documents else [[] for _ in range(n)],
            metadatas=out_metas if spec.metadatas else [[] for _ in range(n)],
            distances=out_dists if spec.distances else [[] for _ in range(n)],
            embeddings=out_embeds if spec.embeddings else None,
        )

    def _split_rows(
        self, rows: list[dict], spec: _IncludeSpec
    ) -> tuple[list[str], list[str], list[dict], list[float], list[list[float]]]:
        """Split a list of Surreal result rows into the per-field lists the
        :class:`QueryResult` constructor expects.

        Missing optional fields degrade to empty strings / dicts / ``0.0`` so
        the downstream ``zip(ids, docs, metas, dists)`` loop in
        ``searcher.py`` never hits a ``None``.
        """
        ids_i: list[str] = []
        docs_i: list[str] = []
        metas_i: list[dict] = []
        dists_i: list[float] = []
        embs_i: list[list[float]] = []
        for row in rows:
            ids_i.append(row.get("id_ext") or "")
            if spec.documents:
                docs_i.append(row.get("document") or "")
            if spec.metadatas:
                metas_i.append(dict(row.get("metadata") or {}))
            if spec.distances:
                dist = row.get("_distance")
                dists_i.append(float(dist) if dist is not None else 0.0)
            if spec.embeddings:
                emb = row.get("embedding")
                embs_i.append(list(emb) if emb is not None else [])
        return ids_i, docs_i, metas_i, dists_i, embs_i

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
        elif op == "$search":
            # ``$search`` is handled upstream in :meth:`SurrealCollection.query`
            # where it promotes the call to a dedicated BM25 path; it must
            # never reach the AND-combined WHERE clause a ``get()`` or
            # ``delete()`` would build, because the FULLTEXT operator needs
            # an index reference (``@0@``) that only the query helper knows.
            raise UnsupportedFilterError(
                "where_document={'$search': ...} is only valid on query(); "
                "get()/delete() should use {'$contains': ...}"
            )
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
            # mp-j19: HNSW vector KNN + FULLTEXT BM25 both land in-DB.
            "supports_vector_search",
            "supports_bm25_search",
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

    def _ensure_bootstrap(
        self,
        db_name: str,
        conn,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """Apply schema DDL once per (palace, process). Idempotent on Surreal.

        When ``embedding_dim`` is ``None`` (the default), ``palace_meta:main``
        is created with ``embedding_dim`` left unset so the first real write
        can lock the dim atomically via the race-free conditional UPDATE
        (mp-hlo). Passing an explicit dim — e.g. from
        ``options={"embedding_dim": ...}`` — pre-seeds the field; subsequent
        writes with a different dim raise
        :class:`DimensionMismatchError`.
        """
        with self._lock:
            tables = self._bootstrapped.setdefault(db_name, set())
            if "drawer" in tables and "closet" in tables:
                return
        conn.query(_bootstrap_ddl(embedding_dim))
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
            # Only pre-seed embedding_dim when the caller explicitly asks
            # for one — otherwise leave it NONE so first-write locks the
            # dim atomically (mp-hlo).
            embedding_dim: Optional[int] = None
            if options and isinstance(options, dict) and "embedding_dim" in options:
                embedding_dim = int(options["embedding_dim"])
            self._ensure_bootstrap(db_name, conn, embedding_dim=embedding_dim)

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
