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
    dim_field = f"{embedding_dim}" if embedding_dim is not None else "NONE"
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

    UPSERT palace_meta:main SET
        schema_version = {_SCHEMA_VERSION},
        embedder_name  = '{_DEFAULT_EMBEDDER}',
        embedding_dim  = {dim_field},
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
        """Create the HNSW index on first embedding write (mp-j19).

        SurrealDB 3.0.4 DEFINE INDEX with HNSW bakes the dimension into the
        index and silently drops mismatched writes, so we defer creation
        until we see a real vector. On subsequent calls, the index already
        exists and the ``IF NOT EXISTS`` guard makes this a no-op.

        The dim is also written back to ``palace_meta:main.embedding_dim``
        so :meth:`_expected_embedding_dim` can enforce
        ``DimensionMismatchError`` on query.
        """
        if self._hnsw_checked:
            return
        index_name = f"{self._table}_vec"
        ddl = (
            f"DEFINE INDEX IF NOT EXISTS {index_name} ON {self._table} "
            f"FIELDS embedding HNSW DIMENSION {observed_dim} "
            f"DIST COSINE M 16 EFC 150;"
        )
        try:
            self._db.query(ddl)
            self._db.query(
                "UPSERT palace_meta:main SET embedding_dim = $d;",
                {"d": int(observed_dim)},
            )
        except Exception:
            logger.exception("failed to create HNSW index %s", index_name)
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

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_writes(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        # Lazy-create the HNSW index using the first non-empty embedding's
        # dimension (mp-j19) — Surreal bakes the dim into DEFINE INDEX.
        if embeddings:
            first = next((e for e in embeddings if e), None)
            if first is not None:
                self._ensure_hnsw_index(len(first))
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
        if embeddings:
            first = next((e for e in embeddings if e), None)
            if first is not None:
                self._ensure_hnsw_index(len(first))
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
            return self._bm25_query(
                query_texts=query_texts,
                search_term=search_term,
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

    def _bm25_query(
        self,
        *,
        query_texts: list[str],
        search_term: str,
        n_results: int,
        where: Optional[dict],
        where_document: Optional[dict],
        spec: _IncludeSpec,
    ) -> QueryResult:
        """Run a BM25 query across the FULLTEXT index.

        ``search_term`` is the literal token string passed to the
        ``document @0@ $term`` operator. The BM25 score is turned into a
        pseudo-distance ``1 / (1 + score)`` so ``searcher.py``'s
        ``similarity = 1 - distance`` convention keeps producing sensible
        numbers: a higher BM25 score -> lower pseudo-distance -> higher
        similarity. The outer shape still honors ``len(query_texts)`` so
        batched callers get the right number of result rows — each batch
        element runs the same BM25 search (this matches Chroma's broadcast
        when multiple query_texts share the same DB).
        """
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

        filter_clause, bindings = self._build_filter_clause(
            where=where, where_document=where_document
        )
        bindings["term"] = search_term
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

        # Convert BM25 score -> pseudo-distance (searcher.py expects cosine-
        # like 0..N distance). Using ``1 / (1 + score)`` keeps it in (0, 1]
        # where 1.0 = no match and distances strictly decrease with score.
        def _score_to_dist(row):
            s = row.get("_score") or 0.0
            return 1.0 / (1.0 + float(s))

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

        # Broadcast across the outer query dim — same rows, one copy per
        # query_text. Chroma does the same shape when the callers share DB.
        n = len(query_texts)
        return QueryResult(
            ids=[list(ids_i) for _ in range(n)],
            documents=[list(docs_i) for _ in range(n)]
            if spec.documents
            else [[] for _ in range(n)],
            metadatas=(
                [list(metas_i) for _ in range(n)] if spec.metadatas else [[] for _ in range(n)]
            ),
            distances=(
                [list(dists_i) for _ in range(n)] if spec.distances else [[] for _ in range(n)]
            ),
            embeddings=[list(embs_i) for _ in range(n)] if spec.embeddings else None,
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
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Apply schema DDL once per (palace, process). Idempotent on Surreal.

        ``embedding_dim`` is baked into the HNSW index definition and the
        ``palace_meta:main`` row. The dimension is fixed for the lifetime of
        the palace — re-bootstrap with a different dim after data exists will
        leave the HNSW index inconsistent and is a caller error (RFC 001
        ``DimensionMismatchError`` territory).
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
            embedding_dim = _DEFAULT_EMBEDDING_DIM
            if options and isinstance(options, dict):
                embedding_dim = int(options.get("embedding_dim", embedding_dim))
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
