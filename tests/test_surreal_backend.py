"""Tests for the SurrealDB backend drawer CRUD (mp-6xi).

These tests run against a **real local Surreal server** as configured in
``docs/surrealdb-local.md`` (``http://127.0.0.1:8000``, ``root``/``root``).
The entire test namespace is created and torn down per test so repeated
runs do not leak state. If the server is not reachable the whole module
is skipped — it stays out of the default ``pytest`` run otherwise.
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest

from mempalace.backends import (
    DimensionMismatchError,
    GetResult,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
)


def _surreal_reachable() -> bool:
    host, port = "127.0.0.1", 8000
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _surreal_reachable(),
    reason="local SurrealDB not running on 127.0.0.1:8000 (see docs/surrealdb-local.md)",
)


@pytest.fixture()
def surreal_backend():
    """Fresh SurrealBackend instance bound to a throwaway namespace."""
    from mempalace.backends.surreal import SurrealBackend

    test_ns = f"mp_test_{uuid.uuid4().hex[:12]}"
    os.environ["MEMPALACE_SURREAL_NS"] = test_ns
    backend = SurrealBackend(namespace=test_ns)
    try:
        yield backend
    finally:
        # Drop the entire namespace so nothing leaks between runs.
        try:
            conn = backend._connect("cleanup_dummy")
            conn.query(f"REMOVE NAMESPACE IF EXISTS {test_ns};")
        except Exception:
            pass
        backend.close()


@pytest.fixture()
def palace_ref():
    return PalaceRef(id=f"palace_{uuid.uuid4().hex[:10]}", local_path=None)


@pytest.fixture()
def drawer_collection(surreal_backend, palace_ref):
    return surreal_backend.get_collection(
        palace=palace_ref,
        collection_name="mempalace_drawers",
        create=True,
    )


# ---------------------------------------------------------------------------
# Bootstrap / lifecycle
# ---------------------------------------------------------------------------


def test_create_true_bootstraps_schema(surreal_backend, palace_ref):
    col = surreal_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=True
    )
    # Schema is idempotent: second call on the same palace must not error.
    col2 = surreal_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_closets", create=True
    )
    assert col._table == "drawer"
    assert col2._table == "closet"


def test_create_false_raises_for_unknown_palace(surreal_backend):
    from mempalace.backends import PalaceNotFoundError

    missing = PalaceRef(id=f"missing_{uuid.uuid4().hex[:8]}")
    with pytest.raises(PalaceNotFoundError):
        surreal_backend.get_collection(
            palace=missing, collection_name="mempalace_drawers", create=False
        )


def test_unknown_collection_name_raises(surreal_backend, palace_ref):
    with pytest.raises(ValueError):
        surreal_backend.get_collection(
            palace=palace_ref, collection_name="does_not_exist", create=True
        )


# ---------------------------------------------------------------------------
# add / get / count
# ---------------------------------------------------------------------------


def test_add_then_get_roundtrip(drawer_collection):
    drawer_collection.add(
        documents=["first doc", "second doc"],
        ids=["id1", "id2"],
        metadatas=[{"wing": "w1"}, {"wing": "w2"}],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    assert drawer_collection.count() == 2

    result = drawer_collection.get(ids=["id1", "id2"])
    assert isinstance(result, GetResult)
    assert sorted(result.ids) == ["id1", "id2"]
    # Ordered by id_ext -> stable.
    by_id = dict(zip(result.ids, result.documents))
    assert by_id == {"id1": "first doc", "id2": "second doc"}


def test_get_with_include_embeddings(drawer_collection):
    drawer_collection.add(
        documents=["a"],
        ids=["x"],
        metadatas=[{"k": "v"}],
        embeddings=[[0.9, 0.8]],
    )
    result = drawer_collection.get(ids=["x"], include=["documents", "metadatas", "embeddings"])
    assert result.ids == ["x"]
    assert result.documents == ["a"]
    assert result.metadatas == [{"k": "v"}]
    assert result.embeddings == [[0.9, 0.8]]


def test_get_include_filters_fields(drawer_collection):
    drawer_collection.add(
        documents=["doc"],
        ids=["only"],
        metadatas=[{"k": "v"}],
    )
    r = drawer_collection.get(ids=["only"], include=["documents"])
    assert r.documents == ["doc"]
    # Metadatas not requested -> empty list (typed result shape).
    assert r.metadatas == []
    assert r.embeddings is None


def test_get_empty_returns_empty_result(drawer_collection):
    r = drawer_collection.get(ids=["nope"])
    assert r.ids == []
    assert r.documents == []
    assert r.metadatas == []


def test_count_empty_collection(drawer_collection):
    assert drawer_collection.count() == 0


# ---------------------------------------------------------------------------
# add semantics — duplicate rejection
# ---------------------------------------------------------------------------


def test_add_rejects_duplicate_id(drawer_collection):
    drawer_collection.add(documents=["a"], ids=["dup"])
    with pytest.raises(Exception):
        drawer_collection.add(documents=["b"], ids=["dup"])


def test_add_duplicate_raises_specific_duplicate_error(drawer_collection):
    """mp-bac: duplicate-id detection uses the wire-level status/kind
    envelope, not string-sniffing. We raise a typed ``DuplicateIdError``
    with the offending id embedded, so callers can catch it specifically
    instead of a bare ``Exception``.
    """
    from mempalace.backends.surreal import DuplicateIdError

    drawer_collection.add(documents=["a"], ids=["dup"])
    with pytest.raises(DuplicateIdError) as exc:
        drawer_collection.add(documents=["b"], ids=["dup"])
    # The raised error carries the duplicate id for observability.
    assert "'dup'" in str(exc.value) or '"dup"' in str(exc.value)
    # The original record is untouched — duplicate rejection must not
    # overwrite the existing document.
    r = drawer_collection.get(ids=["dup"])
    assert r.documents == ["a"]


def test_add_does_not_string_sniff_error_messages(drawer_collection, monkeypatch):
    """mp-bac: regression guard — success must not be mis-classified as an
    error just because the SDK happens to return a string.

    The old implementation raised ``RuntimeError`` on any ``str`` return
    from ``self._db.create``. The new implementation only flags errors
    reported via the wire-protocol ``status == "ERR"`` envelope, so a
    legitimate success path is never misread.
    """
    from mempalace.backends.surreal import _raise_on_statement_error

    # Wire-protocol OK envelope with a string payload: the helper must
    # treat this as success, not error. Pre-fix behaviour would have
    # rejected any ``str`` return — a latent false-positive.
    ok_response = {"result": [{"status": "OK", "result": "some-ok-string", "time": "1µs"}]}
    assert _raise_on_statement_error(ok_response, "probe") == "some-ok-string"


def test_add_statement_err_raises_backend_error(drawer_collection):
    """mp-bac: non-duplicate ERR statuses raise ``BackendError`` — not a
    ``DuplicateIdError`` and not a silent success.
    """
    from mempalace.backends import BackendError
    from mempalace.backends.surreal import DuplicateIdError, _raise_on_statement_error

    err_response = {
        "result": [{"status": "ERR", "kind": "Thrown", "result": "some failure", "time": "1µs"}]
    }
    with pytest.raises(BackendError) as exc:
        _raise_on_statement_error(err_response, "probe")
    # BackendError hierarchy: DuplicateIdError is a subclass, so catch the
    # parent and confirm the type is NOT DuplicateIdError.
    assert not isinstance(exc.value, DuplicateIdError)


def test_add_length_mismatch_raises(drawer_collection):
    with pytest.raises(ValueError):
        drawer_collection.add(documents=["a", "b"], ids=["x"])
    with pytest.raises(ValueError):
        drawer_collection.add(documents=["a"], ids=["x"], metadatas=[{"k": "v"}, {"k": "w"}])


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def test_upsert_creates_and_replaces(drawer_collection):
    drawer_collection.upsert(documents=["v1"], ids=["key"], metadatas=[{"n": 1}])
    drawer_collection.upsert(documents=["v2"], ids=["key"], metadatas=[{"n": 2}])
    r = drawer_collection.get(ids=["key"])
    assert r.documents == ["v2"]
    assert r.metadatas == [{"n": 2}]
    assert drawer_collection.count() == 1


# ---------------------------------------------------------------------------
# update (atomic merge)
# ---------------------------------------------------------------------------


def test_update_merges_metadata_preserves_other_keys(drawer_collection):
    drawer_collection.add(documents=["doc"], ids=["key"], metadatas=[{"a": 1, "b": 2}])
    drawer_collection.update(ids=["key"], metadatas=[{"b": 99, "c": 3}])
    r = drawer_collection.get(ids=["key"])
    assert r.metadatas == [{"a": 1, "b": 99, "c": 3}]
    # Document untouched.
    assert r.documents == ["doc"]


def test_update_requires_at_least_one_field(drawer_collection):
    drawer_collection.add(documents=["x"], ids=["id1"])
    with pytest.raises(ValueError):
        drawer_collection.update(ids=["id1"])


def test_update_length_mismatch_raises(drawer_collection):
    drawer_collection.add(documents=["x"], ids=["id1"])
    with pytest.raises(ValueError):
        drawer_collection.update(ids=["id1", "id2"], documents=["a"])


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_by_ids(drawer_collection):
    drawer_collection.add(documents=["a", "b"], ids=["d1", "d2"])
    drawer_collection.delete(ids=["d1"])
    assert drawer_collection.count() == 1
    r = drawer_collection.get(ids=["d1", "d2"])
    assert r.ids == ["d2"]


def test_delete_by_where(drawer_collection):
    drawer_collection.add(
        documents=["a", "b", "c"],
        ids=["x1", "x2", "x3"],
        metadatas=[{"wing": "w1"}, {"wing": "w2"}, {"wing": "w1"}],
    )
    drawer_collection.delete(where={"wing": "w1"})
    r = drawer_collection.get()
    assert r.ids == ["x2"]


def test_delete_without_args_is_noop(drawer_collection):
    drawer_collection.add(documents=["a"], ids=["d"])
    drawer_collection.delete()
    assert drawer_collection.count() == 1


# ---------------------------------------------------------------------------
# where-clause translation
# ---------------------------------------------------------------------------


def test_where_eq_shortcut(drawer_collection):
    drawer_collection.add(
        documents=["a", "b"],
        ids=["1", "2"],
        metadatas=[{"wing": "alpha"}, {"wing": "beta"}],
    )
    r = drawer_collection.get(where={"wing": "alpha"})
    assert r.ids == ["1"]


def test_where_in(drawer_collection):
    drawer_collection.add(
        documents=["a", "b", "c"],
        ids=["1", "2", "3"],
        metadatas=[{"wing": "a"}, {"wing": "b"}, {"wing": "c"}],
    )
    r = drawer_collection.get(where={"wing": {"$in": ["a", "c"]}})
    assert sorted(r.ids) == ["1", "3"]


def test_where_and_or(drawer_collection):
    drawer_collection.add(
        documents=["a", "b", "c"],
        ids=["1", "2", "3"],
        metadatas=[
            {"wing": "w", "kind": "p"},
            {"wing": "w", "kind": "q"},
            {"wing": "z", "kind": "p"},
        ],
    )
    r = drawer_collection.get(where={"$and": [{"wing": "w"}, {"kind": "p"}]})
    assert r.ids == ["1"]
    r = drawer_collection.get(where={"$or": [{"wing": "z"}, {"kind": "q"}]})
    assert sorted(r.ids) == ["2", "3"]


def test_where_unknown_operator_raises(drawer_collection):
    with pytest.raises(UnsupportedFilterError):
        drawer_collection.get(where={"$weird": "x"})


def test_where_document_contains(drawer_collection):
    drawer_collection.add(
        documents=["hello world", "goodbye world", "hello again"],
        ids=["1", "2", "3"],
    )
    r = drawer_collection.get(where_document={"$contains": "hello"})
    assert sorted(r.ids) == ["1", "3"]


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------


def test_limit_and_offset(drawer_collection):
    drawer_collection.add(
        documents=[f"d{i}" for i in range(5)],
        ids=[f"id{i}" for i in range(5)],
    )
    r1 = drawer_collection.get(limit=2)
    assert r1.ids == ["id0", "id1"]
    r2 = drawer_collection.get(limit=2, offset=2)
    assert r2.ids == ["id2", "id3"]


# ---------------------------------------------------------------------------
# query() — hybrid BM25 + vector (mp-j19)
# ---------------------------------------------------------------------------


def _seed_vector_corpus(collection):
    """Seed a small 3-dim vector corpus for deterministic KNN assertions.

    Vectors are laid out so that ``[1.0, 0.0, 0.0]`` cleanly prefers ``v1``
    (same direction), ``[0.0, 1.0, 0.0]`` prefers ``v2``, etc. — no ties.
    """
    collection.add(
        documents=[
            "the quick brown fox jumps over the lazy dog",
            "a second document about cats and birds",
            "machine learning pipelines use vector embeddings",
            "totally unrelated content about baking bread",
        ],
        ids=["v1", "v2", "v3", "v4"],
        metadatas=[
            {"wing": "w1", "room": "r1"},
            {"wing": "w2", "room": "r1"},
            {"wing": "w1", "room": "r2"},
            {"wing": "w3", "room": "r3"},
        ],
        embeddings=[
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.5, 0.5, 0.5],
        ],
    )


def test_query_vector_only_returns_typed_shape(drawer_collection):
    """Pure vector path: query_embeddings -> HNSW KNN, QueryResult shape."""
    _seed_vector_corpus(drawer_collection)

    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=2,
    )
    assert isinstance(result, QueryResult)
    # Outer dim = 1 query, inner dim = up to n_results hits.
    assert len(result.ids) == 1
    assert len(result.ids[0]) == 2
    # v1 has embedding [0.9, 0.1, 0.1] — closest to [1.0, 0.0, 0.0].
    assert result.ids[0][0] == "v1"
    assert result.documents[0][0].startswith("the quick brown")
    # Distances are cosine-like: closer = smaller.
    assert result.distances[0][0] <= result.distances[0][1]


def test_query_vector_respects_where_filter(drawer_collection):
    _seed_vector_corpus(drawer_collection)
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=5,
        where={"wing": "w1"},
    )
    # Only v1 and v3 are in wing w1.
    assert set(result.ids[0]) == {"v1", "v3"}


def test_query_vector_multi_query_outer_dim(drawer_collection):
    _seed_vector_corpus(drawer_collection)
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        n_results=1,
    )
    # Two queries -> outer dim 2, each with 1 hit.
    assert len(result.ids) == 2
    assert result.ids[0][0] == "v1"
    assert result.ids[1][0] == "v2"


def test_query_vector_dim_mismatch_raises(drawer_collection):
    _seed_vector_corpus(drawer_collection)
    with pytest.raises(DimensionMismatchError):
        drawer_collection.query(
            query_embeddings=[[1.0, 0.0]],  # wrong dim (2 vs. 3)
            n_results=1,
        )


def test_query_empty_collection_returns_empty_inner(drawer_collection):
    """Querying before anything is inserted must not crash — empty inner lists."""
    # Force-create the HNSW index with a throwaway write we then delete so the
    # index exists but the row set is empty.
    drawer_collection.add(
        documents=["temp"],
        ids=["temp_id"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    drawer_collection.delete(ids=["temp_id"])
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=5,
    )
    assert result.ids == [[]]
    assert result.documents == [[]]
    assert result.distances == [[]]


def test_query_bm25_only_via_where_document_search(drawer_collection):
    """where_document={'$search': ...} promotes to BM25-only full-text."""
    _seed_vector_corpus(drawer_collection)
    # Top up with extra docs so BM25 idf actually scores non-zero.
    drawer_collection.add(
        documents=[
            "filler document about gardening",
            "another filler talking about cooking",
            "unrelated filler covering travel",
        ],
        ids=["f1", "f2", "f3"],
        embeddings=[[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
    )
    result = drawer_collection.query(
        query_texts=["brown"],
        where_document={"$search": "brown"},
        n_results=5,
    )
    assert "v1" in result.ids[0]
    # Highest-scoring doc is v1 ("brown fox"); pseudo-distance strictly < 1.
    idx = result.ids[0].index("v1")
    assert result.distances[0][idx] < 1.0


def test_query_bm25_ranks_by_score(drawer_collection):
    """BM25 path returns rows ordered best-score first (lowest pseudo-dist)."""
    _seed_vector_corpus(drawer_collection)
    drawer_collection.add(
        documents=["pad one", "pad two", "pad three", "pad four"],
        ids=["p1", "p2", "p3", "p4"],
        embeddings=[[0.1, 0.1, 0.1]] * 4,
    )
    result = drawer_collection.query(
        query_texts=["machine"],
        where_document={"$search": "machine"},
        n_results=3,
    )
    # v3's document contains "machine" — expect it first.
    assert result.ids[0][0] == "v3"
    # Distances must be monotonically non-decreasing (best first).
    dists = result.distances[0]
    assert all(a <= b for a, b in zip(dists, dists[1:]))


def test_query_requires_exactly_one_input(drawer_collection):
    with pytest.raises(ValueError):
        drawer_collection.query()
    with pytest.raises(ValueError):
        drawer_collection.query(query_texts=[], query_embeddings=None)
    with pytest.raises(ValueError):
        drawer_collection.query(query_texts=["foo"], query_embeddings=[[0.1, 0.2, 0.3]])


def test_query_with_include_distances_embeddings(drawer_collection):
    _seed_vector_corpus(drawer_collection)
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    assert result.embeddings is not None
    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) == 1
    assert len(result.embeddings[0][0]) == 3


def test_query_include_omits_unrequested_fields(drawer_collection):
    _seed_vector_corpus(drawer_collection)
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
        include=["documents"],
    )
    # documents requested, metadatas/distances/embeddings not.
    assert result.documents[0]
    assert result.metadatas == [[]]
    assert result.distances == [[]]
    assert result.embeddings is None


def test_query_search_on_get_path_raises(drawer_collection):
    """``$search`` only makes sense on query(); get()/delete() must reject it."""
    drawer_collection.add(documents=["x"], ids=["id"])
    with pytest.raises(UnsupportedFilterError):
        drawer_collection.get(where_document={"$search": "x"})


def test_query_various_n_results(drawer_collection):
    """n_results bounds the inner list length; asking for more than present
    should return what exists, never crash (parity with Chroma)."""
    _seed_vector_corpus(drawer_collection)
    # 4 docs seeded; ask for 10.
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=10,
    )
    assert len(result.ids[0]) == 4
    # And ask for 1 — only the best match.
    result = drawer_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
    )
    assert len(result.ids[0]) == 1
    assert result.ids[0][0] == "v1"


def test_query_hybrid_where_and_search(drawer_collection):
    """BM25 path must respect a ``where=`` metadata filter alongside $search."""
    _seed_vector_corpus(drawer_collection)
    drawer_collection.add(
        documents=["machine model in wing w3", "ignore this one"],
        ids=["m1", "m2"],
        metadatas=[{"wing": "w3"}, {"wing": "w3"}],
        embeddings=[[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
    )
    result = drawer_collection.query(
        query_texts=["machine"],
        where={"wing": "w3"},
        where_document={"$search": "machine"},
        n_results=5,
    )
    # Only w3's "machine model" document should come back.
    assert result.ids[0] == ["m1"]


def test_query_text_path_uses_default_embedder(drawer_collection):
    """query_texts without $search triggers the text-embed vector path.

    We can't easily assert ranking without loading the real embedder (slow),
    so this test only checks that the call completes and returns the typed
    QueryResult shape. The embedder model is shared with Chroma by design —
    parity is verified at the integration layer, not here.
    """
    # Seed with 384-dim embeddings (matching DefaultEmbeddingFunction output)
    # so the lazily-created HNSW index has the right dim.
    import random

    random.seed(0)
    dim = 384
    n = 4
    drawer_collection.add(
        documents=[
            "the cat sat on the mat",
            "dogs chase the ball in the park",
            "a recipe for chocolate cake",
            "the fox and the hound are friends",
        ],
        ids=[f"t{i}" for i in range(n)],
        embeddings=[[random.random() for _ in range(dim)] for _ in range(n)],
    )
    result = drawer_collection.query(query_texts=["cat"], n_results=2)
    assert isinstance(result, QueryResult)
    assert len(result.ids) == 1
    assert len(result.ids[0]) == 2


def test_query_embeddings_empty_list_raises(drawer_collection):
    with pytest.raises(ValueError):
        drawer_collection.query(query_embeddings=[])


# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------


def test_close_palace_evicts_connection(surreal_backend, palace_ref):
    surreal_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=True
    )
    from mempalace.backends.surreal import _safe_db_name

    db_name = _safe_db_name(palace_ref)
    assert db_name in surreal_backend._conns
    surreal_backend.close_palace(palace_ref)
    assert db_name not in surreal_backend._conns


def test_close_marks_backend_closed(surreal_backend, palace_ref):
    from mempalace.backends import BackendClosedError

    surreal_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=True
    )
    surreal_backend.close()
    with pytest.raises(BackendClosedError):
        surreal_backend.get_collection(
            palace=palace_ref, collection_name="mempalace_drawers", create=True
        )


def test_health_healthy_by_default(surreal_backend):
    assert surreal_backend.health().ok is True


def test_detect_always_false():
    """Surreal palaces have no on-disk marker in the palace path."""
    from mempalace.backends.surreal import SurrealBackend

    assert SurrealBackend.detect("/tmp/any/path") is False


# ---------------------------------------------------------------------------
# P0 regression guards — mp-o7v, mp-s58, mp-hlo, mp-93e, mp-15y
# ---------------------------------------------------------------------------


def test_hnsw_ddl_failure_raises_and_retries(drawer_collection, monkeypatch):
    """mp-o7v: HNSW DDL failure MUST raise, not silently swallow.

    Silent swallow turns every subsequent vector query into a zero-row
    result for the life of the process — the opposite of the 100% recall
    principle. The fix raises :class:`HnswIndexCreationError` and leaves
    ``_hnsw_checked`` unset so the next call retries.
    """
    from mempalace.backends.surreal import HnswIndexCreationError

    real_query = drawer_collection._db.query
    call_state = {"fail_ddl": True}

    def fake_query(stmt, *args, **kwargs):
        # Only fail the DEFINE INDEX ... HNSW statement; everything else
        # (palace_meta updates, SELECTs) passes through so the test can
        # exercise the actual DDL-failure branch.
        if "DEFINE INDEX" in stmt and "HNSW" in stmt and call_state["fail_ddl"]:
            raise RuntimeError("simulated DDL failure")
        return real_query(stmt, *args, **kwargs)

    monkeypatch.setattr(drawer_collection._db, "query", fake_query)

    with pytest.raises(HnswIndexCreationError):
        drawer_collection.add(
            documents=["x"],
            ids=["a"],
            embeddings=[[0.1, 0.2, 0.3]],
        )
    # Flag stays False so the next call retries — critical for recall.
    assert drawer_collection._hnsw_checked is False

    # Stop forcing the failure; retry should succeed cleanly.
    call_state["fail_ddl"] = False
    drawer_collection.add(
        documents=["x"],
        ids=["a"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    assert drawer_collection._hnsw_checked is True


def test_add_wrong_dim_after_lock_raises_dimension_mismatch(drawer_collection):
    """mp-s58: add() must reject wrong-dim vectors once dim is locked."""
    # Lock dim at 3.
    drawer_collection.add(
        documents=["first"],
        ids=["id1"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    # Wrong dim on a subsequent add must raise BEFORE hitting Surreal —
    # otherwise HNSW silently drops the row and recall is broken.
    with pytest.raises(DimensionMismatchError):
        drawer_collection.add(
            documents=["bad"],
            ids=["id2"],
            embeddings=[[0.1, 0.2]],  # dim 2 != locked 3
        )
    # Also enforced on upsert.
    with pytest.raises(DimensionMismatchError):
        drawer_collection.upsert(
            documents=["bad"],
            ids=["id3"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],  # dim 4
        )
    # And update(embeddings=...) too.
    with pytest.raises(DimensionMismatchError):
        drawer_collection.update(
            ids=["id1"],
            embeddings=[[0.5, 0.6]],
        )


def test_concurrent_first_write_dim_lock_race(surreal_backend, palace_ref):
    """mp-hlo: two processes racing to lock the embedding_dim — one wins.

    Using threads here instead of processes since the SDK is blocking and
    the race hits the same Surreal database. The conditional UPDATE
    ``SET embedding_dim = $d WHERE embedding_dim IS NONE`` serialises at
    the DB level so exactly one dim sticks; the loser observes the
    stored dim and raises :class:`DimensionMismatchError`.
    """
    import threading

    col = surreal_backend.get_collection(
        palace=palace_ref, collection_name="mempalace_drawers", create=True
    )
    # Fresh collection — dim must still be unlocked.
    assert col._expected_embedding_dim() is None

    # Two collection instances racing a first-write. We use the same
    # underlying connection (single-threaded http) but independent
    # collection wrappers so each has its own ``_hnsw_checked`` flag.
    from mempalace.backends.surreal import SurrealCollection

    col_a = SurrealCollection(col._db, col._table)
    col_b = SurrealCollection(col._db, col._table)

    barrier = threading.Barrier(2)
    results: dict[str, Exception | None] = {"a": None, "b": None}

    def writer(which: str, collection, dim: int):
        barrier.wait()
        try:
            collection.add(
                documents=[f"doc-{which}"],
                ids=[f"id-{which}"],
                embeddings=[[0.1] * dim],
            )
        except Exception as e:
            results[which] = e

    t1 = threading.Thread(target=writer, args=("a", col_a, 3))
    t2 = threading.Thread(target=writer, args=("b", col_b, 5))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one writer should have failed with DimensionMismatchError.
    errors = [(k, v) for k, v in results.items() if v is not None]
    successes = [k for k, v in results.items() if v is None]
    assert len(errors) == 1, f"expected one failure, got results={results}"
    assert len(successes) == 1
    _, err = errors[0]
    assert isinstance(err, DimensionMismatchError), f"wrong exception type: {err!r}"

    # Winner's dim is durably stored; further writes at that dim succeed,
    # and at the loser's dim still fail.
    locked = col._expected_embedding_dim()
    assert locked in (3, 5)
    with pytest.raises(DimensionMismatchError):
        col.add(
            documents=["dim-mismatch"],
            ids=["late"],
            embeddings=[[0.2] * (3 if locked == 5 else 5)],
        )


def test_concurrent_metadata_updates_both_visible(surreal_backend, palace_ref):
    """mp-93e: concurrent metadata updates must both land — no lost writes.

    The old implementation SELECTed, merged in Python, then MERGEd back,
    so two writers racing the same record could each observe the same
    pre-image and clobber each other. The fix pushes the merge into one
    SurrealQL ``UPDATE ... SET metadata = object::extend(...)`` statement
    so the merge is server-side and per-record atomic.

    We use two independent backend instances (each with its own
    SurrealDB connection) to get genuine server-side concurrency — the
    blocking-HTTP SDK is not thread-safe across a single connection, so
    driving the race through separate connections is the accurate shape
    of "two processes racing" in production.
    """
    import threading

    from mempalace.backends.surreal import SurrealBackend

    ns = os.environ.get("MEMPALACE_SURREAL_NS")
    backend_a = SurrealBackend(namespace=ns)
    backend_b = SurrealBackend(namespace=ns)
    try:
        col_a = backend_a.get_collection(
            palace=palace_ref, collection_name="mempalace_drawers", create=True
        )
        col_b = backend_b.get_collection(
            palace=palace_ref, collection_name="mempalace_drawers", create=True
        )

        # Seed the row via one of the collections.
        col_a.add(documents=["seed"], ids=["row"], metadatas=[{"k0": 0}])

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def writer(collection, patch):
            barrier.wait()
            try:
                collection.update(ids=["row"], metadatas=[patch])
            except Exception as e:  # pragma: no cover - surface any race crash
                errors.append(e)

        # Repeat a handful of times — if the merge is truly atomic we
        # never lose a key; the SELECT-then-MERGE racer loses keys
        # probabilistically and fails reliably within a few iterations.
        for i in range(5):
            col_a.update(ids=["row"], metadatas=[{"from_a": None, "from_b": None}])
            t1 = threading.Thread(target=writer, args=(col_a, {"from_a": f"A{i}"}))
            t2 = threading.Thread(target=writer, args=(col_b, {"from_b": f"B{i}"}))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            assert not errors, f"concurrent update raised: {errors}"

            r = col_a.get(ids=["row"])
            meta = r.metadatas[0]
            assert meta.get("from_a") == f"A{i}", f"iter {i}: lost writer A's key: {meta!r}"
            assert meta.get("from_b") == f"B{i}", f"iter {i}: lost writer B's key: {meta!r}"
            # Seed key survives every iteration.
            assert meta.get("k0") == 0, f"iter {i}: lost seed key: {meta!r}"
    finally:
        backend_a.close()
        backend_b.close()


def test_bm25_multi_query_per_term_ranking(drawer_collection):
    """mp-15y: multi-element query_texts with $search must rank per-query.

    The old implementation ran a single BM25 query and broadcast the
    same row set across every outer element. That's wrong whenever the
    two queries target different best matches. The fix accepts a list of
    terms and loops one independent search per outer element — matching
    Chroma's per-query behaviour.
    """
    # Seed so each term has a distinct top hit.
    drawer_collection.add(
        documents=[
            "alpha wolf runs through forests at dawn",
            "beta cat watches birds from windowsills",
            "gamma fox burrows into leaf piles",
            "delta mouse scurries beneath floorboards",
        ],
        ids=["w1", "w2", "w3", "w4"],
        embeddings=[[0.1, 0.2, 0.3]] * 4,
    )
    result = drawer_collection.query(
        query_texts=["wolf", "cat"],
        where_document={"$search": ["wolf", "cat"]},
        n_results=3,
    )
    # Two independent rankings — not a broadcast of the same row set.
    assert len(result.ids) == 2
    # First query ranks "wolf" hit first; second ranks "cat" hit first.
    assert result.ids[0][0] == "w1", f"query 0 (wolf) wrong top: {result.ids[0]}"
    assert result.ids[1][0] == "w2", f"query 1 (cat) wrong top: {result.ids[1]}"
    # Confirm the two rankings differ — proves per-query execution.
    assert result.ids[0] != result.ids[1] or (result.distances[0] != result.distances[1]), (
        "per-query rankings must not be a simple broadcast"
    )


def test_bm25_multi_query_length_mismatch_raises(drawer_collection):
    """mp-15y: list-form $search must match len(query_texts) or raise."""
    drawer_collection.add(
        documents=["foo bar baz"],
        ids=["only"],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    with pytest.raises(UnsupportedFilterError):
        drawer_collection.query(
            query_texts=["foo", "bar"],
            where_document={"$search": ["foo"]},  # len 1 vs 2
            n_results=3,
        )
