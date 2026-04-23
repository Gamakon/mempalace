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
