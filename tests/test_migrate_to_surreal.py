"""Tests for ``mempalace migrate-to-surreal`` (mp-ciw).

Moves drawers + embeddings from a ChromaDB palace into SurrealDB. These
tests exercise the real Chroma read path end-to-end and the real
SurrealDB write path when a local SurrealDB server is reachable. If the
server is not reachable the module is skipped — the migration requires
Surreal to be running by definition.
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest

from mempalace.backends.chroma import ChromaBackend
from mempalace.migrate import (
    TargetCollisionError,
    _derive_surreal_db_name,
    migrate_to_surreal,
)


# ---------------------------------------------------------------------------
# Server reachability — if SurrealDB isn't running, skip the whole module.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_WING_ROOMS = [
    ("project_a", "2026-04-01"),
    ("project_a", "2026-04-02"),
    ("project_a", "2026-04-03"),
    ("project_b", "2026-04-01"),
    ("project_b", "2026-04-02"),
    ("project_b", "2026-04-03"),
]

_EMBED_DIM = 4


def _stable_embedding(seed: int) -> list[float]:
    """Generate a deterministic low-dim embedding from an integer seed.

    Kept tiny (dim=4) so tests are fast and the HNSW index builds cleanly
    on first write. Values are spread across the unit cube to avoid all-
    zero vectors (which some indexes reject).
    """
    return [
        0.1 + (seed % 7) * 0.05,
        0.2 + (seed % 5) * 0.07,
        0.3 + (seed % 3) * 0.11,
        0.4 + (seed % 11) * 0.03,
    ]


@pytest.fixture()
def chroma_palace(tmp_path):
    """Build a small on-disk Chroma palace with ~20 drawers across 2 wings x 3 rooms.

    Uses real ChromaBackend writes (no mocks) so the migration pipeline
    exercises the same Chroma ``get(...)`` code path a production palace
    hits — including pagination, metadata round-trip, and embedding
    retrieval via ``include=["embeddings"]``.
    """
    palace_path = tmp_path / "src_palace"
    palace_path.mkdir()
    backend = ChromaBackend()

    drawer_col = backend.get_or_create_collection(str(palace_path), "mempalace_drawers")
    closet_col = backend.get_or_create_collection(str(palace_path), "mempalace_closets")

    # 20 drawers spread across (wing, room) pairs, each with a stable dim=4 embedding.
    drawer_ids: list[str] = []
    drawer_docs: list[str] = []
    drawer_metas: list[dict] = []
    drawer_embeds: list[list[float]] = []
    for i in range(20):
        wing, room = _WING_ROOMS[i % len(_WING_ROOMS)]
        drawer_id = f"drawer_{wing}_{room}_{i:02d}"
        drawer_ids.append(drawer_id)
        drawer_docs.append(f"verbatim content for drawer {i} in {wing}/{room}")
        drawer_metas.append(
            {
                "wing": wing,
                "room": room,
                "source_file": f"/tmp/fake/{wing}_{i}.md",
                "chunk_index": i,
                "normalize_version": 2,
            }
        )
        drawer_embeds.append(_stable_embedding(i))
    drawer_col.add(
        ids=drawer_ids,
        documents=drawer_docs,
        metadatas=drawer_metas,
        embeddings=drawer_embeds,
    )

    # A handful of closets so the migration also exercises the second
    # collection path. Distinct metadata to prove round-trip fidelity.
    closet_ids = [f"closet_{i:02d}" for i in range(3)]
    closet_col.add(
        ids=closet_ids,
        documents=[f"aaak pointer line {i}" for i in range(3)],
        metadatas=[
            {"wing": "project_a", "room": "2026-04-01", "closet_kind": "topic"} for _ in range(3)
        ],
        embeddings=[_stable_embedding(100 + i) for i in range(3)],
    )

    # Close the backend so the sqlite file is safely reopened by the migration.
    backend.close()

    return {
        "path": str(palace_path),
        "drawer_ids": drawer_ids,
        "drawer_docs": drawer_docs,
        "drawer_metas": drawer_metas,
        "drawer_embeds": drawer_embeds,
        "closet_ids": closet_ids,
    }


@pytest.fixture()
def surreal_target():
    """Throwaway Surreal NS so each test cleans up after itself."""
    ns = f"mp_mig_{uuid.uuid4().hex[:10]}"
    os.environ["MEMPALACE_SURREAL_NS"] = ns
    yield ns
    # Cleanup: drop the NS so repeated runs don't accumulate DBs.
    try:
        from mempalace.backends.surreal import SurrealBackend

        b = SurrealBackend(namespace=ns)
        conn = b._connect("cleanup_dummy")
        conn.query(f"REMOVE NAMESPACE IF EXISTS {ns};")
        b.close()
    except Exception:
        pass
    os.environ.pop("MEMPALACE_SURREAL_NS", None)


# ---------------------------------------------------------------------------
# Core migration behavior
# ---------------------------------------------------------------------------


def test_derive_surreal_db_name_sanitizes_path(tmp_path):
    weird = tmp_path / "weird-palace.v2"
    weird.mkdir()
    name = _derive_surreal_db_name(str(weird))
    # Only [A-Za-z0-9_] is valid in a Surreal USE DB identifier.
    assert name
    assert all(ch.isalnum() or ch == "_" for ch in name)


def test_full_migration_counts_match(chroma_palace, surreal_target):
    """End-to-end: every Chroma drawer lands in Surreal with matching payload."""
    result = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        progress=False,
    )
    assert result["dry_run"] is False
    # 20 drawers + 3 closets = 23 total.
    assert result["total"] == 23
    assert result["migrated"] == 23
    assert result["verified"] is True
    assert result["errors"] == []
    assert result["by_collection"]["mempalace_drawers"] == 20
    assert result["by_collection"]["mempalace_closets"] == 3


def test_migration_spot_check_documents_and_metadata(chroma_palace, surreal_target):
    """Sampled drawers round-trip document + metadata + embedding dim."""
    from mempalace.backends.base import PalaceRef
    from mempalace.backends.surreal import SurrealBackend

    migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        progress=False,
    )

    # Re-open Surreal to verify payload (the migration already verified
    # internally; this is a belt-and-suspenders check from the outside).
    db_name = _derive_surreal_db_name(chroma_palace["path"])
    backend = SurrealBackend(namespace=surreal_target)
    try:
        col = backend.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_drawers",
            create=False,
        )
        first_id = chroma_palace["drawer_ids"][0]
        got = col.get(
            ids=[first_id],
            include=["documents", "metadatas", "embeddings"],
        )
        assert got.ids == [first_id]
        assert got.documents[0] == chroma_palace["drawer_docs"][0]
        src_meta = chroma_palace["drawer_metas"][0]
        for k, v in src_meta.items():
            assert got.metadatas[0].get(k) == v, f"meta mismatch on {k}"
        assert got.embeddings is not None
        assert len(got.embeddings[0]) == _EMBED_DIM
    finally:
        backend.close()


def test_dry_run_performs_no_writes(chroma_palace, surreal_target):
    """--dry-run must not create any records in Surreal."""
    from mempalace.backends.base import PalaceNotFoundError, PalaceRef
    from mempalace.backends.surreal import SurrealBackend

    result = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        dry_run=True,
        progress=False,
    )
    assert result["dry_run"] is True
    assert result["migrated"] == 0
    assert result["total"] == 23

    # Opening the target palace with create=False must fail — nothing was written.
    db_name = _derive_surreal_db_name(chroma_palace["path"])
    backend = SurrealBackend(namespace=surreal_target)
    try:
        with pytest.raises((PalaceNotFoundError, Exception)):
            backend.get_collection(
                palace=PalaceRef(id=db_name, namespace=db_name),
                collection_name="mempalace_drawers",
                create=False,
            )
    finally:
        backend.close()


def test_migration_is_idempotent(chroma_palace, surreal_target):
    """Running the migration twice must not duplicate rows.

    Surreal keys drawers by ``drawer:<id>`` record id + UNIQUE index on
    ``id_ext``, and we drive writes through ``upsert``, so a second run
    should no-op at the persistence layer.
    """
    r1 = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        progress=False,
    )
    r2 = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        progress=False,
    )
    assert r1["migrated"] == r2["migrated"] == 23
    assert r2["verified"] is True

    # Count on the Surreal side should be exactly 20 drawers + 3 closets
    # after the second run — no duplicates.
    from mempalace.backends.base import PalaceRef
    from mempalace.backends.surreal import SurrealBackend

    db_name = _derive_surreal_db_name(chroma_palace["path"])
    backend = SurrealBackend(namespace=surreal_target)
    try:
        drawers = backend.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_drawers",
            create=False,
        )
        closets = backend.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_closets",
            create=False,
        )
        assert drawers.count() == 20
        assert closets.count() == 3
    finally:
        backend.close()


def test_resumability_after_partial_interrupt(chroma_palace, surreal_target, monkeypatch):
    """Simulate a mid-migration crash and confirm a re-run completes cleanly.

    We patch :class:`mempalace.backends.surreal.SurrealCollection` so the
    first run raises after the second upsert batch. The second call to
    ``migrate_to_surreal`` must finish the migration and leave the palace
    in a consistent state — no duplicates, full counts.
    """
    from mempalace.backends.surreal import SurrealCollection

    original_upsert = SurrealCollection.upsert
    call_count = {"n": 0}

    def flaky_upsert(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-migration interrupt (mp-ciw test)")
        return original_upsert(self, *args, **kwargs)

    # Use a small batch size so we definitely split the 20-drawer corpus
    # into multiple batches before tripping the simulated failure.
    monkeypatch.setattr(SurrealCollection, "upsert", flaky_upsert)
    with pytest.raises(RuntimeError, match="simulated mid-migration"):
        migrate_to_surreal(
            source_palace=chroma_palace["path"],
            target_ns=surreal_target,
            batch_size=4,
            progress=False,
        )

    # Restore the real upsert and re-run — idempotent upsert means we
    # finish the job without duplicating anything already written.
    monkeypatch.setattr(SurrealCollection, "upsert", original_upsert)
    result = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        batch_size=4,
        progress=False,
    )
    assert result["migrated"] == 23
    assert result["verified"] is True

    from mempalace.backends.base import PalaceRef
    from mempalace.backends.surreal import SurrealBackend

    db_name = _derive_surreal_db_name(chroma_palace["path"])
    backend = SurrealBackend(namespace=surreal_target)
    try:
        drawers = backend.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_drawers",
            create=False,
        )
        assert drawers.count() == 20
    finally:
        backend.close()


def test_missing_source_palace_raises(tmp_path, surreal_target):
    """Pointing --source at a non-palace directory must raise cleanly."""
    bogus = tmp_path / "not_a_palace"
    bogus.mkdir()
    with pytest.raises(FileNotFoundError):
        migrate_to_surreal(
            source_palace=str(bogus),
            target_ns=surreal_target,
            progress=False,
        )


# ---------------------------------------------------------------------------
# mp-2v9: collision guard against --target-db that already has drawers.
# ---------------------------------------------------------------------------


def test_target_db_populated_refuses_without_allow_merge(chroma_palace, surreal_target):
    """Explicit --target-db + already-populated target -> refuse.

    Round-trip through the real Surreal server: first migration fills
    ``shared_target``, second migration (with explicit --target-db
    pointing at the same name) must raise TargetCollisionError.
    """
    shared_db = f"collide_{uuid.uuid4().hex[:6]}"

    # First migration: target is empty, proceeds normally.
    r1 = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        target_db=shared_db,
        progress=False,
    )
    assert r1["migrated"] == 23

    # Second migration: same NS/DB but pretend it's a different source
    # palace — copy it to a new directory with a different basename so
    # the default derivation would NOT collide. The explicit --target-db
    # drives the collision.
    import shutil

    second_path = chroma_palace["path"] + "_clone"
    shutil.copytree(chroma_palace["path"], second_path)

    with pytest.raises(TargetCollisionError):
        migrate_to_surreal(
            source_palace=second_path,
            target_ns=surreal_target,
            target_db=shared_db,
            progress=False,
        )


def test_target_db_populated_proceeds_with_allow_merge(chroma_palace, surreal_target):
    """--allow-merge lets a caller upsert into an existing Surreal DB.

    This is the escape hatch for deliberate merges (e.g. rebuilding an
    index while keeping old drawers around). The second migration's
    idempotent upsert means counts stay at 23 after it completes.
    """
    shared_db = f"merge_{uuid.uuid4().hex[:6]}"
    r1 = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        target_db=shared_db,
        progress=False,
    )
    assert r1["migrated"] == 23

    r2 = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        target_db=shared_db,
        allow_merge=True,
        progress=False,
    )
    assert r2["migrated"] == 23
    assert r2["verified"] is True


def test_target_db_fresh_proceeds_silently(chroma_palace, surreal_target):
    """Explicit --target-db pointing at an empty NS/DB proceeds cleanly."""
    fresh_db = f"fresh_{uuid.uuid4().hex[:6]}"
    result = migrate_to_surreal(
        source_palace=chroma_palace["path"],
        target_ns=surreal_target,
        target_db=fresh_db,
        progress=False,
    )
    assert result["migrated"] == 23
    assert result["verified"] is True


# ---------------------------------------------------------------------------
# mp-8me: Numpy float32 embeddings along the migration path.
#
# ChromaDB stores embeddings internally as float32. When ``get(include=
# ['embeddings'])`` returns them, the type is ``numpy.ndarray`` whose
# elements are numpy scalars (float32 on older chromadb, float64 on
# 1.5.8). The migration wraps each row with ``list(e)``, which preserves
# the numpy scalar type. The SurrealDB Python SDK 1.0.8's CBOR encoder
# has NO path for ``numpy.float32`` (it raises ``BufferError``), so the
# migration would crash silently on any palace whose chroma binding
# returns float32. We guard by coercing inside the backend; these tests
# prove the guard holds end-to-end along the migration code path.
# ---------------------------------------------------------------------------


def test_migration_preserves_numpy_float32_embedding_values(chroma_palace, surreal_target):
    """A Chroma palace whose embeddings come back as ``numpy.float32`` must
    migrate cleanly with exact (within float32 tolerance) value preservation.

    We spy on the raw Chroma collection's ``get`` so each row's embedding
    is re-materialized as a real ``np.float32`` ndarray before the
    migration sees it — exactly what older chromadb builds do natively.
    Without the ``_coerce_embedding_to_py_floats`` guard in the Surreal
    backend this raises::

        BufferError: ('no encoder for type ', <class 'numpy.float32'>)

    With the guard in place, the migration completes and spot-checked
    drawers round-trip their vectors within float32 tolerance.
    """
    import numpy as np

    from mempalace.backends.base import PalaceRef
    from mempalace.backends.chroma import ChromaCollection
    from mempalace.backends.surreal import SurrealBackend

    original_get = ChromaCollection.get

    def float32_get(self, **kwargs):
        res = original_get(self, **kwargs)
        # Rebuild each embedding as a real numpy.float32 ndarray — the
        # migration wraps with ``list(e)``, yielding a list of np.float32
        # scalars, which is the payload shape that broke the SDK.
        if res.embeddings is None:
            return res
        new_embs = [np.asarray(e, dtype=np.float32) for e in res.embeddings]
        from dataclasses import replace

        return replace(res, embeddings=new_embs)

    # Grab the original float values before the migration so we can verify
    # round-trip fidelity within float32 tolerance.
    want_first = chroma_palace["drawer_embeds"][0]
    first_id = chroma_palace["drawer_ids"][0]

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(ChromaCollection, "get", float32_get)
        result = migrate_to_surreal(
            source_palace=chroma_palace["path"],
            target_ns=surreal_target,
            progress=False,
        )
    assert result["migrated"] == 23
    assert result["verified"] is True

    # Spot-check the first drawer's vector survives within float32 tolerance.
    db_name = _derive_surreal_db_name(chroma_palace["path"])
    backend = SurrealBackend(namespace=surreal_target)
    try:
        col = backend.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_drawers",
            create=False,
        )
        got = col.get(ids=[first_id], include=["embeddings"])
        assert got.ids == [first_id]
        assert got.embeddings is not None
        got_vec = got.embeddings[0]
        # Output must be plain Python floats (coercion contract).
        assert all(type(x) is float for x in got_vec)
        # Values equal within float32 mantissa (~1e-7 relative).
        assert len(got_vec) == len(want_first)
        for g, w in zip(got_vec, want_first):
            assert abs(float(g) - float(w)) <= 1e-5, (
                f"value drift beyond float32 tolerance: {g!r} vs {w!r}"
            )
    finally:
        backend.close()


def test_migration_with_numpy_float32_seeded_palace(tmp_path, surreal_target):
    """Seed a Chroma palace by writing ``np.float32`` ndarrays directly.

    This exercises the write-through path rather than spying on the
    read — chroma accepts ``np.asarray(..., dtype=np.float32)`` as an
    embedding and stores it natively. The migration must succeed and
    the final Surreal rows must be plain Python floats (not numpy
    scalars) within float32 tolerance of the seed values.
    """
    import numpy as np

    from mempalace.backends.base import PalaceRef
    from mempalace.backends.chroma import ChromaBackend
    from mempalace.backends.surreal import SurrealBackend

    palace_path = tmp_path / "src_palace_f32"
    palace_path.mkdir()
    backend = ChromaBackend()
    drawer_col = backend.get_or_create_collection(str(palace_path), "mempalace_drawers")

    ids = [f"f32_{i:02d}" for i in range(6)]
    docs = [f"doc {i}" for i in range(6)]
    metas = [{"wing": "w", "room": "r", "chunk_index": i} for i in range(6)]
    # Real NumPy float32 ndarrays — the type Chroma stores internally.
    seed_vecs = [
        np.asarray([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i], dtype=np.float32) for i in range(6)
    ]
    drawer_col.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=seed_vecs,
    )
    backend.close()

    result = migrate_to_surreal(
        source_palace=str(palace_path),
        target_ns=surreal_target,
        progress=False,
    )
    assert result["migrated"] == 6
    assert result["verified"] is True

    # Verify from Surreal that every vector landed and values match.
    db_name = _derive_surreal_db_name(str(palace_path))
    surreal_b = SurrealBackend(namespace=surreal_target)
    try:
        col = surreal_b.get_collection(
            palace=PalaceRef(id=db_name, namespace=db_name),
            collection_name="mempalace_drawers",
            create=False,
        )
        got = col.get(ids=ids, include=["embeddings"])
        assert sorted(got.ids) == sorted(ids)
        got_by_id = dict(zip(got.ids, got.embeddings))
        for i, gid in enumerate(ids):
            gvec = got_by_id[gid]
            assert all(type(x) is float for x in gvec), (
                f"numpy scalar leaked through on {gid}: {[type(x).__name__ for x in gvec]}"
            )
            for g, w in zip(gvec, seed_vecs[i].tolist()):
                assert abs(g - w) <= 1e-5, f"drift on {gid}: {g!r} vs {w!r}"
    finally:
        surreal_b.close()
