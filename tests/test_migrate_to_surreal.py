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
