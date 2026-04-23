"""Tests for destructive-operation safety in mempalace.migrate."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mempalace.migrate import (
    TargetCollisionError,
    _derive_surreal_db_name,
    migrate,
    migrate_to_surreal,
)


def test_migrate_requires_palace_database(tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()

    result = migrate(str(palace_dir))

    out = capsys.readouterr().out
    assert result is False
    assert "No palace database found" in out


def test_migrate_aborts_without_confirmation(tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    # Presence of chroma.sqlite3 is the safety gate; validity is mocked below.
    (palace_dir / "chroma.sqlite3").write_text("db")

    mock_chromadb = SimpleNamespace(
        __version__="0.6.0",
        PersistentClient=MagicMock(side_effect=Exception("unreadable")),
    )

    with (
        patch.dict("sys.modules", {"chromadb": mock_chromadb}),
        patch("mempalace.migrate.detect_chromadb_version", return_value="0.5.x"),
        patch(
            "mempalace.migrate.extract_drawers_from_sqlite",
            return_value=[{"id": "id1", "document": "doc", "metadata": {"wing": "w", "room": "r"}}],
        ),
        patch("builtins.input", return_value="n"),
        patch("mempalace.migrate.shutil.copytree") as mock_copytree,
        patch("mempalace.migrate.shutil.rmtree") as mock_rmtree,
    ):
        result = migrate(str(palace_dir))

    out = capsys.readouterr().out
    assert result is False
    assert "Aborted." in out
    mock_copytree.assert_not_called()
    mock_rmtree.assert_not_called()


# ---------------------------------------------------------------------------
# mp-2v9: --target-db collision guard
# ---------------------------------------------------------------------------


def test_derive_surreal_db_name_disambiguates_same_basename(tmp_path):
    """Two palaces at different paths with the same basename must produce
    different target-db names.

    This is the core mp-2v9 fix: pre-mp-2v9, the name was just the
    basename, so ``/a/mem`` and ``/b/mem`` both resolved to ``mem`` and a
    migration upsert-merged them silently.
    """
    a = tmp_path / "a" / "mem"
    b = tmp_path / "b" / "mem"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    name_a = _derive_surreal_db_name(str(a))
    name_b = _derive_surreal_db_name(str(b))

    # Both keep the human-readable basename prefix...
    assert name_a.startswith("mem_")
    assert name_b.startswith("mem_")
    # ...but the path-hash suffix diverges so they cannot alias.
    assert name_a != name_b
    # Deterministic: same path in -> same name out (hash is stable).
    assert _derive_surreal_db_name(str(a)) == name_a


def test_derive_surreal_db_name_is_surreal_safe(tmp_path):
    """The derived name must contain only ``[A-Za-z0-9_]`` (Surreal USE DB)."""
    weird = tmp_path / "weird-palace.v2"
    weird.mkdir()
    name = _derive_surreal_db_name(str(weird))
    assert name
    assert all(ch.isalnum() or ch == "_" for ch in name), name


def test_derive_surreal_db_name_handles_empty_basename(tmp_path):
    """A path that normalises to an empty basename (e.g. ``/``) still gets
    a deterministic name built from the hash slug."""
    # We can't use "/" in tests; simulate with a trailing-slash string.
    # os.path.normpath("/") -> "/", basename("/") -> "".
    name = _derive_surreal_db_name("/")
    assert name.startswith("palace_")
    # Deterministic per-call.
    assert _derive_surreal_db_name("/") == name


def _build_chroma_palace(tmp_path, name: str = "src") -> str:
    """Create a tiny Chroma palace on disk for collision-guard tests.

    Kept intentionally small (2 drawers, dim=2 vectors) because these
    tests only exercise the guard logic — the collision check runs
    before any Chroma read, so palace size is irrelevant to what we
    are asserting.
    """
    from mempalace.backends.chroma import ChromaBackend

    path = tmp_path / name
    path.mkdir()
    backend = ChromaBackend()
    col = backend.get_or_create_collection(str(path), "mempalace_drawers")
    col.add(
        ids=["d1", "d2"],
        documents=["one", "two"],
        metadatas=[{"wing": "w", "room": "r"}, {"wing": "w", "room": "r"}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    backend.close()
    return str(path)


def test_migrate_refuses_when_target_db_populated_without_allow_merge(tmp_path, monkeypatch):
    """If --target-db is explicit AND the target already has rows, refuse
    unless --allow-merge is set.

    We patch ``_inspect_target_collision`` because exercising the real
    Surreal side would require a running server; the guard logic itself
    is what mp-2v9 is fixing, and it is testable in isolation.
    """
    palace = _build_chroma_palace(tmp_path)

    def fake_inspect(_kwargs, db_name, *, sample_size=3):
        # Pretend the target DB already holds 12 drawers.
        return 12, ["drawer_a", "drawer_b"], {"mempalace_drawers": 12}

    monkeypatch.setattr("mempalace.migrate._inspect_target_collision", fake_inspect)

    with pytest.raises(TargetCollisionError) as excinfo:
        migrate_to_surreal(
            source_palace=palace,
            target_db="other_palace",
            progress=False,
        )
    assert "already has 12" in str(excinfo.value)


def test_migrate_proceeds_with_allow_merge_on_populated_target(tmp_path, monkeypatch):
    """--allow-merge skips the collision check entirely; migration proceeds.

    We stub out the Surreal write path too — we only care that the
    TargetCollisionError does NOT fire when the caller opts in.
    """
    palace = _build_chroma_palace(tmp_path)

    inspect_calls: list[tuple] = []

    def fake_inspect(kwargs, db_name, *, sample_size=3):
        inspect_calls.append((db_name,))
        return 12, ["x"], {"mempalace_drawers": 12}

    monkeypatch.setattr("mempalace.migrate._inspect_target_collision", fake_inspect)

    # Stub the Surreal write path so the test stays offline. The guard
    # runs before these are instantiated, so a successful call past the
    # guard proves the opt-in works.
    class _FakeBackend:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    # migrate_to_surreal opens SurrealBackend inside the function — patch it
    # at import site. The actual write loop would then run; we short-circuit
    # by raising a sentinel exception the test catches to assert "we got
    # past the guard".
    class _PastGuard(Exception):
        pass

    def raise_past_guard(**_kwargs):
        raise _PastGuard("past the guard")

    monkeypatch.setattr("mempalace.backends.surreal.SurrealBackend", raise_past_guard)

    with pytest.raises(_PastGuard):
        migrate_to_surreal(
            source_palace=palace,
            target_db="other_palace",
            allow_merge=True,
            progress=False,
        )

    # --allow-merge short-circuits the inspection entirely.
    assert inspect_calls == []


def test_migrate_proceeds_silently_when_target_empty(tmp_path, monkeypatch):
    """Explicit --target-db + empty target -> no refusal, no noise."""
    palace = _build_chroma_palace(tmp_path)

    def fake_inspect(_kwargs, _db_name, *, sample_size=3):
        return 0, [], {}

    monkeypatch.setattr("mempalace.migrate._inspect_target_collision", fake_inspect)

    class _PastGuard(Exception):
        pass

    def raise_past_guard(**_kwargs):
        raise _PastGuard("past the guard")

    monkeypatch.setattr("mempalace.backends.surreal.SurrealBackend", raise_past_guard)

    with pytest.raises(_PastGuard):
        migrate_to_surreal(
            source_palace=palace,
            target_db="fresh_palace",
            progress=False,
        )


def test_migrate_skips_collision_check_when_target_db_defaulted(tmp_path, monkeypatch):
    """If --target-db is NOT passed, the collision check is skipped.

    The default-derived name already embeds an 8-char sha256 slug of the
    absolute palace path, so a collision would be intentional. We only
    gate the explicit-override path.
    """
    palace = _build_chroma_palace(tmp_path)

    def fake_inspect(_kwargs, _db_name, *, sample_size=3):
        # Should never be called on the default-derivation path.
        raise AssertionError("collision check should not run without --target-db")

    monkeypatch.setattr("mempalace.migrate._inspect_target_collision", fake_inspect)

    class _PastGuard(Exception):
        pass

    def raise_past_guard(**_kwargs):
        raise _PastGuard("past the guard")

    monkeypatch.setattr("mempalace.backends.surreal.SurrealBackend", raise_past_guard)

    with pytest.raises(_PastGuard):
        migrate_to_surreal(
            source_palace=palace,
            # No target_db — derivation owns the name.
            progress=False,
        )
