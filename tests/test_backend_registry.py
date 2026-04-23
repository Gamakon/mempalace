"""Tests for the backend registry + ``MEMPALACE_BACKEND`` config wiring (mp-2i2).

Covers the three pieces a user needs to pick their storage backend:

* :func:`mempalace.backends.get_backend` returns a live instance for each
  in-tree backend (``chroma`` + ``surreal``).
* Unknown names raise a clear :class:`KeyError` that lists available
  options — no silent fallback.
* Missing optional package (``surrealdb``) surfaces as a
  :class:`MissingBackendDependencyError` with an exact ``pip install``
  hint rather than a deep SDK ImportError.
* :class:`mempalace.config.MempalaceConfig` reads ``MEMPALACE_BACKEND``
  env var, then ``backend`` field in ``config.json``, then default.
"""

from __future__ import annotations

import importlib
import json

import pytest

from mempalace.backends import (
    ChromaBackend,
    MissingBackendDependencyError,
    SurrealBackend,
    available_backends,
    get_backend,
    reset_backends,
)
from mempalace.config import DEFAULT_BACKEND, MempalaceConfig


@pytest.fixture(autouse=True)
def _clean_backend_cache():
    """Reset the per-name backend instance cache between tests.

    Without this, ``SurrealBackend`` instantiated in one test would be
    returned by ``get_backend("surreal")`` in the next test even after we
    simulate uninstalling ``surrealdb``, because the cache short-circuits
    the dependency check.
    """
    reset_backends()
    yield
    reset_backends()


# ---------------------------------------------------------------------------
# Registry: lookups
# ---------------------------------------------------------------------------


def test_available_backends_lists_both_builtins():
    names = available_backends()
    assert "chroma" in names
    assert "surreal" in names


def test_get_backend_chroma_returns_chroma_instance():
    backend = get_backend("chroma")
    assert isinstance(backend, ChromaBackend)
    assert backend.name == "chroma"


def test_get_backend_surreal_returns_surreal_instance():
    backend = get_backend("surreal")
    assert isinstance(backend, SurrealBackend)
    assert backend.name == "surreal"


def test_get_backend_unknown_name_raises_keyerror():
    with pytest.raises(KeyError) as exc_info:
        get_backend("definitely-not-a-real-backend")
    # The error should list the supported options so the user can self-correct.
    assert "chroma" in str(exc_info.value)
    assert "surreal" in str(exc_info.value)


def test_get_backend_caches_instances():
    a = get_backend("chroma")
    b = get_backend("chroma")
    assert a is b


# ---------------------------------------------------------------------------
# Registry: missing optional dep
# ---------------------------------------------------------------------------


def test_missing_surrealdb_package_raises_with_install_hint(monkeypatch):
    """Simulate ``surrealdb`` not being installed.

    We patch ``importlib.util.find_spec`` so ``_check_optional_dependency``
    sees the package as absent. The raised error must point at the extras
    install command rather than letting an ImportError bubble up from
    ``surrealdb`` somewhere deeper.
    """
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "surrealdb":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(MissingBackendDependencyError) as exc_info:
        get_backend("surreal")
    msg = str(exc_info.value)
    assert "surrealdb" in msg
    assert 'pip install -e ".[surreal]"' in msg


def test_missing_dep_error_is_importerror_subclass():
    """Callers that already guard on ImportError should keep working."""
    assert issubclass(MissingBackendDependencyError, ImportError)


# ---------------------------------------------------------------------------
# Config: MEMPALACE_BACKEND env var + config.json override
# ---------------------------------------------------------------------------


def test_config_default_backend_is_chroma(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.backend == DEFAULT_BACKEND == "chroma"


def test_config_reads_backend_from_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"backend": "surreal"}))
    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.backend == "surreal"


def test_config_env_var_overrides_file(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"backend": "chroma"}))
    monkeypatch.setenv("MEMPALACE_BACKEND", "surreal")
    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.backend == "surreal"


def test_config_env_var_is_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "SURREAL")
    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.backend == "surreal"


def test_config_rejects_unknown_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "qdrant")
    cfg = MempalaceConfig(config_dir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        _ = cfg.backend
    assert "qdrant" in str(exc_info.value)
    assert "chroma" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Surreal backend: env-driven connection defaults
# ---------------------------------------------------------------------------


def test_surreal_backend_respects_connection_env_vars(monkeypatch):
    """The Surreal backend's default constructor picks up ``MEMPALACE_SURREAL_*``.

    Those env vars are consulted at module-import time, so we reload the
    module inside the patched environment, verify the new defaults, and
    restore the original module constants (while env is still patched)
    before letting the fixture tear down. If we reimported *after*
    monkeypatch unsets the env vars instead, the module's module-level
    ``DEFAULT_URL = os.environ.get(...)`` would bake in the patched values
    and leak into the rest of the test session.
    """
    import mempalace.backends.surreal as surreal_mod

    orig_url = surreal_mod.DEFAULT_URL
    orig_user = surreal_mod.DEFAULT_USER
    orig_pass = surreal_mod.DEFAULT_PASS

    monkeypatch.setenv("MEMPALACE_SURREAL_URL", "http://example.test:9000")
    monkeypatch.setenv("MEMPALACE_SURREAL_USER", "alice")
    monkeypatch.setenv("MEMPALACE_SURREAL_PASS", "s3cret")

    try:
        reloaded = importlib.reload(surreal_mod)

        assert reloaded.DEFAULT_URL == "http://example.test:9000"
        assert reloaded.DEFAULT_USER == "alice"
        assert reloaded.DEFAULT_PASS == "s3cret"

        backend = reloaded.SurrealBackend()
        assert backend._url == "http://example.test:9000"
        assert backend._username == "alice"
        assert backend._password == "s3cret"
    finally:
        # Restore the pre-test env vars *before* reloading, so the module's
        # import-time reads land on the pristine values regardless of when
        # monkeypatch's own teardown runs.
        monkeypatch.setenv("MEMPALACE_SURREAL_URL", orig_url)
        monkeypatch.setenv("MEMPALACE_SURREAL_USER", orig_user)
        monkeypatch.setenv("MEMPALACE_SURREAL_PASS", orig_pass)
        importlib.reload(surreal_mod)
        # The reload replaced ``SurrealBackend`` with a new class object;
        # point the registry at the fresh class so later tests that
        # ``isinstance(x, SurrealBackend)`` check against the live module
        # keep passing.
        from mempalace.backends import register as _register

        _register("surreal", surreal_mod.SurrealBackend)
