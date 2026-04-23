"""Backend registry + entry-point discovery (RFC 001 §3).

Third-party backends ship as installable packages that declare a
``mempalace.backends`` entry point::

    # pyproject.toml of mempalace-postgres
    [project.entry-points."mempalace.backends"]
    postgres = "mempalace_postgres:PostgresBackend"

MemPalace discovers them at process start. In-tree tests and local development
can register manually via :func:`register`. Explicit registration wins on
name conflict (matches RFC 001 §3.2).
"""

from __future__ import annotations

import importlib.util
import logging
from importlib import metadata
from threading import Lock
from typing import Optional, Type

from .base import BaseBackend

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "mempalace.backends"

# Backends whose constructor / runtime path requires an optional third-party
# package. Mapped name -> (import_module, extras_hint) so :func:`get_backend`
# can surface a helpful install command before hitting an ImportError deep
# inside the backend's connection path.
_OPTIONAL_BACKEND_DEPS: dict[str, tuple[str, str]] = {
    "surreal": ("surrealdb", 'pip install -e ".[surreal]"'),
}


class MissingBackendDependencyError(ImportError):
    """Raised when the optional package for a backend is not installed.

    Subclasses :class:`ImportError` so callers that already guard against
    missing imports keep working; the message carries the exact
    ``pip install`` incantation the user needs to run.
    """


_registry: dict[str, Type[BaseBackend]] = {}
_instances: dict[str, BaseBackend] = {}
_explicit: set[str] = set()
_discovered = False
_lock = Lock()


def register(name: str, backend_cls: Type[BaseBackend]) -> None:
    """Register ``backend_cls`` under ``name``.

    Explicit registration wins over entry-point discovery on conflict
    (RFC 001 §3.2).
    """
    with _lock:
        _registry[name] = backend_cls
        _explicit.add(name)
        # Invalidate any cached instance so the new class is used on next get.
        _instances.pop(name, None)


def unregister(name: str) -> None:
    """Remove a backend registration (primarily for tests)."""
    with _lock:
        _registry.pop(name, None)
        _explicit.discard(name)
        _instances.pop(name, None)


def _discover_entry_points() -> None:
    """Load entry-point-declared backends once per process."""
    global _discovered
    if _discovered:
        return
    with _lock:
        if _discovered:
            return
        try:
            eps = metadata.entry_points()
            # Py ≥ 3.10 returns an EntryPoints object; older versions returned a dict.
            group = (
                eps.select(group=_ENTRY_POINT_GROUP)
                if hasattr(eps, "select")
                else eps.get(_ENTRY_POINT_GROUP, [])
            )
        except Exception:
            logger.exception("entry-point discovery for %s failed", _ENTRY_POINT_GROUP)
            group = []
        for ep in group:
            if ep.name in _explicit:
                continue  # explicit registration wins
            try:
                cls = ep.load()
            except Exception:
                logger.exception("failed to load backend entry point %r", ep.name)
                continue
            if not isinstance(cls, type) or not issubclass(cls, BaseBackend):
                logger.warning(
                    "entry point %r did not resolve to a BaseBackend subclass (got %r)",
                    ep.name,
                    cls,
                )
                continue
            _registry.setdefault(ep.name, cls)
        _discovered = True


def available_backends() -> list[str]:
    """Return sorted list of all registered backend names."""
    _discover_entry_points()
    return sorted(_registry.keys())


def get_backend_class(name: str) -> Type[BaseBackend]:
    """Return the registered backend class for ``name``."""
    _discover_entry_points()
    try:
        return _registry[name]
    except KeyError as e:
        raise KeyError(f"unknown backend {name!r}; available: {available_backends()}") from e


def _check_optional_dependency(name: str) -> None:
    """Raise :class:`MissingBackendDependencyError` if an optional dep is absent.

    Keeps the error at registry-boundary level so callers get a single clear
    install hint instead of a deep SDK import traceback the first time they
    hit a query path.
    """
    spec = _OPTIONAL_BACKEND_DEPS.get(name)
    if spec is None:
        return
    module, hint = spec
    if importlib.util.find_spec(module) is None:
        raise MissingBackendDependencyError(
            f"backend {name!r} requires the {module!r} package; install it with: {hint}"
        )


def get_backend(name: str) -> BaseBackend:
    """Return a long-lived instance of the named backend.

    Instances are cached per-name; repeated calls return the same object.
    Call :func:`reset_backends` in tests that need isolation.

    Raises :class:`KeyError` for unknown backends and
    :class:`MissingBackendDependencyError` when a backend's optional
    third-party package is not installed.
    """
    _discover_entry_points()
    with _lock:
        inst = _instances.get(name)
        if inst is not None:
            return inst
        cls = _registry.get(name)
        if cls is None:
            raise KeyError(f"unknown backend {name!r}; available: {sorted(_registry.keys())}")
        _check_optional_dependency(name)
        inst = cls()
        _instances[name] = inst
        return inst


def reset_backends() -> None:
    """Close and drop all cached backend instances (primarily for tests)."""
    with _lock:
        for inst in _instances.values():
            try:
                inst.close()
            except Exception:
                logger.exception("error closing backend during reset")
        _instances.clear()


def resolve_backend_for_palace(
    *,
    explicit: Optional[str] = None,
    config_value: Optional[str] = None,
    env_value: Optional[str] = None,
    palace_path: Optional[str] = None,
    default: str = "chroma",
) -> str:
    """Resolve the backend name for a palace per RFC 001 §3.3 priority order.

    1. Explicit kwarg / CLI flag
    2. Per-palace config value
    3. ``MEMPALACE_BACKEND`` env var
    4. Auto-detect from on-disk artifacts (migration/upgrade path only)
    5. Default (``chroma``)

    Auto-detection is strictly a migration aid: it fires only when a local path
    is presented, no earlier rule has chosen a backend, AND the path already
    contains backend-identifiable artifacts. For new palaces, (5) wins.
    """
    for candidate in (explicit, config_value, env_value):
        if candidate:
            return candidate

    _discover_entry_points()
    if palace_path:
        for name, cls in _registry.items():
            try:
                if cls.detect(palace_path):
                    return name
            except Exception:
                logger.exception("detect() raised on backend %r", name)
                continue
    return default


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register the in-tree backends (chroma default, surreal optional).

    The ``surreal`` backend ships in-tree but its runtime depends on the
    ``surrealdb`` package which is an optional extra (``pip install -e
    ".[surreal]"``). The class itself imports lazily, so registering it
    unconditionally is safe — the install check fires in :func:`get_backend`.
    """
    from .chroma import ChromaBackend
    from .surreal import SurrealBackend

    # Use setdefault semantics so a caller that pre-registered for tests wins.
    if "chroma" not in _registry:
        _registry["chroma"] = ChromaBackend
    if "surreal" not in _registry:
        _registry["surreal"] = SurrealBackend


_register_builtins()
