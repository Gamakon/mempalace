"""
conftest.py — Shared fixtures for MemPalace tests.

Provides isolated palace and knowledge graph instances so tests never
touch the user's real data or leak temp files on failure.

HOME is redirected to a temp directory at module load time — before any
mempalace imports — so that module-level initialisations (e.g.
``_kg = KnowledgeGraph()`` in mcp_server) write to a throwaway location
instead of the real user profile.
"""

import os
import random
import shutil
import sys
import tempfile

# ── Isolate HOME before any mempalace imports ──────────────────────────
_original_env = {}
_session_tmp = tempfile.mkdtemp(prefix="mempalace_session_")

for _var in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
    _original_env[_var] = os.environ.get(_var)

os.environ["HOME"] = _session_tmp
os.environ["USERPROFILE"] = _session_tmp
os.environ["HOMEDRIVE"] = os.path.splitdrive(_session_tmp)[0] or "C:"
os.environ["HOMEPATH"] = os.path.splitdrive(_session_tmp)[1] or _session_tmp

# Now it is safe to import mempalace modules that trigger initialisation.
import chromadb  # noqa: E402
import pytest  # noqa: E402

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.knowledge_graph import KnowledgeGraph  # noqa: E402


# ── Chaos-seed CLI option (mp-ou9) ─────────────────────────────────────
#
# Tests that introduce deliberate randomness (e.g. the SIGKILL timing in
# ``tests/test_surreal_multiprocess.py``) must be reproducible. A
# `--chaos-seed` CLI option lets a developer replay a flaky CI run by
# re-invoking pytest with the exact seed printed in the original log.
#
# When not supplied, a fresh random seed is generated for each session
# and printed to stdout so CI logs always capture it.


def pytest_addoption(parser):
    """Register the --chaos-seed option for reproducible chaos tests."""
    parser.addoption(
        "--chaos-seed",
        action="store",
        default=None,
        type=int,
        help=(
            "Seed for randomised chaos tests (e.g. SIGKILL timing in "
            "test_surreal_multiprocess). Default: fresh random int, "
            "printed at session start so CI logs capture it."
        ),
    )


def pytest_configure(config):
    """Resolve the chaos seed once per session and print it to stdout."""
    seed = config.getoption("--chaos-seed")
    if seed is None:
        # 63-bit positive int — wide enough to avoid collisions, narrow
        # enough to copy-paste from a CI log without scientific notation.
        seed = random.SystemRandom().randrange(1, 2**63)
    config._chaos_seed = int(seed)
    # Print unconditionally so both passing and failing runs capture it.
    # Use sys.stdout.write (not print) so the message survives even when
    # pytest is configured with -s/--capture=no restrictions.
    sys.stdout.write(f"\n[chaos-seed] using seed: {config._chaos_seed}\n")
    sys.stdout.flush()


@pytest.fixture(scope="session")
def chaos_seed(request) -> int:
    """Return the session-wide chaos seed resolved in pytest_configure."""
    return int(request.config._chaos_seed)


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    """Reset the MCP server's cached backend state between tests.

    Chroma: clears the cached ``PersistentClient`` / collection so the next
    test picks up its own ``palace_path``.

    Surreal: drops the server-side database (via ``drop_palace``) and clears
    the cached ``PalaceRef`` so two tests with distinct tmp palace paths
    never share a Surreal DB. The long-lived ``SurrealBackend`` itself is
    retained — constructing a fresh one per test would re-open the
    websocket each time and slow the suite significantly (mp-1y1).
    """

    def _clear_cache():
        try:
            from mempalace import mcp_server

            # Chroma-side caches
            mcp_server._client_cache = None
            mcp_server._collection_cache = None

            # Surreal-side caches: drop the previously used palace from the
            # server so leftover drawer/diary rows cannot leak into the
            # next test, then forget the cached ref so it is re-derived
            # from the next test's ``_config.palace_path``.
            backend = getattr(mcp_server, "_surreal_backend", None)
            ref = getattr(mcp_server, "_surreal_palace_ref", None)
            if backend is not None and ref is not None:
                try:
                    backend.drop_palace(ref)
                except Exception:
                    pass
            mcp_server._surreal_palace_ref = None
            mcp_server._surreal_palace_ref_path = None
        except (ImportError, AttributeError):
            pass

    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture(scope="session", autouse=True)
def _isolate_home():
    """Ensure HOME points to a temp dir for the entire test session.

    The env vars were already set at module level (above) so that
    module-level initialisations are captured.  This fixture simply
    restores the originals on teardown and cleans up the temp dir.
    """
    yield
    for var, orig in _original_env.items():
        if orig is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = orig
    shutil.rmtree(_session_tmp, ignore_errors=True)


@pytest.fixture
def tmp_dir():
    """Create and auto-cleanup a temporary directory."""
    d = tempfile.mkdtemp(prefix="mempalace_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def palace_path(tmp_dir):
    """Path to an empty palace directory inside tmp_dir."""
    p = os.path.join(tmp_dir, "palace")
    os.makedirs(p)
    return p


@pytest.fixture
def config(tmp_dir, palace_path):
    """A MempalaceConfig pointing at the temp palace."""
    cfg_dir = os.path.join(tmp_dir, "config")
    os.makedirs(cfg_dir)
    import json

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump({"palace_path": palace_path}, f)
    return MempalaceConfig(config_dir=cfg_dir)


@pytest.fixture
def collection(palace_path):
    """A ChromaDB collection pre-seeded in the temp palace."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    yield col
    client.delete_collection("mempalace_drawers")
    del client


# Sample drawers reused by the backend-agnostic ``seeded_collection`` fixture.
# Kept at module scope so both the Chroma and Surreal paths seed the *same*
# documents/metadata — otherwise a test that passes against Chroma might fail
# against Surreal purely because the inputs drifted (mp-om7).
_SEED_IDS = [
    "drawer_proj_backend_aaa",
    "drawer_proj_backend_bbb",
    "drawer_proj_frontend_ccc",
    "drawer_notes_planning_ddd",
]
_SEED_DOCUMENTS = [
    "The authentication module uses JWT tokens for session management. "
    "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies.",
    "Database migrations are handled by Alembic. We use PostgreSQL 15 "
    "with connection pooling via pgbouncer.",
    "The React frontend uses TanStack Query for server state management. "
    "All API calls go through a centralized fetch wrapper.",
    "Sprint planning: migrate auth to passkeys by Q3. "
    "Evaluate ChromaDB alternatives for vector search.",
]
_SEED_METADATAS = [
    {
        "wing": "project",
        "room": "backend",
        "source_file": "auth.py",
        "chunk_index": 0,
        "added_by": "miner",
        "filed_at": "2026-01-01T00:00:00",
    },
    {
        "wing": "project",
        "room": "backend",
        "source_file": "db.py",
        "chunk_index": 0,
        "added_by": "miner",
        "filed_at": "2026-01-02T00:00:00",
    },
    {
        "wing": "project",
        "room": "frontend",
        "source_file": "App.tsx",
        "chunk_index": 0,
        "added_by": "miner",
        "filed_at": "2026-01-03T00:00:00",
    },
    {
        "wing": "notes",
        "room": "planning",
        "source_file": "sprint.md",
        "chunk_index": 0,
        "added_by": "miner",
        "filed_at": "2026-01-04T00:00:00",
    },
]


@pytest.fixture
def seeded_collection(config, palace_path, collection, monkeypatch):
    """A drawer collection pre-seeded on the currently-configured backend (mp-om7).

    Honours ``MempalaceConfig().backend`` so the same tests exercise either
    backend by setting ``MEMPALACE_BACKEND=surreal`` (unset = Chroma default).
    Yields an object implementing :class:`mempalace.backends.base.BaseCollection`;
    tests should rely only on that abstract surface, not on backend-specific
    extras.

    Seeding invariants (identical on both backends):

    * Four drawers with deterministic IDs (``drawer_proj_backend_aaa`` etc.)
    * Metadata carrying ``wing``, ``room``, ``source_file``, ``chunk_index``,
      ``added_by``, ``filed_at``.
    * Documents stored verbatim — the MemPalace invariant.

    The parallel ``collection`` fixture is consumed (even under Surreal) and
    seeded with the same rows. Some Chroma-only call sites — notably
    ``searcher.search_memories`` (used by ``test_searcher.py``) hard-wire
    :class:`ChromaBackend` regardless of ``_config.backend`` — still need a
    populated Chroma palace. Seeding both backends side-by-side keeps them
    working without adding a compatibility shim at the searcher layer.

    Teardown:

    * Chroma — the ``collection`` fixture deletes its collection on exit.
    * Surreal — ``backend.drop_palace`` issues ``REMOVE DATABASE`` so no
      rows leak into the next test. The outer ``_reset_mcp_cache`` autouse
      fixture issues a second ``drop_palace`` as belt-and-braces (mp-1y1).
    """
    # Always populate Chroma at ``palace_path`` — test_searcher.py and any
    # other Chroma-pinned caller need this even under Surreal.
    collection.add(
        ids=list(_SEED_IDS),
        documents=list(_SEED_DOCUMENTS),
        metadatas=[dict(m) for m in _SEED_METADATAS],
    )

    if config.backend != "surreal":
        # Chroma path: return the ChromaCollection adapter so the yielded
        # value satisfies :class:`BaseCollection` and tests don't depend on
        # the raw chromadb API.
        from mempalace.backends.chroma import ChromaCollection

        yield ChromaCollection(collection)
        return

    # Surreal path: build/attach a backend targeting the same palace the MCP
    # server will see, seed drawers on it, and yield its collection handle.
    import hashlib

    from mempalace import mcp_server
    from mempalace.backends.base import PalaceRef
    from mempalace.backends.surreal import SurrealBackend, _embed_texts

    backend = getattr(mcp_server, "_surreal_backend", None)
    if backend is None:
        backend = SurrealBackend(
            url=os.environ.get("MEMPALACE_SURREAL_URL", "ws://127.0.0.1:8000"),
            username=os.environ.get("MEMPALACE_SURREAL_USER", "root"),
            password=os.environ.get("MEMPALACE_SURREAL_PASS", "root"),
        )
        monkeypatch.setattr(mcp_server, "_surreal_backend", backend)

    # Must match ``mcp_server._get_surreal_backend``'s derivation exactly —
    # a different prefix would point the fixture at a different Surreal DB
    # than the one MCP tool handlers read, and the seed would be invisible
    # to the tests (mp-1y1).
    palace_id = "mcp_" + hashlib.sha256(palace_path.encode()).hexdigest()[:16]
    palace_ref = PalaceRef(id=palace_id, local_path=palace_path)

    # Prime the MCP server caches so ``_get_collection()`` — called by every
    # tool under test — lands on the same palace we seed here.
    monkeypatch.setattr(mcp_server, "_surreal_palace_ref", palace_ref)
    monkeypatch.setattr(mcp_server, "_surreal_palace_ref_path", palace_path)
    monkeypatch.setattr(mcp_server, "_collection_cache", None)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None)
    monkeypatch.setattr(mcp_server, "_metadata_cache_time", 0)

    # Start from a clean DB. ``drop_palace`` is idempotent so this is cheap
    # insurance against a prior run's leftover tables.
    try:
        backend.drop_palace(palace_ref)
    except Exception:
        pass

    col = backend.get_collection(
        palace=palace_ref,
        collection_name=config.collection_name,
        create=True,
    )

    # Surreal's HNSW index is dim-locked on first write, and the collection
    # does not auto-embed documents (Chroma does). Pre-compute vectors with
    # the same embedder the MCP server uses so vector search at query time
    # walks the same embedding space.
    embeddings = _embed_texts(list(_SEED_DOCUMENTS))
    col.add(
        ids=list(_SEED_IDS),
        documents=list(_SEED_DOCUMENTS),
        metadatas=[dict(m) for m in _SEED_METADATAS],
        embeddings=embeddings,
    )
    try:
        yield col
    finally:
        try:
            backend.drop_palace(palace_ref)
        except Exception:
            pass


@pytest.fixture
def kg(tmp_dir):
    """An isolated KnowledgeGraph using a temp SQLite file."""
    db_path = os.path.join(tmp_dir, "test_kg.sqlite3")
    graph = KnowledgeGraph(db_path=db_path)
    yield graph
    graph.close()


@pytest.fixture
def seeded_kg(kg):
    """KnowledgeGraph pre-loaded with sample triples."""
    kg.add_entity("Alice", entity_type="person")
    kg.add_entity("Max", entity_type="person")
    kg.add_entity("swimming", entity_type="activity")
    kg.add_entity("chess", entity_type="activity")

    kg.add_triple("Alice", "parent_of", "Max", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "does", "chess", valid_from="2024-06-01")
    kg.add_triple("Alice", "works_at", "Acme Corp", valid_from="2020-01-01", valid_to="2024-12-31")
    kg.add_triple("Alice", "works_at", "NewCo", valid_from="2025-01-01")

    return kg
