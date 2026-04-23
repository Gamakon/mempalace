#!/usr/bin/env python3
"""
mempalace migrate — Recover a palace created with a different ChromaDB version.

Reads documents and metadata directly from the palace's SQLite database
(bypassing ChromaDB's API, which fails on version-mismatched palaces),
then re-imports everything into a fresh palace using the currently installed
ChromaDB version.

Since mempalace 3.2.0 (chromadb>=1.5.4), chromadb automatically migrates
0.4.1+ databases on first open — no manual migration needed for upgrades.
Use this command only when downgrading chromadb (e.g. rolling back to an
older mempalace release) or if automatic migration fails.

Usage:
    mempalace migrate                          # migrate default palace
    mempalace migrate --palace /path/to/palace  # migrate specific palace
    mempalace migrate --dry-run                # show what would be migrated

Also houses :func:`migrate_to_surreal` (mp-ciw): a one-shot, idempotent
drawer + embedding transfer from an existing ChromaDB palace into
SurrealDB. See its docstring for the CLI wiring.
"""

import hashlib
import os
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional


def extract_drawers_from_sqlite(db_path: str) -> list:
    """Read all drawers directly from ChromaDB's SQLite, bypassing the API.

    Works regardless of which ChromaDB version created the database.
    Returns list of dicts with 'id', 'document', and 'metadata' keys.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get all embedding IDs and their documents
    rows = conn.execute(
        """
        SELECT e.embedding_id,
               MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END) as document
        FROM embeddings e
        JOIN embedding_metadata em ON em.id = e.id
        GROUP BY e.embedding_id
    """
    ).fetchall()

    drawers = []
    for row in rows:
        embedding_id = row["embedding_id"]
        document = row["document"]
        if not document:
            continue

        # Get metadata for this embedding
        meta_rows = conn.execute(
            """
            SELECT em.key, em.string_value, em.int_value, em.float_value, em.bool_value
            FROM embedding_metadata em
            JOIN embeddings e ON e.id = em.id
            WHERE e.embedding_id = ?
              AND em.key NOT LIKE 'chroma:%'
        """,
            (embedding_id,),
        ).fetchall()

        metadata = {}
        for mr in meta_rows:
            key = mr["key"]
            if mr["string_value"] is not None:
                metadata[key] = mr["string_value"]
            elif mr["int_value"] is not None:
                metadata[key] = mr["int_value"]
            elif mr["float_value"] is not None:
                metadata[key] = mr["float_value"]
            elif mr["bool_value"] is not None:
                metadata[key] = bool(mr["bool_value"])

        drawers.append(
            {
                "id": embedding_id,
                "document": document,
                "metadata": metadata,
            }
        )

    conn.close()
    return drawers


def detect_chromadb_version(db_path: str) -> str:
    """Detect which ChromaDB version created the database by checking schema."""
    conn = sqlite3.connect(db_path)
    try:
        # 1.x has schema_str column in collections table
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collections)").fetchall()]
        if "schema_str" in cols:
            return "1.x"
        # 0.6.x has embeddings_queue but no schema_str
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        if "embeddings_queue" in tables:
            return "0.6.x"
        return "unknown"
    finally:
        conn.close()


def contains_palace_database(path: str) -> bool:
    """Return True when path looks like a MemPalace ChromaDB directory."""
    return os.path.isfile(os.path.join(path, "chroma.sqlite3"))


def confirm_destructive_action(
    operation_name: str, palace_path: str, assume_yes: bool = False
) -> bool:
    """Require confirmation before destructive palace operations."""
    if assume_yes:
        return True

    print(f"\n  {operation_name} will replace data in: {palace_path}")
    print("  A backup will be created first, then the palace will be rebuilt.")
    try:
        answer = input("  Continue? [y/N]: ").strip().lower()
    except EOFError:
        print("  Aborted. Re-run with --yes to confirm destructive changes.")
        return False

    if answer not in {"y", "yes"}:
        print("  Aborted.")
        return False
    return True


def migrate(palace_path: str, dry_run: bool = False, confirm: bool = False):
    """Migrate a palace to the currently installed ChromaDB version."""
    from .backends.chroma import ChromaBackend

    palace_path = os.path.abspath(os.path.expanduser(palace_path))
    db_path = os.path.join(palace_path, "chroma.sqlite3")

    if not os.path.isdir(palace_path) or not contains_palace_database(palace_path):
        print(f"\n  No palace database found at {db_path}")
        return False

    print(f"\n{'=' * 60}")
    print("  MemPalace Migrate")
    print(f"{'=' * 60}\n")
    print(f"  Palace:    {palace_path}")
    print(f"  Database:  {db_path}")
    print(f"  DB size:   {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")

    # Detect version
    source_version = detect_chromadb_version(db_path)
    target_version = ChromaBackend.backend_version()
    print(f"  Source:    ChromaDB {source_version}")
    print(f"  Target:    ChromaDB {target_version}")

    # Try reading with current chromadb first
    try:
        col = ChromaBackend().get_collection(palace_path, "mempalace_drawers")
        count = col.count()
        print(f"\n  Palace is already readable by chromadb {target_version}.")
        print(f"  {count} drawers found. No migration needed.")
        return True
    except Exception:
        print(f"\n  Palace is NOT readable by chromadb {target_version}.")
        print("  Extracting from SQLite directly...")

    # Extract all drawers via raw SQL
    drawers = extract_drawers_from_sqlite(db_path)
    print(f"  Extracted {len(drawers)} drawers from SQLite")

    if not drawers:
        print("  Nothing to migrate.")
        return True

    # Show summary
    wings = defaultdict(lambda: defaultdict(int))
    for d in drawers:
        w = d["metadata"].get("wing", "?")
        r = d["metadata"].get("room", "?")
        wings[w][r] += 1

    print("\n  Summary:")
    for wing, rooms in sorted(wings.items()):
        total = sum(rooms.values())
        print(f"    WING: {wing} ({total} drawers)")
        for room, count in sorted(rooms.items(), key=lambda x: -x[1]):
            print(f"      ROOM: {room:30} {count:5}")

    if dry_run:
        print("\n  DRY RUN — no changes made.")
        print(f"  Would migrate {len(drawers)} drawers.")
        return True

    if not confirm_destructive_action("Migration", palace_path, assume_yes=confirm):
        return False

    # Backup the old palace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{palace_path}.pre-migrate.{timestamp}"
    print(f"\n  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    # Build fresh palace in a temp directory (avoids chromadb reading old state)
    import tempfile

    temp_palace = tempfile.mkdtemp(prefix="mempalace_migrate_")
    print(f"  Creating fresh palace in {temp_palace}...")
    fresh_backend = ChromaBackend()
    col = fresh_backend.get_or_create_collection(temp_palace, "mempalace_drawers")

    # Re-import in batches
    batch_size = 500
    imported = 0
    for i in range(0, len(drawers), batch_size):
        batch = drawers[i : i + batch_size]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        imported += len(batch)
        print(f"  Imported {imported}/{len(drawers)} drawers...")

    # Verify before swapping
    final_count = col.count()
    del col
    del fresh_backend

    # Swap: remove old palace, move new one into place
    print("  Swapping old palace for migrated version...")
    shutil.rmtree(palace_path)
    shutil.move(temp_palace, palace_path)

    print("\n  Migration complete.")
    print(f"  Drawers migrated: {final_count}")
    print(f"  Backup at: {backup_path}")

    if final_count != len(drawers):
        print(f"  WARNING: Expected {len(drawers)}, got {final_count}")

    print(f"\n{'=' * 60}\n")
    return True


# ---------------------------------------------------------------------------
# mp-ciw: Chroma -> SurrealDB drawer + embedding migration
# ---------------------------------------------------------------------------

_DB_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _derive_surreal_db_name(palace_path: str) -> str:
    """Derive a collision-resistant Surreal DB name from a palace path.

    Surreal ``USE DB`` identifiers are restricted to ``[A-Za-z0-9_]``. We
    want two properties from a default-derived name:

    1. **Human-readable** — a glance at the Surreal DB list tells the
       operator which palace it corresponds to. So we keep the palace
       directory basename as a prefix.
    2. **Collision-resistant** — two palaces at different paths but with
       the same basename (e.g. ``/home/a/mem`` vs ``/home/b/mem``) must
       NOT silently share a target DB. Upsert semantics would merge the
       two sets of drawers with no warning. mp-2v9 fixes that by
       appending an 8-char sha256 slug of the absolute path.

    Layout: ``<cleaned-basename>_<sha8>`` where ``sha8`` is the first 8
    hex chars of ``sha256(abspath(palace_path))``. If the cleaned
    basename is empty (e.g. ``/``) we fall back to ``palace`` as the
    prefix — the hash still disambiguates.

    The full absolute path (not just the passed-in string) is hashed so
    ``foo/../mem`` and ``mem`` resolve to the same DB.
    """
    abspath = os.path.abspath(os.path.expanduser(palace_path))
    base = os.path.basename(os.path.normpath(abspath)) or "palace"
    cleaned = _DB_NAME_SAFE_RE.sub("_", base).strip("_") or "palace"
    path_hash = hashlib.sha256(abspath.encode()).hexdigest()[:8]
    return f"{cleaned}_{path_hash}"


# The ChromaDB miner/sweeper produces two collections per palace:
#   * ``mempalace_drawers`` — verbatim chunks, primary payload.
#   * ``mempalace_closets`` — AAAK compressed pointers.
# Both are drawer-shaped on the Surreal side (`drawer` and `closet` tables).
# KG triples live in a separate SQLite store (``knowledge_graph.py``) and
# are out of scope for this migration (mp-3lc handles them).
_COLLECTION_MAP = (
    ("mempalace_drawers", "drawer"),
    ("mempalace_closets", "closet"),
)


def _open_chroma_readonly(palace_path: str):
    """Return a ChromaBackend for read-only use against ``palace_path``.

    ChromaDB's PersistentClient has no explicit read-only mode, so we treat
    the backend as read-only by discipline: the migration pipeline only
    calls ``count()`` and ``get(...)`` on the returned backend's
    collections — never ``add``/``upsert``/``update``/``delete``. The
    palace directory is not chmod'd; callers are responsible for honoring
    the read-only contract.
    """
    from .backends.chroma import ChromaBackend

    return ChromaBackend()


def _iter_chroma_batches(col, batch_size: int):
    """Yield batches of ``(ids, documents, metadatas, embeddings)`` from a
    Chroma collection using offset-paginated ``get(...)`` calls.

    ``include=["documents", "metadatas", "embeddings"]`` is explicit — the
    default omits embeddings, and omitting them here would silently drop
    the vector payload we are supposed to migrate.
    """
    offset = 0
    total = col.count()
    while offset < total:
        res = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = list(res.get("ids") or [])
        if not ids:
            break
        docs = list(res.get("documents") or [])
        metas = list(res.get("metadatas") or [])
        embeds = res.get("embeddings")
        # Pad to match ids length (Chroma pads too, but defensive).
        if len(docs) < len(ids):
            docs += [""] * (len(ids) - len(docs))
        if len(metas) < len(ids):
            metas += [{}] * (len(ids) - len(metas))
        embed_list: Optional[list]
        if embeds is None:
            embed_list = None
        else:
            embed_list = [list(e) if e is not None else None for e in embeds]
        yield ids, docs, metas, embed_list
        offset += len(ids)


def _normalize_metadata_for_surreal(meta: Optional[dict]) -> dict:
    """Coerce a Chroma metadata dict into a Surreal-safe dict.

    * ``None`` becomes ``{}`` (Surreal's ``metadata`` field is optional but
      downstream code prefers a dict).
    * Values are copied as-is; Chroma only stores str/int/float/bool so the
      types round-trip cleanly through the Surreal ``FLEXIBLE object``.
    """
    if not meta:
        return {}
    return {str(k): v for k, v in meta.items()}


def _sample_ids(ids: list, n: int) -> list:
    """Deterministic sample of up to ``n`` ids, evenly spaced."""
    if not ids:
        return []
    if len(ids) <= n:
        return list(ids)
    step = max(1, len(ids) // n)
    out = [ids[i] for i in range(0, len(ids), step)][:n]
    # Always include first + last so we exercise boundary rows too.
    if ids[0] not in out:
        out[0] = ids[0]
    if ids[-1] not in out:
        out[-1] = ids[-1]
    return out


def _verify_migration(
    chroma_col,
    surreal_col,
    *,
    label: str,
    sample_size: int = 10,
) -> tuple[bool, list[str]]:
    """Verify a migrated collection.

    * Counts: Surreal >= Chroma.
    * Spot-check: ``sample_size`` drawers fetched from Chroma must exist
      in Surreal with matching document, metadata, and equal embedding
      dim (not equal vector — float-float equality is brittle, and the
      dim is the load-bearing contract).

    Returns ``(ok, errors)``. ``errors`` is always a list so the caller
    can print the first few for triage.
    """
    errors: list[str] = []
    chroma_count = chroma_col.count()
    surreal_count = surreal_col.count()
    if surreal_count < chroma_count:
        errors.append(f"{label}: surreal count {surreal_count} < chroma count {chroma_count}")

    # Pull a stable id list to sample from. We fetch ids only (not vectors)
    # so this is cheap even on large palaces.
    id_page = chroma_col.get(
        limit=max(sample_size * 4, 40),
        include=[],
    )
    source_ids = list(id_page.get("ids") or [])
    sample = _sample_ids(source_ids, sample_size)
    if not sample:
        return (len(errors) == 0, errors)

    src = chroma_col.get(
        ids=sample,
        include=["documents", "metadatas", "embeddings"],
    )
    dst = surreal_col.get(
        ids=sample,
        include=["documents", "metadatas", "embeddings"],
    )
    src_by_id = {
        rid: (
            src.documents[i] if i < len(src.documents) else "",
            src.metadatas[i] if i < len(src.metadatas) else {},
            (src.embeddings or [None] * len(sample))[i] if src.embeddings is not None else None,
        )
        for i, rid in enumerate(src.ids)
    }
    dst_by_id = {
        rid: (
            dst.documents[i] if i < len(dst.documents) else "",
            dst.metadatas[i] if i < len(dst.metadatas) else {},
            (dst.embeddings or [None] * len(dst.ids))[i] if dst.embeddings is not None else None,
        )
        for i, rid in enumerate(dst.ids)
    }
    for rid in sample:
        if rid not in dst_by_id:
            errors.append(f"{label}: id {rid!r} missing from surreal")
            continue
        s_doc, s_meta, s_emb = src_by_id.get(rid, ("", {}, None))
        d_doc, d_meta, d_emb = dst_by_id[rid]
        if s_doc != d_doc:
            errors.append(f"{label}: id {rid!r} document mismatch")
        # Metadata comparison: Surreal may have added a ``filed_at`` or
        # similar system field that Chroma never carried. Compare only
        # the keys the source knows about.
        norm_src = _normalize_metadata_for_surreal(s_meta)
        for k, v in norm_src.items():
            if d_meta.get(k) != v:
                errors.append(
                    f"{label}: id {rid!r} metadata[{k!r}] source={v!r} dest={d_meta.get(k)!r}"
                )
                break
        # Embedding dim check (not element equality — float jitter).
        src_dim = len(s_emb) if s_emb is not None else 0
        dst_dim = len(d_emb) if d_emb is not None else 0
        if src_dim != dst_dim:
            errors.append(f"{label}: id {rid!r} embedding dim source={src_dim} dest={dst_dim}")
    return (len(errors) == 0, errors)


class TargetCollisionError(RuntimeError):
    """Raised when the target Surreal DB already has drawers and the caller
    did not pass ``allow_merge=True`` (CLI: ``--allow-merge``).

    mp-2v9: merging two palaces into the same Surreal DB via upsert is a
    silent data-corruption risk — two drawers that happen to share an id
    (same chunk hash, same chunk_index in a different palace) would
    collide and one would overwrite the other. Refusing by default is
    the safe behaviour; the caller must opt-in explicitly.
    """


def _inspect_target_collision(
    surreal_kwargs: dict,
    db_name: str,
    *,
    sample_size: int = 3,
) -> tuple[int, list[str], dict[str, int]]:
    """Return ``(total_existing, sampled_ids, per_collection_counts)``.

    Opens the target Surreal DB with ``create=False`` for each known
    collection; a :class:`PalaceNotFoundError` (nothing there yet) is
    treated as an empty target and returns ``(0, [], {})``. Any other
    error is surfaced so the caller can fail loudly rather than silently
    proceeding into an unknown state.

    Deliberately side-effect-free: we never bootstrap or create anything
    during the collision check, even if the DB is present but empty.
    """
    from .backends.base import PalaceNotFoundError, PalaceRef
    from .backends.surreal import SurrealBackend

    backend = SurrealBackend(**surreal_kwargs)
    palace_ref = PalaceRef(id=db_name, local_path=None, namespace=db_name)
    per_counts: dict[str, int] = {}
    sampled_ids: list[str] = []
    total = 0
    try:
        for src_name, _dst in _COLLECTION_MAP:
            try:
                col = backend.get_collection(
                    palace=palace_ref,
                    collection_name=src_name,
                    create=False,
                )
            except PalaceNotFoundError:
                # Palace doesn't exist at all — definitely empty.
                return (0, [], {})
            except Exception as e:
                # Any other failure: a FileNotFoundError wrapper around a
                # missing db, a transient surreal hiccup, etc. Treat as
                # "not present" only when the error string clearly says
                # so; otherwise re-raise so the caller sees the real bug.
                msg = str(e).lower()
                if "does not exist" in msg or "not found" in msg or "no such" in msg:
                    return (0, [], {})
                raise
            cnt = col.count()
            per_counts[src_name] = cnt
            total += cnt
            if cnt and len(sampled_ids) < sample_size:
                page = col.get(limit=max(1, sample_size - len(sampled_ids)), include=[])
                sampled_ids.extend(list(page.get("ids") or []))
    finally:
        try:
            backend.close()
        except Exception:
            pass
    return total, sampled_ids, per_counts


def _resolve_surreal_connection(
    surreal_url: Optional[str],
    surreal_user: Optional[str],
    surreal_pass: Optional[str],
    namespace: str,
) -> dict:
    """Build kwargs for :class:`SurrealBackend` with env-var fallback.

    Explicit kwargs win over env vars, env vars win over the backend
    defaults. Factored out so :func:`migrate_to_surreal` stays under the
    complexity cap.
    """
    from .backends.surreal import DEFAULT_PASS, DEFAULT_URL, DEFAULT_USER

    return {
        "namespace": namespace,
        "url": (surreal_url or os.environ.get("MEMPALACE_SURREAL_URL") or DEFAULT_URL),
        "username": (surreal_user or os.environ.get("MEMPALACE_SURREAL_USER") or DEFAULT_USER),
        "password": (surreal_pass or os.environ.get("MEMPALACE_SURREAL_PASS") or DEFAULT_PASS),
    }


def _open_source_collections(
    chroma_backend,
    source_palace: str,
    progress: bool,
) -> tuple[dict[str, object], dict[str, int]]:
    """Open every known Chroma collection and snapshot its count.

    Counts are taken up front so progress output is meaningful ("migrated
    400/12345") and the caller sees total work in the summary line. A
    missing collection (e.g. legacy palace with no closets) is silently
    skipped rather than failing the migration.
    """
    source_counts: dict[str, int] = {}
    source_collections: dict[str, object] = {}
    for src_name, _dst_table in _COLLECTION_MAP:
        try:
            col = chroma_backend.get_collection(source_palace, src_name, create=False)
        except FileNotFoundError:
            continue
        except Exception as e:
            # Legacy palaces without the closets collection raise on get.
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                continue
            raise
        cnt = col.count()
        source_collections[src_name] = col
        source_counts[src_name] = cnt
        if progress:
            print(f"  {src_name}: {cnt} drawers in source")
    return source_collections, source_counts


def _migrate_one_collection(
    *,
    src_name: str,
    dst_table: str,
    src_col,
    src_count: int,
    surreal_backend,
    palace_ref,
    batch_size: int,
    progress: bool,
) -> tuple[int, bool, list[str]]:
    """Run the copy + verify cycle for a single Chroma collection.

    Returns ``(migrated_count, verified_ok, errors)``. Raises on write
    failure so :func:`migrate_to_surreal` can short-circuit the loop and
    surface the failure site; the upsert path is idempotent so a re-run
    continues cleanly from where we stopped.
    """
    dst_col = surreal_backend.get_collection(
        palace=palace_ref,
        collection_name=src_name,
        create=True,
    )
    col_migrated = 0
    errors: list[str] = []
    last_batch_ids: list[str] = []
    try:
        for batch_ids, batch_docs, batch_metas, batch_embeds in _iter_chroma_batches(
            src_col, batch_size
        ):
            last_batch_ids = batch_ids
            norm_metas = [_normalize_metadata_for_surreal(m) for m in batch_metas]
            # Upsert is the idempotency guarantee: running this whole
            # function twice produces the same final state with no
            # duplicate rows (Surreal keys by ``drawer:<id>`` record id
            # + UNIQUE index on ``id_ext``).
            dst_col.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=norm_metas,
                embeddings=batch_embeds,
            )
            col_migrated += len(batch_ids)
            if progress:
                print(f"  [{src_name} -> {dst_table}] migrated {col_migrated}/{src_count} drawers")
    except Exception as e:
        msg = (
            f"{src_name}: failed after {col_migrated}/{src_count} drawers; "
            f"last batch size {len(last_batch_ids)}; error: {e}"
        )
        errors.append(msg)
        if progress:
            print(f"\n  ERROR: {msg}")
        raise

    ok, verify_errors = _verify_migration(src_col, dst_col, label=src_name)
    errors.extend(verify_errors)
    if not ok and progress:
        for err in verify_errors[:5]:
            print(f"  VERIFY FAIL: {err}")
    return col_migrated, ok, errors


def migrate_to_surreal(
    source_palace: str,
    *,
    target_ns: Optional[str] = None,
    target_db: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 200,
    surreal_url: Optional[str] = None,
    surreal_user: Optional[str] = None,
    surreal_pass: Optional[str] = None,
    progress: bool = True,
    allow_merge: bool = False,
) -> dict:
    """Migrate drawers + embeddings from a Chroma palace into SurrealDB (mp-ciw).

    The source Chroma palace is treated as read-only: this function only
    calls ``count()`` and ``get(...)`` on Chroma collections — no delete,
    no upsert, no write of any kind. Running the command twice is safe
    because all writes go through ``SurrealCollection.upsert`` keyed on
    the Chroma drawer id, which is idempotent by construction.

    Scope (per task mp-ciw):

    * ``mempalace_drawers`` -> Surreal ``drawer`` table.
    * ``mempalace_closets`` -> Surreal ``closet`` table (if present).
    * KG triples and a first-class ``entity`` / ``triple`` port are
      out of scope — mp-3lc handles them.

    mp-2v9 collision guard
    ----------------------
    When ``target_db`` is supplied explicitly and the target already
    contains drawers (or closets), we refuse to migrate unless the
    caller passes ``allow_merge=True``. This prevents silently merging
    two palaces into the same Surreal DB via upsert — which would be
    indistinguishable from data corruption if the two palaces happen to
    share drawer ids. The default-derived target DB name includes an
    8-char sha256 slug of the absolute source path, so two different
    palaces with the same basename never collide by default.

    Returns a dict summary with ``{total, migrated, duration_s, verified,
    errors}`` suitable for programmatic callers (tests, future GUI).
    """
    from .backends.base import PalaceRef
    from .backends.surreal import SurrealBackend

    source_palace = os.path.abspath(os.path.expanduser(source_palace))
    if not os.path.isdir(source_palace) or not contains_palace_database(source_palace):
        raise FileNotFoundError(
            f"source palace not found or has no chroma.sqlite3: {source_palace}"
        )

    target_db_explicit = target_db is not None
    db_name = target_db or _derive_surreal_db_name(source_palace)
    namespace = target_ns or os.environ.get("MEMPALACE_SURREAL_NS", "mempalace")

    if progress:
        print(f"\n{'=' * 60}")
        print("  MemPalace Migrate -> SurrealDB")
        print(f"{'=' * 60}\n")
        print(f"  Source palace: {source_palace}")
        print(f"  Target NS/DB:  {namespace} / {db_name}")
        print(f"  Batch size:    {batch_size}")
        if dry_run:
            print("  Mode:          DRY RUN (no writes)")

    chroma_backend = _open_chroma_readonly(source_palace)
    source_collections, source_counts = _open_source_collections(
        chroma_backend, source_palace, progress
    )

    total_source = sum(source_counts.values())
    if progress:
        print(f"  Total source drawers: {total_source}\n")

    if dry_run:
        if progress:
            print("  DRY RUN — no writes. Exiting.\n")
        return {
            "total": total_source,
            "migrated": 0,
            "by_collection": source_counts,
            "duration_s": 0.0,
            "verified": True,
            "errors": [],
            "dry_run": True,
        }

    if total_source == 0:
        if progress:
            print("  Nothing to migrate.\n")
        return {
            "total": 0,
            "migrated": 0,
            "by_collection": {},
            "duration_s": 0.0,
            "verified": True,
            "errors": [],
            "dry_run": False,
        }

    surreal_kwargs = _resolve_surreal_connection(surreal_url, surreal_user, surreal_pass, namespace)

    # mp-2v9 guardrail: if the caller supplied --target-db explicitly and
    # the target already has data, refuse to merge without opt-in. We only
    # trigger on the explicit-override path because the default derivation
    # now embeds a sha8 of the absolute path, making accidental collisions
    # astronomically unlikely. A user who typed out --target-db, however,
    # may have picked a name that already belongs to another palace.
    if target_db_explicit and not allow_merge:
        existing_total, sampled_ids, per_counts = _inspect_target_collision(surreal_kwargs, db_name)
        if existing_total > 0:
            if progress:
                print(
                    f"\n  REFUSING to migrate: target DB {db_name!r} already contains "
                    f"{existing_total} row(s) across: "
                    f"{', '.join(f'{k}={v}' for k, v in per_counts.items()) or '(none)'}"
                )
                if sampled_ids:
                    print(f"  Sampled existing ids: {sampled_ids[:3]}")
                print(
                    "  Pass --allow-merge to upsert into the existing DB on purpose,\n"
                    "  or pick a different --target-db. Aborting.\n"
                )
            raise TargetCollisionError(
                f"target Surreal DB {db_name!r} already has {existing_total} "
                f"row(s) ({per_counts}); pass allow_merge=True to proceed"
            )
        if progress:
            print(f"  Target DB {db_name!r} is empty — proceeding.")
    elif target_db_explicit and allow_merge and progress:
        print(f"  --allow-merge set: will upsert into existing DB {db_name!r} if present.")

    surreal_backend = SurrealBackend(**surreal_kwargs)
    palace_ref = PalaceRef(id=db_name, local_path=None, namespace=db_name)

    start = time.monotonic()
    migrated = 0
    errors: list[str] = []
    verified = True
    summary_by_collection: dict[str, int] = {}

    try:
        for src_name, dst_table in _COLLECTION_MAP:
            if src_name not in source_collections:
                continue
            src_count = source_counts[src_name]
            if src_count == 0:
                summary_by_collection[src_name] = 0
                continue
            col_migrated, ok, col_errors = _migrate_one_collection(
                src_name=src_name,
                dst_table=dst_table,
                src_col=source_collections[src_name],
                src_count=src_count,
                surreal_backend=surreal_backend,
                palace_ref=palace_ref,
                batch_size=batch_size,
                progress=progress,
            )
            summary_by_collection[src_name] = col_migrated
            migrated += col_migrated
            if not ok:
                verified = False
                errors.extend(col_errors)
    finally:
        try:
            surreal_backend.close()
        except Exception:
            pass
        try:
            chroma_backend.close()
        except Exception:
            pass

    duration = time.monotonic() - start

    if progress:
        print("\n  Summary")
        print("  -------")
        print(f"  total source drawers: {total_source}")
        print(f"  migrated:             {migrated}")
        print(f"  duration:             {duration:.2f}s")
        print(f"  verified:             {verified}")
        if errors:
            print(f"  errors ({len(errors)}):")
            for err in errors[:10]:
                print(f"    - {err}")
        print(f"\n{'=' * 60}\n")

    return {
        "total": total_source,
        "migrated": migrated,
        "by_collection": summary_by_collection,
        "duration_s": duration,
        "verified": verified,
        "errors": errors,
        "dry_run": False,
    }
