"""
verify_migration.py — Independent data-parity audit of a ChromaDB+SQLite -> SurrealDB migration (mp-dju).

This is the "prove it really worked" step: a full audit that runs
*independently* of the migration code so the two cannot conspire to hide
a bug. ``mempalace migrate-to-surreal`` / ``migrate-kg-to-surreal`` each
carry an internal spot-check; this module:

* Re-reads the Chroma source via raw ``get()``.
* Re-reads the SQLite KG via read-only ``mode=ro`` URI.
* Re-reads the Surreal target via its own ``get()`` / raw SurrealQL.
* Cross-checks counts, field-by-field payloads for a random sample, and
  reverse-checks that no Surreal row exists without a source row
  (phantom-row detection).

Exit codes (surfaced via ``VerifyReport.exit_code``):

* ``0`` — clean. Every check passed.
* ``1`` — count mismatch (drawer total or triple total).
* ``2`` — sample mismatch (one or more sampled records differ in
  document / metadata / embedding / triple fields).
* ``3`` — phantom rows found (a Surreal record with no matching source).

Precedence: we return the *lowest-severity* non-zero code that fires, to
keep the exit-code signal stable. The report body always contains the
full mismatch detail regardless.

Type-coercion rules tolerated by the audit
------------------------------------------
The migration (mp-ciw) runs user metadata through
``_normalize_metadata_for_surreal`` which converts keys to ``str`` but
leaves values untouched. Chroma itself only stores ``str/int/float/bool``
values — so the value type of every round-tripped metadata key must be
preserved. We therefore compare metadata values with ``==`` and a float
tolerance for floats, and we normalise both sides' keys to ``str`` before
diffing.

The KG migration (mp-3lc) casts ``extracted_at`` into a SurrealDB
``datetime`` via ``_normalize_extracted_at``. Surreal then round-trips it
as a datetime-bearing string whose *prefix* contains the source
``YYYY-MM-DD``. Bit-exact string equality is impossible across those
formats, so the audit compares ``extracted_at`` by "source date prefix
appears in target representation" — the same rule the migration's own
test uses (``test_extracted_at_preserved_not_migration_timestamp``).

All other triple fields (``valid_from``, ``valid_to``, ``confidence``,
``source_drawer_id``, ``adapter_name``) are required to match exactly
(floats within ``1e-6``).
"""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional


_FLOAT_TOL = 1e-6


# ── Slug / predicate normalisation (mirrors KG write paths) ───────────────


def _entity_slug(name: str) -> str:
    """Match :meth:`KnowledgeGraph._entity_id` / :meth:`KnowledgeGraphSurreal._entity_id`.

    Separate copy (rather than importing the methods) so the audit does
    not depend on any migration code — defence-in-depth: a bug in the
    migration's slug path cannot hide itself in the audit's slug path.
    """
    return name.lower().replace(" ", "_").replace("'", "")


def _normalize_predicate(predicate: str) -> str:
    """Match the KG write path's predicate normalisation."""
    return predicate.lower().replace(" ", "_")


# ── Report dataclasses ────────────────────────────────────────────────────


@dataclass
class Mismatch:
    """A single field-level mismatch between source and target."""

    category: str  # "drawer" | "triple"
    kind: str  # "document" | "metadata" | "embedding_dim" | ...
    identifier: str  # drawer id, or "<sub> -[pred]-> <obj>"
    source: Any
    target: Any
    detail: str = ""


@dataclass
class VerifyReport:
    """Final audit output.

    Attributes
    ----------
    drawer_count_source, drawer_count_target:
        Total drawer rows visible from each side.
    triple_count_source, triple_count_target:
        Total triple rows visible from each side.
    sampled_drawers, sampled_triples:
        How many rows were deep-compared (capped by ``--sample-size``).
    mismatches:
        Field-level mismatches found during deep compare.
    phantom_drawers, phantom_triples:
        Target-side rows with no matching source row (detected by
        reverse-lookup from a random sample of Surreal rows).
    count_delta_drawers, count_delta_triples:
        ``target - source``. Negative means target is missing rows.
    """

    drawer_count_source: int = 0
    drawer_count_target: int = 0
    triple_count_source: int = 0
    triple_count_target: int = 0
    sampled_drawers: int = 0
    sampled_triples: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    phantom_drawers: list[str] = field(default_factory=list)
    phantom_triples: list[str] = field(default_factory=list)
    kg_checked: bool = False

    @property
    def count_delta_drawers(self) -> int:
        return self.drawer_count_target - self.drawer_count_source

    @property
    def count_delta_triples(self) -> int:
        return self.triple_count_target - self.triple_count_source

    @property
    def count_mismatch(self) -> bool:
        if self.count_delta_drawers != 0:
            return True
        if self.kg_checked and self.count_delta_triples != 0:
            return True
        return False

    @property
    def sample_mismatch(self) -> bool:
        return bool(self.mismatches)

    @property
    def phantom_found(self) -> bool:
        return bool(self.phantom_drawers or self.phantom_triples)

    @property
    def exit_code(self) -> int:
        """Return the lowest-severity non-zero code that fires, or 0.

        Precedence: ``1`` (count) < ``2`` (sample) < ``3`` (phantom).
        """
        if self.count_mismatch:
            return 1
        if self.sample_mismatch:
            return 2
        if self.phantom_found:
            return 3
        return 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# ── Source readers ────────────────────────────────────────────────────────


def _open_chroma_source(palace_path: str):
    """Open the source palace via ChromaBackend — read-only by discipline."""
    from .backends.chroma import ChromaBackend

    backend = ChromaBackend()
    return backend


def _iter_source_drawers(backend, palace_path: str):
    """Yield ``(collection_name, collection_handle)`` for each source collection.

    Mirrors the pair used by :mod:`migrate_to_surreal` so the audit sees
    exactly the same surface.
    """
    for name in ("mempalace_drawers", "mempalace_closets"):
        try:
            col = backend.get_collection(palace_path, name, create=False)
        except FileNotFoundError:
            continue
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                continue
            raise
        yield name, col


def _open_sqlite_readonly(path: str) -> sqlite3.Connection:
    """Open the source KG via ``?mode=ro`` — writes raise SQLITE_READONLY."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ── Metadata comparison ───────────────────────────────────────────────────


def _normalise_meta_keys(meta: Optional[dict]) -> dict:
    """Coerce keys to ``str`` the same way ``_normalize_metadata_for_surreal`` does.

    Values are compared unchanged; Chroma's type set is str/int/float/bool,
    all of which round-trip through Surreal's FLEXIBLE object unchanged.
    """
    if not meta:
        return {}
    return {str(k): v for k, v in meta.items()}


def _value_equal(a: Any, b: Any) -> bool:
    """Compare two metadata values with float tolerance."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= _FLOAT_TOL
        except (TypeError, ValueError):
            return False
    return a == b


def _compare_metadata(drawer_id: str, src_meta: dict, dst_meta: dict) -> list[Mismatch]:
    """Every source key must appear in the target with an equal value.

    Extra target keys (e.g. migration-injected timestamps) are tolerated
    — the audit's job is "nothing was lost," not "nothing was added."
    """
    src = _normalise_meta_keys(src_meta)
    dst = _normalise_meta_keys(dst_meta)
    out: list[Mismatch] = []
    for k, v in src.items():
        if k not in dst:
            out.append(
                Mismatch(
                    category="drawer",
                    kind="metadata_missing_key",
                    identifier=drawer_id,
                    source={k: v},
                    target=None,
                    detail=f"target metadata is missing key {k!r}",
                )
            )
            continue
        if not _value_equal(v, dst[k]):
            out.append(
                Mismatch(
                    category="drawer",
                    kind="metadata_value",
                    identifier=drawer_id,
                    source={k: v},
                    target={k: dst[k]},
                    detail=f"metadata[{k!r}] differs: src={v!r} dst={dst[k]!r}",
                )
            )
    return out


def _compare_embedding(
    drawer_id: str,
    src_emb: Optional[list],
    dst_emb: Optional[list],
) -> list[Mismatch]:
    """Compare embedding dimension + element-wise max delta."""
    out: list[Mismatch] = []
    if src_emb is None and dst_emb is None:
        return out
    if src_emb is None or dst_emb is None:
        out.append(
            Mismatch(
                category="drawer",
                kind="embedding_presence",
                identifier=drawer_id,
                source="present" if src_emb is not None else None,
                target="present" if dst_emb is not None else None,
                detail=(
                    "embedding present on one side only: "
                    f"src={src_emb is not None}, dst={dst_emb is not None}"
                ),
            )
        )
        return out
    if len(src_emb) != len(dst_emb):
        out.append(
            Mismatch(
                category="drawer",
                kind="embedding_dim",
                identifier=drawer_id,
                source=len(src_emb),
                target=len(dst_emb),
                detail=f"embedding dim src={len(src_emb)} dst={len(dst_emb)}",
            )
        )
        return out
    max_delta = 0.0
    for a, b in zip(src_emb, dst_emb):
        try:
            delta = abs(float(a) - float(b))
        except (TypeError, ValueError):
            out.append(
                Mismatch(
                    category="drawer",
                    kind="embedding_type",
                    identifier=drawer_id,
                    source=type(a).__name__,
                    target=type(b).__name__,
                    detail=(f"non-numeric embedding value: src={a!r} dst={b!r}"),
                )
            )
            return out
        if delta > max_delta:
            max_delta = delta
    if max_delta > _FLOAT_TOL:
        out.append(
            Mismatch(
                category="drawer",
                kind="embedding_delta",
                identifier=drawer_id,
                source=f"dim={len(src_emb)}",
                target=f"dim={len(dst_emb)}",
                detail=f"max element-wise delta {max_delta:.3e} > tol {_FLOAT_TOL:.0e}",
            )
        )
    return out


# ── Drawer audit ──────────────────────────────────────────────────────────


def _collect_source_ids(col) -> list[str]:
    """Page through a source collection and return every id."""
    ids: list[str] = []
    total = col.count()
    offset = 0
    batch = 1000
    while offset < total:
        got = col.get(limit=batch, offset=offset, include=[])
        chunk = list(got.get("ids") or [])
        if not chunk:
            break
        ids.extend(chunk)
        offset += len(chunk)
    return ids


def _audit_drawer_collection(
    *,
    src_col,
    dst_col,
    src_name: str,
    sample_size: int,
    rng: random.Random,
    report: VerifyReport,
) -> None:
    """Count + sample + phantom-check for one drawer-shaped collection."""
    src_count = src_col.count()
    dst_count = dst_col.count()
    report.drawer_count_source += src_count
    report.drawer_count_target += dst_count

    if src_count == 0:
        # Nothing to sample; phantom check still runs below so a target
        # row without any source at all is caught.
        pass

    src_ids = _collect_source_ids(src_col) if src_count else []

    # ── Forward sample: source → target deep compare ──────────────────
    if src_ids and sample_size > 0:
        sample = rng.sample(src_ids, min(sample_size, len(src_ids)))
        src_got = src_col.get(
            ids=sample,
            include=["documents", "metadatas", "embeddings"],
        )
        # Chroma returns dicts; Surreal returns GetResult — handle both.
        src_ids_got, src_docs, src_metas, src_embs = _unpack_get(src_got, len(sample))
        src_by_id = {
            rid: (
                src_docs[i] if i < len(src_docs) else "",
                src_metas[i] if i < len(src_metas) else {},
                src_embs[i] if src_embs is not None and i < len(src_embs) else None,
            )
            for i, rid in enumerate(src_ids_got)
        }
        dst_got = dst_col.get(
            ids=sample,
            include=["documents", "metadatas", "embeddings"],
        )
        dst_ids_got, dst_docs, dst_metas, dst_embs = _unpack_get(dst_got, len(sample))
        dst_by_id = {
            rid: (
                dst_docs[i] if i < len(dst_docs) else "",
                dst_metas[i] if i < len(dst_metas) else {},
                dst_embs[i] if dst_embs is not None and i < len(dst_embs) else None,
            )
            for i, rid in enumerate(dst_ids_got)
        }
        for rid in sample:
            report.sampled_drawers += 1
            if rid not in dst_by_id:
                report.mismatches.append(
                    Mismatch(
                        category="drawer",
                        kind="missing_in_target",
                        identifier=rid,
                        source=f"present in {src_name}",
                        target=None,
                        detail=f"id {rid!r} missing from surreal target",
                    )
                )
                continue
            s_doc, s_meta, s_emb = src_by_id[rid]
            d_doc, d_meta, d_emb = dst_by_id[rid]
            if s_doc != d_doc:
                report.mismatches.append(
                    Mismatch(
                        category="drawer",
                        kind="document",
                        identifier=rid,
                        source=s_doc[:80],
                        target=d_doc[:80],
                        detail="document string differs",
                    )
                )
            report.mismatches.extend(_compare_metadata(rid, s_meta, d_meta))
            report.mismatches.extend(_compare_embedding(rid, s_emb, d_emb))

    # ── Reverse phantom check: sample target, require source exists ───
    if dst_count > 0 and sample_size > 0:
        dst_sample_page = dst_col.get(
            limit=max(sample_size * 2, 20),
            include=[],
        )
        dst_sample_ids = list(getattr(dst_sample_page, "ids", None) or [])
        if not dst_sample_ids and isinstance(dst_sample_page, dict):
            dst_sample_ids = list(dst_sample_page.get("ids") or [])
        if len(dst_sample_ids) > sample_size:
            dst_sample_ids = rng.sample(dst_sample_ids, sample_size)
        src_id_set = set(src_ids)
        for dst_id in dst_sample_ids:
            if dst_id not in src_id_set:
                report.phantom_drawers.append(dst_id)


def _unpack_get(got, expected_len: int):
    """Normalise a Chroma dict or Surreal ``GetResult`` into a uniform tuple."""
    if hasattr(got, "ids"):
        ids = list(got.ids or [])
        docs = list(got.documents or [])
        metas = list(got.metadatas or [])
        embs = got.embeddings
        if embs is not None:
            embs = [list(e) if e is not None else None for e in embs]
    else:
        ids = list(got.get("ids") or [])
        docs = list(got.get("documents") or [])
        metas = list(got.get("metadatas") or [])
        embs = got.get("embeddings")
        if embs is not None:
            embs = [list(e) if e is not None else None for e in embs]
    # Pad to parallel length so indexed access is safe.
    while len(docs) < len(ids):
        docs.append("")
    while len(metas) < len(ids):
        metas.append({})
    return ids, docs, metas, embs


# ── KG audit ──────────────────────────────────────────────────────────────


def _count_sqlite(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _count_surreal_triples(surreal) -> int:
    rows = surreal._db.query("SELECT count() FROM triple GROUP ALL") or []
    if not rows:
        return 0
    row = rows[0] if isinstance(rows, list) else rows
    if isinstance(row, dict):
        return int(row.get("count") or 0)
    return 0


def _iter_sqlite_triples(conn: sqlite3.Connection):
    """Yield every triple row joined with entity display names.

    Returns a dict per row with both slug and display-name columns so the
    audit can dual-check ``(subject_slug, predicate, object_slug)`` against
    Surreal's ``(in.id, predicate, out.id)``.
    """
    cur = conn.execute(
        """
        SELECT
            t.id                AS id,
            t.subject           AS subject_slug,
            t.predicate         AS predicate,
            t.object            AS object_slug,
            t.valid_from        AS valid_from,
            t.valid_to          AS valid_to,
            t.confidence        AS confidence,
            t.source_closet     AS source_closet,
            t.source_file       AS source_file,
            t.source_drawer_id  AS source_drawer_id,
            t.adapter_name      AS adapter_name,
            t.extracted_at      AS extracted_at,
            s.name              AS subject_name,
            o.name              AS object_name
        FROM triples t
        LEFT JOIN entities s ON t.subject = s.id
        LEFT JOIN entities o ON t.object  = o.id
        ORDER BY t.id
        """
    )
    for row in cur:
        yield {
            "id": row["id"],
            "subject_slug": row["subject_slug"],
            "object_slug": row["object_slug"],
            "subject_name": row["subject_name"] or row["subject_slug"],
            "object_name": row["object_name"] or row["object_slug"],
            "predicate": row["predicate"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
            "source_closet": row["source_closet"],
            "source_file": row["source_file"],
            "source_drawer_id": row["source_drawer_id"],
            "adapter_name": row["adapter_name"],
            "extracted_at": row["extracted_at"],
        }


def _find_surreal_triple(surreal, tri: dict) -> Optional[dict]:
    """Return the Surreal row for ``tri`` keyed on (sub_slug, pred, obj_slug).

    Closed triples additionally constrain on ``(valid_from, valid_to)`` so
    we pick the matching historical record when multiple span the same
    endpoint pair.
    """
    from surrealdb import RecordID

    sub_rec = RecordID("entity", _entity_slug(tri["subject_name"]))
    obj_rec = RecordID("entity", _entity_slug(tri["object_name"]))
    pred = _normalize_predicate(tri["predicate"])

    if tri["valid_to"] is None:
        rows = surreal._db.query(
            (
                "SELECT predicate, valid_from, valid_to, confidence, "
                "source_closet, source_file, source_drawer_id, "
                "adapter_name, extracted_at "
                "FROM triple WHERE in = $sub AND out = $obj "
                "AND predicate = $pred AND valid_to IS NONE"
            ),
            {"sub": sub_rec, "obj": obj_rec, "pred": pred},
        )
    else:
        rows = surreal._db.query(
            (
                "SELECT predicate, valid_from, valid_to, confidence, "
                "source_closet, source_file, source_drawer_id, "
                "adapter_name, extracted_at "
                "FROM triple WHERE in = $sub AND out = $obj "
                "AND predicate = $pred AND valid_from = $vf "
                "AND valid_to = $vt"
            ),
            {
                "sub": sub_rec,
                "obj": obj_rec,
                "pred": pred,
                "vf": tri["valid_from"],
                "vt": tri["valid_to"],
            },
        )
    if not rows:
        return None
    return rows[0]


def _compare_triple(tri: dict, row: dict) -> list[Mismatch]:
    """Deep-compare a source triple against its Surreal row."""
    ident = f"{tri['subject_name']} -[{tri['predicate']}]-> {tri['object_name']}"
    out: list[Mismatch] = []
    if row.get("predicate") != _normalize_predicate(tri["predicate"]):
        out.append(
            Mismatch(
                category="triple",
                kind="predicate",
                identifier=ident,
                source=tri["predicate"],
                target=row.get("predicate"),
                detail="predicate differs after normalisation",
            )
        )
    if row.get("valid_from") != tri["valid_from"]:
        out.append(
            Mismatch(
                category="triple",
                kind="valid_from",
                identifier=ident,
                source=tri["valid_from"],
                target=row.get("valid_from"),
            )
        )
    if row.get("valid_to") != tri["valid_to"]:
        out.append(
            Mismatch(
                category="triple",
                kind="valid_to",
                identifier=ident,
                source=tri["valid_to"],
                target=row.get("valid_to"),
            )
        )
    src_conf = tri["confidence"]
    dst_conf = row.get("confidence")
    try:
        if abs(float(src_conf) - float(dst_conf)) > _FLOAT_TOL:
            out.append(
                Mismatch(
                    category="triple",
                    kind="confidence",
                    identifier=ident,
                    source=src_conf,
                    target=dst_conf,
                )
            )
    except (TypeError, ValueError):
        out.append(
            Mismatch(
                category="triple",
                kind="confidence",
                identifier=ident,
                source=src_conf,
                target=dst_conf,
                detail="non-numeric confidence value",
            )
        )
    for field_ in ("source_drawer_id", "adapter_name", "source_closet", "source_file"):
        if (row.get(field_) or None) != (tri[field_] or None):
            out.append(
                Mismatch(
                    category="triple",
                    kind=field_,
                    identifier=ident,
                    source=tri[field_],
                    target=row.get(field_),
                )
            )
    # ``extracted_at`` crosses a format boundary (SQLite 'YYYY-MM-DD HH:MM:SS'
    # -> Surreal datetime). The migration's own test accepts "source date
    # prefix appears in target representation"; so do we.
    src_ext = tri["extracted_at"]
    dst_ext = row.get("extracted_at")
    if src_ext is not None:
        if dst_ext is None:
            out.append(
                Mismatch(
                    category="triple",
                    kind="extracted_at",
                    identifier=ident,
                    source=src_ext,
                    target=None,
                    detail="extracted_at missing on target",
                )
            )
        else:
            src_date = src_ext.split(" ")[0].split("T")[0]
            if src_date not in str(dst_ext):
                out.append(
                    Mismatch(
                        category="triple",
                        kind="extracted_at",
                        identifier=ident,
                        source=src_ext,
                        target=dst_ext,
                        detail=f"source date {src_date!r} not in target {dst_ext!r}",
                    )
                )
    return out


def _audit_kg(
    *,
    sqlite_path: str,
    surreal,
    sample_size: int,
    rng: random.Random,
    report: VerifyReport,
) -> None:
    """Count + sample + phantom-check the KG side of the migration."""
    report.kg_checked = True
    conn = _open_sqlite_readonly(sqlite_path)
    try:
        report.triple_count_source = _count_sqlite(conn, "triples")
        report.triple_count_target = _count_surreal_triples(surreal)

        all_triples = list(_iter_sqlite_triples(conn))
        if all_triples and sample_size > 0:
            sample = rng.sample(all_triples, min(sample_size, len(all_triples)))
            for tri in sample:
                report.sampled_triples += 1
                row = _find_surreal_triple(surreal, tri)
                if row is None:
                    ident = f"{tri['subject_name']} -[{tri['predicate']}]-> {tri['object_name']}"
                    report.mismatches.append(
                        Mismatch(
                            category="triple",
                            kind="missing_in_target",
                            identifier=ident,
                            source="present in SQLite",
                            target=None,
                            detail="no matching triple in Surreal",
                        )
                    )
                    continue
                report.mismatches.extend(_compare_triple(tri, row))

        # ── Phantom check: sample Surreal triples and require source hit.
        if report.triple_count_target > 0 and sample_size > 0:
            limit = max(sample_size * 2, 20)
            dst_rows = (
                surreal._db.query(
                    (
                        "SELECT predicate, valid_from, valid_to, "
                        "in.id AS sub_id, out.id AS obj_id "
                        f"FROM triple LIMIT {int(limit)}"
                    )
                )
                or []
            )
            if len(dst_rows) > sample_size:
                dst_rows = rng.sample(dst_rows, sample_size)
            # Build the source lookup once, keyed on (sub_slug, pred,
            # obj_slug, valid_from, valid_to) — full key so historical
            # closed-triple versions are distinguishable.
            src_key_set: set[tuple] = set()
            for tri in all_triples:
                src_key_set.add(
                    (
                        _entity_slug(tri["subject_name"]),
                        _normalize_predicate(tri["predicate"]),
                        _entity_slug(tri["object_name"]),
                        tri["valid_from"],
                        tri["valid_to"],
                    )
                )
            for row in dst_rows:
                sub_id_raw = row.get("sub_id")
                obj_id_raw = row.get("obj_id")
                # Surreal RecordID stringifies as ``entity:slug``; extract slug.
                sub_slug = _strip_record_id(sub_id_raw)
                obj_slug = _strip_record_id(obj_id_raw)
                key = (
                    sub_slug,
                    row.get("predicate"),
                    obj_slug,
                    row.get("valid_from"),
                    row.get("valid_to"),
                )
                if key not in src_key_set:
                    report.phantom_triples.append(
                        f"{sub_slug} -[{row.get('predicate')}]-> {obj_slug}"
                        f" @ ({row.get('valid_from')}, {row.get('valid_to')})"
                    )
    finally:
        conn.close()


def _strip_record_id(value: Any) -> str:
    """Return the slug portion of a Surreal RecordID-ish value.

    The surrealdb Python client returns a ``RecordID`` with ``.id``
    attribute, but queries that ``SELECT in.id AS sub_id`` flatten it
    to a raw string like ``"alice"``. Accept both shapes.
    """
    if value is None:
        return ""
    if hasattr(value, "id"):
        return str(value.id)
    s = str(value)
    if ":" in s:
        return s.split(":", 1)[1]
    return s


# ── Public entry point ────────────────────────────────────────────────────


def verify_migration(
    *,
    source_palace: str,
    source_kg: Optional[str] = None,
    target_ns: Optional[str] = None,
    target_db: Optional[str] = None,
    sample_size: int = 50,
    surreal_url: Optional[str] = None,
    surreal_user: Optional[str] = None,
    surreal_pass: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> VerifyReport:
    """Run the full audit and return a populated :class:`VerifyReport`.

    This function never mutates either side. The Chroma source is treated
    read-only by discipline (only ``count()`` + ``get()`` are called); the
    SQLite KG is opened via ``mode=ro`` URI so writes raise at the SQL
    layer.

    Parameters
    ----------
    source_palace:
        Chroma palace directory.
    source_kg:
        SQLite KG file. When ``None`` the default
        ``~/.mempalace/knowledge_graph.sqlite3`` is used if present;
        otherwise the KG section is skipped and the report carries only
        drawer-side results.
    target_ns, target_db:
        Target SurrealDB namespace + database. When ``None`` the namespace
        falls back to ``MEMPALACE_SURREAL_NS`` env (or ``"mempalace"``),
        and the database name is derived from ``source_palace`` the same
        way :func:`migrate_to_surreal` does.
    sample_size:
        How many rows to deep-compare on each side (forward) and how many
        target rows to reverse-check (phantom). Default ``50``.
    rng:
        Optional deterministic RNG so a failing audit is reproducible.
    """
    from .backends.base import PalaceRef
    from .backends.surreal import (
        DEFAULT_PASS,
        DEFAULT_URL,
        DEFAULT_USER,
        SurrealBackend,
    )
    from .migrate import _derive_surreal_db_name  # type: ignore[attr-defined]

    rng = rng or random.Random()
    source_palace = os.path.abspath(os.path.expanduser(source_palace))
    db_name = target_db or _derive_surreal_db_name(source_palace)
    namespace = target_ns or os.environ.get("MEMPALACE_SURREAL_NS") or "mempalace"

    surreal_backend = SurrealBackend(
        url=surreal_url or os.environ.get("MEMPALACE_SURREAL_URL") or DEFAULT_URL,
        username=surreal_user or os.environ.get("MEMPALACE_SURREAL_USER") or DEFAULT_USER,
        password=surreal_pass or os.environ.get("MEMPALACE_SURREAL_PASS") or DEFAULT_PASS,
        namespace=namespace,
    )
    palace_ref = PalaceRef(id=db_name, local_path=None, namespace=db_name)

    report = VerifyReport()

    chroma_backend = _open_chroma_source(source_palace)
    try:
        for src_name, src_col in _iter_source_drawers(chroma_backend, source_palace):
            try:
                dst_col = surreal_backend.get_collection(
                    palace=palace_ref,
                    collection_name=src_name,
                    create=False,
                )
            except Exception as e:
                report.mismatches.append(
                    Mismatch(
                        category="drawer",
                        kind="target_collection_missing",
                        identifier=src_name,
                        source=f"{src_col.count()} rows",
                        target=None,
                        detail=f"opening target collection failed: {e}",
                    )
                )
                report.drawer_count_source += src_col.count()
                continue
            _audit_drawer_collection(
                src_col=src_col,
                dst_col=dst_col,
                src_name=src_name,
                sample_size=sample_size,
                rng=rng,
                report=report,
            )
    finally:
        try:
            chroma_backend.close()
        except Exception:
            pass

    # ── KG side ───────────────────────────────────────────────────────
    kg_path = source_kg
    if kg_path is None:
        default_kg = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")
        if os.path.isfile(default_kg):
            kg_path = default_kg
    if kg_path and os.path.isfile(kg_path):
        from .kg_surreal import KnowledgeGraphSurreal

        surreal_kg = KnowledgeGraphSurreal(
            url=surreal_url or DEFAULT_URL,
            user=surreal_user or DEFAULT_USER,
            password=surreal_pass or DEFAULT_PASS,
            namespace=namespace,
            database=db_name,
        )
        try:
            _audit_kg(
                sqlite_path=kg_path,
                surreal=surreal_kg,
                sample_size=sample_size,
                rng=rng,
                report=report,
            )
        finally:
            surreal_kg.close()

    try:
        surreal_backend.close()
    except Exception:
        pass

    return report


# ── Pretty printing ───────────────────────────────────────────────────────


def format_report(report: VerifyReport) -> str:
    """Render a human-readable summary + mismatch table."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  MemPalace verify-migration report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Counts")
    lines.append("  ------")
    lines.append(
        f"    drawers: source={report.drawer_count_source} "
        f"target={report.drawer_count_target} "
        f"delta={report.count_delta_drawers:+d}"
    )
    if report.kg_checked:
        lines.append(
            f"    triples: source={report.triple_count_source} "
            f"target={report.triple_count_target} "
            f"delta={report.count_delta_triples:+d}"
        )
    else:
        lines.append("    triples: (KG not checked — source KG not provided)")
    lines.append("")
    lines.append("  Samples deep-compared")
    lines.append("  ---------------------")
    lines.append(f"    drawers: {report.sampled_drawers}")
    lines.append(f"    triples: {report.sampled_triples}")
    lines.append("")
    lines.append(f"  Mismatches found: {len(report.mismatches)}")
    lines.append(
        f"  Phantom rows:     {len(report.phantom_drawers)} drawers, "
        f"{len(report.phantom_triples)} triples"
    )
    lines.append("")

    if report.mismatches:
        lines.append("  Mismatches")
        lines.append("  ----------")
        for m in report.mismatches:
            lines.append(
                f"    [{m.category}/{m.kind}] {m.identifier}: "
                f"src={m.source!r} dst={m.target!r}" + (f"  — {m.detail}" if m.detail else "")
            )
        lines.append("")

    if report.phantom_drawers:
        lines.append("  Phantom drawers (target rows with no source match)")
        for rid in report.phantom_drawers:
            lines.append(f"    - {rid}")
        lines.append("")
    if report.phantom_triples:
        lines.append("  Phantom triples (target rows with no source match)")
        for t in report.phantom_triples:
            lines.append(f"    - {t}")
        lines.append("")

    if report.ok:
        lines.append("  RESULT: CLEAN — no mismatches, no phantoms, counts match.")
    else:
        lines.append(
            f"  RESULT: PROBLEMS FOUND — exit_code={report.exit_code} "
            f"(count_mismatch={report.count_mismatch}, "
            f"sample_mismatch={report.sample_mismatch}, "
            f"phantom_found={report.phantom_found})"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
