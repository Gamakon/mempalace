# SurrealDB Schema for MemPalace (Draft)

**Issue:** mp-6gu
**Status:** Doc only — no implementation
**Goal:** Replace ChromaDB (drawer storage + vectors) and SQLite (`knowledge_graph.py`) with a single SurrealDB instance that serves both document + vector + graph needs.

Namespace/Database layout: one SurrealDB `NS mempalace DB <palace_id>` per palace. All tables below live inside one database. Local-first deployment uses embedded/file-backed SurrealDB (`surrealkv://<palace_path>/palace.skv`) so the privacy-by-architecture principle still holds.

---

## 1. Tables / Records

| Table             | Kind     | Purpose                                                                                  |
|-------------------|----------|------------------------------------------------------------------------------------------|
| `drawer`          | NORMAL   | Verbatim text chunk + embedding. Primary payload. Replaces Chroma `mempalace_drawers`.    |
| `closet`          | NORMAL   | Compressed AAAK index pointers. Replaces Chroma `mempalace_closets`.                      |
| `wing`            | NORMAL   | Wing (person/project) identity; parent of rooms.                                         |
| `room`            | NORMAL   | Room (day/topic) identity; parent of drawers.                                            |
| `entity`          | NORMAL   | Person/project/tool/concept node — replaces KG `entities`.                               |
| `triple`          | EDGE     | `RELATE` edge: `entity -> triple -> entity`. Replaces KG `triples` with native graph.   |
| `attribute`       | NORMAL   | Temporal key/value properties on entities (from `docs/schema.sql`).                      |
| `palace_meta`     | NORMAL   | Single-row config: schema version, embedder identity, embedding dim, hnsw_space.         |

Two edge tables are sufficient; everything else is normal records. Wings/rooms are modelled as records (not edges) so full-text search on wing/room names works and metadata like creation date has somewhere to live. Drawers reference their room via record link; no explicit `RELATE` needed for that parent–child path (cheaper for the hot write path).

---

## 2. Field Definitions

### `drawer` (the Chroma replacement — 99% of payload)

```surrealql
DEFINE TABLE drawer SCHEMAFULL PERMISSIONS NONE;

DEFINE FIELD id_ext          ON drawer TYPE string;            -- legacy Chroma id (e.g. "drawer_<wing>_<room>_<sha>")
DEFINE FIELD document        ON drawer TYPE string ASSERT $value != NONE;  -- verbatim text
DEFINE FIELD embedding       ON drawer TYPE array<float> ASSERT array::len($value) = palace_meta:main.embedding_dim;  -- dim locked to palace_meta.embedding_dim (default 384, all-MiniLM-L6-v2)

-- structural pointers
DEFINE FIELD wing            ON drawer TYPE record<wing>;
DEFINE FIELD room            ON drawer TYPE record<room>;
DEFINE FIELD hall            ON drawer TYPE option<string>;    -- detect_hall() tag
DEFINE FIELD chunk_index     ON drawer TYPE int;

-- provenance (mirrors today's Chroma metadata)
DEFINE FIELD source_file     ON drawer TYPE option<string>;
DEFINE FIELD source_mtime    ON drawer TYPE option<float>;
DEFINE FIELD added_by        ON drawer TYPE option<string>;
DEFINE FIELD filed_at        ON drawer TYPE datetime VALUE $value OR time::now() DEFAULT time::now();  -- VALUE clause required: DEFAULT alone does not survive UPSERT CONTENT in SurrealDB 3.0.4
DEFINE FIELD normalize_version ON drawer TYPE int DEFAULT 2;
DEFINE FIELD entities        ON drawer TYPE option<array<string>>; -- flat names, for cheap filter

-- catch-all for stragglers without a migration per new key
DEFINE FIELD metadata        ON drawer TYPE option<object> FLEXIBLE;  -- NOTE: FLEXIBLE goes AFTER TYPE in SurrealDB 3.0.4; `FLEXIBLE TYPE option<object>` is a parse error
```

### `closet`

Same shape as `drawer` — different table so queries can target one or the other. The AAAK index layer gets its own embedding space (can be smaller dim if needed).

### `wing` / `room`

```surrealql
DEFINE TABLE wing SCHEMAFULL;
DEFINE FIELD name       ON wing TYPE string ASSERT $value != NONE;
DEFINE FIELD kind       ON wing TYPE string DEFAULT 'unknown';  -- person | project | topic
DEFINE FIELD created_at ON wing TYPE datetime VALUE $value OR time::now() DEFAULT time::now();
DEFINE INDEX wing_name  ON wing FIELDS name UNIQUE;

DEFINE TABLE room SCHEMAFULL;
DEFINE FIELD name       ON room TYPE string ASSERT $value != NONE;
DEFINE FIELD wing       ON room TYPE record<wing>;
DEFINE FIELD date       ON room TYPE option<string>;            -- ISO day for day-rooms
DEFINE FIELD created_at ON room TYPE datetime VALUE $value OR time::now() DEFAULT time::now();
DEFINE INDEX room_wing_name ON room FIELDS wing, name UNIQUE;
```

### `entity` (from `knowledge_graph.py`)

```surrealql
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD slug       ON entity TYPE string;                  -- lowercased/underscored "max_morgan"
DEFINE FIELD name       ON entity TYPE string;                  -- display name
DEFINE FIELD type       ON entity TYPE string DEFAULT 'unknown';
DEFINE FIELD properties ON entity TYPE object FLEXIBLE DEFAULT {};  -- FLEXIBLE follows TYPE; reverse order is a parse error in SurrealDB 3.0.4
DEFINE FIELD created_at ON entity TYPE datetime VALUE $value OR time::now() DEFAULT time::now();
DEFINE INDEX entity_slug ON entity FIELDS slug UNIQUE;
```

### `triple` (edge — KG relationship with temporal validity)

```surrealql
DEFINE TABLE triple TYPE RELATION FROM entity TO entity SCHEMAFULL;
DEFINE FIELD predicate        ON triple TYPE string ASSERT $value != NONE;
DEFINE FIELD valid_from       ON triple TYPE option<string>;   -- ISO date/datetime string (e.g. "2015-04-01"); matches SQLite caller semantics — kg_surreal.py passes plain ISO strings, not Surreal `d'...'` literals
DEFINE FIELD valid_to         ON triple TYPE option<string>;   -- ditto; keeping as string avoids forcing callers to emit `d'2015-04-01'` datetime literals
DEFINE FIELD confidence       ON triple TYPE float DEFAULT 1.0;
DEFINE FIELD source_drawer    ON triple TYPE option<record<drawer>>;  -- replaces source_drawer_id
DEFINE FIELD source_closet    ON triple TYPE option<record<closet>>;
DEFINE FIELD source_file      ON triple TYPE option<string>;
DEFINE FIELD adapter_name     ON triple TYPE option<string>;
DEFINE FIELD extracted_at     ON triple TYPE datetime VALUE $value OR time::now() DEFAULT time::now();
```

### `attribute`

```surrealql
DEFINE TABLE attribute SCHEMAFULL;
DEFINE FIELD entity     ON attribute TYPE record<entity>;
DEFINE FIELD key        ON attribute TYPE string;
DEFINE FIELD value      ON attribute TYPE option<string>;
DEFINE FIELD valid_from ON attribute TYPE option<string>;   -- ISO date/datetime string (see `triple.valid_from` note)
DEFINE FIELD valid_to   ON attribute TYPE option<string>;
DEFINE INDEX attribute_pk ON attribute FIELDS entity, key, valid_from UNIQUE;
```

### `palace_meta`

```surrealql
DEFINE TABLE palace_meta SCHEMAFULL;
DEFINE FIELD schema_version  ON palace_meta TYPE int;
DEFINE FIELD embedder_name   ON palace_meta TYPE string;        -- e.g. "all-MiniLM-L6-v2"
DEFINE FIELD embedding_dim   ON palace_meta TYPE int;           -- e.g. 384
DEFINE FIELD hnsw_space      ON palace_meta TYPE string DEFAULT 'cosine';
DEFINE FIELD created_at      ON palace_meta TYPE datetime VALUE $value OR time::now() DEFAULT time::now();
```
Single canonical row at `palace_meta:main`. Used by the backend to enforce `DimensionMismatchError` / `EmbedderIdentityMismatchError` on write (RFC 001 contract).

---

## 3. Graph Edges (RELATE)

KG triples use native SurrealDB edges — replaces the SQLite `(subject, predicate, object)` join pattern:

```surrealql
-- Write (valid_from/valid_to are strings — plain ISO dates; no `d'...'` literal required)
RELATE entity:max->triple->entity:alice
  SET predicate  = 'child_of',
      valid_from = '2015-04-01',
      confidence = 1.0,
      source_drawer = drawer:⟨drawer_personal_2026-04-23_abc123⟩;

-- Query "everything about Max, outgoing, valid on 2026-01-15"
-- String comparison works because ISO 8601 date strings sort lexicographically.
SELECT *, out.name AS object_name
FROM entity:max->triple
WHERE (valid_from IS NONE OR valid_from <= '2026-01-15')
  AND (valid_to   IS NONE OR valid_to   >= '2026-01-15');

-- Incoming: `<-triple<-entity`. Both directions: union.
-- Invalidate: UPDATE triple WHERE in=... AND predicate=... AND valid_to IS NONE SET valid_to = '...';
```

Semantic parity with `knowledge_graph.py::query_entity(as_of=, direction=)` is direct. No join table needed; graph traversal is first-class.

---

## 4. Indexes

```surrealql
-- Vector: HNSW for sub-ms semantic search on the drawer hot path
DEFINE INDEX drawer_vec ON drawer FIELDS embedding HNSW DIMENSION 384 DIST COSINE M 16 EFC 150;
DEFINE INDEX closet_vec ON closet FIELDS embedding HNSW DIMENSION 384 DIST COSINE M 16 EFC 150;

-- Full-text / BM25 — replaces the $contains fast-path Chroma advertises
DEFINE ANALYZER mp_text TOKENIZERS blank, class FILTERS lowercase, ascii, snowball(english);
DEFINE INDEX drawer_ft  ON drawer FIELDS document SEARCH ANALYZER mp_text BM25 HIGHLIGHTS;
DEFINE INDEX closet_ft  ON closet FIELDS document SEARCH ANALYZER mp_text BM25 HIGHLIGHTS;

-- Metadata filters (RFC 001 `where=` equality path — the where a user hits most)
DEFINE INDEX drawer_wing_room ON drawer FIELDS wing, room;
DEFINE INDEX drawer_source    ON drawer FIELDS source_file;
DEFINE INDEX drawer_id_ext    ON drawer FIELDS id_ext UNIQUE;

-- KG hot paths (mirrors idx_triples_* in schema.sql)
DEFINE INDEX triple_predicate ON triple FIELDS predicate;
DEFINE INDEX triple_valid     ON triple FIELDS valid_from, valid_to;
```

SurrealDB's HNSW + BM25 in one store means the backend can run hybrid search (`SELECT ... FROM drawer WHERE document @@ $q OR embedding <|k,cosine|> $vec`) without the separate orchestration `searcher.py` currently does.

---

## 5. Bootstrap DDL (condensed, apply once per palace)

```surrealql
USE NS mempalace DB $palace_id;

-- meta
DEFINE TABLE palace_meta SCHEMAFULL;
-- ... (fields from §2)
CREATE palace_meta:main SET schema_version=1, embedder_name='all-MiniLM-L6-v2', embedding_dim=384, hnsw_space='cosine';

-- structural
DEFINE TABLE wing SCHEMAFULL; DEFINE TABLE room SCHEMAFULL;
-- ... fields + indexes from §2, §4

-- payload
DEFINE TABLE drawer SCHEMAFULL; DEFINE TABLE closet SCHEMAFULL;
-- ... fields + HNSW + BM25 + wing/room indexes from §2, §4

-- graph
DEFINE TABLE entity SCHEMAFULL; DEFINE TABLE triple TYPE RELATION FROM entity TO entity SCHEMAFULL;
DEFINE TABLE attribute SCHEMAFULL;
-- ... fields + indexes
```

The `surreal.py` backend should idempotently run these on first `get_collection(create=True)`, gated by a check against `palace_meta:main.schema_version`.

---

## 6. Translation Notes / Open Questions

- **Chroma `where=` operators.** `$eq/$ne/$in/$nin/$and/$or/$gt/$gte/$lt/$lte` all map 1:1 to SurrealQL (`=`, `!=`, `INSIDE`, `NOT INSIDE`, `AND`, `OR`, comparison operators). `$contains` maps to the BM25 index (`@@`) — faster than Chroma's substring scan but semantics shift from "substring" to "tokenized match". Needs a compat test; may require a raw `string::contains(document, $v)` fallback to preserve RFC 001 conformance exactly.
- **Include / typed results.** `QueryResult`/`GetResult` already cover embeddings+documents+metadatas+distances — all available via one SurrealQL `SELECT *, vector::distance::cosine(embedding, $q) AS _distance`. No loss.
- **Per-palace isolation.** Today each palace is its own Chroma dir. In Surreal we use one DB per palace (not one table) so drop/rebuild/repair is a `REMOVE DATABASE` — mirrors the "filesystem directory = palace" model.
- **Embedding dim lock.** Enforced via `palace_meta.embedding_dim` + `ASSERT array::len($value) = …` on `drawer.embedding`. That's the RFC 001 `DimensionMismatchError` source of truth, and it's stronger than Chroma (which stores dim only implicitly in HNSW).
- **Entity id scheme.** Today `knowledge_graph._entity_id()` lowercases + underscores names. In Surreal we keep that as `entity.slug` (unique index) and use Surreal's own `entity:<slug>` record id. Disambiguation fields from `entity_registry.py` (DOB, aliases, source, confidence, wiki_cache) can live inside `entity.properties` (flexible object) — no schema churn if onboarding adds new keys.
- **Closets vs. drawers.** Two tables, same shape. The AAAK compression is applied to `closet.document` by the ingest pipeline — the schema stays agnostic.
- **HNSW tuning.** `M=16, EFC=150` is a reasonable default for palaces up to ~1M drawers; needs benchmarking on the 135K-drawer fork palace referenced in `chroma.py::quarantine_stale_hnsw`. Open: does SurrealDB HNSW exhibit the same stale-segment class of bug? Needs a soak test before shipping.
- **BM25 as hybrid primary.** Today `searcher.py` runs BM25 in-process. Moving it into the DB removes a whole module but changes the latency envelope — budget is ≤500ms hooks, ≤100ms startup; bench the BM25+HNSW `OR` path before removing the external BM25.
- **No Rust-segment crash class.** Chroma's BLOB-seq-id bug and stale-HNSW quarantine (`chroma.py`) go away entirely. Biggest single reliability win of the migration.

---

## 7. Schema doc lessons learned

This schema has been real-world tested: the fixes below landed after implementing the SurrealDB backend (mp-6xi) and the KG port (mp-4yf) against SurrealDB 3.0.4. Future readers can trust the DDL above compiles and round-trips through `UPSERT CONTENT` / `RELATE` without surprises.

- **mp-6zy — FLEXIBLE clause order.** SurrealDB 3.0.4 parses `DEFINE FIELD ... FLEXIBLE TYPE option<object>` as an error. The working form is `TYPE option<object> FLEXIBLE` (FLEXIBLE after the type). Applied to `drawer.metadata` and `entity.properties`.
- **mp-5js — duplicate `drawer.embedding`.** The original doc defined `drawer.embedding` twice; the second (with the `array::len` ASSERT against `palace_meta:main.embedding_dim`) is the correct one. Duplicate removed.
- **mp-m1z — `DEFAULT time::now()` does not survive `UPSERT CONTENT`.** The second UPSERT fails with "Expected `datetime` but found `NONE`" because `UPSERT CONTENT` overlays the incoming object and DEFAULT is only evaluated on CREATE. The working pattern is `VALUE $value OR time::now() DEFAULT time::now()` — the `VALUE` clause re-fills on every write. Applied to every `*_at` datetime field (drawer, wing, room, entity, triple, palace_meta).
- **mp-7wg — `valid_from` / `valid_to` retyped to `option<string>`.** Callers (the KG port in `kg_surreal.py`, matching the existing SQLite semantics of `knowledge_graph.py`) pass plain ISO date strings like `"2015-04-01"`, not Surreal `d'...'` datetime literals. Typing as `datetime` forced every call site to emit the `d'...'` prefix and made SCHEMALESS the only escape hatch. Typing as `option<string>` keeps SCHEMAFULL and matches caller reality; ISO 8601 strings sort lexicographically so the temporal validity range query still works as-is.
