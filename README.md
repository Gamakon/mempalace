> [!CAUTION]
> **Scam alert.** The only official sources for MemPalace are this
> [GitHub repository](https://github.com/MemPalace/mempalace), the
> [PyPI package](https://pypi.org/project/mempalace/), and the docs site at
> **[mempalaceofficial.com](https://mempalaceofficial.com)**. Any other
> domain — including `mempalace.tech` — is an impostor and may distribute
> malware. Details and timeline: [docs/HISTORY.md](docs/HISTORY.md).

---

> [!IMPORTANT]
> **Gamakon fork — shared MemPalace over SurrealDB.**
>
> This fork adds a **SurrealDB backend** so a team can share one palace
> across **multiple concurrent Claude Code and OpenClaw sessions**. The
> upstream ChromaDB backend takes a single-writer SQLite lock — fine for
> solo use, but fails the moment two agents write at once. SurrealDB lifts
> that restriction: any number of Claude/OpenClaw processes can read and
> write the same palace simultaneously.
>
> **Quick start (local, solo)**
>
> ```bash
> # 1. Install SurrealDB server (macOS; see https://surrealdb.com/install for other OSes)
> brew install surrealdb/tap/surreal
>
> # 2. Clone this fork and install with the Surreal extra
> git clone https://github.com/Gamakon/mempalace.git
> cd mempalace
> pip install -e ".[surreal]"
>
> # 3. Start a local SurrealDB server (see docs/surrealdb-local.md)
> mkdir -p ~/.mempalace/surreal
> nohup surreal start --user root --pass root --bind 127.0.0.1:8000 \
>   surrealkv:///Users/$(whoami)/.mempalace/surreal/surreal.db \
>   > ~/.mempalace/surreal/server.log 2>&1 &
>
> # 4. Tell MemPalace to use Surreal (edit ~/.mempalace/config.json)
> #    {"backend": "surreal", "palace_path": "..."}
>
> # 5. (Optional) Migrate an existing Chroma palace
> mempalace migrate-to-surreal --include-kg
> # If the existing Chroma HNSW index is corrupt/huge:
> mempalace migrate-to-surreal --sqlite-direct --include-kg
> ```
>
> **Shared / networked deployment (team mode)**
>
> Run SurrealDB on a host every teammate can reach — laptop, LAN box, or
> dedicated server. Every client points at the same endpoint.
>
> ```bash
> # On the DB host — bind to a reachable address and set real credentials
> surreal start --user <ADMIN_USER> --pass <ADMIN_PASS> \
>   --bind 0.0.0.0:8000 \
>   surrealkv:///var/lib/mempalace/surreal.db
>
> # Firewall: open TCP/8000 to trusted clients only. Run behind a VPN or
> # reverse proxy with TLS in production — SurrealDB's root auth has no
> # TLS by default.
> ```
>
> **On each teammate's machine**, set these env vars (or put them in
> `~/.mempalace/config.json` under `backend`, `surreal_url`, etc.):
>
> ```bash
> export MEMPALACE_BACKEND=surreal
> export MEMPALACE_SURREAL_URL=ws://db.yourlan:8000   # or wss://... behind TLS
> export MEMPALACE_SURREAL_USER=<ADMIN_USER>
> export MEMPALACE_SURREAL_PASS=<ADMIN_PASS>
> export MEMPALACE_SURREAL_NS=mempalace                # shared namespace for the team
> ```
>
> All Claude Code / OpenClaw sessions on every laptop now write to the
> same palace. Concurrent writes are race-free (see mp-33y / mp-85q in
> the Gamakon test suite — 4-process × 100-drawer stress, SIGKILL chaos,
> 100/100 ops in 4.89s across two sessions).
>
> **Gotchas**
>
> - The default backend is still `chroma` for upstream compatibility.
>   You MUST set `backend: surreal` (config.json or env) or nothing changes.
> - Use `ws://` / `wss://`, **not** `http://` — SurrealDB 3.0.4's HTTP
>   transport has a cross-session NS/DB routing race under concurrent
>   writers (see [`docs/surrealdb-upstream-bug-http-ns-race.md`](docs/surrealdb-upstream-bug-http-ns-race.md)).
> - HNSW index rebuild after a 100k+ drawer migration can take minutes.
>   First search may block while Surreal settles the index.
> - If a teammate's MCP server spawns with `"backend": null` (default
>   Chroma) against a palace that only exists in Surreal, searches
>   return empty or the CLI segfaults on a corrupt Chroma HNSW. Always
>   set the backend explicitly.
>
> **Upstream docs below describe the single-user, Chroma-backend
> experience.** Everything still works that way — this fork adds an
> option, it doesn't remove one.

---

<div align="center">

<img src="assets/mempalace_logo.png" alt="MemPalace" width="240">

# MemPalace

Local-first AI memory. Verbatim storage, pluggable backend, 96.6% R@5 raw on LongMemEval — zero API calls.

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]
[![][discord-shield]][discord-link]

</div>

---

## What it is

MemPalace stores your conversation history as verbatim text and retrieves
it with semantic search. It does not summarize, extract, or paraphrase.
The index is structured — people and projects become *wings*, topics
become *rooms*, and original content lives in *drawers* — so searches
can be scoped rather than run against a flat corpus.

The retrieval layer is pluggable. The current default is ChromaDB; a
[SurrealDB backend](docs/surrealdb-local.md) ships in-tree for
multi-process / concurrent-writer setups. The interface is defined in
[`mempalace/backends/base.py`](mempalace/backends/base.py) and alternative
backends can be dropped in without touching the rest of the system.

Nothing leaves your machine unless you opt in.

Architecture, concepts, and mining flows:
[mempalaceofficial.com/concepts/the-palace](https://mempalaceofficial.com/concepts/the-palace.html).

---

## Install

```bash
pip install mempalace
mempalace init ~/projects/myapp
```

### From source (with SurrealDB backend)

For concurrent multi-process use (see **Multi-process / team use** below),
install from source with the optional `surreal` extra:

```bash
# 1. Install the SurrealDB server (macOS)
brew install surrealdb/tap/surreal

# 2. Clone and install the package with the Surreal Python client
git clone https://github.com/Gamakon/mempalace.git
cd mempalace
pip install -e ".[surreal]"
```

Linux/Windows: see https://surrealdb.com/install for the server install.
Start the server and configure MemPalace following
[`docs/surrealdb-local.md`](docs/surrealdb-local.md).

## Quickstart

```bash
# Mine content into the palace
mempalace mine ~/projects/myapp                    # project files
mempalace mine ~/.claude/projects/ --mode convos   # Claude Code sessions (scope with --wing per project)

# Search
mempalace search "why did we switch to GraphQL"

# Load context for a new session
mempalace wake-up
```

For Claude Code, Gemini CLI, MCP-compatible tools, and local models, see
[mempalaceofficial.com/guide/getting-started](https://mempalaceofficial.com/guide/getting-started.html).

---

## Benchmarks

All numbers below are reproducible from this repository with the commands
in [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md). Full
per-question result files are committed under `benchmarks/results_*`.

**LongMemEval — retrieval recall (R@5, 500 questions):**

| Mode | R@5 | LLM required |
|---|---|---|
| Raw (semantic search, no heuristics, no LLM) | **96.6%** | None |
| Hybrid v4, held-out 450q (tuned on 50 dev, not seen during training) | **98.4%** | None |
| Hybrid v4 + LLM rerank (full 500) | ≥99% | Any capable model |

The raw 96.6% requires no API key, no cloud, and no LLM at any stage. The
hybrid pipeline adds keyword boosting, temporal-proximity boosting, and
preference-pattern extraction; the held-out 98.4% is the honest
generalisable figure.

The rerank pipeline promotes the best candidate out of the top-20
retrieved sessions using an LLM reader. It works with any reasonably
capable model — we have reproduced it with Claude Haiku, Claude Sonnet,
and minimax-m2.7 via Ollama Cloud (no Anthropic dependency). The gap
between raw and reranked is model-agnostic; we do not headline a "100%"
number because the last 0.6% was reached by inspecting specific wrong
answers, which `benchmarks/BENCHMARKS.md` flags as teaching to the test.

**Other benchmarks (full results in [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md)):**

| Benchmark | Metric | Score | Notes |
|---|---|---|---|
| LoCoMo (session, top-10, no rerank) | R@10 | 60.3% | 1,986 questions |
| LoCoMo (hybrid v5, top-10, no rerank) | R@10 | 88.9% | Same set |
| ConvoMem (all categories, 250 items) | Avg recall | 92.9% | 50 per category |
| MemBench (ACL 2025, 8,500 items) | R@5 | 80.3% | All categories |

We deliberately do not include a side-by-side comparison against Mem0,
Mastra, Hindsight, Supermemory, or Zep. Those projects publish different
metrics on different splits, and placing retrieval recall next to
end-to-end QA accuracy is not an honest comparison. See each project's
own research page for their published numbers.

**Reproducing every result:**

```bash
git clone https://github.com/MemPalace/mempalace.git
cd mempalace
pip install -e ".[dev]"
# see benchmarks/README.md for dataset download commands
python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json
```

---

## Knowledge graph

MemPalace includes a temporal entity-relationship graph with validity
windows — add, query, invalidate, timeline — backed by local SQLite.
Usage and tool reference:
[mempalaceofficial.com/concepts/knowledge-graph](https://mempalaceofficial.com/concepts/knowledge-graph.html).

## MCP server

29 MCP tools cover palace reads/writes, knowledge-graph operations,
cross-wing navigation, drawer management, and agent diaries. Installation
and the full tool list:
[mempalaceofficial.com/reference/mcp-tools](https://mempalaceofficial.com/reference/mcp-tools.html).

## Agents

Each specialist agent gets its own wing and diary in the palace.
Discoverable at runtime via `mempalace_list_agents` — no bloat in your
system prompt:
[mempalaceofficial.com/concepts/agents](https://mempalaceofficial.com/concepts/agents.html).

## Auto-save hooks

Two Claude Code hooks save periodically and before context compression:
[mempalaceofficial.com/guide/hooks](https://mempalaceofficial.com/guide/hooks.html).

## Multi-process / team use

The default Chroma backend uses a single-writer SQLite lock — fine for a
single MCP/CLI process at a time. When multiple Claude Code sessions,
agents, or hooks may write to the same palace concurrently, switch to the
**SurrealDB backend**, which supports multi-process concurrent writes:

```bash
# Prerequisites: SurrealDB server + mempalace with the [surreal] extra.
brew install surrealdb/tap/surreal       # macOS; see https://surrealdb.com/install for Linux/Windows
pip install -e ".[surreal]"              # from a source checkout

export MEMPALACE_BACKEND=surreal
# or set "backend": "surreal" in ~/.mempalace/config.json
```

Local server setup: [`docs/surrealdb-local.md`](docs/surrealdb-local.md).
Existing Chroma palaces migrate in place with:

```bash
mempalace migrate-to-surreal --include-kg     # drawers + knowledge graph
mempalace verify-migration                    # audit parity after migration
```

Large palaces (100k+ drawers) complete successfully; the HNSW rebuild at
that scale is a known SurrealDB scaling behavior and may take time.

---

## Requirements

- Python 3.9+
- A vector-store backend (ChromaDB by default; SurrealDB optional for concurrent writes)
- ~300 MB disk for the default embedding model

No API key is required for the core benchmark path.

## Docs

- Getting started → [mempalaceofficial.com/guide/getting-started](https://mempalaceofficial.com/guide/getting-started.html)
- CLI reference → [mempalaceofficial.com/reference/cli](https://mempalaceofficial.com/reference/cli.html)
- Python API → [mempalaceofficial.com/reference/python-api](https://mempalaceofficial.com/reference/python-api.html)
- Full benchmark methodology → [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)
- Release notes → [CHANGELOG.md](CHANGELOG.md)
- Corrections and public notices → [docs/HISTORY.md](docs/HISTORY.md)

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-3.3.2-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/MemPalace/mempalace/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/MemPalace/mempalace/blob/main/LICENSE
[discord-shield]: https://img.shields.io/badge/discord-join-5865F2?style=flat-square&labelColor=0a0e14&logo=discord&logoColor=5865F2
[discord-link]: https://discord.com/invite/ycTQQCu6kn
