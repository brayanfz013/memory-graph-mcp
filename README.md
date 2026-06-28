# memory-graph-mcp

> Unified semantic memory for AI agents. One MCP server that gives Claude Code / Codex / Cursor / any MCP client a per-repo knowledge store with **semantic recall**, a **typed knowledge graph**, an **auto-crystallized wiki**, and **cross-agent coordination**.

[![CI](https://github.com/brayanfz013/memory-graph-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/brayanfz013/memory-graph-mcp/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

## Why

Most agent workflows lose context between sessions. Memory is either:
- **In-prompt only** — vanishes when the conversation resets, and bloats tokens.
- **A flat vector DB** — finds similar text but loses *relationships* between decisions.

`memory-graph` gives you **four reinforcing layers in one MCP server**:

| Layer | What it stores | Why it matters |
|---|---|---|
| **Vector memories** | Findings with embeddings | Semantic recall of past solutions / decisions |
| **Knowledge graph** | Typed nodes + edges (`SOLVES`, `SUPERSEDES`, …) + PageRank | Surfaces *which decision matters most* + lineage |
| **Wiki** | Long-form crystallized docs auto-generated from canonical nodes | Curated knowledge survives turnover |
| **Collective + cache** | Cross-agent state with TTL | Multi-agent coordination without re-running expensive tools |

Storage is **workspace-scoped DuckDB** at `<repo>/.memory-graph/memory.duckdb` — no cloud, no API keys (embeddings run locally with `fastembed`).

---

## Install

### As a Claude Code plugin (recommended)

Zero-install — the plugin's `.mcp.json` invokes `uvx` to bootstrap the server straight from the GitHub release tag. **Prerequisite:** `uv` on PATH (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

Inside Claude Code:

```bash
/plugin marketplace add brayanfz013/memory-graph-mcp
/plugin install memory-graph@memory-graph-marketplace
```

First launch downloads + caches the wheel (≈30–60 s); subsequent launches reuse `uv`'s cache and start instantly. No `pip install` step needed.

The plugin auto-wires the MCP server, ships a `memory-graph` skill with the usage protocol, and adds a `/memory-recall` slash command. See [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json).

### As a standalone Python package

```bash
# With uv (fast)
uv pip install memory-graph-mcp

# Or with pip
pip install memory-graph-mcp

# Optional: Vertex AI embeddings instead of local fastembed
uv pip install "memory-graph-mcp[vertex]"
```

This installs the `memory-graph` CLI entry point, which speaks MCP over stdio.

### From source

```bash
git clone https://github.com/brayanfz013/memory-graph-mcp.git
cd memory-graph-mcp
uv pip install -e ".[dev]"
```

---

## Wire it into any MCP client

Add to your MCP config (`.vscode/mcp.json`, `~/.codex/config.toml`, Claude Desktop, etc.):

```json
{
  "servers": {
    "memory-graph": {
      "type": "stdio",
      "command": "memory-graph",
      "env": {
        "MEMORY_GRAPH_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

`MEMORY_GRAPH_WORKSPACE` controls **where the per-repo DuckDB lives**. The server auto-detects it from (in priority order):
1. `MEMORY_GRAPH_WORKSPACE` (explicit)
2. `CLAUDE_PROJECT_DIR` (set by Claude Code)
3. `CODEX_WORKSPACE_DIR` (set by Codex)
4. `PWD` at launch

---

## Quickstart

Once the server is running, three calls cover 90% of use cases:

```python
# 1. Before working — check if this problem was solved before
recall(query="duckdb concurrent write lock", top_k=5)

# 2. After solving — persist the finding (creates vector memory + KG node + auto-edges)
record_finding(
    finding_type="solution",
    title="DuckDB lock retry with exponential backoff",
    content="Wrap writes in retry decorator (3 attempts, 0.2s base delay). Validated with 3 concurrent agents.",
    related_files=["memory_graph/db.py"],
    tags=["duckdb", "concurrency"],
)

# 3. Later — fetch the full curated wiki page
wiki_get("solution.duckdb-lock-retry")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete walkthrough.

---

## Troubleshooting

If the MCP server fails to start, check Claude Code's MCP stderr log. Common causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `WorkspaceResolutionError` / "not a directory" | `MEMORY_GRAPH_WORKSPACE` or `CLAUDE_PROJECT_DIR` unset, or contains an unexpanded `${...}` literal | Set `MEMORY_GRAPH_WORKSPACE` explicitly in your MCP config to the absolute path of your project root |
| Stalled for >2 minutes on first run | fastembed ONNX model download from HuggingFace (~100 MB) in progress, or blocked by proxy/firewall | Verify `https://huggingface.co` is reachable; check `~/.cache/fastembed` for partial downloads and delete it to retry |
| `RuntimeError: fastembed failed to initialize` | Corporate proxy with TLS interception, or HuggingFace CDN unreachable | Set `HF_HUB_OFFLINE=0` and pre-download the model with `python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"` from a network with internet access |
| `duckdb.IOException` on startup | DB file locked by another process, or corrupted from a half-write | Close other Claude Code sessions on the same workspace; if it persists, back up and delete `<workspace>/.memory-graph/memory.duckdb` |

For the full usage protocol (what to recall, when to record, how to traverse the graph), see [`PROTOCOL.md`](PROTOCOL.md).

---

## Tool surface (v0.4.1)

**Primary (use these by default):**
- `recall(query, scope, ...)` — fused semantic search across memories + KG + wiki
- `record_finding(...)` — store + index + auto-edges + auto-crystallize
- `wiki_get(canonical_id_or_title)` — fetch a wiki page in full

**Graph navigation:**
- `kg_neighbors(node_id, ...)` — immediate neighbors
- `kg_path(from_id, to_id, ...)` — shortest path between two nodes
- `kg_resolve(canonical_id)` — lookup by stable slug

**Lifecycle / advanced writes:**
- `kg_promote(node_id, status)` — draft → canonical → superseded
- `kg_add_edge(from, to, rel_type)` — manual typed edges (`SOLVES`, `SUPERSEDES`, etc.)
- `wiki_ingest(title, body, ...)` — manual long-form authoring

**Cross-agent state:**
- `collective_store / get / list` — share state across agents with TTL
- `cache_check / store` — memoize expensive tool calls

**Embedding provider administration (v0.4.1):**
- `embedding_status()` — active vs DB-stored provider + mismatch diff + registry
- `embedding_migrate(target_provider, target_model, dry_run, batch_size)` — re-embed everything under a new model
- `embedding_benchmark(providers, eval_set_path, top_k)` — score providers on Recall@1 / Recall@5 / MRR / latency

**Health / admin:**
- `memory_report()` — counts + top influential + wiki lint
- `memory_consolidate(dry_run)` — purge expired + recompute PageRank
- `memory_stats()` — table row counts
- `wiki_bootstrap()` — seed wiki from workspace + canonicals

**Legacy (kept for backward compat):**
- `memory_recall`, `kg_query`, `kg_influential`

Full schema: [`memory_graph/server.py`](memory_graph/server.py).

---

## Configuration

All settings via env vars (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `MEMORY_GRAPH_WORKSPACE` | auto-detected | Project root where DuckDB lives |
| `MEMORY_GRAPH_DB_PATH` | `<workspace>/.memory-graph/memory.duckdb` | Override DB location |
| `MEMORY_GRAPH_EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (local) · `ollama` (local HTTP) · `vertex` (Google Cloud) |
| `MEMORY_GRAPH_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | See `PROVIDER_REGISTRY` for options (incl. `BAAI/bge-base-en-v1.5`, `intfloat/multilingual-e5-base`) |
| `MEMORY_GRAPH_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `MEMORY_GRAPH_OLLAMA_MODEL` | `nomic-embed-text` | Must be pulled first: `ollama pull <model>` |
| `MEMORY_GRAPH_GOOGLE_PROJECT_ID` | — | Required if provider = `vertex` |
| `MEMORY_GRAPH_MAX_ENTRIES` | `10000` | Collective memory LRU cap |
| `MEMORY_GRAPH_PAGERANK_DAMPING` | `0.85` | PageRank damping factor |

### Swapping embedding providers

Different models give different retrieval quality (especially across languages). `memory-graph` ships with three providers and tooling to swap between them safely without losing your stored content.

**1. See what's available + what's currently in the DB:**

```python
embedding_status()
```

Returns active env identity, DB-stored identity, mismatch flag, and the full `PROVIDER_REGISTRY` (`fastembed`, `ollama`, `vertex` × their supported models with dimensions and language hints).

**2. Benchmark before swapping** — score Recall@1 / Recall@5 / MRR against the bundled eval set (or your own):

```python
embedding_benchmark(providers=[
    {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
    {"provider": "fastembed", "model": "BAAI/bge-base-en-v1.5"},
    {"provider": "fastembed", "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"},
])
```

The bundled eval set ([`eval/eval_set.json`](eval/eval_set.json)) has 12 seeds + 13 queries across English and Spanish. Extend it with your own corpus for reliable numbers on *your* content.

**3. Migrate when you've picked a winner** — re-embeds every stored memory and wiki page under the new model. Source content stays; only vectors are replaced.

```python
# Plan first (safe — no rewrite)
embedding_migrate(
    target_provider="fastembed",
    target_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    dry_run=True,
)

# Apply the rewrite
embedding_migrate(
    target_provider="fastembed",
    target_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    dry_run=False,
)
```

After migration, `embedding_meta` records a new generation. Next process restart auto-detects the new identity. Mismatches between active env and DB-stored identity are logged as warnings on connection so you never silently mix vectors from different models.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ MCP client (Claude Code / Codex / Cursor / Claude API)  │
└────────────────┬────────────────────────────────────────┘
                 │ stdio (JSON-RPC)
       ┌─────────▼─────────┐
       │  server.py        │  ← FastMCP tool surface
       │  (≈20 tools)      │
       └─────┬──────┬──────┘
             │      │
   ┌─────────▼──┐   │   ┌────────────────┐
   │ unified.py │   └──▶│ intelligence.py│  ← record_finding pipeline
   │ (recall)   │       └────────┬───────┘
   └─────┬──────┘                │
         │                       │
   ┌─────▼─────────┬─────────────▼─────┬──────────────┐
   │ vector_store  │ knowledge_graph   │ wiki         │
   │ (embeddings)  │ (nodes+edges+PR)  │ (crystallize)│
   └─────┬─────────┴────────┬──────────┴──────┬───────┘
         │                  │                 │
         └────────┬─────────┴─────────────────┘
                  ▼
         ┌─────────────────┐
         │ DuckDB + VSS    │  ← single file per workspace
         │ <ws>/.memory-graph/memory.duckdb
         └─────────────────┘
```

Lifecycle: `draft → canonical → superseded`. Auto-promote at `reuse_count ≥ 3`. Canonical promotion auto-triggers wiki crystallization.

---

## Development

```bash
git clone https://github.com/brayanfz013/memory-graph-mcp.git
cd memory-graph-mcp
uv pip install -e ".[dev]"

# Run tests
pytest

# Run the server in stdio mode (for manual MCP client testing)
memory-graph

# Lint
ruff check memory_graph/
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Built on [Model Context Protocol](https://modelcontextprotocol.io), [DuckDB](https://duckdb.org) (with VSS extension for vector search), and [fastembed](https://github.com/qdrant/fastembed) for local embeddings.

Protocol inspired by working patterns from [Anthropic Claude Code](https://claude.com/claude-code), [OpenAI Codex](https://github.com/openai/codex), and observation of multi-agent collaboration anti-patterns. Trajectory-distillation extensions on the [v0.5.0 roadmap](CHANGELOG.md) draw on Google Research's [ReasoningBank](https://github.com/google-research/reasoning-bank) (arXiv:2509.25140).

The v0.5.0 knowledge-organization features (hierarchical topic mind map, outline-first wiki pages, source grounding, and the `memory_gaps` coverage critic) adapt ideas from Stanford OVAL's [STORM / Co-STORM](https://github.com/stanford-oval/storm) (MIT) — the *concepts*, re-expressed locally with no LLM and no `dspy` dependency. See [docs/INSPIRATION.md](docs/INSPIRATION.md) for what was adopted and what was left out.
