# Changelog

All notable changes to `memory-graph-mcp` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] — 2026-06-28

Reliability fix. The MCP server crashed at **import time** — before `main()`'s try/except could run — whenever it was launched from a directory that failed workspace validation. This left Claude (and other agents) unable to connect to the server at all in those directories, with `MCP error -32000: Connection closed` in the logs.

### Fixed (server failed to start outside project roots)

- **`memory_graph/settings.py`**: workspace resolution no longer raises during module import. `MemoryGraphSettings`'s `db_path` / `lock_path` / `workspace_path` defaults previously called the strict `resolve_workspace_path()`, which raised `WorkspaceResolutionError` for the home directory (`refusing to use broad path`), editor install dirs, and any folder without a project marker (`.git`, `pyproject.toml`, etc.). Because `settings = MemoryGraphSettings()` runs at import, the exception killed the whole process before the server could speak JSON-RPC, so the entire MCP connection dropped.
- New `resolve_storage()` degrades gracefully: a valid project root still gets workspace-scoped storage at `<workspace>/.memory-graph/`; any other directory falls back to a per-path global dir at `~/.memory-graph/fallback/<name>-<hash>` and logs a `WARNING` to stderr instead of crashing. The strict `resolve_workspace_path()` is retained (and still raises) for callers that want it; the security intent — never silently writing memory into the home root or an editor install dir — is preserved by the fallback location.

### Why this needed a release

Users launching Claude Code from `~` or from a folder without project markers saw memory-graph fail to connect on every session, and no memory was recorded. Storage now always resolves to a writable location, so the server starts everywhere.

### Tests

- Added `GracefulStorageFallbackTests` covering project-root scoping and the no-raise markerless fallback. Full suite: 53 passing.

## [0.4.3] — 2026-05-19

Plugin install fix. `/plugin install` was failing schema validation against the official Claude Code plugin manifest because v0.4.2 used fields that do not exist in the spec.

### Fixed (plugin install — v0.4.2 was unusable)

- **`.claude-plugin/plugin.json`**: removed the non-spec `components` wrapper, `requirements` block, and `installNotes` field. The MCP server, skills, and commands are now picked up via Claude Code's default auto-discovery from `.mcp.json`, `skills/`, and `commands/` respectively (per [Plugins reference — file locations](https://code.claude.com/docs/en/plugins-reference#file-locations-reference)). Added `$schema` for editor validation.
- **`.claude-plugin/marketplace.json`**: changed `"source": "."` to `"source": "./"` (the spec requires relative paths to start with `./`). Removed the non-spec `owner.url` field.

### Why this needed a release

`/plugin marketplace add` worked because the marketplace manifest is read on registration, but `/plugin install` validates the full plugin manifest before copying it into `~/.claude/plugins/cache`, and the schema mismatch caused a hard failure. Users on v0.4.2 saw the marketplace registered but no plugin installable.

### How to upgrade if you already added the v0.4.2 marketplace

```text
/plugin marketplace update memory-graph-marketplace
/plugin install memory-graph@memory-graph-marketplace
```

`marketplace update` re-fetches the manifest from this repo's HEAD; `install` then reads the new, spec-compliant `plugin.json`.

## [0.4.2] — 2026-05-19

Zero-install plugin distribution + security & reliability hardening. The plugin now bootstraps itself via `uvx` straight from a pinned commit SHA in this repo, so end users no longer need to `pip install memory-graph-mcp` before `/plugin install`.

### Changed

- **`.mcp.json`**: `command` switched from `"memory-graph"` (required pre-installed binary) to `"uvx"` with `--from git+https://github.com/brayanfz013/memory-graph-mcp@<commit-sha>`. We pin a **full commit SHA** rather than a tag name because git tags are mutable — pinning the SHA makes the install reference immutable and prevents tag-takeover attacks from silently delivering new code on subsequent installs. Single-command install: `/plugin marketplace add brayanfz013/memory-graph-mcp` + `/plugin install memory-graph@memory-graph-marketplace`. Requires `uv` on PATH; first launch downloads + caches the wheel (≈30–60 s), subsequent launches use uv's cache.
- **README**: "As a Claude Code plugin" section rewritten to reflect the new flow and call out the `uv` prerequisite up front.

### Fixed (security)

- **CI hardening**: added `permissions: contents: read` to [`.github/workflows/ci.yml`](.github/workflows/ci.yml) so the workflow runs with least privilege and PR runs from forks cannot escalate to write scopes on the `GITHUB_TOKEN`.
- **Supply-chain integrity**: `.mcp.json` now references a commit SHA, not a tag. Even if a maintainer key is compromised, the install reference cannot be silently retargeted via `git push --force <tag>`.

### Fixed (reliability)

- **Schema migration order**: in [`memory_graph/db.py`](memory_graph/db.py) `_migrate_schema`, the `< 4` block now runs before the `< 5` block so upgrades from v0.3.x databases apply the wiki/canonical migration before the embedding-meta migration. Previously a partial failure during the v4 step could leave the DB at a half-applied state.
- **Startup error visibility**: [`memory_graph/server.py`](memory_graph/server.py) `main()` is now wrapped in a top-level `try/except` that writes a human-readable error message to stderr and exits with code 1. Previously a `WorkspaceResolutionError` (e.g., `CLAUDE_PROJECT_DIR` unset or not a directory) caused the MCP server to die before logging anything, leaving users with an opaque "MCP server failed to start" in Claude Code.
- **fastembed cold-start UX**: [`memory_graph/embeddings.py`](memory_graph/embeddings.py) `_FastEmbedProvider.__init__` now logs the model-name + cache directory + expected duration *before* the ONNX model download starts, and wraps the download in a `try/except` that re-raises with a clear actionable message naming common causes (offline, corporate proxy, partial cache). Previously a stalled download or proxy failure surfaced only as a generic exception with no user guidance.

### Notes

- The `memory-graph` CLI entry point still works for users who prefer `pip install` — point your MCP client at the binary in PATH and the plugin behavior is identical.
- **First-run network requirement**: on first use in any workspace, the server downloads the BGE-small-en-v1.5 ONNX model (~100 MB) from HuggingFace. Requires `https://huggingface.co` reachable. Subsequent starts use the local cache at `~/.cache/fastembed` and require no network access.
- On every release, bump the SHA reference inside `.mcp.json` so `uvx` pins to a known immutable commit.

## [0.4.1] — 2026-05-19

Pluggable embedding providers + identity tracking + benchmark harness. Lets users swap between local fastembed models, local Ollama, or Google Vertex AI safely without losing stored content.

Added:

- **Three embedding providers behind a uniform interface**: `fastembed` (local ONNX, default), `ollama` (local HTTP API), `vertex` (Google Cloud). Each exposes `provider / model_name / dimensions / embed_texts / embed_query`.
- **`PROVIDER_REGISTRY`** in `settings.py` listing supported models with `{dim, lang, note}` metadata. fastembed: BGE-small-en (384), BGE-base-en (768), paraphrase-multilingual-mpnet-base-v2 (768, multi), intfloat/multilingual-e5-large (1024, multi), jinaai/jina-embeddings-v2-base-es (768, ES), paraphrase-multilingual-MiniLM-L12-v2 (384, multi). Ollama: nomic-embed-text (768), mxbai-embed-large (1024). Vertex: text-embedding-005 (768).
- **Schema v5 — `embedding_meta` table** tracking provider, model_name, dimensions, generation, is_active. Auto-seeds on first connection; mismatch with active env is logged as a warning so you never silently mix vector spaces.
- **`embedding_status()` MCP tool** — reports active env identity, DB-stored identity, mismatch flag, count of embeddings at risk, full registry, and actionable guidance.
- **`embedding_migrate(target_provider, target_model, dry_run, batch_size)` MCP tool** — re-embeds every memory + wiki page under a new provider/model, recreates vector tables with the right dimensions, records a new generation in `embedding_meta`. Default `dry_run=True` returns the plan without rewriting.
- **`embedding_benchmark(providers, eval_set_path, top_k)` MCP tool** — runs each (provider, model) combo against `eval/eval_set.json` in an isolated temp workspace and reports Recall@1, Recall@5, MRR, mean_latency_ms, cold_warmup_ms.
- **Bundled `eval/eval_set.json`** — 12 seeds + 13 queries across English and Spanish; extensible for your own corpus.
- **`tests/test_embedding_providers.py`** — provider registry checks, FastEmbed identity contract, Ollama unreachable-error contract, embedding_meta auto-seed, migrate dry-run plan validation, benchmark end-to-end.
- **README + PROTOCOL.md docs** for the swap workflow.

## [Unreleased] — v0.5.0 roadmap

Planned (inspired by Google Research's [ReasoningBank](https://github.com/google-research/reasoning-bank), arXiv:2509.25140):

- **`distill_trajectory(steps, outcome)` tool** — LLM-as-judge converts a sequence of agent actions into 1–3 distilled reasoning items via `record_finding`.
- **New finding types: `strategy` and `pitfall`** — first-class strategy-level reasoning and negative signals from failed trajectories.
- **`judgment` column + `judge_finding(node_id, evidence)` tool** — explicit Success/Failure verdict that populates `confidence` (today derived only from `reuse_count`).
- **`recall_with_contrast(query, n)`** — MaTTS-style retrieval grouping items by agreement/disagreement for agent self-contrast.
- **`trajectories` table** — auditable evidence linking distilled items back to their source execution.
- **Time-decay on PageRank** — older nodes lose influence unless reused.
- **Conflict detection** — flag contradictory findings on overlapping topics.

## [0.4.0] — 2026-05-08

Major refactor (v0.3.0 → v0.4.0): unified semantic recall + auto-edges + integrated wiki + tool-surface consolidation.

### Added
- **`unified.recall(query, scope, ...)`** — single fused recall across `memories + KG + wiki` by semantic similarity. Returns `top_canonicals` ready to feed into `wiki_get`.
- **`intelligence._auto_infer_edges`** — after every `record_finding`, embeds the new node and creates up to 3 `RELATED_TO` edges to existing nodes with cosine ≥ 0.62.
- **Auto-crystallize on canonical promotion** — when a node transitions to `canonical`, a wiki page is generated automatically.
- **`wiki_get(canonical_id_or_title)`** — fetch a wiki page in full (no truncation) plus its linked KG node.
- **`wiki_bootstrap`** now also crystallizes the top 200 KG nodes, not just repo structure.
- **DuckDB schema v4** with `wiki_pages` + `wiki_embeddings` created at init (not lazily) plus NULL-safe defaults for `status / reuse_count / confidence`.

### Changed
- **`kg_query`** now embeddings-first; `ILIKE` is fallback only when semantic results are below threshold.
- **Tool surface consolidated** from 19 → 3 primary (`recall`, `record_finding`, `wiki_get`) + 3 navigation + 3 lifecycle + 5 collective/cache + 4 health. Legacy primitives (`memory_recall`, `kg_query`, `kg_influential`) kept exposed for backward compatibility.
- **`kg_compute_pagerank`** is now run automatically inside `memory_consolidate(dry_run=False)`.
- **`wiki_lint`** is now included in `memory_report()` output instead of being a standalone tool.
- **`memory_record_finding`** kept as an alias for `record_finding` (legacy callers don't break).

### Fixed
- `canonical_id` was occasionally `NULL` on raw `kg_add_node` calls — now auto-generated via `make_canonical_id` when missing.
- `reuse_count` was `NULL` for older rows, blocking auto-promotion — backfilled to `0` and given a default at INSERT.
- Doubled prefixes like `solution.solution.x` were repaired in the v3 → v4 migration.
- `_bfs_expand` now selects all 11 perfector columns so `canonical_id` is never lost during graph traversal.

### Packaging (this release)
- Added `LICENSE` (Apache 2.0).
- Added `README.md`, `PROTOCOL.md`, `CHANGELOG.md`.
- Added `.claude-plugin/plugin.json` + bundled `skills/memory-graph/SKILL.md` + `commands/memory-recall.md` so this can install as a Claude Code plugin.
- Added `examples/` with quickstart and `.mcp.json` sample.
- Expanded `.gitignore` to exclude `.memory-graph/`, `*.egg-info/`, `*.duckdb*`, `.pytest_cache/`.
- Enriched `pyproject.toml` with author, license, classifiers, URLs, dev extras, and a second `memory-graph-mcp` script alias.

## [0.3.1] — 2026-04 (Perfector)

- Canonical dedup, wiki layer, lifecycle states (`draft / canonical / superseded`), multi-level compression (`tldr_32 / brief_96 / summary_256`).
- Cross-agent `collective_*` + `cache_*` tools.
- Workspace-scoped DuckDB with file lock for concurrent-writer safety.

## [0.3.0] and earlier

Initial vector-memory + KG-only design (pre-wiki). See git history for details.
