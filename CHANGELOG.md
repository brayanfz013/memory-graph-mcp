# Changelog

All notable changes to `memory-graph-mcp` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
