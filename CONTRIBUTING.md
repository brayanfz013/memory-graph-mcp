# Contributing to memory-graph-mcp

Thanks for taking the time to consider contributing. This document covers the dev setup, the standards your change has to clear before merge, and how releases work.

## Dev setup

```bash
git clone https://github.com/brayanfz013/memory-graph-mcp.git
cd memory-graph-mcp
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                          # 48+ tests, ~30s
uvx ruff check memory_graph tests examples
```

The MCP server runs over stdio:

```bash
memory-graph                    # waits for JSON-RPC on stdin
```

For end-to-end manual testing without an MCP client, see [`examples/quickstart.py`](examples/quickstart.py).

## Architecture in 60 seconds

Read [`PROTOCOL.md`](PROTOCOL.md) first — that documents the three-pillar usage philosophy (recall → record → link) the whole project is built around.

Then [`README.md`](README.md) for tool surface and architecture diagram.

Module map:
- `memory_graph/server.py` — MCP tool surface (FastMCP)
- `memory_graph/settings.py` — config + `PROVIDER_REGISTRY`
- `memory_graph/db.py` — DuckDB schema + migrations + connection
- `memory_graph/embeddings.py` — pluggable providers (fastembed, ollama, vertex)
- `memory_graph/vector_store.py` — semantic memory CRUD
- `memory_graph/knowledge_graph.py` — KG nodes + typed edges + PageRank
- `memory_graph/wiki.py` — long-form crystallized pages
- `memory_graph/unified.py` — fused `recall()` across memories+KG+wiki
- `memory_graph/intelligence.py` — `record_finding` pipeline (vector + KG + auto-edges + auto-crystallize)
- `memory_graph/embedding_admin.py` — `embedding_status`, `embedding_migrate`
- `memory_graph/benchmark.py` — Recall@k harness
- `memory_graph/collective.py` — cross-agent state
- `memory_graph/tool_cache.py` — tool memoization

## Definition of done for a PR

Your change is mergeable when:

1. **Tests pass**: `pytest` → all green. Add tests for behavior changes; the [`tests/`](tests/) folder has examples of the isolation pattern (`MemoryGraphTestCase` with `addCleanup`). Don't skip the parent-package pop — see the inline comments in `setUp` for the trap that catches new test code.
2. **Lint passes**: `uvx ruff check memory_graph tests examples` → no errors.
3. **No accidental coupling to a specific embedding provider**. New code should call `embed_query` / `embed_texts` from `memory_graph.embeddings` — never import a provider class directly.
4. **No new dependencies** unless you justify them in the PR description. Local-first is a design value: `fastembed` is the default for a reason.
5. **CHANGELOG entry** under `[Unreleased]` describing what changed and why. Keep it terse.
6. **No secrets / PII / hardcoded internal identifiers** anywhere. The repo is public Apache-2.0; treat everything you commit as world-readable.

## How the test isolation pattern works (read before adding tests)

`MemoryGraphTestCase` in [`tests/test_end_to_end.py`](tests/test_end_to_end.py) sets up per-test isolation by:
1. Saving the previous `MEMORY_GRAPH_WORKSPACE` env var.
2. Creating a temp workspace, registering `shutil.rmtree` via `addCleanup`.
3. Setting the env var to the new workspace.
4. Popping every `memory_graph.*` submodule **AND the `memory_graph` package itself** from `sys.modules`.
5. Re-importing fresh.

Step 4's package-pop is non-obvious but critical. Python caches sub-module references as attributes on the package, so `from . import vector_store as vs_mod` returns the stale module via the package attribute even after `sys.modules['memory_graph.vector_store']` was popped. Without popping the package, tests pass in isolation but fail when run as a suite. There's a comment in the setUp explaining this.

## Adding an embedding provider

1. Add the provider class to `memory_graph/embeddings.py`. It must expose `provider`, `model_name`, `dimensions` attributes and `embed_texts`/`embed_query` methods.
2. Wire it in `_build_provider()`.
3. Add an entry to `PROVIDER_REGISTRY` in `memory_graph/settings.py` with `{dim, lang, note}`.
4. Add tests under `tests/test_embedding_providers.py` mirroring the existing FastEmbed / Ollama patterns.
5. If the provider requires extra deps, declare them under `[project.optional-dependencies]` in `pyproject.toml`.
6. Document the new provider in `README.md` and `PROTOCOL.md`.

## Release flow (maintainers)

1. Update `version` in [`pyproject.toml`](pyproject.toml), [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json), [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json), and the log line in [`memory_graph/server.py`](memory_graph/server.py).
2. Move the `[Unreleased]` section in [`CHANGELOG.md`](CHANGELOG.md) under a new `[X.Y.Z] — YYYY-MM-DD` header.
3. Commit: `chore(release): vX.Y.Z`.
4. Tag: `git tag vX.Y.Z && git push origin main --tags`.
5. CI builds and validates. To publish to PyPI manually:
   ```bash
   uv pip install build twine
   python -m build
   twine upload dist/*
   ```

## Code of conduct

Be civil. Critique the code, not the person. If a contributor needs context to learn something, give them that context — that's how the project grows.

## License

By contributing you agree your changes are licensed under Apache License 2.0 (see [LICENSE](LICENSE)).
