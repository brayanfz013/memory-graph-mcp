"""memory-graph MCP server — unified memory layer.

High-level tool surface (v0.5.1):

  Primary (use these by default):
    - recall(query, scope, ...)         — semantic search across memories+KG+wiki
    - record_finding(...)               — store + index + auto-edges + auto-crystallize
    - wiki_get(canonical_id_or_title)   — fetch a wiki page in full

  Graph navigation:
    - kg_neighbors(node_id, ...)        — immediate neighbors in the graph
    - kg_path(from_id, to_id, ...)      — shortest path between two nodes
    - kg_resolve(canonical_id)          — lookup node by stable slug

  Lifecycle / advanced:
    - kg_promote(node_id, status)       — draft → canonical → superseded
    - kg_add_edge(from, to, rel_type)   — manual edge override (auto-edges
                                          are created by record_finding)
    - wiki_ingest(title, body, ...)     — manual wiki authoring

  Cross-agent state:
    - collective_store / collective_get / collective_list
    - cache_check / cache_store

  Topic map + coverage:
    - memory_map(rebuild, top_k)        — hierarchical topic mind map of the KG
    - memory_gaps(limit)                — coverage critic + recommended actions

  Health / admin:
    - memory_report()                   — counts + types + top influential
    - memory_consolidate(dry_run)       — purge expired entries + rebuild topics
    - wiki_bootstrap(force, max_dirs)   — seed wiki from repo + canonicals

  Embedding provider administration:
    - embedding_status()                — active vs stored provider + diff
    - embedding_migrate(provider, model, dry_run) — re-embed under new model
    - embedding_benchmark(providers)    — score Recall@k against eval set

Background-only (NOT exposed as tools):
    - kg_compute_pagerank               — fired automatically after edges land
    - wiki_lint                         — included in memory_report

Search modes:
    All search tools use embeddings as primary signal (semantic), falling
    back to text ILIKE only when embeddings yield no above-threshold match.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import (
    benchmark,
    collective,
    embedding_admin,
    intelligence,
    knowledge_graph,
    tool_cache,
    topics,
    unified,
    wiki,
)
from .db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("memory-graph")

mcp = FastMCP(
    "memory-graph",
    instructions=(
        "Unified semantic memory for AI agents. "
        "Use `recall` for cross-scope semantic search (memories+KG+wiki) — pass "
        "compact=True to cut tokens (dedupes facts, trims redundant blobs) and "
        "group_topics=True to see results clustered by theme. "
        "`record_finding` persists new knowledge with auto-edges; pass "
        "sources=[file:line, URL, commit] to ground claims. "
        "`wiki_get` fetches a page — use outline_only=True or section='…' to "
        "load just what you need. Call `memory_primer` once at task start to "
        "orient (top topics + key canonicals + coverage, token-cheap) — ideal "
        "after a long chat is compacted. `memory_map` shows the topic mind map; "
        "`memory_gaps` reports coverage holes with recommended actions. "
        "`kg_neighbors`/`kg_path` traverse relationships; `collective_*`/"
        "`cache_*` share state across agents. Lower-level primitives "
        "(kg_add_edge, wiki_ingest, kg_promote) remain available but "
        "`record_finding` covers most write paths automatically."
    ),
)


# ── PRIMARY: Unified semantic recall ──────────────────────────────


@mcp.tool()
def recall(
    query: str,
    scope: str = "all",
    top_k: int = 10,
    min_score: float = 0.45,
    node_type: str | None = None,
    hops: int = 1,
    compact: bool = False,
    group_topics: bool = False,
) -> dict[str, Any]:
    """Semantic search across memories, KG nodes, and wiki pages.

    scope: 'memories' | 'kg' | 'wiki' | 'all' (default).
    Returns fused, ranked results plus `top_canonicals` (most-relevant
    canonical_ids across scopes — feed these to `wiki_get` for detail).

    Token-saving options (recommended for large stores):
      compact=True       — drop redundant raw content from KG nodes, shorten
                           wiki snippets, and remove memories already covered by
                           a returned KG node. Same shape, far fewer tokens.
      group_topics=True  — add a `topics` block clustering the KG hits by theme
                           (the mind map) so you see structure, not a flat list.

    Prefer this over the legacy memory_recall/kg_query/wiki_query trio.
    """
    return unified.recall(
        query=query, scope=scope, top_k=top_k,
        min_score=min_score, node_type=node_type, hops=hops,
        compact=compact, group_topics=group_topics,
    )


# ── PRIMARY: Record findings (write) ──────────────────────────────


@mcp.tool()
def record_finding(
    finding_type: str,
    title: str,
    content: str,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
    source_agent: str | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a finding — vector memory + KG node + auto-edges + auto-crystallize.

    finding_type: solution | decision | insight | problem | pattern | context.
    Pipeline:
      1. memory_store (with embedding) for semantic recall.
      2. kg_add_node with deduplicated canonical_id.
      3. auto-infers up to 3 RELATED_TO edges to semantically similar nodes.
      4. auto-promotes draft → canonical when reuse_count ≥ 3.
      5. auto-crystallizes a wiki page when the node becomes canonical.

    `sources` are grounding anchors (file:line, URLs, commit SHAs) rendered as a
    numbered Sources section when the page crystallizes. Grounded findings are
    more trustworthy and keep `memory_gaps` from flagging them as unproven.

    Replaces the older memory_record_finding (kept as alias).
    """
    return intelligence.memory_record_finding(
        finding_type, title, content, related_files, tags, source_agent, sources,
    )


# Legacy alias retained for callers still using the old name.
@mcp.tool()
def memory_record_finding(
    finding_type: str,
    title: str,
    content: str,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
    source_agent: str | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Alias for `record_finding` (kept for backward compatibility)."""
    return intelligence.memory_record_finding(
        finding_type, title, content, related_files, tags, source_agent, sources,
    )


# ── PRIMARY: Wiki access ──────────────────────────────────────────


@mcp.tool()
def wiki_get(
    canonical_id_or_title: str,
    outline_only: bool = False,
    section: str | None = None,
) -> dict[str, Any]:
    """Fetch a wiki page by canonical_id (preferred) or exact title.

    Default returns the full body. Token-saving modes:
      outline_only=True — title + preamble + section headings (`outline`) only,
                          to decide if the page is worth loading in full.
      section="Summary" — return just that one section's body.

    Use after `recall` to expand a `top_canonicals[i]` into its full doc.
    """
    return wiki.wiki_get(canonical_id_or_title, outline_only=outline_only, section=section)


# ── Topic mind map + coverage critic ──────────────────────────────


@mcp.tool()
def memory_primer(topics_k: int = 8, canonicals_k: int = 8) -> dict[str, Any]:
    """Compact session orientation — call ONCE at task start to ground yourself.

    Returns a token-cheap snapshot: top topics (mind map), most influential
    canonical knowledge, a one-line coverage summary, and store size — labels
    and ids only, no bodies. Then drill in with recall(compact=True) and
    wiki_get(outline_only=True). Ideal for a SessionStart hook so context never
    starts cold, even after a long chat is compacted.
    """
    return intelligence.memory_primer(topics_k=topics_k, canonicals_k=canonicals_k)


@mcp.tool()
def memory_map(rebuild: bool = False, top_k: int = 20) -> dict[str, Any]:
    """Hierarchical topic map of the knowledge graph (Co-STORM-style mind map).

    Clusters KG nodes into topics → subtopics over the RELATED_TO affinity
    graph (no LLM). Reads the persisted map; pass rebuild=True to recompute.
    Call this to orient by theme before a deep `recall`.
    """
    return topics.memory_map(rebuild=rebuild, top_k=top_k)


@mcp.tool()
def memory_gaps(limit: int = 20) -> dict[str, Any]:
    """Coverage critic — where the knowledge graph is thin and what to do.

    Surfaces isolated nodes, ungrounded canonicals, missing wiki pages,
    promote-ready drafts, stale canonicals, and orphan wiki pages, each with a
    concrete recommended action. Deterministic and cheap (no LLM).
    """
    return intelligence.memory_gaps(limit)


# ── Graph navigation ──────────────────────────────────────────────


@mcp.tool()
def kg_neighbors(
    node_id: str,
    direction: str = "both",
    rel_type: str | None = None,
) -> dict[str, Any]:
    """Immediate neighbors of a KG node (optionally filtered by edge type)."""
    return intelligence.kg_neighbors(node_id, direction, rel_type)


@mcp.tool()
def kg_path(
    from_id: str,
    to_id: str,
    max_depth: int = 5,
) -> dict[str, Any]:
    """Shortest path between two KG nodes (BFS, undirected)."""
    return intelligence.kg_path(from_id, to_id, max_depth)


@mcp.tool()
def kg_resolve(canonical_id: str) -> dict[str, Any]:
    """Resolve a node by stable canonical_id slug."""
    return knowledge_graph.kg_resolve(canonical_id)


# ── Lifecycle / advanced writes ───────────────────────────────────


@mcp.tool()
def kg_promote(node_id: str, status: str) -> dict[str, Any]:
    """Transition KG node lifecycle: draft → canonical → superseded.

    Promotion to canonical also triggers wiki crystallization in the
    next record_finding call (or run wiki_bootstrap to backfill).
    """
    return knowledge_graph.kg_promote(node_id, status)


@mcp.tool()
def kg_add_edge(
    from_id: str,
    to_id: str,
    rel_type: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Manual edge override.

    Valid relations: SOLVES, CAUSED_BY, DEPENDS_ON, RELATED_TO, USES_TOOL,
    SUPERSEDES. record_finding auto-infers RELATED_TO edges; use this for
    typed edges (SOLVES, CAUSED_BY, etc.) the system can't infer.
    """
    return knowledge_graph.kg_add_edge(from_id, to_id, rel_type, weight)


@mcp.tool()
def wiki_ingest(
    title: str,
    body: str,
    canonical_id: str | None = None,
    tags: list[str] | None = None,
    author: str = "agent",
) -> dict[str, Any]:
    """Author a wiki page manually (auto-embedded for semantic recall).

    Most cases should let record_finding crystallize wikis automatically.
    Use this only when you need long-form, hand-written documentation.
    """
    return wiki.wiki_ingest(title, body, canonical_id, tags, author)


# ── Cross-agent state ─────────────────────────────────────────────


@mcp.tool()
def collective_store(
    type: str,
    key: str,
    value: Any,
    scope: str = "global",
) -> dict[str, Any]:
    """Share a value across agents with type-based TTL.

    Types: knowledge|result|consensus|system (permanent),
    context (1h), task (30m), error (24h), metric (1h).
    Scopes: global | project | agent.
    """
    return collective.collective_store(type, key, value, scope)


@mcp.tool()
def collective_get(key: str, scope: str = "global") -> dict[str, Any] | None:
    """Retrieve a collective entry. Null if expired or missing."""
    return collective.collective_get(key, scope)


@mcp.tool()
def collective_list(
    type: str | None = None,
    scope: str = "global",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List collective entries (most recently accessed first)."""
    return collective.collective_list(type, scope, limit)


@mcp.tool()
def cache_check(tool_name: str, args_hash: str) -> dict[str, Any]:
    """Check for a cached result before calling an expensive tool."""
    return tool_cache.cache_check(tool_name, args_hash)


@mcp.tool()
def cache_store(
    tool_name: str,
    args_hash: str,
    result: str,
    ttl_seconds: int = 3600,
) -> dict[str, Any]:
    """Cache a tool call result."""
    return tool_cache.cache_store(tool_name, args_hash, result, ttl_seconds)


# ── Health / admin ────────────────────────────────────────────────


@mcp.tool()
def memory_report() -> dict[str, Any]:
    """Health report: counts, top influential nodes, expired entries pending cleanup, wiki lint summary."""
    report = intelligence.memory_report()
    try:
        report["wiki_lint"] = wiki.wiki_lint()
    except Exception as exc:
        report["wiki_lint_error"] = str(exc)
    return report


@mcp.tool()
def memory_consolidate(dry_run: bool = True) -> dict[str, Any]:
    """Remove expired entries + orphaned data + recompute PageRank.

    Default is dry_run=True. Set dry_run=False to actually delete.
    """
    result = intelligence.memory_consolidate(dry_run)
    if not dry_run:
        try:
            pagerank = knowledge_graph.kg_compute_pagerank()
            result["pagerank"] = pagerank
        except Exception as exc:
            result["pagerank_error"] = str(exc)
    return result


@mcp.tool()
def memory_stats() -> dict[str, Any]:
    """Quick row counts for all tables."""
    with get_connection() as conn:
        stats: dict[str, Any] = {}
        tables = [
            "memories", "memory_embeddings", "kg_nodes", "kg_edges",
            "wiki_pages", "wiki_embeddings", "collective_memory", "tool_cache",
        ]
        for table in tables:
            try:
                count = (conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() or (0,))[0]  # noqa: S608
            except Exception:
                count = -1
            stats[table] = count
        stats["total"] = sum(v for v in stats.values() if v >= 0)
        return stats


# ── Embedding provider administration ────────────────────────────


@mcp.tool()
def embedding_status() -> dict[str, Any]:
    """Show the active embedding provider, what is stored in DB, and any mismatch.

    Returns: active_env (current provider/model/dim), stored_in_db (whatever
    was set the last time the workspace was written), mismatch flag,
    embeddings_at_risk count (rows that would need re-embedding), and the
    PROVIDER_REGISTRY listing every supported (provider, model, dim, lang).

    Use this before swapping providers to know whether a migration is needed.
    """
    return embedding_admin.embedding_status()


@mcp.tool()
def embedding_migrate(
    target_provider: str,
    target_model: str,
    dry_run: bool = True,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Re-embed every memory + wiki page under a new provider/model.

    Default is dry_run=True (returns the plan + estimated work).
    Pass dry_run=False to actually rewrite. Source content is untouched —
    only the vectors are recreated with the new model's dimensions.

    target_provider: 'fastembed' | 'ollama' | 'vertex'.
    target_model:    a model name listed in PROVIDER_REGISTRY for that provider.
    """
    return embedding_admin.embedding_migrate(
        target_provider=target_provider,
        target_model=target_model,
        dry_run=dry_run,
        batch_size=batch_size,
    )


@mcp.tool()
def embedding_benchmark(
    providers: list[dict[str, str]] | None = None,
    eval_set_path: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Score multiple (provider, model) combos against the bundled eval set.

    Each combo runs in an ISOLATED temp workspace — your real DB is never touched.
    Returns per-combo Recall@1, Recall@5, MRR, mean_latency_ms, cold_warmup_ms.

    Default `providers` covers 3 fastembed models (BGE-small, BGE-base,
    multilingual-e5-base). Pass your own list to compare Ollama / Vertex.
    """
    return benchmark.embedding_benchmark(
        providers=providers,
        eval_set_path=eval_set_path,
        top_k=top_k,
    )


@mcp.tool()
def wiki_bootstrap(force: bool = False, max_dirs: int = 30) -> dict[str, Any]:
    """Seed wiki from workspace structure + crystallize all canonical KG nodes.

    Idempotent. Re-run safely. Use after first install or after a large batch
    of new canonicals lands.
    """
    return wiki.wiki_bootstrap(force, max_dirs)


# ── Legacy / low-level primitives (kept for compatibility) ────────
# These are still exposed but `recall` and `record_finding` should be
# preferred. They are useful for narrow operations or migrations.


@mcp.tool()
def memory_recall(
    query: str,
    type_filter: str | None = None,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Legacy: semantic recall over memories only. Prefer `recall(scope='memories')`."""
    from . import vector_store as vs
    return vs.memory_recall(query, type_filter, top_k, min_score)


@mcp.tool()
def kg_query(
    query: str,
    node_type: str | None = None,
    hops: int = 1,
    top_k: int = 10,
) -> dict[str, Any]:
    """Legacy: semantic KG search with BFS expansion. Prefer `recall(scope='kg')`."""
    return knowledge_graph.kg_query(query, node_type, hops, top_k)


@mcp.tool()
def kg_influential(
    top_k: int = 10,
    node_type: str | None = None,
) -> list[dict[str, Any]]:
    """Top KG nodes by PageRank score (optionally filtered by node_type)."""
    return intelligence.kg_influential(top_k, node_type)


# ── Entry point ───────────────────────────────────────────────────


def main() -> None:
    """Run the memory-graph MCP server over stdio transport.

    Wrapped in try/except so any startup failure (workspace resolution,
    DuckDB init, embedding provider) surfaces to the user via stderr
    instead of dying silently with no JSON-RPC response.
    """
    import sys
    try:
        logger.info("Starting memory-graph MCP server v0.4.4 (unified recall + auto-edges + wiki integration + pluggable embedding providers)")
        mcp.run(transport="stdio")
    except Exception as exc:  # noqa: BLE001 — top-level catch is the point
        sys.stderr.write(f"\n[memory-graph] FATAL startup error: {type(exc).__name__}: {exc}\n")
        sys.stderr.write("[memory-graph] Common causes:\n")
        sys.stderr.write("  - MEMORY_GRAPH_WORKSPACE / CLAUDE_PROJECT_DIR not set or not a directory\n")
        sys.stderr.write("  - fastembed model download failed (offline / proxy / CDN unreachable)\n")
        sys.stderr.write("  - DuckDB database file locked or corrupt at <workspace>/.memory-graph/\n")
        sys.stderr.write("See https://github.com/brayanfz013/memory-graph-mcp#troubleshooting for details.\n")
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
