"""High-level intelligence tools — reports, consolidation, graph traversal.

These complement the low-level primitives in vector_store, knowledge_graph,
collective, and tool_cache with agent-friendly aggregate operations.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from .db import get_connection, with_retry

logger = logging.getLogger(__name__)

PROMOTION_REUSE_THRESHOLD = 3

AUTO_EDGE_MAX = 3
AUTO_EDGE_MIN_SCORE = 0.62


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to approximately max_words, ending at sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    for sep in (".", "!", "?", ";"):
        last = truncated.rfind(sep)
        if last > len(truncated) // 2:
            return truncated[: last + 1]
    return truncated + "…"


def _generate_tldr(title: str, content: str) -> str:
    """Generate ~32-word TLDR from title + first sentence of content."""
    first_sentence = content.split(". ")[0] if ". " in content else content
    combined = f"{title}: {first_sentence}."
    return _truncate_words(combined, 32)


def _generate_brief(title: str, content: str) -> str:
    """Generate ~96-word brief from title + first paragraph of content."""
    first_para = content.split("\n\n")[0] if "\n\n" in content else content
    combined = f"{title}: {first_para}"
    return _truncate_words(combined, 96)


def memory_report() -> dict[str, Any]:
    """Full health report of the memory system."""
    with get_connection() as conn:
        tables = {
            "memories": "SELECT COUNT(*) FROM memories",
            "kg_nodes": "SELECT COUNT(*) FROM kg_nodes",
            "kg_edges": "SELECT COUNT(*) FROM kg_edges",
            "collective": "SELECT COUNT(*) FROM collective_memory",
            "cache": "SELECT COUNT(*) FROM tool_cache",
        }
        counts: dict[str, int] = {}
        for name, sql in tables.items():
            counts[name] = (conn.execute(sql).fetchone() or (0,))[0]

        mem_types = conn.execute(
            "SELECT type, COUNT(*) FROM memories GROUP BY type ORDER BY COUNT(*) DESC"
        ).fetchall()

        node_types = conn.execute(
            "SELECT node_type, COUNT(*) FROM kg_nodes GROUP BY node_type ORDER BY COUNT(*) DESC"
        ).fetchall()

        top_nodes = conn.execute(
            "SELECT node_id, label, pagerank_score FROM kg_nodes ORDER BY pagerank_score DESC LIMIT 5"
        ).fetchall()

        coll_scopes = conn.execute(
            "SELECT scope, COUNT(*) FROM collective_memory GROUP BY scope"
        ).fetchall()

        active_cache = (conn.execute(
            "SELECT COUNT(*) FROM tool_cache WHERE expires_at > current_timestamp"
        ).fetchone() or (0,))[0]

        expired = conn.execute(
            """SELECT
                 (SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at < current_timestamp),
                 (SELECT COUNT(*) FROM collective_memory WHERE expires_at IS NOT NULL AND expires_at < current_timestamp),
                 (SELECT COUNT(*) FROM tool_cache WHERE expires_at < current_timestamp)"""
        ).fetchone() or (0, 0, 0)

        return {
            "counts": counts,
            "total_entries": sum(counts.values()),
            "memory_types": {r[0]: r[1] for r in mem_types},
            "kg_node_types": {r[0]: r[1] for r in node_types},
            "kg_top_nodes": [
                {"node_id": r[0], "label": r[1], "pagerank": round(float(r[2]), 6)}
                for r in top_nodes
            ],
            "collective_scopes": {r[0]: r[1] for r in coll_scopes},
            "cache_active": active_cache,
            "expired_pending": {
                "memories": expired[0],
                "collective": expired[1],
                "cache": expired[2],
            },
        }


def memory_consolidate(
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove expired entries and rebuild indexes.

    Set dry_run=False to actually delete. Default is preview only.
    """
    with get_connection() as conn:
        expired_memories = (conn.execute(
            "SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at < current_timestamp"
        ).fetchone() or (0,))[0]
        expired_collective = (conn.execute(
            "SELECT COUNT(*) FROM collective_memory WHERE expires_at IS NOT NULL AND expires_at < current_timestamp"
        ).fetchone() or (0,))[0]
        expired_cache = (conn.execute(
            "SELECT COUNT(*) FROM tool_cache WHERE expires_at < current_timestamp"
        ).fetchone() or (0,))[0]

        orphaned_embeddings = (conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE id NOT IN (SELECT id FROM memories)"
        ).fetchone() or (0,))[0]

        orphaned_edges = (conn.execute(
            """SELECT COUNT(*) FROM kg_edges
               WHERE from_id NOT IN (SELECT node_id FROM kg_nodes)
                  OR to_id NOT IN (SELECT node_id FROM kg_nodes)"""
        ).fetchone() or (0,))[0]

        result: dict[str, Any] = {
            "dry_run": dry_run,
            "would_remove": {
                "expired_memories": expired_memories,
                "expired_collective": expired_collective,
                "expired_cache": expired_cache,
                "orphaned_embeddings": orphaned_embeddings,
                "orphaned_edges": orphaned_edges,
            },
            "total": (
                expired_memories + expired_collective + expired_cache
                + orphaned_embeddings + orphaned_edges
            ),
        }

        if not dry_run:
            conn.execute(
                "DELETE FROM memory_embeddings WHERE id IN "
                "(SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < current_timestamp)"
            )
            conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < current_timestamp"
            )
            conn.execute(
                "DELETE FROM collective_memory WHERE expires_at IS NOT NULL AND expires_at < current_timestamp"
            )
            conn.execute(
                "DELETE FROM tool_cache WHERE expires_at < current_timestamp"
            )
            conn.execute(
                "DELETE FROM memory_embeddings WHERE id NOT IN (SELECT id FROM memories)"
            )
            conn.execute(
                """DELETE FROM kg_edges
                   WHERE from_id NOT IN (SELECT node_id FROM kg_nodes)
                      OR to_id NOT IN (SELECT node_id FROM kg_nodes)"""
            )
            result["status"] = "cleaned"
            logger.info("Consolidation: removed %d entries", result["total"])
        else:
            result["status"] = "preview"

        return result


def kg_neighbors(
    node_id: str,
    direction: str = "both",
    rel_type: str | None = None,
) -> dict[str, Any]:
    """Get immediate neighbors of a node in the knowledge graph."""
    with get_connection() as conn:
        node_row = conn.execute(
            "SELECT node_id, node_type, label, pagerank_score FROM kg_nodes WHERE node_id = ?",
            [node_id],
        ).fetchone()
        if not node_row:
            return {"error": f"Node not found: {node_id}"}

        edges: list[dict[str, Any]] = []
        neighbor_ids: set[str] = set()

        if direction in ("outgoing", "both"):
            params: list[Any] = [node_id]
            sql = "SELECT from_id, to_id, rel_type, weight FROM kg_edges WHERE from_id = ?"
            if rel_type:
                sql += " AND rel_type = ?"
                params.append(rel_type)
            for r in conn.execute(sql, params).fetchall():
                edges.append({"from_id": r[0], "to_id": r[1], "rel_type": r[2], "weight": r[3]})
                neighbor_ids.add(r[1])

        if direction in ("incoming", "both"):
            params = [node_id]
            sql = "SELECT from_id, to_id, rel_type, weight FROM kg_edges WHERE to_id = ?"
            if rel_type:
                sql += " AND rel_type = ?"
                params.append(rel_type)
            for r in conn.execute(sql, params).fetchall():
                edges.append({"from_id": r[0], "to_id": r[1], "rel_type": r[2], "weight": r[3]})
                neighbor_ids.add(r[0])

        neighbors: list[dict[str, Any]] = []
        if neighbor_ids:
            ph = ", ".join("?" for _ in neighbor_ids)
            rows = conn.execute(
                f"SELECT node_id, node_type, label, pagerank_score FROM kg_nodes WHERE node_id IN ({ph})",
                list(neighbor_ids),
            ).fetchall()
            neighbors = [
                {"node_id": r[0], "node_type": r[1], "label": r[2], "pagerank": round(float(r[3]), 6)}
                for r in rows
            ]

        return {
            "node": {"node_id": node_row[0], "node_type": node_row[1], "label": node_row[2]},
            "neighbors": sorted(neighbors, key=lambda n: n["pagerank"], reverse=True),
            "edges": edges,
            "count": len(neighbors),
        }


def kg_path(
    from_id: str,
    to_id: str,
    max_depth: int = 5,
) -> dict[str, Any]:
    """Find shortest path between two nodes via BFS."""
    with get_connection() as conn:
        found = conn.execute(
            "SELECT node_id FROM kg_nodes WHERE node_id IN (?, ?)", [from_id, to_id]
        ).fetchall()
        found_ids = {r[0] for r in found}
        if from_id not in found_ids:
            return {"error": f"Source node not found: {from_id}"}
        if to_id not in found_ids:
            return {"error": f"Target node not found: {to_id}"}
        if from_id == to_id:
            return {"path": [from_id], "edges": [], "length": 0}

        all_edges = conn.execute(
            "SELECT from_id, to_id, rel_type, weight FROM kg_edges"
        ).fetchall()
        adj: dict[str, list[tuple[str, str, str, str, float]]] = defaultdict(list)
        for src, tgt, rel, w in all_edges:
            adj[src].append((tgt, src, tgt, rel, w))
            adj[tgt].append((src, src, tgt, rel, w))

        visited = {from_id}
        queue: list[tuple[str, list[str], list[dict[str, Any]]]] = [
            (from_id, [from_id], [])
        ]

        while queue:
            current, path, edge_list = queue.pop(0)
            if len(path) > max_depth + 1:
                break

            for neighbor, orig_src, orig_tgt, rel, w in adj.get(current, []):
                edge_record = {
                    "from_id": orig_src,
                    "to_id": orig_tgt,
                    "rel_type": rel,
                    "weight": w,
                }
                if neighbor == to_id:
                    return {
                        "path": path + [neighbor],
                        "edges": edge_list + [edge_record],
                        "length": len(path),
                    }
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path + [neighbor],
                        edge_list + [edge_record],
                    ))

        return {"path": [], "edges": [], "length": -1, "message": "No path found"}


def kg_influential(
    top_k: int = 10,
    node_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most influential nodes by PageRank score."""
    with get_connection() as conn:
        sql = "SELECT node_id, node_type, label, properties_json, pagerank_score FROM kg_nodes"
        params: list[Any] = []
        if node_type:
            sql += " WHERE node_type = ?"
            params.append(node_type)
        sql += " ORDER BY pagerank_score DESC LIMIT ?"
        params.append(top_k)

        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "node_id": r[0],
                "node_type": r[1],
                "label": r[2],
                "properties": json.loads(r[3]) if r[3] else {},
                "pagerank": round(float(r[4]), 6),
            }
            for r in rows
        ]


def memory_record_finding(
    finding_type: str,
    title: str,
    content: str,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
    source_agent: str | None = None,
) -> dict[str, Any]:
    """Record a structured finding — vector memory + KG node + auto-edges + auto-crystallize.

    Pipeline:
      1. memory_store: persist content with embedding for semantic recall.
      2. kg_add_node: create/upsert KG node with canonical_id (auto-generated if missing).
      3. auto-infer edges (RELATED_TO) to top-3 semantically similar existing nodes.
      4. maybe_promote: draft → canonical if reuse_count >= threshold.
      5. on canonical promotion: auto-crystallize wiki page (idempotent).

    Each sub-call opens its own connection via get_connection().
    """
    from . import knowledge_graph as kg_mod, vector_store as vs_mod

    metadata: dict[str, Any] = {}
    if related_files:
        metadata["files"] = related_files
    if tags:
        metadata["tags"] = tags
    if source_agent:
        metadata["agent"] = source_agent

    type_map = {
        "solution": "Solution",
        "decision": "Decision",
        "insight": "Pattern",
        "problem": "Problem",
        "pattern": "Pattern",
        "context": "Entity",
    }
    node_type = type_map.get(finding_type, "Entity")
    canonical_id = kg_mod.make_canonical_id(node_type, title)

    tldr_32 = _generate_tldr(title, content)
    brief_96 = _generate_brief(title, content)

    mem_result = vs_mod.memory_store(
        type=finding_type,
        content=f"{title}: {content}",
        metadata=metadata,
    )

    kg_result = kg_mod.kg_add_node(
        node_type=node_type,
        label=title,
        properties={
            "content": content,
            "memory_id": mem_result.get("id"),
            **(metadata or {}),
        },
        canonical_id=canonical_id,
        tldr_32=tldr_32,
        brief_96=brief_96,
    )

    node_id = kg_result.get("node_id")
    logger.info(
        "Finding recorded: %s → mem=%s, kg=%s, canonical=%s",
        title, mem_result.get("id"), node_id, canonical_id,
    )

    auto_edges = _auto_infer_edges(node_id, title, content)

    promotion = maybe_promote(canonical_id)

    crystallized: dict[str, Any] | None = None
    if promotion or kg_result.get("status") == "canonical":
        try:
            from . import wiki as wiki_mod
            crystallized = wiki_mod.wiki_crystallize(canonical_id)
        except Exception as exc:
            logger.warning("Auto-crystallize failed for %s: %s", canonical_id, exc)

    result: dict[str, Any] = {
        "memory_id": mem_result.get("id"),
        "node_id": node_id,
        "canonical_id": canonical_id,
        "type": finding_type,
        "title": title,
        "status": kg_result.get("status", "draft"),
        "tldr_32": tldr_32,
        "auto_edges": auto_edges,
    }
    if promotion:
        result["promoted"] = promotion
    if crystallized and "page_id" in crystallized:
        result["wiki_page_id"] = crystallized["page_id"]
    return result


@with_retry()
def _auto_infer_edges(
    node_id: str,
    title: str,
    content: str,
) -> list[dict[str, Any]]:
    """Create RELATED_TO edges to top-N semantically similar existing nodes.

    Uses memory embeddings as proxy for KG similarity (since KG nodes carry
    memory_id in properties). Skips self-edges and duplicates.
    Returns the list of edges actually created.
    """
    from .embeddings import embed_query

    if not node_id:
        return []

    try:
        qvec = embed_query(f"{title}\n{content}")
    except Exception as exc:
        logger.warning("Auto-edge embed failed for %s: %s", node_id, exc)
        return []

    dim = len(qvec)
    edges_created: list[dict[str, Any]] = []

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT n.node_id, n.label,
                   array_cosine_similarity(e.vector, ?::FLOAT[{dim}]) AS score
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.id
            JOIN kg_nodes n ON n.properties_json LIKE '%' || m.id || '%'
            WHERE n.node_id != ?
              AND (m.expires_at IS NULL OR m.expires_at > current_timestamp)
            ORDER BY score DESC
            LIMIT ?
            """,
            [qvec, node_id, AUTO_EDGE_MAX * 3],
        ).fetchall()

        for row in rows:
            if len(edges_created) >= AUTO_EDGE_MAX:
                break
            target_id, target_label, score = row[0], row[1], float(row[2])
            if score < AUTO_EDGE_MIN_SCORE:
                continue
            existing = conn.execute(
                """SELECT 1 FROM kg_edges
                   WHERE from_id = ? AND to_id = ? AND rel_type = 'RELATED_TO'""",
                [node_id, target_id],
            ).fetchone()
            if existing:
                continue
            try:
                conn.execute(
                    """INSERT INTO kg_edges (from_id, to_id, rel_type, weight)
                       VALUES (?, ?, 'RELATED_TO', ?)
                       ON CONFLICT (from_id, to_id, rel_type) DO UPDATE SET weight = ?""",
                    [node_id, target_id, score, score],
                )
                edges_created.append({
                    "to_id": target_id,
                    "to_label": target_label,
                    "score": round(score, 4),
                })
            except Exception as exc:
                logger.debug("Skip auto-edge %s -> %s: %s", node_id, target_id, exc)

    if edges_created:
        logger.info("Auto-inferred %d RELATED_TO edges from %s", len(edges_created), node_id)
    return edges_created


@with_retry()
def maybe_promote(canonical_id: str) -> dict[str, Any] | None:
    """Auto-promote a node from draft → canonical if reuse threshold is met."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT node_id, status, reuse_count FROM kg_nodes WHERE canonical_id = ?",
            [canonical_id],
        ).fetchone()
        if not row:
            return None

        node_id, current_status, reuse_count = row[0], row[1], row[2] or 0
        if current_status != "draft" or reuse_count < PROMOTION_REUSE_THRESHOLD:
            return None

    from . import knowledge_graph as kg_mod
    result = kg_mod.kg_promote(node_id, "canonical")
    logger.info("Auto-promoted %s → canonical (reuse_count=%d)", canonical_id, reuse_count)
    return result
