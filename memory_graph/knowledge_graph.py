"""Knowledge Graph with PageRank — store decisions, solutions, and relationships."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .db import get_connection, with_retry

logger = logging.getLogger(__name__)

VALID_NODE_TYPES = {"Decision", "Solution", "Problem", "Tool", "Pattern", "Entity"}
VALID_EDGE_TYPES = {"SOLVES", "CAUSED_BY", "DEPENDS_ON", "RELATED_TO", "USES_TOOL", "SUPERSEDES"}
VALID_STATUSES = {"draft", "canonical", "superseded"}


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug for canonical IDs."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80]


def make_canonical_id(node_type: str, label: str) -> str:
    """Generate a canonical perfector_id from node_type + label."""
    return f"{node_type.lower()}.{slugify(label)}"


@with_retry()
def kg_add_node(
    node_type: str,
    label: str,
    properties: dict[str, Any] | None = None,
    canonical_id: str | None = None,
    status: str = "draft",
    tldr_32: str | None = None,
    brief_96: str | None = None,
    summary_256: str | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    """Add a node to the knowledge graph.

    Update priority (first match wins):
      1. node_id — update existing node in place (COALESCE for metadata)
      2. canonical_id — upsert by canonical_id (deduplicated)
      3. Neither — auto-generates canonical_id from (node_type, label) for dedup
    """
    if node_type not in VALID_NODE_TYPES:
        return {"error": f"Invalid node_type. Must be one of: {sorted(VALID_NODE_TYPES)}"}
    if status not in VALID_STATUSES:
        return {"error": f"Invalid status. Must be one of: {sorted(VALID_STATUSES)}"}

    if not canonical_id and not node_id:
        canonical_id = make_canonical_id(node_type, label)

    with get_connection() as conn:
        props_json = json.dumps(properties or {})

        if node_id:
            existing = conn.execute(
                "SELECT node_id, canonical_id FROM kg_nodes WHERE node_id = ?", [node_id]
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE kg_nodes
                       SET label = ?, properties_json = ?, status = ?,
                           tldr_32 = COALESCE(?, tldr_32),
                           brief_96 = COALESCE(?, brief_96),
                           summary_256 = COALESCE(?, summary_256)
                       WHERE node_id = ?""",
                    [label, props_json, status, tldr_32, brief_96, summary_256, node_id],
                )
                logger.info("KG node updated (node_id): %s", node_id)
                return {
                    "node_id": node_id,
                    "canonical_id": existing[1],
                    "node_type": node_type,
                    "label": label,
                    "status": status,
                    "action": "updated",
                }

        if canonical_id:
            existing = conn.execute(
                "SELECT node_id FROM kg_nodes WHERE canonical_id = ?", [canonical_id]
            ).fetchone()

            if existing:
                node_id = existing[0]
                conn.execute(
                    """UPDATE kg_nodes
                       SET label = ?, properties_json = ?, status = ?,
                           tldr_32 = COALESCE(?, tldr_32),
                           brief_96 = COALESCE(?, brief_96),
                           summary_256 = COALESCE(?, summary_256)
                       WHERE node_id = ?""",
                    [label, props_json, status, tldr_32, brief_96, summary_256, node_id],
                )
                logger.info("KG node updated (canonical): %s → %s", canonical_id, node_id)
                return {
                    "node_id": node_id,
                    "canonical_id": canonical_id,
                    "node_type": node_type,
                    "label": label,
                    "status": status,
                    "action": "updated",
                }

        node_id = f"{node_type.lower()}.{uuid.uuid4().hex[:8]}"

        conn.execute(
            """INSERT INTO kg_nodes
               (node_id, node_type, label, properties_json, canonical_id, status,
                tldr_32, brief_96, summary_256, reuse_count, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0)
               ON CONFLICT (node_id) DO UPDATE
               SET label = ?, properties_json = ?, status = ?,
                   tldr_32 = COALESCE(?, tldr_32),
                   brief_96 = COALESCE(?, brief_96),
                   summary_256 = COALESCE(?, summary_256)""",
            [
                node_id, node_type, label, props_json, canonical_id, status,
                tldr_32, brief_96, summary_256,
                label, props_json, status, tldr_32, brief_96, summary_256,
            ],
        )

        logger.info("KG node added: %s (canonical=%s, status=%s)", node_id, canonical_id, status)
        return {
            "node_id": node_id,
            "canonical_id": canonical_id,
            "node_type": node_type,
            "label": label,
            "status": status,
            "action": "created",
        }


@with_retry()
def kg_add_edge(
    from_id: str,
    to_id: str,
    rel_type: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Add a directed edge between two nodes."""
    if rel_type not in VALID_EDGE_TYPES:
        return {"error": f"Invalid rel_type. Must be one of: {sorted(VALID_EDGE_TYPES)}"}

    with get_connection() as conn:
        existing = conn.execute("SELECT node_id FROM kg_nodes WHERE node_id IN (?, ?)", [from_id, to_id]).fetchall()
        found_ids = {row[0] for row in existing}
        missing = {from_id, to_id} - found_ids
        if missing:
            return {"error": f"Node(s) not found: {sorted(missing)}"}

        conn.execute(
            """INSERT INTO kg_edges (from_id, to_id, rel_type, weight)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (from_id, to_id, rel_type) DO UPDATE SET weight = ?""",
            [from_id, to_id, rel_type, weight, weight],
        )

        logger.info("KG edge added: %s -[%s]-> %s", from_id, rel_type, to_id)
        return {"from_id": from_id, "to_id": to_id, "rel_type": rel_type, "weight": weight}


@with_retry()
def kg_query(
    query: str,
    node_type: str | None = None,
    hops: int = 1,
    top_k: int = 10,
    min_score: float = 0.45,
) -> dict[str, Any]:
    """Search KG nodes by semantic similarity (via memory embeddings) + PageRank.

    Strategy:
      1. Embed `query`, find top semantic matches over linked memories
         (kg_nodes.properties_json.memory_id) → seed nodes.
      2. Fall back to ILIKE on label/properties for keyword-only seeds.
      3. Expand seeds via BFS up to `hops` to surface related context.
      4. Rank by combined PageRank × semantic score.
    """
    from .embeddings import embed_query

    with get_connection() as conn:
        seed_nodes: list[dict[str, Any]] = []
        seed_ids: set[str] = set()
        score_by_id: dict[str, float] = {}

        try:
            qvec = embed_query(query)
            dim = len(qvec)
            sem_filters = ["(m.expires_at IS NULL OR m.expires_at > current_timestamp)"]
            sem_params: list[Any] = [qvec]
            if node_type and node_type in VALID_NODE_TYPES:
                sem_filters.append("n.node_type = ?")
                sem_params.append(node_type)
            sem_where = " AND ".join(sem_filters)
            sem_sql = f"""
                SELECT n.node_id, n.node_type, n.label, n.properties_json, n.pagerank_score,
                       n.canonical_id, n.status, n.tldr_32, n.brief_96, n.reuse_count, n.confidence,
                       array_cosine_similarity(e.vector, ?::FLOAT[{dim}]) AS score
                FROM memory_embeddings e
                JOIN memories m ON m.id = e.id
                JOIN kg_nodes n ON n.properties_json LIKE '%' || m.id || '%'
                WHERE {sem_where}
                ORDER BY score DESC
                LIMIT ?
            """
            sem_rows = conn.execute(sem_sql, sem_params + [top_k * 2]).fetchall()
            for row in sem_rows:
                score = float(row[11])
                if score < min_score:
                    continue
                node = _row_to_node(row[:11])
                node["semantic_score"] = round(score, 4)
                seed_nodes.append(node)
                seed_ids.add(node["node_id"])
                score_by_id[node["node_id"]] = score
        except Exception as exc:
            logger.warning("Semantic kg_query failed, falling back to ILIKE: %s", exc)

        if len(seed_nodes) < top_k:
            ilike_filters = ["(label ILIKE ? OR properties_json ILIKE ?)"]
            ilike_params: list[Any] = [f"%{query}%", f"%{query}%"]
            if node_type and node_type in VALID_NODE_TYPES:
                ilike_filters.append("node_type = ?")
                ilike_params.append(node_type)
            if seed_ids:
                placeholders = ", ".join("?" for _ in seed_ids)
                ilike_filters.append(f"node_id NOT IN ({placeholders})")
                ilike_params.extend(seed_ids)
            ilike_where = " AND ".join(ilike_filters)
            ilike_rows = conn.execute(
                f"""SELECT node_id, node_type, label, properties_json, pagerank_score,
                           canonical_id, status, tldr_32, brief_96, reuse_count, confidence
                    FROM kg_nodes WHERE {ilike_where}
                    ORDER BY pagerank_score DESC LIMIT ?""",
                ilike_params + [top_k - len(seed_nodes)],
            ).fetchall()
            for row in ilike_rows:
                node = _row_to_node(row)
                node["semantic_score"] = 0.0
                seed_nodes.append(node)
                seed_ids.add(node["node_id"])

        for nid in seed_ids:
            conn.execute(
                "UPDATE kg_nodes SET reuse_count = COALESCE(reuse_count, 0) + 1 WHERE node_id = ?",
                [nid],
            )

        expanded_nodes, expanded_edges = _bfs_expand(conn, seed_ids, hops)

        all_nodes = {n["node_id"]: n for n in seed_nodes}
        for n in expanded_nodes:
            n.setdefault("semantic_score", 0.0)
            if n["node_id"] not in all_nodes:
                all_nodes[n["node_id"]] = n

        ranked = sorted(
            all_nodes.values(),
            key=lambda n: (n.get("semantic_score", 0.0), n["pagerank_score"]),
            reverse=True,
        )[:top_k]

        return {
            "nodes": ranked,
            "edges": expanded_edges,
            "seed_count": len(seed_nodes),
            "expanded_count": len(expanded_nodes),
            "search_mode": "semantic+ilike",
        }


@with_retry()
def kg_compute_pagerank() -> dict[str, Any]:
    """Recompute PageRank scores for all nodes."""
    from .settings import settings

    with get_connection() as conn:
        scores, iterations = _run_pagerank(conn, settings.pagerank_damping, settings.pagerank_max_iter)

        for node_id, score in scores.items():
            conn.execute("UPDATE kg_nodes SET pagerank_score = ? WHERE node_id = ?", [score, node_id])

        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("PageRank computed: %d nodes, %d iterations", len(scores), iterations)

        return {
            "nodes_updated": len(scores),
            "iterations": iterations,
            "top_5": [{"node_id": nid, "score": round(s, 6)} for nid, s in top_5],
        }


def _run_pagerank(
    conn: Any,
    damping: float = 0.85,
    max_iter: int = 100,
) -> tuple[dict[str, float], int]:
    """Iterative PageRank with weighted edges."""
    nodes = [r[0] for r in conn.execute("SELECT node_id FROM kg_nodes").fetchall()]
    edges = conn.execute("SELECT from_id, to_id, weight FROM kg_edges").fetchall()

    n = len(nodes)
    if n == 0:
        return {}, 0

    scores = {nid: 1.0 / n for nid in nodes}
    out_weight: dict[str, float] = defaultdict(float)
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for src, tgt, w in edges:
        out_weight[src] += w
        adj[src].append((tgt, w))

    for i in range(max_iter):
        new_scores = {nid: (1 - damping) / n for nid in nodes}

        for src in nodes:
            if out_weight[src] == 0:
                continue
            for tgt, w in adj[src]:
                if tgt in new_scores:
                    new_scores[tgt] += damping * scores[src] * w / out_weight[src]

        diff = max(abs(new_scores[nid] - scores[nid]) for nid in nodes)
        scores = new_scores
        if diff < 1e-6:
            return scores, i + 1

    return scores, max_iter


def _bfs_expand(
    conn: Any,
    seed_ids: set[str],
    hops: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """BFS expansion from seed nodes up to N hops."""
    visited = set(seed_ids)
    frontier = set(seed_ids)
    all_edges: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []

    for _ in range(hops):
        if not frontier:
            break

        placeholders = ", ".join("?" for _ in frontier)
        edge_rows = conn.execute(
            f"""SELECT from_id, to_id, rel_type, weight
                FROM kg_edges
                WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})""",
            list(frontier) + list(frontier),
        ).fetchall()

        next_frontier: set[str] = set()
        for src, tgt, rel, w in edge_rows:
            all_edges.append({"from_id": src, "to_id": tgt, "rel_type": rel, "weight": w})
            for nid in (src, tgt):
                if nid not in visited:
                    next_frontier.add(nid)
                    visited.add(nid)

        if next_frontier:
            ph = ", ".join("?" for _ in next_frontier)
            node_rows = conn.execute(
                f"""SELECT node_id, node_type, label, properties_json, pagerank_score,
                           canonical_id, status, tldr_32, brief_96, reuse_count, confidence
                    FROM kg_nodes WHERE node_id IN ({ph})""",
                list(next_frontier),
            ).fetchall()
            all_nodes.extend(_row_to_node(r) for r in node_rows)

        frontier = next_frontier

    return all_nodes, all_edges


def _row_to_node(row: tuple) -> dict[str, Any]:
    """Convert a DB row tuple to a node dict (extended with perfector columns)."""
    result: dict[str, Any] = {
        "node_id": row[0],
        "node_type": row[1],
        "label": row[2],
        "properties": json.loads(row[3]) if row[3] else {},
        "pagerank_score": float(row[4]) if row[4] is not None else 0.0,
    }
    if len(row) > 5:
        result["canonical_id"] = row[5]
        result["status"] = row[6] or "draft"
        result["tldr_32"] = row[7]
        result["brief_96"] = row[8]
        result["reuse_count"] = row[9] if row[9] is not None else 0
        result["confidence"] = float(row[10]) if row[10] is not None else 0.0
    return result


# ── Perfector extensions ───────────────────────────────────────

@with_retry()
def kg_promote(
    node_id: str,
    status: str,
) -> dict[str, Any]:
    """Transition a node's lifecycle status: draft → canonical → superseded."""
    if status not in VALID_STATUSES:
        return {"error": f"Invalid status. Must be one of: {sorted(VALID_STATUSES)}"}

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT node_id, status, canonical_id, label FROM kg_nodes WHERE node_id = ?",
            [node_id],
        ).fetchone()
        if not existing:
            return {"error": f"Node not found: {node_id}"}

        old_status = existing[1]
        now = datetime.now(timezone.utc)

        conn.execute(
            "UPDATE kg_nodes SET status = ?, last_validated_at = ? WHERE node_id = ?",
            [status, now, node_id],
        )
        logger.info("KG node promoted: %s (%s → %s)", node_id, old_status, status)
        return {
            "node_id": node_id,
            "canonical_id": existing[2],
            "label": existing[3],
            "old_status": old_status,
            "new_status": status,
        }


@with_retry()
def kg_resolve(canonical_id: str) -> dict[str, Any]:
    """Look up a node by its stable perfector_id (canonical_id)."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT node_id, node_type, label, properties_json, pagerank_score,
                      canonical_id, status, tldr_32, brief_96, reuse_count, confidence
               FROM kg_nodes WHERE canonical_id = ?""",
            [canonical_id],
        ).fetchone()
        if not row:
            return {"error": f"No node with canonical_id: {canonical_id}"}
        return _row_to_node(row)
