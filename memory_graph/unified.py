"""Unified semantic recall across memories, KG nodes, and wiki pages.

This module replaces three brittle text-search tools (memory_recall, kg_query,
wiki_query) with a single semantic search that fuses results from all three
backends and ranks them by combined relevance + influence.

Why this exists: agents asking "what do we know about X?" should not have to
choose between three search functions or reconcile their disjoint results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .db import get_connection
from .embeddings import embed_query
from .freshness import annotate as annotate_freshness
from .freshness import head_commit
from .knowledge_graph import (
    RESOLVED_PENALTY,
    VALID_NODE_TYPES,
    _bfs_expand,
    _row_to_node,
    resolved_by,
)

logger = logging.getLogger(__name__)

VALID_SCOPES = {"memories", "kg", "wiki", "all"}


DEFAULT_MIN_SCORE = 0.45


def effective_min_score() -> float:
    """Recall floor: env RECALL_MIN_SCORE > the active model's calibrated
    `recall_floor` in PROVIDER_REGISTRY > DEFAULT_MIN_SCORE.

    A fixed floor is meaningless across models: cosine scales differ per model
    (an English model over Spanish text puts unrelated queries at ~0.7). The
    floor must be calibrated per model with negative queries (embedding_benchmark).
    """
    from .embeddings import _adopt_store_identity
    from .settings import PROVIDER_REGISTRY, settings

    _adopt_store_identity()  # the floor belongs to the store's model, not the default one
    if settings.recall_min_score is not None:
        return float(settings.recall_min_score)
    provider = settings.embedding_provider
    model = {"fastembed": settings.fastembed_model, "ollama": settings.ollama_model}.get(
        provider, "text-embedding-005")
    floor = PROVIDER_REGISTRY.get(provider, {}).get(model, {}).get("recall_floor")
    return float(floor) if floor is not None else DEFAULT_MIN_SCORE


def recall(
    query: str,
    scope: str = "all",
    top_k: int = 10,
    min_score: float | None = None,
    node_type: str | None = None,
    hops: int = 1,
    compact: bool = False,
    group_topics: bool = False,
) -> dict[str, Any]:
    """Unified semantic recall over memories + KG nodes + wiki pages.

    Args:
        query: natural language search.
        scope: 'memories' | 'kg' | 'wiki' | 'all' (default).
        top_k: max results per scope.
        min_score: cosine similarity floor (0–1). None (default) = effective_min_score(),
            the active model's calibrated floor; below it recall abstains.
        node_type: optional KG node_type filter (Decision/Solution/...).
        hops: BFS expansion from KG seed nodes (0 to disable).
        compact: token-saving mode. Drops the redundant raw `content` blob from
            KG node properties (tldr_32/brief_96 already summarise it), shortens
            wiki snippets, and drops any memory already represented by a returned
            KG node (same underlying fact). Same shape, far fewer tokens.
        group_topics: add a `topics` block grouping the returned KG nodes by
            their persisted topic (the Co-STORM-style mind map) so the caller
            sees the thematic structure instead of a flat list.

    Returns:
        {
          'memories': [{id, type, content, score, ...}, ...],
          'kg': {'nodes': [...], 'edges': [...]},
          'wiki': [{page_id, title, snippet, score, ...}, ...],
          'top_canonicals': [<canonical_id>, ...],  # most-relevant cross-scope ids
          'topics': [...]   # only when group_topics=True
        }
    """
    if scope not in VALID_SCOPES:
        return {"error": f"Invalid scope. Must be one of: {sorted(VALID_SCOPES)}"}

    if min_score is None:
        min_score = effective_min_score()
    qvec = embed_query(query)
    dim = len(qvec)

    out: dict[str, Any] = {"query": query, "scope": scope}
    canonicals_seen: dict[str, float] = {}

    if scope in {"memories", "all"}:
        out["memories"] = _recall_memories(qvec, dim, top_k, min_score)

    if scope in {"kg", "all"}:
        kg_result = _recall_kg(query, qvec, dim, top_k, min_score, node_type, hops)
        out["kg"] = kg_result
        for n in kg_result.get("nodes", []):
            cid = n.get("canonical_id")
            if cid:
                canonicals_seen[cid] = max(
                    canonicals_seen.get(cid, 0.0),
                    n.get("rank_score", n.get("semantic_score", 0.0)),
                )

    if scope in {"wiki", "all"}:
        out["wiki"] = _recall_wiki(query, qvec, dim, top_k, min_score)
        for w in out["wiki"]:
            cid = w.get("canonical_id")
            if cid:
                canonicals_seen[cid] = max(
                    canonicals_seen.get(cid, 0.0),
                    w.get("score", 0.0),
                )

    out["top_canonicals"] = [
        cid for cid, _ in sorted(
            canonicals_seen.items(), key=lambda kv: kv[1], reverse=True,
        )[:top_k]
    ]
    out["search_mode"] = "semantic"
    out["min_score"] = min_score
    out["abstained"] = not (
        out.get("memories") or (out.get("kg") or {}).get("nodes") or out.get("wiki")
    )

    if group_topics and "kg" in out:
        out["topics"] = _group_by_topic(out["kg"].get("nodes", []))

    if compact:
        _apply_compaction(out)

    return out


def _group_by_topic(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group recalled KG nodes by their persisted topic_id (the mind map).

    Nodes without a topic land under a synthetic 'ungrouped' bucket. Returns
    topics largest-first so the most prominent theme reads first.
    """
    if not nodes:
        return []
    node_ids = [n["node_id"] for n in nodes]
    topic_by_node: dict[str, str | None] = {}
    label_by_topic: dict[str, str] = {}
    with get_connection() as conn:
        ph = ", ".join("?" for _ in node_ids)
        rows = conn.execute(
            f"SELECT node_id, topic_id FROM kg_nodes WHERE node_id IN ({ph})",
            node_ids,
        ).fetchall()
        for nid, tid in rows:
            topic_by_node[nid] = tid
        topic_ids = {t for t in topic_by_node.values() if t}
        if topic_ids:
            tph = ", ".join("?" for _ in topic_ids)
            for tid, label in conn.execute(
                f"SELECT topic_id, label FROM kg_topics WHERE topic_id IN ({tph})",
                list(topic_ids),
            ).fetchall():
                label_by_topic[tid] = label

    buckets: dict[str, dict[str, Any]] = {}
    for n in nodes:
        tid = topic_by_node.get(n["node_id"]) or "ungrouped"
        bucket = buckets.setdefault(
            tid,
            {
                "topic_id": None if tid == "ungrouped" else tid,
                "label": label_by_topic.get(tid, "(ungrouped)"),
                "members": [],
            },
        )
        bucket["members"].append({
            "node_id": n["node_id"],
            "label": n.get("label"),
            "score": n.get("semantic_score", 0.0),
        })
    return sorted(buckets.values(), key=lambda b: len(b["members"]), reverse=True)


_MEMORY_SNIPPET_CHARS = 240


def _strip_title(text: str | None, label: str | None) -> str:
    """Drop a leading 'Title: ' that <=0.5.1 baked into stored summaries."""
    text = (text or "").strip()
    if label and text.startswith(label):
        text = text[len(label):].lstrip(" :—-")
    return text


def _compact_node(n: dict[str, Any]) -> dict[str, Any]:
    """Only what a reader acts on: identity, scores, a short gist, and flags."""
    out: dict[str, Any] = {
        "node_id": n.get("node_id"),
        "canonical_id": n.get("canonical_id"),
        "node_type": n.get("node_type"),
        "label": n.get("label"),
        "status": n.get("status"),
        "semantic_score": n.get("semantic_score"),
        "rank_score": n.get("rank_score", n.get("semantic_score")),
        "tldr": _strip_title(n.get("tldr_32"), n.get("label")),
    }
    files = (n.get("properties") or {}).get("files")
    if files:
        out["files"] = files
    fr = n.get("freshness")
    if fr and fr.get("stale"):
        out["freshness"] = fr
    if n.get("resolved_by"):
        out["resolved_by"] = n["resolved_by"]
    return out


def _apply_compaction(out: dict[str, Any]) -> None:
    """Trim the payload in-place to cut token cost (see recall(compact=…)).

    KG nodes keep only actionable keys (no properties, no brief, no title restated
    in the gist); memories already represented by a returned node are dropped and
    the rest are clipped; wiki snippets are clipped.
    """
    kg = out.get("kg") if isinstance(out.get("kg"), dict) else {}
    kg_nodes = kg.get("nodes", []) if kg else []

    represented_mem_ids = {
        (n.get("properties") or {}).get("memory_id") for n in kg_nodes
    } - {None}
    if kg:
        kg["nodes"] = [_compact_node(n) for n in kg_nodes]

    if "memories" in out:
        kept = []
        for m in out["memories"]:
            if m.get("id") in represented_mem_ids:
                continue
            m = dict(m)
            m.pop("metadata", None)
            c = m.get("content") or ""
            if len(c) > _MEMORY_SNIPPET_CHARS:
                m["content"] = c[:_MEMORY_SNIPPET_CHARS] + "…"
            kept.append(m)
        out["memories"] = kept

    for w in out.get("wiki", []) or []:
        snip = w.get("snippet") or ""
        if len(snip) > 200:
            w["snippet"] = snip[:200] + "…"

    out["compact"] = True


def _recall_memories(
    qvec: list[float],
    dim: int,
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Vector recall over memories table."""
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT
                m.id, m.type, m.content, m.metadata_json, m.created_at,
                array_cosine_similarity(e.vector, ?::FLOAT[{dim}]) AS score
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.id
            WHERE (m.expires_at IS NULL OR m.expires_at > current_timestamp)
            ORDER BY score DESC
            LIMIT ?
            """,
            [qvec, top_k * 2],
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        score = float(row[5])
        if score < min_score:
            continue
        results.append({
            "id": row[0],
            "type": row[1],
            "content": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
            "created_at": str(row[4]),
            "score": round(score, 4),
        })
        if len(results) >= top_k:
            break
    return results


def _recall_kg(
    query: str,
    qvec: list[float],
    dim: int,
    top_k: int,
    min_score: float,
    node_type: str | None,
    hops: int,
) -> dict[str, Any]:
    """Semantic KG search via memory embeddings, with BFS expansion."""
    with get_connection() as conn:
        sem_filters = ["(m.expires_at IS NULL OR m.expires_at > current_timestamp)"]
        sem_params: list[Any] = [qvec]
        if node_type and node_type in VALID_NODE_TYPES:
            sem_filters.append("n.node_type = ?")
            sem_params.append(node_type)
        sem_where = " AND ".join(sem_filters)
        rows = conn.execute(
            f"""
            SELECT n.node_id, n.node_type, n.label, n.properties_json, n.pagerank_score,
                   n.canonical_id, n.status, n.tldr_32, n.brief_96, n.reuse_count, n.confidence,
                   array_cosine_similarity(e.vector, ?::FLOAT[{dim}]) AS score
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.id
            JOIN kg_nodes n ON n.properties_json LIKE '%' || m.id || '%'
            WHERE {sem_where}
            ORDER BY score DESC
            LIMIT ?
            """,
            sem_params + [top_k * 2],
        ).fetchall()

        seed_nodes: list[dict[str, Any]] = []
        seed_ids: set[str] = set()
        for row in rows:
            score = float(row[11])
            if score < min_score:
                continue
            node = _row_to_node(row[:11])
            node["semantic_score"] = round(score, 4)
            seed_nodes.append(node)
            seed_ids.add(node["node_id"])
            if len(seed_nodes) >= top_k:
                break

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

        # recall is a READ: returning a node is not using it. Reuse is counted only
        # when the caller opens a result (kg_resolve / wiki_get → intelligence.register_use).

        expanded_nodes, expanded_edges = _bfs_expand(conn, seed_ids, hops)

        all_nodes = {n["node_id"]: n for n in seed_nodes}
        for n in expanded_nodes:
            n.setdefault("semantic_score", 0.0)
            if n["node_id"] not in all_nodes:
                all_nodes[n["node_id"]] = n

        head = head_commit()  # once per recall, not once per node
        for n in all_nodes.values():
            annotate_freshness(n, head=head)
        fixes = resolved_by(conn, list(all_nodes))
        for nid, cids in fixes.items():
            n = all_nodes[nid]
            n["resolved_by"] = cids
            n["rank_score"] = round(n.get("rank_score", 0.0) * RESOLVED_PENALTY, 4)
        ranked = sorted(
            all_nodes.values(),
            key=lambda n: (n.get("rank_score", 0.0), n["pagerank_score"]),
            reverse=True,
        )[:top_k]

    return {
        "nodes": ranked,
        "edges": expanded_edges,
        "seed_count": len(seed_nodes),
        "expanded_count": len(expanded_nodes),
    }


def _recall_wiki(
    query: str,
    qvec: list[float],
    dim: int,
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Semantic wiki recall via wiki_embeddings, with ILIKE fallback."""
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with get_connection() as conn:
        try:
            sem_rows = conn.execute(
                f"""
                SELECT p.page_id, p.canonical_id, p.title, p.body, p.tags_json, p.updated_at,
                       array_cosine_similarity(we.vector, ?::FLOAT[{dim}]) AS score
                FROM wiki_embeddings we
                JOIN wiki_pages p ON p.page_id = we.page_id
                WHERE p.status = 'active'
                ORDER BY score DESC
                LIMIT ?
                """,
                [qvec, top_k * 2],
            ).fetchall()
            for row in sem_rows:
                score = float(row[6])
                if score < min_score:
                    continue
                results.append(_format_wiki_row(row[:6], score))
                seen_ids.add(row[0])
                if len(results) >= top_k:
                    break
        except Exception as exc:
            logger.debug("Wiki semantic search skipped: %s", exc)

        if len(results) < top_k:
            placeholders = ""
            params: list[Any] = [f"%{query}%", f"%{query}%"]
            if seen_ids:
                placeholders = "AND page_id NOT IN (" + ", ".join("?" for _ in seen_ids) + ")"
                params.extend(seen_ids)
            try:
                ilike_rows = conn.execute(
                    f"""SELECT page_id, canonical_id, title, body, tags_json, updated_at
                        FROM wiki_pages
                        WHERE (title ILIKE ? OR body ILIKE ?)
                          AND status = 'active'
                          {placeholders}
                        ORDER BY updated_at DESC
                        LIMIT ?""",
                    params + [top_k - len(results)],
                ).fetchall()
                for row in ilike_rows:
                    results.append(_format_wiki_row(row, 0.0))
            except Exception as exc:
                logger.debug("Wiki ILIKE fallback skipped: %s", exc)

    return results


def _format_wiki_row(row: tuple, score: float) -> dict[str, Any]:
    """Format a wiki page row into the response shape."""
    body: str = row[3] or ""
    return {
        "page_id": row[0],
        "canonical_id": row[1],
        "title": row[2],
        "snippet": body[:500] + ("…" if len(body) > 500 else ""),
        "tags": json.loads(row[4]) if row[4] else [],
        "updated_at": str(row[5]),
        "score": round(score, 4),
    }
