"""Hierarchical topic clustering over the knowledge graph.

Inspired by Co-STORM's dynamic "mind map": instead of a flat list of nodes,
organise the KG into a hierarchy of topics → subtopics → nodes so agents can
navigate and recall by theme rather than wading through near-duplicate facts.

Design (no LLM, no new dependencies, no vector recomputation):

  record_finding already builds a kNN similarity graph for free — it inserts
  RELATED_TO edges whose `weight` is the cosine similarity to the top-N most
  similar existing nodes. We treat that edge set as the affinity graph and run
  single-linkage agglomerative clustering at TWO cosine thresholds via
  union-find:

    * `loose` threshold  → coarse *topics*    (broad themes)
    * `tight` threshold  → fine  *subtopics*  (tight clusters within a topic)

  Each topic is labelled by its most central member (highest PageRank, then
  reuse_count). Topics are persisted (kg_nodes.topic_id + kg_topics table) so
  `recall` can group/dedupe cheaply and `memory_gaps` can reason about coverage.

Complexity is O(E·α(N)) — linear in the number of edges — so it stays cheap
even on large graphs, and it never re-embeds anything.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .db import get_connection, with_retry
from .knowledge_graph import slugify

logger = logging.getLogger(__name__)

# Default cosine thresholds. `loose` must be ≤ AUTO_EDGE_MIN_SCORE (0.62) or no
# edge would survive it; we deliberately sit at/just-below the edge floor for
# topics and above it for subtopics.
DEFAULT_LOOSE = 0.55
DEFAULT_TIGHT = 0.68
_CLUSTER_REL_TYPES = ("RELATED_TO",)


class _UnionFind:
    """Minimal union-find with path compression + union by size."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _components(
    edges: list[tuple[str, str, float]],
    threshold: float,
    restrict: set[str] | None = None,
) -> dict[str, set[str]]:
    """Connected components over edges with weight ≥ threshold.

    Only nodes touched by a qualifying edge appear. `restrict`, when given,
    limits both endpoints to that node set (used to find subtopics inside a
    coarse topic).
    """
    uf = _UnionFind()
    touched: set[str] = set()
    for src, tgt, weight in edges:
        if weight < threshold:
            continue
        if restrict is not None and (src not in restrict or tgt not in restrict):
            continue
        uf.union(src, tgt)
        touched.add(src)
        touched.add(tgt)
    groups: dict[str, set[str]] = defaultdict(set)
    for nid in touched:
        groups[uf.find(nid)].add(nid)
    return groups


def _load_graph(conn: Any) -> tuple[list[tuple[str, str, float]], dict[str, dict[str, Any]]]:
    """Load the RELATED_TO affinity edges and node metadata needed for labelling."""
    rel_placeholders = ", ".join("?" for _ in _CLUSTER_REL_TYPES)
    edge_rows = conn.execute(
        f"""SELECT from_id, to_id, weight FROM kg_edges
            WHERE rel_type IN ({rel_placeholders})""",
        list(_CLUSTER_REL_TYPES),
    ).fetchall()
    edges = [(r[0], r[1], float(r[2]) if r[2] is not None else 0.0) for r in edge_rows]

    node_rows = conn.execute(
        """SELECT node_id, label, node_type, canonical_id, status,
                  pagerank_score, reuse_count
           FROM kg_nodes"""
    ).fetchall()
    meta: dict[str, dict[str, Any]] = {}
    for r in node_rows:
        meta[r[0]] = {
            "node_id": r[0],
            "label": r[1],
            "node_type": r[2],
            "canonical_id": r[3],
            "status": r[4],
            "pagerank": float(r[5]) if r[5] is not None else 0.0,
            "reuse_count": int(r[6]) if r[6] is not None else 0,
        }
    return edges, meta


def _representative(node_ids: set[str], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick the most central node of a cluster as its label-bearer."""
    members = [meta[n] for n in node_ids if n in meta]
    members.sort(
        key=lambda m: (m["pagerank"], m["reuse_count"], m["label"] or ""),
        reverse=True,
    )
    return members[0]


def _ensure_topic_schema(conn: Any) -> None:
    """Create kg_topics and the kg_nodes.topic_id column if missing (idempotent).

    Normally created by db._migrate_schema (v6); kept here so the module is
    self-sufficient on older DBs, mirroring wiki._ensure_wiki_table.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS kg_topics (
               topic_id    VARCHAR PRIMARY KEY,
               label       VARCHAR NOT NULL,
               summary     VARCHAR,
               size        INTEGER DEFAULT 0,
               subtopics   INTEGER DEFAULT 0,
               top_node_id VARCHAR,
               created_at  TIMESTAMP DEFAULT current_timestamp
           )"""
    )
    try:
        conn.execute("ALTER TABLE kg_nodes ADD COLUMN topic_id VARCHAR")
    except Exception:
        pass  # already present


@with_retry()
def build_topics(
    tight: float = DEFAULT_TIGHT,
    loose: float = DEFAULT_LOOSE,
    persist: bool = True,
) -> dict[str, Any]:
    """(Re)compute the topic hierarchy from the RELATED_TO affinity graph.

    Returns a tree:
      {
        "topics": [
          {"topic_id", "label", "node_type", "size", "node_ids",
           "subtopics": [{"label", "size", "node_ids"}, ...]},
          ...
        ],
        "unclustered": [node_id, ...],   # nodes with no qualifying affinity edge
        "stats": {"topics", "clustered_nodes", "unclustered_nodes", ...}
      }
    """
    if tight < loose:
        tight, loose = loose, tight  # tolerate swapped args

    with get_connection() as conn:
        _ensure_topic_schema(conn)
        edges, meta = _load_graph(conn)

        coarse = _components(edges, loose)
        # Keep only real clusters (size ≥ 2); singletons are "unclustered".
        coarse = {root: members for root, members in coarse.items() if len(members) >= 2}

        topics: list[dict[str, Any]] = []
        clustered: set[str] = set()
        used_ids: set[str] = set()

        for members in sorted(coarse.values(), key=len, reverse=True):
            rep = _representative(members, meta)
            base = f"topic.{slugify(rep['label']) or 'untitled'}"
            topic_id = base
            n = 2
            while topic_id in used_ids:
                topic_id = f"{base}-{n}"
                n += 1
            used_ids.add(topic_id)

            sub_groups = _components(edges, tight, restrict=members)
            covered = {nid for grp in sub_groups.values() for nid in grp}
            subtopics: list[dict[str, Any]] = []
            for grp in sorted(sub_groups.values(), key=len, reverse=True):
                if len(grp) < 2:
                    continue
                srep = _representative(grp, meta)
                subtopics.append({
                    "label": srep["label"],
                    "size": len(grp),
                    "node_ids": sorted(grp),
                })
            # nodes in the topic that didn't form a tight subtopic stay loose
            loose_members = sorted(members - covered)

            member_labels = [
                meta[n]["label"] for n in sorted(
                    members, key=lambda x: meta[x]["pagerank"] if x in meta else 0.0,
                    reverse=True,
                ) if n in meta
            ][:8]
            summary = f"{len(members)} nodes: " + ", ".join(member_labels)

            topics.append({
                "topic_id": topic_id,
                "label": rep["label"],
                "node_type": rep["node_type"],
                "size": len(members),
                "top_node_id": rep["node_id"],
                "subtopics": subtopics,
                "loose_node_ids": loose_members,
                "node_ids": sorted(members),
                "summary": summary,
            })
            clustered |= members

        unclustered = sorted(set(meta) - clustered)

        if persist:
            conn.execute("UPDATE kg_nodes SET topic_id = NULL")
            conn.execute("DELETE FROM kg_topics")
            now = datetime.now(timezone.utc)
            for t in topics:
                conn.execute(
                    """INSERT INTO kg_topics
                       (topic_id, label, summary, size, subtopics, top_node_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [t["topic_id"], t["label"], t["summary"], t["size"],
                     len(t["subtopics"]), t["top_node_id"], now],
                )
                ph = ", ".join("?" for _ in t["node_ids"])
                conn.execute(
                    f"UPDATE kg_nodes SET topic_id = ? WHERE node_id IN ({ph})",
                    [t["topic_id"], *t["node_ids"]],
                )

    logger.info(
        "Topics rebuilt: %d topics over %d clustered nodes (%d unclustered)",
        len(topics), len(clustered), len(unclustered),
    )
    return {
        "topics": topics,
        "unclustered": unclustered,
        "stats": {
            "topics": len(topics),
            "clustered_nodes": len(clustered),
            "unclustered_nodes": len(unclustered),
            "total_nodes": len(meta),
            "tight": tight,
            "loose": loose,
        },
    }


def memory_map(rebuild: bool = False, top_k: int = 20) -> dict[str, Any]:
    """Return the persisted topic hierarchy (the KG "mind map").

    By default reads the stored topics; pass rebuild=True to recompute first.
    Topics are returned largest-first, capped at top_k. Use this to orient
    before a deep `recall`, or to see how knowledge clusters by theme.
    """
    if rebuild:
        built = build_topics()
        topics = built["topics"][:top_k]
        return {
            "topics": topics,
            "unclustered_count": len(built["unclustered"]),
            "stats": built["stats"],
            "rebuilt": True,
        }

    with get_connection() as conn:
        _ensure_topic_schema(conn)
        rows = conn.execute(
            """SELECT topic_id, label, summary, size, subtopics, top_node_id
               FROM kg_topics ORDER BY size DESC LIMIT ?""",
            [top_k],
        ).fetchall()
        if not rows:
            # Lazy first build so callers never see an empty map on a populated graph.
            built = build_topics()
            return {
                "topics": built["topics"][:top_k],
                "unclustered_count": len(built["unclustered"]),
                "stats": built["stats"],
                "rebuilt": True,
            }
        unclustered = (conn.execute(
            "SELECT COUNT(*) FROM kg_nodes WHERE topic_id IS NULL"
        ).fetchone() or (0,))[0]

    topics = [
        {
            "topic_id": r[0],
            "label": r[1],
            "summary": r[2],
            "size": r[3],
            "subtopics": r[4],
            "top_node_id": r[5],
        }
        for r in rows
    ]
    return {
        "topics": topics,
        "unclustered_count": unclustered,
        "rebuilt": False,
    }
