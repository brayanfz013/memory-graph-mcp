"""HNSW vector memory — store and recall memories by semantic similarity."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import get_connection
from .embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)


def memory_store(
    type: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Embed and store a memory for later semantic recall."""
    with get_connection() as conn:
        mem_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)) if ttl_seconds else None

        conn.execute(
            """INSERT INTO memories (id, type, content, metadata_json, created_at, updated_at, ttl_seconds, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [mem_id, type, content, json.dumps(metadata or {}), now, now, ttl_seconds, expires_at],
        )

        vectors = embed_texts([content])
        conn.execute(
            "INSERT INTO memory_embeddings (id, vector) VALUES (?, ?)",
            [mem_id, vectors[0]],
        )

        _rebuild_hnsw_if_needed(conn)

        logger.info("Stored memory %s (type=%s, ttl=%s)", mem_id, type, ttl_seconds)
        return {"id": mem_id, "type": type, "created_at": now.isoformat()}


def memory_recall(
    query: str,
    type_filter: str | None = None,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Recall memories by semantic similarity to the query."""
    with get_connection() as conn:
        query_vec = embed_query(query)
        dim = len(query_vec)

        filters: list[str] = ["(m.expires_at IS NULL OR m.expires_at > current_timestamp)"]
        params: list[Any] = []

        if type_filter:
            filters.append("m.type = ?")
            params.append(type_filter)

        where = " AND ".join(filters)

        sql = f"""
            SELECT
                m.id, m.type, m.content, m.metadata_json, m.created_at,
                array_cosine_similarity(e.vector, ?::FLOAT[{dim}]) AS score
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.id
            WHERE {where}
            ORDER BY score DESC
            LIMIT ?
        """

        rows = conn.execute(sql, [query_vec] + params + [top_k]).fetchall()

        results = []
        for row in rows:
            score = float(row[5])
            if score < min_score:
                continue
            results.append({
                "id": row[0],
                "type": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]),
                "created_at": str(row[4]),
                "score": round(score, 4),
            })

        return results


def _rebuild_hnsw_if_needed(conn: Any) -> None:
    """Rebuild HNSW index periodically (every 50 inserts)."""
    count = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
    if count > 0 and count % 50 == 0:
        try:
            conn.execute("DROP INDEX IF EXISTS idx_memory_hnsw")
            conn.execute("""
                CREATE INDEX idx_memory_hnsw
                ON memory_embeddings USING HNSW (vector)
                WITH (metric = 'cosine')
            """)
            logger.info("HNSW index rebuilt (%d embeddings)", count)
        except Exception as exc:
            logger.warning("HNSW rebuild failed: %s", exc)
