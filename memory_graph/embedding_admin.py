"""Embedding provider administration — status, migrate, registry exposure.

These functions back the `embedding_status`, `embedding_migrate`, and
`embedding_benchmark` MCP tools. The benchmark logic lives in `benchmark.py`.

Key behaviors:
  - `embedding_status()`        — show DB-stored vs active env identity + diff.
  - `embedding_migrate(...)`    — re-embed every memory + wiki page under a new
                                   provider/model, recreating the vector tables
                                   with the right dimensions.

Migration is in-place (no multi-generation storage in MVP). The original
content (`memories.content`, `wiki_pages.body`) stays — only the vectors are
replaced. The new generation is recorded in `embedding_meta` so subsequent
startups detect the new identity.
"""

from __future__ import annotations

import logging
from typing import Any

from .db import (
    get_active_embedding_meta,
    get_connection,
    record_embedding_generation,
    reset_embedding_dimensions_cache,
)
from .settings import PROVIDER_REGISTRY, settings

logger = logging.getLogger(__name__)


def embedding_status() -> dict[str, Any]:
    """Report the active provider, DB-stored identity, and any mismatch.

    Also lists every model in the registry so callers know what they can swap to.
    """
    from .embeddings import get_identity  # late import to avoid eager provider load

    try:
        current = get_identity()
    except Exception as exc:
        current = {"error": f"Cannot load active provider: {exc}"}

    with get_connection() as conn:
        active = get_active_embedding_meta(conn)
        memory_count_row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        memories = int((memory_count_row or (0,))[0])
        wiki_count_row = conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()
        wikis = int((wiki_count_row or (0,))[0])

    mismatch = False
    if active and "error" not in current:
        mismatch = (
            active["provider"] != current.get("provider")
            or active["model_name"] != current.get("model")
            or active["dimensions"] != current.get("dimensions")
        )

    return {
        "active_env": current,
        "stored_in_db": active,
        "mismatch": mismatch,
        "embeddings_at_risk": memories + wikis if mismatch else 0,
        "registry": PROVIDER_REGISTRY,
        "guidance": (
            "Mismatch detected. Run embedding_migrate(target_provider, target_model) "
            "to re-embed all memories + wiki pages under the new provider. "
            "Recall still works against the old vectors in the meantime."
            if mismatch
            else "Active provider matches the DB. No action needed."
        ),
    }


def embedding_migrate(
    target_provider: str,
    target_model: str,
    dry_run: bool = True,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Re-embed every memory and wiki page under a new provider/model.

    Steps:
      1. Mutate `settings.embedding_provider` + `settings.{fastembed_model|ollama_model}`.
      2. Clear provider + dimensions caches.
      3. Probe the new provider (loads model, detects dims).
      4. Drop and recreate `memory_embeddings` / `wiki_embeddings` with new dim.
      5. Embed all rows in batches, INSERT into the recreated table.
      6. Record a new `embedding_meta` generation marking the new identity active.

    `dry_run=True` (default) skips steps 4-6 and just returns the plan + estimated
    work. Pass `dry_run=False` to actually rewrite.
    """
    from . import embeddings as emb

    # Validate target
    if target_provider not in PROVIDER_REGISTRY:
        return {
            "ok": False,
            "error": f"Unknown provider {target_provider!r}. Known: {list(PROVIDER_REGISTRY)}",
        }
    if target_model not in PROVIDER_REGISTRY[target_provider]:
        known = list(PROVIDER_REGISTRY[target_provider])
        return {
            "ok": False,
            "error": f"Unknown model {target_model!r} for {target_provider}. Known: {known}",
        }

    # 1. Mutate settings in-memory (the active process picks them up via reset)
    settings.embedding_provider = target_provider
    if target_provider == "fastembed":
        settings.fastembed_model = target_model
    elif target_provider == "ollama":
        settings.ollama_model = target_model
    # vertex picks model_name from the class itself; nothing to set

    # 2. Reset caches
    emb.reset_provider_cache()
    reset_embedding_dimensions_cache()

    # 3. Probe new provider
    try:
        new_identity = emb.get_identity()
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not load target provider {target_provider}/{target_model}: {exc}",
        }
    new_dim = int(new_identity["dimensions"])

    # 4-6. Inventory work
    with get_connection() as conn:
        active = get_active_embedding_meta(conn)
        mem_count = int((conn.execute("SELECT COUNT(*) FROM memories").fetchone() or (0,))[0])
        wiki_count = int((conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone() or (0,))[0])

        plan = {
            "from": active,
            "to": new_identity,
            "memories_to_reembed": mem_count,
            "wiki_pages_to_reembed": wiki_count,
            "dry_run": dry_run,
            "batch_size": batch_size,
        }

        if dry_run:
            plan["ok"] = True
            plan["message"] = "Dry run — re-run with dry_run=False to actually rewrite."
            return plan

        # 4. Recreate vector tables with new dim
        conn.execute("DROP INDEX IF EXISTS idx_memory_hnsw")
        conn.execute("DROP TABLE IF EXISTS memory_embeddings")
        conn.execute(f"""
            CREATE TABLE memory_embeddings (
                id     VARCHAR PRIMARY KEY,
                vector FLOAT[{new_dim}],
                FOREIGN KEY (id) REFERENCES memories(id)
            )
        """)
        conn.execute("DROP TABLE IF EXISTS wiki_embeddings")
        conn.execute(f"""
            CREATE TABLE wiki_embeddings (
                page_id VARCHAR PRIMARY KEY,
                vector  FLOAT[{new_dim}],
                FOREIGN KEY (page_id) REFERENCES wiki_pages(page_id)
            )
        """)

        # 5. Re-embed memories in batches
        memories_done = _reembed_table(
            conn=conn,
            select_sql="SELECT id, content FROM memories",
            insert_sql="INSERT INTO memory_embeddings (id, vector) VALUES (?, ?)",
            batch_size=batch_size,
        )

        # Re-embed wiki pages
        wiki_done = _reembed_table(
            conn=conn,
            select_sql="SELECT page_id, title || ' ' || COALESCE(body, '') FROM wiki_pages",
            insert_sql="INSERT INTO wiki_embeddings (page_id, vector) VALUES (?, ?)",
            batch_size=batch_size,
        )

        # 6. Restore HNSW index + record new generation
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_hnsw "
                "ON memory_embeddings USING HNSW (vector) WITH (metric = 'cosine')"
            )
        except Exception as exc:
            logger.warning("Could not recreate HNSW index after migration: %s", exc)

        new_gen = record_embedding_generation(
            conn=conn,
            provider=new_identity["provider"],
            model_name=new_identity["model"],
            dimensions=new_dim,
        )

    plan.update({
        "ok": True,
        "message": (
            f"Migrated to generation {new_gen} "
            f"({new_identity['provider']}/{new_identity['model']}, dim={new_dim}). "
            f"Re-embedded {memories_done} memories + {wiki_done} wiki pages."
        ),
        "memories_reembedded": memories_done,
        "wiki_pages_reembedded": wiki_done,
        "new_generation": new_gen,
    })
    return plan


def _reembed_table(
    conn: Any,
    select_sql: str,
    insert_sql: str,
    batch_size: int,
) -> int:
    """Re-embed all rows from `select_sql` into the target table via `insert_sql`.

    `select_sql` must yield `(id, text)` pairs.
    `insert_sql` must accept `(id, vector)`.
    """
    from . import embeddings as emb

    rows = conn.execute(select_sql).fetchall()
    if not rows:
        return 0

    done = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] or "" for r in batch]
        vectors = emb.embed_texts(texts)
        for row_id, vector in zip(ids, vectors):
            conn.execute(insert_sql, [row_id, vector])
        done += len(batch)
    return done


__all__ = ["embedding_status", "embedding_migrate"]
