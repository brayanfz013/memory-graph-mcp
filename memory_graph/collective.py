"""Collective memory — LRU + TTL shared knowledge base with 8 memory types."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import get_connection
from .settings import settings

logger = logging.getLogger(__name__)

TTL_DEFAULTS: dict[str, int | None] = {
    "knowledge": None,       # permanent
    "context": 3600,         # 1 hour
    "task": 1800,            # 30 minutes
    "result": None,          # permanent
    "error": 86400,          # 24 hours
    "metric": 3600,          # 1 hour
    "consensus": None,       # permanent
    "system": None,          # permanent
}

PERMANENT_TYPES = {k for k, v in TTL_DEFAULTS.items() if v is None}


def collective_store(
    type: str,
    key: str,
    value: Any,
    scope: str = "global",
) -> dict[str, Any]:
    """Store a value in collective memory with type-based TTL."""
    if type not in TTL_DEFAULTS:
        return {"error": f"Invalid type. Must be one of: {sorted(TTL_DEFAULTS)}"}

    with get_connection() as conn:
        now = datetime.now(timezone.utc)
        ttl = TTL_DEFAULTS[type]
        expires_at = (now + timedelta(seconds=ttl)) if ttl else None
        value_json = json.dumps(value, ensure_ascii=False, default=str)

        conn.execute(
            """INSERT INTO collective_memory
               (key, scope, type, value_json, created_at, updated_at, last_accessed_at, access_count, ttl_seconds, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT (key, scope) DO UPDATE SET
                   value_json = ?, type = ?, updated_at = ?, last_accessed_at = ?,
                   ttl_seconds = ?, expires_at = ?""",
            [key, scope, type, value_json, now, now, now, ttl, expires_at,
             value_json, type, now, now, ttl, expires_at],
        )

        _evict_if_needed(conn)

        logger.info("Collective store: %s/%s (type=%s)", scope, key, type)
        return {"key": key, "scope": scope, "type": type, "expires_at": str(expires_at) if expires_at else None}


def collective_get(key: str, scope: str = "global") -> dict[str, Any] | None:
    """Retrieve a value from collective memory, updating access tracking."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT key, scope, type, value_json, access_count, created_at, updated_at
               FROM collective_memory
               WHERE key = ? AND scope = ?
                 AND (expires_at IS NULL OR expires_at > current_timestamp)""",
            [key, scope],
        ).fetchone()

        if row is None:
            return None

        conn.execute(
            """UPDATE collective_memory
               SET last_accessed_at = current_timestamp, access_count = access_count + 1
               WHERE key = ? AND scope = ?""",
            [key, scope],
        )

        return {
            "key": row[0],
            "scope": row[1],
            "type": row[2],
            "value": json.loads(row[3]),
            "access_count": row[4] + 1,
            "created_at": str(row[5]),
            "updated_at": str(row[6]),
        }


def collective_list(
    type: str | None = None,
    scope: str = "global",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List collective memory entries, filtered by type and scope."""
    with get_connection() as conn:
        filters = ["scope = ?", "(expires_at IS NULL OR expires_at > current_timestamp)"]
        params: list[Any] = [scope]

        if type:
            filters.append("type = ?")
            params.append(type)

        where = " AND ".join(filters)
        rows = conn.execute(
            f"""SELECT key, type, value_json, access_count, last_accessed_at
                FROM collective_memory
                WHERE {where}
                ORDER BY last_accessed_at DESC LIMIT ?""",
            params + [limit],
        ).fetchall()

        return [
            {
                "key": r[0],
                "type": r[1],
                "value": json.loads(r[2]),
                "access_count": r[3],
                "last_accessed_at": str(r[4]),
            }
            for r in rows
        ]


def collective_cleanup() -> dict[str, int]:
    """Remove all expired entries."""
    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM collective_memory").fetchone()[0]

        conn.execute(
            "DELETE FROM collective_memory WHERE expires_at IS NOT NULL AND expires_at <= current_timestamp"
        )

        after = conn.execute("SELECT COUNT(*) FROM collective_memory").fetchone()[0]
        removed = before - after
        logger.info("Collective cleanup: removed %d expired entries", removed)
        return {"removed": removed}


def _evict_if_needed(conn: Any) -> None:
    """LRU eviction when entry count exceeds max_entries."""
    count = conn.execute("SELECT COUNT(*) FROM collective_memory").fetchone()[0]
    if count <= settings.max_entries:
        return

    evict_count = count - settings.max_entries + 100
    permanent_types = ", ".join(f"'{t}'" for t in PERMANENT_TYPES)

    conn.execute(
        f"""DELETE FROM collective_memory
            WHERE (key, scope) IN (
                SELECT key, scope FROM collective_memory
                WHERE type NOT IN ({permanent_types})
                ORDER BY last_accessed_at ASC
                LIMIT ?
            )""",
        [evict_count],
    )
    logger.info("LRU eviction: removed %d entries (count was %d)", evict_count, count)
