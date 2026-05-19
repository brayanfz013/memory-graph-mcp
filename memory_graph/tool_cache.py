"""Tool call result caching — avoid redundant MCP tool executions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import get_connection

logger = logging.getLogger(__name__)


def cache_check(tool_name: str, args_hash: str) -> dict[str, Any]:
    """Check if a cached result exists for the given tool call."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT result_json, created_at
               FROM tool_cache
               WHERE tool_name = ? AND args_hash = ?
                 AND (expires_at IS NULL OR expires_at > current_timestamp)""",
            [tool_name, args_hash],
        ).fetchone()

        if row is None:
            return {"hit": False}

        conn.execute(
            """UPDATE tool_cache SET last_accessed_at = current_timestamp
               WHERE tool_name = ? AND args_hash = ?""",
            [tool_name, args_hash],
        )

        return {"hit": True, "result": json.loads(row[0]), "cached_at": str(row[1])}


def cache_store(
    tool_name: str,
    args_hash: str,
    result: str,
    ttl_seconds: int = 3600,
) -> dict[str, Any]:
    """Cache a tool call result."""
    with get_connection() as conn:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)

        conn.execute(
            """INSERT INTO tool_cache (tool_name, args_hash, result_json, created_at, last_accessed_at, ttl_seconds, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (tool_name, args_hash) DO UPDATE SET
                   result_json = ?, last_accessed_at = ?, ttl_seconds = ?, expires_at = ?""",
            [tool_name, args_hash, result, now, now, ttl_seconds, expires_at,
             result, now, ttl_seconds, expires_at],
        )

        logger.info("Cache stored: %s (hash=%s, ttl=%ds)", tool_name, args_hash[:8], ttl_seconds)
        return {"tool_name": tool_name, "args_hash": args_hash, "expires_at": expires_at.isoformat()}
