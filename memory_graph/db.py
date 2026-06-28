"""DuckDB connection and schema management for memory-graph."""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, TypeVar

import duckdb

from .settings import settings

logger = logging.getLogger(__name__)

_dimensions: int | None = None
_schema_ready: bool = False   # run init + migrate only once per process
_vss_installed: bool = False  # INSTALL vss only once per process

SCHEMA_VERSION = 6

_T = TypeVar("_T")

_TRANSIENT_MESSAGES = frozenset({
    "Could not set lock",
    "database is locked",
    "lock on the write-ahead log",
    "write-ahead log",
    "unable to acquire",
})


def _is_transient(exc: Exception) -> bool:
    """Return True if the DuckDB error looks like a transient lock conflict."""
    msg = str(exc).lower()
    return any(fragment.lower() in msg for fragment in _TRANSIENT_MESSAGES)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
):
    """Decorator: retry a function on transient DuckDB errors with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (duckdb.IOException, duckdb.OperationalError, OSError) as exc:
                    if not _is_transient(exc) or attempt == max_attempts - 1:
                        raise
                    last_exc = exc
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "Transient DuckDB error (attempt %d/%d), retrying in %.2fs: %s",
                        attempt + 1, max_attempts, delay, exc,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


@contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Open a DuckDB connection, yield it, then close it.

    Schema init runs once per process via _schema_ready flag.
    VSS extension is installed once (_vss_installed) and loaded per connection.
    DuckDB's own WAL-based write lock handles concurrent access across processes;
    transient lock errors are retried by the with_retry decorator on callers.
    """
    global _schema_ready, _vss_installed
    conn = duckdb.connect(settings.db_path)
    try:
        if not _vss_installed:
            conn.execute("INSTALL vss;")
            _vss_installed = True
        conn.execute("LOAD vss;")
        conn.execute("SET hnsw_enable_experimental_persistence = true;")
        if not _schema_ready:
            _init_schema(conn)
            _migrate_schema(conn)
            _ensure_embedding_generation_seeded(conn)
            _schema_ready = True
        yield conn
    finally:
        conn.close()


def _ensure_embedding_generation_seeded(conn: duckdb.DuckDBPyConnection) -> None:
    """If embedding_meta is empty, seed it with the current active provider+model+dim.

    If a row exists and doesn't match the active provider, log a warning but don't
    raise here — embedding_admin.embedding_status surfaces the diff and
    embedding_admin.embedding_migrate handles the rewrite. Recall paths still work.
    """
    try:
        active = get_active_embedding_meta(conn)
        from .embeddings import get_identity
        current = get_identity()
        if active is None:
            record_embedding_generation(
                conn,
                provider=current["provider"],
                model_name=current["model"],
                dimensions=current["dimensions"],
            )
            logger.info(
                "Seeded embedding_meta gen=1 with %s/%s (dim=%d)",
                current["provider"], current["model"], current["dimensions"],
            )
            return
        mismatch = (
            active["provider"] != current["provider"]
            or active["model_name"] != current["model"]
            or active["dimensions"] != current["dimensions"]
        )
        if mismatch:
            logger.warning(
                "Embedding provider mismatch: DB has gen=%d %s/%s (dim=%d), env has %s/%s (dim=%d). "
                "Run embedding_migrate to reconcile before writing new findings.",
                active["generation"], active["provider"], active["model_name"], active["dimensions"],
                current["provider"], current["model"], current["dimensions"],
            )
    except Exception as exc:
        # Never let identity bookkeeping block a connection — log and continue.
        logger.warning("Could not seed/check embedding_meta: %s", exc)


def get_embedding_dimensions() -> int:
    """Return the embedding dimensions (detected from provider on first use)."""
    global _dimensions
    if _dimensions is None:
        from .embeddings import get_dimensions
        _dimensions = get_dimensions()
    return _dimensions


def reset_embedding_dimensions_cache() -> None:
    """Force re-detection on next call. Used after a provider swap."""
    global _dimensions
    _dimensions = None


def get_active_embedding_meta(conn: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    """Return the currently-active embedding_meta row, or None if no row exists."""
    row = conn.execute(
        "SELECT generation, provider, model_name, dimensions, created_at "
        "FROM embedding_meta WHERE is_active = true "
        "ORDER BY generation DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {
        "generation": row[0],
        "provider": row[1],
        "model_name": row[2],
        "dimensions": row[3],
        "created_at": str(row[4]) if row[4] else None,
    }


def record_embedding_generation(
    conn: duckdb.DuckDBPyConnection,
    provider: str,
    model_name: str,
    dimensions: int,
) -> int:
    """Mark all existing generations inactive, insert a new active one, return its id."""
    conn.execute("UPDATE embedding_meta SET is_active = false")
    next_gen_row = conn.execute(
        "SELECT COALESCE(MAX(generation), 0) + 1 FROM embedding_meta"
    ).fetchone()
    next_gen = int((next_gen_row or (1,))[0])
    conn.execute(
        "INSERT INTO embedding_meta (generation, provider, model_name, dimensions, is_active) "
        "VALUES (?, ?, ?, ?, true)",
        [next_gen, provider, model_name, dimensions],
    )
    return next_gen


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables and indices if they don't exist."""
    dim = get_embedding_dimensions()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id            VARCHAR PRIMARY KEY,
            type          VARCHAR NOT NULL,
            content       VARCHAR NOT NULL,
            metadata_json VARCHAR DEFAULT '{}',
            created_at    TIMESTAMP DEFAULT current_timestamp,
            updated_at    TIMESTAMP DEFAULT current_timestamp,
            ttl_seconds   INTEGER,
            expires_at    TIMESTAMP
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            id     VARCHAR PRIMARY KEY,
            vector FLOAT[{dim}],
            FOREIGN KEY (id) REFERENCES memories(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_nodes (
            node_id          VARCHAR PRIMARY KEY,
            node_type        VARCHAR NOT NULL,
            label            VARCHAR NOT NULL,
            properties_json  VARCHAR DEFAULT '{}',
            pagerank_score   DOUBLE DEFAULT 0.0,
            created_at       TIMESTAMP DEFAULT current_timestamp,
            canonical_id     VARCHAR UNIQUE,
            status           VARCHAR DEFAULT 'draft',
            tldr_32          VARCHAR,
            brief_96         VARCHAR,
            summary_256      VARCHAR,
            wiki_page        VARCHAR,
            confidence       DOUBLE DEFAULT 0.0,
            reuse_count      INTEGER DEFAULT 0,
            last_validated_at TIMESTAMP,
            topic_id         VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_edges (
            from_id    VARCHAR NOT NULL,
            to_id      VARCHAR NOT NULL,
            rel_type   VARCHAR NOT NULL,
            weight     DOUBLE DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (from_id, to_id, rel_type)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS collective_memory (
            key              VARCHAR NOT NULL,
            scope            VARCHAR NOT NULL DEFAULT 'global',
            type             VARCHAR NOT NULL,
            value_json       VARCHAR NOT NULL,
            created_at       TIMESTAMP DEFAULT current_timestamp,
            updated_at       TIMESTAMP DEFAULT current_timestamp,
            last_accessed_at TIMESTAMP DEFAULT current_timestamp,
            access_count     INTEGER DEFAULT 0,
            ttl_seconds      INTEGER,
            expires_at       TIMESTAMP,
            PRIMARY KEY (key, scope)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_cache (
            tool_name        VARCHAR NOT NULL,
            args_hash        VARCHAR NOT NULL,
            result_json      VARCHAR NOT NULL,
            created_at       TIMESTAMP DEFAULT current_timestamp,
            last_accessed_at TIMESTAMP DEFAULT current_timestamp,
            ttl_seconds      INTEGER,
            expires_at       TIMESTAMP,
            PRIMARY KEY (tool_name, args_hash)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS wiki_pages (
            page_id      VARCHAR PRIMARY KEY,
            canonical_id VARCHAR,
            title        VARCHAR NOT NULL,
            body         TEXT NOT NULL,
            tags_json    VARCHAR DEFAULT '[]',
            created_at   TIMESTAMP DEFAULT current_timestamp,
            updated_at   TIMESTAMP DEFAULT current_timestamp,
            author       VARCHAR DEFAULT 'agent',
            status       VARCHAR DEFAULT 'active'
        )
    """)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS wiki_embeddings (
            page_id VARCHAR PRIMARY KEY,
            vector  FLOAT[{dim}],
            FOREIGN KEY (page_id) REFERENCES wiki_pages(page_id)
        )
    """)

    _ensure_hnsw_index(conn)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # v5: embedding identity tracking. One row per generation so swapping
    # providers/models is detectable and migrations are auditable.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_meta (
            generation  INTEGER PRIMARY KEY,
            provider    VARCHAR NOT NULL,
            model_name  VARCHAR NOT NULL,
            dimensions  INTEGER NOT NULL,
            is_active   BOOLEAN NOT NULL DEFAULT true,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # v6: hierarchical topic clustering ("mind map"). Each kg_node may belong
    # to one coarse topic; kg_topics holds the topic registry.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_topics (
            topic_id    VARCHAR PRIMARY KEY,
            label       VARCHAR NOT NULL,
            summary     VARCHAR,
            size        INTEGER DEFAULT 0,
            subtopics   INTEGER DEFAULT 0,
            top_node_id VARCHAR,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)


def _migrate_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply additive schema migrations for existing databases."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current_version = (current[0] if current and current[0] else 0)

    if current_version >= SCHEMA_VERSION:
        return

    if current_version < 2:
        _add_column_safe(conn, "kg_nodes", "canonical_id", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "status", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "tldr_32", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "brief_96", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "summary_256", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "wiki_page", "VARCHAR")
        _add_column_safe(conn, "kg_nodes", "confidence", "DOUBLE")
        _add_column_safe(conn, "kg_nodes", "reuse_count", "INTEGER")
        _add_column_safe(conn, "kg_nodes", "last_validated_at", "TIMESTAMP")
        logger.info("Migrated schema to v2: perfector columns added to kg_nodes")

    if current_version < 3:
        # v3 was a no-op duplicate of v2; kept for version-bookkeeping only.
        logger.info("Migrated schema to v3: noop bookkeeping")

    if current_version < 4:
        dim = get_embedding_dimensions()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_pages (
                page_id      VARCHAR PRIMARY KEY,
                canonical_id VARCHAR,
                title        VARCHAR NOT NULL,
                body         TEXT NOT NULL,
                tags_json    VARCHAR DEFAULT '[]',
                created_at   TIMESTAMP DEFAULT current_timestamp,
                updated_at   TIMESTAMP DEFAULT current_timestamp,
                author       VARCHAR DEFAULT 'agent',
                status       VARCHAR DEFAULT 'active'
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS wiki_embeddings (
                page_id VARCHAR PRIMARY KEY,
                vector  FLOAT[{dim}],
                FOREIGN KEY (page_id) REFERENCES wiki_pages(page_id)
            )
        """)
        # Backfill: any kg_nodes without canonical_id get the node_id as fallback.
        # node_id already encodes type as `<type>.<hex>`, so we don't double-prefix.
        conn.execute("""
            UPDATE kg_nodes
            SET canonical_id = node_id
            WHERE canonical_id IS NULL
        """)
        # Repair v3-era doubled prefixes (e.g. 'solution.solution.<hex>' → 'solution.<hex>').
        conn.execute("""
            UPDATE kg_nodes
            SET canonical_id = SUBSTR(canonical_id, LENGTH(LOWER(node_type)) + 2)
            WHERE canonical_id LIKE LOWER(node_type) || '.' || LOWER(node_type) || '.%'
        """)
        # Backfill: reuse_count NULL → 0 so auto-promotion can fire.
        conn.execute("UPDATE kg_nodes SET reuse_count = 0 WHERE reuse_count IS NULL")
        conn.execute("UPDATE kg_nodes SET confidence = 0.0 WHERE confidence IS NULL")
        # Backfill status NULL → 'draft' for older nodes.
        conn.execute("UPDATE kg_nodes SET status = 'draft' WHERE status IS NULL")
        logger.info("Migrated schema to v4: wiki_pages/wiki_embeddings + canonical/reuse/status backfill")

    if current_version < 5:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_meta (
                generation  INTEGER PRIMARY KEY,
                provider    VARCHAR NOT NULL,
                model_name  VARCHAR NOT NULL,
                dimensions  INTEGER NOT NULL,
                is_active   BOOLEAN NOT NULL DEFAULT true,
                created_at  TIMESTAMP DEFAULT current_timestamp
            )
        """)
        logger.info("Migrated schema to v5: embedding_meta table added")

    if current_version < 6:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_topics (
                topic_id    VARCHAR PRIMARY KEY,
                label       VARCHAR NOT NULL,
                summary     VARCHAR,
                size        INTEGER DEFAULT 0,
                subtopics   INTEGER DEFAULT 0,
                top_node_id VARCHAR,
                created_at  TIMESTAMP DEFAULT current_timestamp
            )
        """)
        _add_column_safe(conn, "kg_nodes", "topic_id", "VARCHAR")
        logger.info("Migrated schema to v6: kg_topics + kg_nodes.topic_id (mind-map clustering)")

    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
        [SCHEMA_VERSION],
    )


def _add_column_safe(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add a column only if it doesn't already exist (idempotent).

    DuckDB does not support constraints (UNIQUE, DEFAULT) in ALTER TABLE ADD COLUMN,
    so we use bare types and apply defaults in application code instead.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")  # noqa: S608
    except Exception:
        pass


def _ensure_hnsw_index(conn: duckdb.DuckDBPyConnection) -> None:
    """Create HNSW index if it doesn't exist. hnsw_enable_experimental_persistence
    is set by get_connection() before this is called."""
    try:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_hnsw
            ON memory_embeddings USING HNSW (vector)
            WITH (metric = 'cosine')
        """)
    except duckdb.CatalogException:
        logger.debug("HNSW index already exists or table empty — skipping.")
