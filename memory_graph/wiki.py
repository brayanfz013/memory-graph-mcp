"""Wiki layer — crystallized, curated knowledge pages.

The wiki stores human-refined knowledge that outlives ephemeral memory.
Each wiki page is linked to a canonical KG node via canonical_id and
provides long-form, authoritative documentation.

Pages live in the `wiki_pages` DuckDB table with full-text search.

The wiki_bootstrap function scans the workspace and auto-populates
the wiki with structural knowledge about the codebase: directory map,
READMEs, module docstrings, config summaries, and architecture docs.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import get_connection, with_retry
from .parsers import parse_code_file
from .settings import settings

logger = logging.getLogger(__name__)

# Supported file extensions for code summarization
_SUPPORTED_EXTS = {
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx", ".mts", ".cts",
    ".cls", ".mac", ".inc", ".rtn",
}


def _ensure_wiki_table(conn: Any) -> None:
    """Create the wiki_pages and wiki_embeddings tables if they don't exist.

    Kept for backward-compatibility with older DBs; normally created by db._init_schema.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wiki_pages (
            page_id VARCHAR PRIMARY KEY,
            canonical_id VARCHAR,
            title VARCHAR NOT NULL,
            body TEXT NOT NULL,
            tags_json VARCHAR DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            author VARCHAR DEFAULT 'agent',
            status VARCHAR DEFAULT 'active'
        )
    """)
    from .db import get_embedding_dimensions
    dim = get_embedding_dimensions()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS wiki_embeddings (
            page_id VARCHAR PRIMARY KEY,
            vector  FLOAT[{dim}],
            FOREIGN KEY (page_id) REFERENCES wiki_pages(page_id)
        )
    """)


def _upsert_wiki_embedding(conn: Any, page_id: str, text: str) -> None:
    """Embed and persist the wiki page vector. Best-effort; failure is non-fatal."""
    try:
        from .embeddings import embed_texts
        vec = embed_texts([text[:8000]])[0]
        existing = conn.execute(
            "SELECT 1 FROM wiki_embeddings WHERE page_id = ?", [page_id],
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE wiki_embeddings SET vector = ? WHERE page_id = ?",
                [vec, page_id],
            )
        else:
            conn.execute(
                "INSERT INTO wiki_embeddings (page_id, vector) VALUES (?, ?)",
                [page_id, vec],
            )
    except Exception as exc:
        logger.warning("wiki_embedding upsert skipped for %s: %s", page_id, exc)


@with_retry()
def wiki_ingest(
    title: str,
    body: str,
    canonical_id: str | None = None,
    tags: list[str] | None = None,
    author: str = "agent",
) -> dict[str, Any]:
    """Create or update a wiki page + embedding. Idempotent on canonical_id."""
    with get_connection() as conn:
        _ensure_wiki_table(conn)
        now = datetime.now(timezone.utc)
        embed_text = f"{title}\n{body}"

        if canonical_id:
            existing = conn.execute(
                "SELECT page_id FROM wiki_pages WHERE canonical_id = ?",
                [canonical_id],
            ).fetchone()
            if existing:
                page_id = existing[0]
                conn.execute(
                    """UPDATE wiki_pages
                       SET title = ?, body = ?, tags_json = ?, updated_at = ?, author = ?
                       WHERE page_id = ?""",
                    [title, body, json.dumps(tags or []), now, author, page_id],
                )
                conn.execute(
                    "UPDATE kg_nodes SET wiki_page = ? WHERE canonical_id = ?",
                    [page_id, canonical_id],
                )
                _upsert_wiki_embedding(conn, page_id, embed_text)
                logger.info("Wiki page updated: %s (%s)", title, page_id)
                return {"page_id": page_id, "action": "updated", "title": title}

        page_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO wiki_pages
               (page_id, canonical_id, title, body, tags_json, created_at, updated_at, author)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [page_id, canonical_id, title, body, json.dumps(tags or []), now, now, author],
        )

        if canonical_id:
            conn.execute(
                "UPDATE kg_nodes SET wiki_page = ? WHERE canonical_id = ?",
                [page_id, canonical_id],
            )

        _upsert_wiki_embedding(conn, page_id, embed_text)
        logger.info("Wiki page created: %s (%s)", title, page_id)
        return {"page_id": page_id, "action": "created", "title": title}


@with_retry()
def wiki_crystallize(canonical_id: str) -> dict[str, Any]:
    """Promote a KG node to wiki by extracting its content into a wiki page."""
    with get_connection() as conn:
        _ensure_wiki_table(conn)

        row = conn.execute(
            """SELECT node_id, label, properties_json, tldr_32, brief_96, summary_256,
                      status, canonical_id
               FROM kg_nodes WHERE canonical_id = ?""",
            [canonical_id],
        ).fetchone()
        if not row:
            return {"error": f"No KG node with canonical_id: {canonical_id}"}

        label = row[1]
        props = json.loads(row[2]) if row[2] else {}
        tldr_32 = row[3] or ""
        brief_96 = row[4] or ""
        summary_256 = row[5] or ""
        content = props.get("content", "")

        sections = [f"# {label}\n"]
        if tldr_32:
            sections.append(f"**TL;DR:** {tldr_32}\n")
        if brief_96:
            sections.append(f"## Summary\n{brief_96}\n")
        if summary_256:
            sections.append(f"## Detail\n{summary_256}\n")
        if content and content not in (brief_96, summary_256):
            sections.append(f"## Full Content\n{content}\n")

        files = props.get("files", [])
        if files:
            file_list = "\n".join(f"- `{f}`" for f in files)
            sections.append(f"## Related Files\n{file_list}\n")

        tags = props.get("tags", [])
        body = "\n".join(sections)

    result = wiki_ingest(
        title=label,
        body=body,
        canonical_id=canonical_id,
        tags=tags,
        author="auto-crystallize",
    )
    result["source_node_id"] = row[0]
    return result


@with_retry()
def wiki_get(canonical_id_or_title: str) -> dict[str, Any]:
    """Fetch the full wiki page (no truncation) by canonical_id or exact title.

    Lookup priority:
      1. canonical_id (preferred — stable slug)
      2. exact title match (case-insensitive)
    Returns body in full, plus tags, author, timestamps, and linked KG node.
    """
    with get_connection() as conn:
        _ensure_wiki_table(conn)
        row = conn.execute(
            """SELECT page_id, canonical_id, title, body, tags_json,
                      created_at, updated_at, author, status
               FROM wiki_pages
               WHERE canonical_id = ? AND status = 'active'
               LIMIT 1""",
            [canonical_id_or_title],
        ).fetchone()

        if not row:
            row = conn.execute(
                """SELECT page_id, canonical_id, title, body, tags_json,
                          created_at, updated_at, author, status
                   FROM wiki_pages
                   WHERE LOWER(title) = LOWER(?) AND status = 'active'
                   ORDER BY updated_at DESC
                   LIMIT 1""",
                [canonical_id_or_title],
            ).fetchone()

        if not row:
            return {"error": f"No wiki page found for: {canonical_id_or_title}"}

        kg_node = None
        if row[1]:
            kg_row = conn.execute(
                """SELECT node_id, node_type, label, status, pagerank_score, reuse_count
                   FROM kg_nodes WHERE canonical_id = ?""",
                [row[1]],
            ).fetchone()
            if kg_row:
                kg_node = {
                    "node_id": kg_row[0],
                    "node_type": kg_row[1],
                    "label": kg_row[2],
                    "status": kg_row[3],
                    "pagerank_score": float(kg_row[4]) if kg_row[4] is not None else 0.0,
                    "reuse_count": kg_row[5] if kg_row[5] is not None else 0,
                }

        return {
            "page_id": row[0],
            "canonical_id": row[1],
            "title": row[2],
            "body": row[3],
            "tags": json.loads(row[4]) if row[4] else [],
            "created_at": str(row[5]),
            "updated_at": str(row[6]),
            "author": row[7],
            "status": row[8],
            "kg_node": kg_node,
        }


@with_retry()
def wiki_query(
    query: str,
    top_k: int = 5,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Search wiki pages by text matching on title and body."""
    with get_connection() as conn:
        _ensure_wiki_table(conn)

        filters = ["(title ILIKE ? OR body ILIKE ?)"]
        params: list[Any] = [f"%{query}%", f"%{query}%"]

        if tags:
            for tag in tags:
                filters.append("tags_json ILIKE ?")
                params.append(f"%{tag}%")

        where = " AND ".join(filters)
        rows = conn.execute(
            f"""SELECT page_id, canonical_id, title, body, tags_json,
                       created_at, updated_at, author, status
                FROM wiki_pages
                WHERE {where} AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT ?""",
            params + [top_k],
        ).fetchall()

        pages = []
        for row in rows:
            pages.append({
                "page_id": row[0],
                "canonical_id": row[1],
                "title": row[2],
                "body": row[3][:500] + ("…" if len(row[3]) > 500 else ""),
                "tags": json.loads(row[4]) if row[4] else [],
                "updated_at": str(row[6]),
                "author": row[7],
            })

        return {"count": len(pages), "pages": pages}


@with_retry()
def wiki_lint() -> dict[str, Any]:
    """Audit wiki health — find orphaned pages, stale content, missing links."""
    with get_connection() as conn:
        _ensure_wiki_table(conn)

        orphaned = conn.execute(
            """SELECT page_id, title FROM wiki_pages
               WHERE canonical_id IS NULL OR canonical_id NOT IN
                     (SELECT canonical_id FROM kg_nodes WHERE canonical_id IS NOT NULL)"""
        ).fetchall()

        missing_wiki = conn.execute(
            """SELECT node_id, canonical_id, label FROM kg_nodes
               WHERE status = 'canonical' AND
                     (wiki_page IS NULL OR wiki_page NOT IN
                      (SELECT page_id FROM wiki_pages))"""
        ).fetchall()

        return {
            "orphaned_pages": [{"page_id": r[0], "title": r[1]} for r in orphaned],
            "canonical_without_wiki": [
                {"node_id": r[0], "canonical_id": r[1], "label": r[2]}
                for r in missing_wiki
            ],
            "issues_total": len(orphaned) + len(missing_wiki),
        }


# ── Wiki Bootstrap — Auto-populate from workspace ─────────────

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".memory-graph", "dist", "build",
    ".tox", ".eggs", "*.egg-info", ".next", ".nuxt", ".output",
    "coverage", ".coverage", ".nyc_output",
}

_DOC_FILES = {
    "README.md", "readme.md", "README.rst", "ARCHITECTURE.md",
    "CONTRIBUTING.md", "CHANGELOG.md", "TODO.md", "DESIGN_DOCUMENT.md",
}

_MAX_FILE_CHARS = 8_000
_MAX_TREE_DEPTH = 4


def _should_skip_dir(name: str) -> bool:
    if name.startswith(".") and name not in {".github", ".vscode", ".claude", ".codex"}:
        return True
    return name in _SKIP_DIRS or name.endswith(".egg-info")


def _read_file_safe(path: Path, max_chars: int = _MAX_FILE_CHARS) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except (OSError, UnicodeDecodeError):
        return None


def _build_dir_tree(root: Path, max_depth: int = _MAX_TREE_DEPTH) -> str:
    lines: list[str] = [f"{root.name}/"]

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return
        dirs = [e for e in entries if e.is_dir() and not _should_skip_dir(e.name)]
        files = [e for e in entries if e.is_file()]
        items = dirs + files[:20]
        for i, entry in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(items) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
        if len(files) > 20:
            lines.append(f"{prefix}    ... and {len(files) - 20} more files")

    _walk(root, "", 0)
    return "\n".join(lines)


def _format_parsed_summary(parsed: dict[str, Any]) -> str:
    """Format a ParsedSummary dict into a markdown string for wiki body."""
    parts: list[str] = []

    if parsed.get("docstrings"):
        parts.append(f"**Module docs:** {parsed['docstrings'][0][:300]}")

    if parsed.get("classes"):
        parts.append(f"**Classes:** {', '.join(parsed['classes'][:15])}")

    if parsed.get("functions"):
        parts.append(f"**Functions:** {', '.join(parsed['functions'][:15])}")

    if parsed.get("methods"):
        parts.append(f"**Methods:** {', '.join(parsed['methods'][:15])}")

    if parsed.get("exports"):
        parts.append(f"**Exports:** {', '.join(parsed['exports'][:15])}")

    if parsed.get("interfaces"):
        parts.append(f"**Interfaces:** {', '.join(parsed['interfaces'][:15])}")

    if parsed.get("namespaces"):
        parts.append(f"**Namespaces:** {', '.join(parsed['namespaces'][:15])}")

    return "\n".join(parts) if parts else "*No public symbols detected.*"


def _summarize_directory(dir_path: Path, workspace_root: Path) -> dict[str, Any] | None:
    rel = dir_path.relative_to(workspace_root)
    readme_content = None
    code_files: list[Path] = []

    for doc_name in ("README.md", "readme.md", "README.rst"):
        readme_path = dir_path / doc_name
        if readme_path.exists():
            readme_content = _read_file_safe(readme_path)
            break

    try:
        code_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS
            and not f.name.startswith("_") or f.name == "__init__.py"
        ]
    except PermissionError:
        pass

    # Always create a page for directories with code files, even without README
    if not readme_content and not code_files:
        return None

    sections: list[str] = [f"# {rel}\n"]
    if readme_content:
        sections.append(f"## README\n{readme_content}\n")

    if code_files:
        summary_lines: list[str] = []
        for code_file in code_files[:10]:
            parsed = parse_code_file(code_file)
            if parsed:  # non-empty means at least some symbols were found
                summary_lines.append(f"### `{code_file.name}`\n{_format_parsed_summary(parsed)}")
        if summary_lines:
            sections.append("## Code Modules\n" + "\n\n".join(summary_lines) + "\n")

    try:
        subdirs = [d.name for d in sorted(dir_path.iterdir())
                    if d.is_dir() and not _should_skip_dir(d.name)]
    except PermissionError:
        subdirs = []
    if subdirs:
        sections.append("## Subdirectories\n" + ", ".join(f"`{d}`" for d in subdirs) + "\n")

    return {
        "rel_path": str(rel),
        "title": f"Directory: {rel}",
        "body": "\n".join(sections),
        "has_readme": readme_content is not None,
    }


@with_retry()
def wiki_bootstrap(
    force: bool = False,
    max_dirs: int = 30,
) -> dict[str, Any]:
    """Auto-populate wiki by scanning the workspace structure.

    Reads READMEs, module docstrings, directory structure, and config files
    to create initial wiki pages. Uses canonical_ids for idempotent re-runs.
    """
    workspace_root = Path(settings.workspace_path)
    if not workspace_root.exists():
        return {"error": f"Workspace not found: {workspace_root}"}

    created = 0
    updated = 0
    skipped = 0
    pages_info: list[dict[str, str]] = []

    # ── 1. Workspace overview page ──
    tree = _build_dir_tree(workspace_root)
    overview_cid = "workspace.overview"
    overview_body = (
        f"# Workspace: {workspace_root.name}\n\n"
        f"**Path:** `{workspace_root}`\n\n"
        f"## Directory Structure\n```\n{tree}\n```\n"
    )

    for doc_name in sorted(_DOC_FILES):
        doc_path = workspace_root / doc_name
        if doc_path.exists():
            content = _read_file_safe(doc_path)
            if content:
                overview_body += f"\n## {doc_name}\n{content}\n"

    with get_connection() as conn:
        _ensure_wiki_table(conn)
        existing = conn.execute(
            "SELECT page_id FROM wiki_pages WHERE canonical_id = ?",
            [overview_cid],
        ).fetchone()

    if existing and not force:
        skipped += 1
    else:
        result = wiki_ingest(
            title=f"Workspace: {workspace_root.name}",
            body=overview_body,
            canonical_id=overview_cid,
            tags=["bootstrap", "workspace", "overview"],
            author="wiki-bootstrap",
        )
        if result["action"] == "created":
            created += 1
        else:
            updated += 1
        pages_info.append({"canonical_id": overview_cid, "action": result["action"]})

    # ── 2. Per-directory pages ──
    dirs_analyzed = 0
    for dirpath, dirnames, _filenames in os.walk(workspace_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        if dirs_analyzed >= max_dirs:
            break

        current = Path(dirpath)
        if current == workspace_root:
            continue

        summary = _summarize_directory(current, workspace_root)
        if not summary:
            continue

        rel = summary["rel_path"]
        cid = f"workspace.dir-{rel.replace('/', '-').replace(' ', '-').lower()}"

        with get_connection() as conn:
            _ensure_wiki_table(conn)
            existing = conn.execute(
                "SELECT page_id FROM wiki_pages WHERE canonical_id = ?",
                [cid],
            ).fetchone()

        if existing and not force:
            skipped += 1
            dirs_analyzed += 1
            continue

        result = wiki_ingest(
            title=summary["title"],
            body=summary["body"],
            canonical_id=cid,
            tags=["bootstrap", "directory", rel.split("/")[0]],
            author="wiki-bootstrap",
        )
        if result["action"] == "created":
            created += 1
        else:
            updated += 1
        pages_info.append({"canonical_id": cid, "action": result["action"]})
        dirs_analyzed += 1

    # ── 3. Config files page ──
    config_sections: list[str] = []
    config_files = [
        ("pyproject.toml", "Python project config"),
        ("package.json", "Node.js project config"),
        (".vscode/mcp.json", "MCP server declarations"),
        (".claude/settings.json", "Claude Code settings"),
    ]
    for config_rel, desc in config_files:
        config_path = workspace_root / config_rel
        if config_path.exists():
            content = _read_file_safe(config_path, max_chars=3000)
            if content:
                config_sections.append(f"### {config_rel}\n{desc}\n```\n{content}\n```\n")

    if config_sections:
        config_cid = "workspace.configs"
        with get_connection() as conn:
            _ensure_wiki_table(conn)
            existing = conn.execute(
                "SELECT page_id FROM wiki_pages WHERE canonical_id = ?",
                [config_cid],
            ).fetchone()

        if existing and not force:
            skipped += 1
        else:
            config_body = "# Workspace Configuration Files\n\n" + "\n".join(config_sections)
            result = wiki_ingest(
                title="Workspace Configurations",
                body=config_body,
                canonical_id=config_cid,
                tags=["bootstrap", "config"],
                author="wiki-bootstrap",
            )
            if result["action"] == "created":
                created += 1
            else:
                updated += 1
            pages_info.append({"canonical_id": config_cid, "action": result["action"]})

    # ── 4. Crystallize canonical KG nodes that lack a wiki page ──
    canon_created = 0
    canon_updated = 0
    canon_pages: list[dict[str, str]] = []
    with get_connection() as conn:
        _ensure_wiki_table(conn)
        rows = conn.execute(
            """SELECT canonical_id, label
               FROM kg_nodes
               WHERE status IN ('canonical', 'draft')
                 AND canonical_id IS NOT NULL
               ORDER BY pagerank_score DESC, reuse_count DESC NULLS LAST
               LIMIT 200"""
        ).fetchall()

    for cid, label in rows:
        if not cid:
            continue
        with get_connection() as conn:
            existing = conn.execute(
                "SELECT page_id FROM wiki_pages WHERE canonical_id = ?", [cid],
            ).fetchone()
        if existing and not force:
            skipped += 1
            continue
        try:
            result = wiki_crystallize(cid)
        except Exception as exc:
            logger.warning("Bootstrap crystallize skipped for %s: %s", cid, exc)
            continue
        if "error" in result:
            continue
        if result.get("action") == "created":
            canon_created += 1
        else:
            canon_updated += 1
        canon_pages.append({"canonical_id": cid, "label": label, "action": result.get("action")})

    created += canon_created
    updated += canon_updated
    pages_info.extend(canon_pages)

    logger.info(
        "Wiki bootstrap complete: %d created, %d updated, %d skipped (incl. %d canonicals)",
        created, updated, skipped, canon_created + canon_updated,
    )
    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "total_pages": created + updated,
        "canonicals_seeded": canon_created + canon_updated,
        "pages": pages_info,
        "workspace": str(workspace_root),
    }
