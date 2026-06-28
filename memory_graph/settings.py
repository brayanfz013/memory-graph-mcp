"""Settings for memory-graph MCP server.

Storage is workspace-scoped: the DB goes to <workspace>/.memory-graph/memory.duckdb
so each project keeps its own knowledge isolated.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Global fallback storage root, used only when the active directory is not a
# safe project root (e.g. the home directory, or a folder with no project
# markers). Keeps memory off the project tree while still letting the server
# start instead of crashing at import time.
GLOBAL_FALLBACK_ROOT = Path.home() / ".memory-graph"

WORKSPACE_MARKERS = (
    ".git",
    ".vscode",
    ".codex",
    ".claude",
    "AGENTS.md",
    "CLAUDE.md",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
)
INVALID_WORKSPACE_PARTS = {
    "microsoft visual studio code",
    "microsoft vs code",
    "microsoft vscode",
    "visual studio code",
}


class WorkspaceResolutionError(RuntimeError):
    """Raised when the workspace path is missing or unsafe."""


def _iter_workspace_candidates() -> Iterable[tuple[str, str]]:
    """Yield workspace candidates in priority order."""
    keys = (
        "MEMORY_GRAPH_WORKSPACE",
        "CLAUDE_PROJECT_DIR",
        "CODEX_WORKSPACE_DIR",
    )
    for key in keys:
        value = os.environ.get(key)
        if value:
            yield key, value
    yield "PWD", os.getcwd()


def _normalize_workspace(raw_workspace: str) -> Path | None:
    """Normalize an env-provided workspace path, dropping unresolved placeholders."""
    if not raw_workspace or "${" in raw_workspace:
        return None
    return Path(raw_workspace).expanduser().resolve()


def _looks_like_workspace(workspace: Path) -> bool:
    """Return True when the path resembles a real project workspace."""
    return any((workspace / marker).exists() for marker in WORKSPACE_MARKERS)


def _validate_workspace(workspace: Path, source: str) -> Path:
    """Validate that the workspace looks like a project root and not an editor install dir."""
    if not workspace.exists() or not workspace.is_dir():
        raise WorkspaceResolutionError(
            f"Invalid workspace from {source}: {workspace} does not exist or is not a directory.",
        )

    workspace_name = workspace.name.casefold()
    if workspace_name in INVALID_WORKSPACE_PARTS:
        raise WorkspaceResolutionError(
            f"Invalid workspace from {source}: {workspace} looks like an editor installation directory.",
        )

    if workspace in {Path.home(), Path("/")}:
        raise WorkspaceResolutionError(
            f"Invalid workspace from {source}: refusing to use broad path {workspace}.",
        )

    if not _looks_like_workspace(workspace):
        raise WorkspaceResolutionError(
            "Invalid workspace from "
            f"{source}: {workspace} does not look like a project root. "
            f"Expected one of {WORKSPACE_MARKERS}.",
        )

    return workspace


def resolve_workspace_path() -> Path:
    """Resolve the active workspace path from the environment."""
    for source, raw_workspace in _iter_workspace_candidates():
        workspace = _normalize_workspace(raw_workspace)
        if workspace is None:
            continue
        return _validate_workspace(workspace, source)

    raise WorkspaceResolutionError(
        "Could not resolve MEMORY_GRAPH_WORKSPACE. Set MEMORY_GRAPH_WORKSPACE, "
        "CLAUDE_PROJECT_DIR, or CODEX_WORKSPACE_DIR to a valid project root.",
    )


def _best_effort_workspace() -> Path:
    """Return the active directory from env/cwd without validating it.

    Never raises. Used as the file-scanning root (e.g. wiki ingest) and as the
    key for fallback storage when the directory is not a safe project root.
    """
    for _source, raw_workspace in _iter_workspace_candidates():
        workspace = _normalize_workspace(raw_workspace)
        if workspace is not None and workspace.exists() and workspace.is_dir():
            return workspace
    return Path.cwd()


def resolve_storage() -> tuple[Path, Path]:
    """Resolve (workspace_path, db_dir) without ever raising.

    When the active directory is a safe project root, memory is scoped to
    ``<workspace>/.memory-graph`` (per-project isolation, the normal case).
    When it is not — the home directory, an editor install dir, or a folder
    with no project markers — we fall back to a per-path directory under
    ``~/.memory-graph/fallback/<name>-<hash>`` and warn on stderr, so the
    server still starts and tools keep working instead of the whole MCP
    connection dying at import time.
    """
    workspace = _best_effort_workspace()
    try:
        validated = resolve_workspace_path()
        db_dir = validated / ".memory-graph"
        db_dir.mkdir(parents=True, exist_ok=True)
        return validated, db_dir
    except WorkspaceResolutionError as exc:
        digest = hashlib.sha1(str(workspace).encode("utf-8")).hexdigest()[:12]
        label = workspace.name or "root"
        db_dir = GLOBAL_FALLBACK_ROOT / "fallback" / f"{label}-{digest}"
        db_dir.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(
            f"[memory-graph] WARNING: {exc} "
            f"Falling back to global storage at {db_dir}. "
            "Add a project marker (e.g. an empty .git or CLAUDE.md) to scope "
            "memory to this directory instead.\n",
        )
        sys.stderr.flush()
        return workspace, db_dir


# Resolved once at import time so workspace_path, db_path and lock_path stay
# mutually consistent (a single fallback decision, not three independent ones).
_WORKSPACE_PATH, _DB_DIR = resolve_storage()


class MemoryGraphSettings(BaseSettings):
    """Configuration loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    workspace_path: str = Field(
        default_factory=lambda: str(_WORKSPACE_PATH),
        description="Active workspace root for repo-scoped memory (file-scanning root)",
    )
    db_path: str = Field(
        default_factory=lambda: str(_DB_DIR / "memory.duckdb"),
        description="Path to DuckDB file",
    )
    lock_path: str = Field(
        default_factory=lambda: str(_DB_DIR / "server.lock"),
        description="Path to workspace-scoped server lock file",
    )
    max_entries: int = Field(default=10_000, description="Max collective memory entries (LRU)")
    pagerank_damping: float = Field(default=0.85, description="PageRank damping factor")
    pagerank_max_iter: int = Field(default=100, description="PageRank max iterations")

    # Embedding provider: "fastembed" (local), "ollama" (local API), or "vertex" (Google Cloud)
    embedding_provider: str = Field(
        default="fastembed",
        description=(
            "Embedding provider: 'fastembed' (local ONNX, no key) | "
            "'ollama' (local HTTP API at localhost:11434) | "
            "'vertex' (Google Cloud API)"
        ),
    )
    # fastembed settings — see PROVIDER_REGISTRY for supported model names
    fastembed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "fastembed model name. Common choices: "
            "BAAI/bge-small-en-v1.5 (384-dim, EN), "
            "BAAI/bge-base-en-v1.5 (768-dim, EN), "
            "intfloat/multilingual-e5-base (768-dim, multilingual incl. ES)"
        ),
    )
    # ollama settings (only used when embedding_provider=ollama)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP API base URL",
    )
    ollama_model: str = Field(
        default="nomic-embed-text",
        description=(
            "Ollama embedding model name. Common: nomic-embed-text (768-dim), "
            "mxbai-embed-large (1024-dim). Must be pulled first: 'ollama pull <model>'"
        ),
    )
    # vertex settings (only used when embedding_provider=vertex).
    # Must be set explicitly via MEMORY_GRAPH_GOOGLE_PROJECT_ID env var or .env
    # file when using vertex. Empty default surfaces a clear error from
    # _VertexProvider rather than silently failing against a wrong project.
    google_project_id: str = Field(default="")
    google_region: str = Field(default="us-central1")


# Registry of known embedding models and their dimensions.
# Used by embedding_admin to surface "what's available" and validate user choices.
# Adding a model here is informational only — the actual loading is done by the
# provider class which detects dimensions on first use.
PROVIDER_REGISTRY: dict[str, dict[str, dict[str, int | str]]] = {
    "fastembed": {
        "BAAI/bge-small-en-v1.5": {"dim": 384, "lang": "en", "note": "smallest, fastest"},
        "BAAI/bge-base-en-v1.5": {"dim": 768, "lang": "en", "note": "stronger EN"},
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
            "dim": 768, "lang": "multi", "note": "multilingual incl. ES at 768-dim",
        },
        "intfloat/multilingual-e5-large": {
            "dim": 1024, "lang": "multi", "note": "top multilingual, larger download",
        },
        "jinaai/jina-embeddings-v2-base-es": {
            "dim": 768, "lang": "es", "note": "Spanish-specialised",
        },
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "dim": 384, "lang": "multi", "note": "tiny multilingual, fast",
        },
    },
    "ollama": {
        "nomic-embed-text": {"dim": 768, "lang": "en", "note": "Matryoshka-capable"},
        "mxbai-embed-large": {"dim": 1024, "lang": "en", "note": "stronger but larger"},
    },
    "vertex": {
        "text-embedding-005": {"dim": 768, "lang": "multi", "note": "Google Cloud"},
    },
}


settings = MemoryGraphSettings()
