"""Settings for memory-graph MCP server.

Storage is workspace-scoped: the DB goes to <workspace>/.memory-graph/memory.duckdb
so each project keeps its own knowledge isolated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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


def _resolve_default_db() -> str:
    """Return a workspace-scoped DB path based on the validated workspace root."""
    workspace = resolve_workspace_path()
    db_dir = Path(workspace) / ".memory-graph"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "memory.duckdb")


class MemoryGraphSettings(BaseSettings):
    """Configuration loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    workspace_path: str = Field(
        default_factory=lambda: str(resolve_workspace_path()),
        description="Validated workspace root for repo-scoped memory",
    )
    db_path: str = Field(default_factory=_resolve_default_db, description="Path to DuckDB file")
    lock_path: str = Field(
        default_factory=lambda: str(resolve_workspace_path() / ".memory-graph" / "server.lock"),
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
