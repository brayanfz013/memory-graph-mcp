"""Embedding abstraction — pluggable providers.

Supported providers (settable via EMBEDDING_PROVIDER):
  - fastembed (default, local ONNX, no API key)
  - ollama    (local HTTP API at localhost:11434)
  - vertex    (Google Cloud Vertex AI)

Each provider exposes a uniform interface:
  - embed_texts(texts, task_type) -> list[list[float]]
  - embed_query(query)            -> list[float]
  - dimensions                    -> int
  - provider                      -> str   (the provider name)
  - model_name                    -> str   (the specific model identifier)

`provider` + `model_name` + `dimensions` form the *embedding identity* —
persisted in DB via `embedding_meta` so a mismatch on startup can be detected
and the user can be guided to run `embedding_migrate`.
"""

from __future__ import annotations

import logging
from typing import Any

from .settings import settings

logger = logging.getLogger(__name__)

_provider_instance: Any = None


def _build_provider() -> Any:
    """Instantiate the configured provider. Called lazily on first use."""
    name = settings.embedding_provider.lower()
    if name == "vertex":
        return _VertexProvider()
    if name == "ollama":
        return _OllamaProvider()
    if name == "fastembed":
        return _FastEmbedProvider()
    raise ValueError(
        f"Unknown embedding_provider {name!r}. "
        "Supported: fastembed, ollama, vertex."
    )


def _get_provider() -> Any:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance
    _provider_instance = _build_provider()
    logger.info(
        "Embedding provider: %s / %s (dims=%d)",
        _provider_instance.provider,
        _provider_instance.model_name,
        _provider_instance.dimensions,
    )
    return _provider_instance


def reset_provider_cache() -> None:
    """Force the next call to re-instantiate the provider.

    Used by `embedding_admin.embedding_migrate` after the env / settings
    change so the new provider takes effect without a process restart.
    """
    global _provider_instance
    _provider_instance = None


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    return _get_provider().embed_texts(texts, task_type)


def embed_query(query: str) -> list[float]:
    return _get_provider().embed_query(query)


def get_dimensions() -> int:
    return _get_provider().dimensions


def get_identity() -> dict[str, Any]:
    """Return the active provider+model+dimensions tuple — used for mismatch checks."""
    p = _get_provider()
    return {"provider": p.provider, "model": p.model_name, "dimensions": p.dimensions}


# ── FastEmbed (local, no API key) ──────────────────────────────


class _FastEmbedProvider:
    """Local embeddings via fastembed (ONNX, no PyTorch)."""

    provider = "fastembed"

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model_name = settings.fastembed_model
        # First-run download (~100 MB for BGE-small-en) happens inside TextEmbedding().
        # Log up front so a user staring at a stalled MCP server has a breadcrumb.
        logger.info(
            "fastembed: initializing model %s (first run may download ~100 MB from HuggingFace CDN, "
            "cache at ~/.cache/fastembed; this can take 1–2 minutes)",
            self.model_name,
        )
        try:
            self._model = TextEmbedding(self.model_name)
        except Exception as exc:  # noqa: BLE001 — surface any download/init failure clearly
            raise RuntimeError(
                f"fastembed failed to initialize model '{self.model_name}'. "
                f"Common causes: offline machine, corporate proxy blocking https://huggingface.co, "
                f"or partial download in ~/.cache/fastembed (delete the cache and retry). "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        # Detect dimensions from a test embedding
        test = list(self._model.embed(["test"]))
        self.dimensions = len(test[0])
        logger.info("fastembed loaded: %s (%d-dim)", self.model_name, self.dimensions)

    def embed_texts(self, texts: list[str], task_type: str = "") -> list[list[float]]:
        del task_type  # part of the provider contract; FastEmbed has no task-type concept
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


# ── Ollama (local HTTP) ────────────────────────────────────────


class _OllamaProvider:
    """Local embeddings via Ollama HTTP API at $OLLAMA_BASE_URL.

    Requires `ollama pull <model>` to have been run for the configured model.
    Falls back to a clear error if the server isn't running.
    """

    provider = "ollama"

    def __init__(self) -> None:
        import urllib.error
        import urllib.request

        self._urllib_request = urllib.request
        self._urllib_error = urllib.error
        self.model_name = settings.ollama_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/api/embeddings"

        # Probe with a tiny embedding to detect dimensions + verify server is up
        try:
            probe = self._raw_embed("test")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama embedding server unreachable at {self._base_url}. "
                f"Start it with `ollama serve` and pull the model with "
                f"`ollama pull {self.model_name}`. Underlying error: {exc}"
            ) from exc
        self.dimensions = len(probe)
        logger.info(
            "Ollama provider ready: model=%s endpoint=%s dims=%d",
            self.model_name, self._endpoint, self.dimensions,
        )

    def _raw_embed(self, text: str) -> list[float]:
        import json

        body = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        req = self._urllib_request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urllib_request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        embedding = payload.get("embedding") or payload.get("embeddings")
        if not embedding:
            raise RuntimeError(f"Ollama returned no embedding for model {self.model_name}: {payload}")
        # When using newer /api/embed endpoints, response is {"embeddings": [[...]]}
        if isinstance(embedding[0], list):
            return embedding[0]
        return embedding

    def embed_texts(self, texts: list[str], task_type: str = "") -> list[list[float]]:
        del task_type  # part of the provider contract; Ollama has no task-type concept
        # Ollama's /api/embeddings is single-prompt; loop for batches.
        return [self._raw_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._raw_embed(query)


# ── Vertex AI (Google Cloud) ──────────────────────────────────


class _VertexProvider:
    """Google Cloud Vertex AI embeddings."""

    provider = "vertex"
    dimensions = 768
    model_name = "text-embedding-005"

    def __init__(self) -> None:
        import os
        import time

        self._time = time

        if not settings.google_project_id:
            raise RuntimeError(
                "Vertex AI embedding provider requires a Google Cloud project. "
                "Set MEMORY_GRAPH_GOOGLE_PROJECT_ID (or google_project_id in .env). "
                "Also export GOOGLE_APPLICATION_CREDENTIALS pointing at a service "
                "account JSON file with the Vertex AI User role."
            )
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS pointing at a "
                "service account JSON file. Either export it or use Application "
                "Default Credentials via `gcloud auth application-default login`."
            )

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            vertexai=True,
            project=settings.google_project_id,
            location=settings.google_region,
        )
        self._batch_size = 100
        logger.info("Vertex AI client initialized (project=%s)", settings.google_project_id)

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=self._types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
            out.extend(list(e.values) for e in resp.embeddings)
            if i + self._batch_size < len(texts):
                self._time.sleep(0.5)
        return out

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], task_type="RETRIEVAL_QUERY")[0]


__all__ = [
    "embed_texts",
    "embed_query",
    "get_dimensions",
    "get_identity",
    "reset_provider_cache",
]
