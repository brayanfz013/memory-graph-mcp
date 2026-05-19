"""Embedding provider quality benchmark.

For each (provider, model) tuple, the benchmark:
  1. Spins up an isolated temp workspace + DuckDB.
  2. Seeds it with the curated `eval/eval_set.json` findings.
  3. Runs each query in the eval set through `unified.recall`.
  4. Computes Recall@1, Recall@5, MRR, mean_latency_ms.
  5. Returns a comparison table.

It does NOT touch the user's real workspace — every benchmark run is
isolated. Safe to call from inside an active project.

Caveats:
  - Eval set is small (12 seeds, 13 queries) — treat as a directional signal,
    not a definitive ranking. Extend `eval/eval_set.json` for your own corpus
    to get reliable numbers on your actual content.
  - Each (provider, model) probe loads the model from scratch. First fastembed
    run downloads ~33-300 MB depending on model. Subsequent runs are cached.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.json"
_MODULE_NAMES = (
    "memory_graph.settings",
    "memory_graph.db",
    "memory_graph.embeddings",
    "memory_graph.vector_store",
    "memory_graph.knowledge_graph",
    "memory_graph.collective",
    "memory_graph.tool_cache",
    "memory_graph.wiki",
    "memory_graph.intelligence",
    "memory_graph.unified",
    "memory_graph.embedding_admin",
)


def _load_eval_set(path: Path | None = None) -> dict[str, Any]:
    src = path or EVAL_SET_PATH
    with open(src, encoding="utf-8") as fh:
        return json.load(fh)


def embedding_benchmark(
    providers: list[dict[str, str]] | None = None,
    eval_set_path: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the eval set against each (provider, model) combo and return metrics.

    `providers` is a list of dicts like
        [{"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
         {"provider": "fastembed", "model": "intfloat/multilingual-e5-base"}]
    If omitted, defaults to the 3 fastembed models in PROVIDER_REGISTRY.

    Each combo runs in an isolated temp workspace — your real workspace is
    never touched. The temp workspace is deleted after the benchmark.
    """
    from .settings import PROVIDER_REGISTRY

    if providers is None:
        providers = [
            {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
            {"provider": "fastembed", "model": "BAAI/bge-base-en-v1.5"},
            {"provider": "fastembed", "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"},
        ]

    eval_set = _load_eval_set(Path(eval_set_path) if eval_set_path else None)
    seeds = eval_set["seeds"]
    queries = eval_set["queries"]

    results = []
    for combo in providers:
        provider = combo["provider"]
        model = combo["model"]
        if provider not in PROVIDER_REGISTRY or model not in PROVIDER_REGISTRY.get(provider, {}):
            results.append({
                "provider": provider,
                "model": model,
                "error": "Not in PROVIDER_REGISTRY",
            })
            continue
        try:
            metrics = _benchmark_one(provider, model, seeds, queries, top_k)
        except Exception as exc:
            logger.exception("Benchmark failed for %s/%s", provider, model)
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
        results.append({"provider": provider, "model": model, **metrics})

    return {
        "top_k": top_k,
        "seeds": len(seeds),
        "queries": len(queries),
        "results": results,
    }


def _benchmark_one(
    provider: str,
    model: str,
    seeds: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Spin up an isolated workspace, seed, run queries, compute metrics."""
    workspace = Path(tempfile.mkdtemp(prefix="mg-bench-"))
    prev_workspace_env = os.environ.get("MEMORY_GRAPH_WORKSPACE")
    try:
        (workspace / ".git").mkdir()
        # Workspace is read directly from this env var by resolve_workspace_path().
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(workspace)

        # Force re-import so settings re-resolves the workspace.
        # IMPORTANT: also pop the `memory_graph` parent package — Python caches
        # sub-module references as attributes on the package, so popping only
        # sys.modules['memory_graph.X'] is not enough to make
        # `from . import X` re-import.
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.pop("memory_graph", None)
        intel = importlib.import_module("memory_graph.intelligence")

        # NB: Pydantic-settings here is configured WITHOUT env_prefix, so
        # MEMORY_GRAPH_FASTEMBED_MODEL / _PROVIDER / _OLLAMA_MODEL are NOT
        # auto-bound. We mutate the freshly-loaded settings instance directly
        # and reset the provider cache so the next embed call rebuilds against
        # the new identity.
        _settings_mod = importlib.import_module("memory_graph.settings")
        _settings = _settings_mod.settings
        _settings.embedding_provider = provider
        if provider == "fastembed":
            _settings.fastembed_model = model
        elif provider == "ollama":
            _settings.ollama_model = model
        # vertex uses model_name from the class itself, no setting needed.
        _emb_mod = importlib.import_module("memory_graph.embeddings")
        _emb_mod.reset_provider_cache()
        _db_mod = importlib.import_module("memory_graph.db")
        _db_mod.reset_embedding_dimensions_cache()
        unified = importlib.import_module("memory_graph.unified")
        emb = importlib.import_module("memory_graph.embeddings")

        # Warm up + measure cold latency
        t0 = time.perf_counter()
        identity = emb.get_identity()
        cold_warmup_ms = (time.perf_counter() - t0) * 1000

        # Seed the workspace
        seed_id_to_node_id: dict[str, str] = {}
        for seed in seeds:
            rec = intel.memory_record_finding(
                finding_type=seed["finding_type"],
                title=seed["title"],
                content=seed["content"],
                tags=[seed.get("language", "en"), "benchmark-seed"],
            )
            seed_id_to_node_id[seed["id"]] = rec.get("node_id") or rec.get("canonical_id") or ""

        # Run queries
        per_query: list[dict[str, Any]] = []
        latencies: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            recall = unified.recall(query=q["query"], top_k=top_k, scope="memories")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            ranked_node_ids = [
                m.get("metadata", {}).get("node_id")
                or m.get("id")
                or _extract_node_id_fallback(m)
                for m in recall.get("memories", [])
            ]
            expected_node_id = seed_id_to_node_id.get(q["expected_seed_id"], "")
            rank = _find_rank(ranked_node_ids, expected_node_id, recall.get("memories", []), q["expected_seed_id"], seeds)
            per_query.append({
                "query": q["query"],
                "expected": q["expected_seed_id"],
                "rank": rank,
                "found": rank is not None and rank <= top_k,
                "latency_ms": round(elapsed_ms, 2),
            })

        # Aggregate metrics
        recall_at_1 = sum(1 for r in per_query if r["rank"] == 1) / len(per_query)
        recall_at_5 = sum(1 for r in per_query if r["rank"] is not None and r["rank"] <= 5) / len(per_query)
        reciprocal_ranks = [1.0 / r["rank"] for r in per_query if r["rank"] is not None]
        mrr = sum(reciprocal_ranks) / len(per_query) if per_query else 0.0
        mean_latency = statistics.mean(latencies) if latencies else 0.0

        return {
            "identity": identity,
            "recall@1": round(recall_at_1, 3),
            "recall@5": round(recall_at_5, 3),
            "mrr": round(mrr, 3),
            "mean_latency_ms": round(mean_latency, 2),
            "cold_warmup_ms": round(cold_warmup_ms, 2),
            "per_query": per_query,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        if prev_workspace_env is None:
            os.environ.pop("MEMORY_GRAPH_WORKSPACE", None)
        else:
            os.environ["MEMORY_GRAPH_WORKSPACE"] = prev_workspace_env
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.pop("memory_graph", None)


def _extract_node_id_fallback(memory_row: dict[str, Any]) -> str:
    """Different code paths store node_id in different places — try harder."""
    md = memory_row.get("metadata") or {}
    if isinstance(md, dict):
        if md.get("node_id"):
            return md["node_id"]
        if md.get("canonical_id"):
            return md["canonical_id"]
    return memory_row.get("memory_id", "") or ""


def _find_rank(
    ranked_ids: list[str],
    expected_node_id: str,
    raw_memories: list[dict[str, Any]],
    expected_seed_id: str,
    seeds: list[dict[str, Any]],
) -> int | None:
    """Return 1-based rank of expected_node_id in ranked_ids, or None if absent.

    Fallback: if node_id matching fails (some paths return memory_id only),
    match by content substring against the seed's content.
    """
    if expected_node_id:
        for i, rid in enumerate(ranked_ids, start=1):
            if rid and (rid == expected_node_id):
                return i
    # Substring fallback: scan recall results for the seed's content
    seed = next((s for s in seeds if s["id"] == expected_seed_id), None)
    if seed is None:
        return None
    needle = (seed["content"][:80] or "").lower()
    for i, mem in enumerate(raw_memories, start=1):
        if needle and needle in (mem.get("content") or "").lower():
            return i
    return None


__all__ = ["embedding_benchmark"]
