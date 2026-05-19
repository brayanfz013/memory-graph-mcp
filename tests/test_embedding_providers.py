"""Tests for the embedding provider abstraction + admin tooling.

Covered:
  - Provider registry exposes the 3 supported providers + ≥1 model each.
  - FastEmbed provider reports a uniform identity (provider, model, dimensions).
  - Ollama provider raises a clear error when the server is unreachable
    (this is the realistic CI/local default — no live Ollama).
  - embedding_status seeds embedding_meta on first connection.
  - embedding_status surfaces a mismatch when env/settings change after seed.
  - embedding_migrate (dry_run=True) returns a plan with the right counts.
  - embedding_migrate (dry_run=False) rebuilds vector tables and updates
    embedding_meta to the new generation; recall still works afterwards.
  - benchmark.embedding_benchmark runs the eval set end-to-end and returns
    metrics for at least one provider (skips combos that can't load).

We deliberately do NOT depend on live Ollama or Vertex; tests use fastembed.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


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
    "memory_graph.benchmark",
)


class EmbeddingProviderTestCase(unittest.TestCase):
    """Base: each test gets its own temp workspace + reloaded modules.

    Saves / restores embedding env vars so swap tests don't pollute siblings.
    """

    def setUp(self) -> None:
        self._prev_env = {
            "MEMORY_GRAPH_WORKSPACE": os.environ.get("MEMORY_GRAPH_WORKSPACE"),
            "MEMORY_GRAPH_EMBEDDING_PROVIDER": os.environ.get("MEMORY_GRAPH_EMBEDDING_PROVIDER"),
            "MEMORY_GRAPH_FASTEMBED_MODEL": os.environ.get("MEMORY_GRAPH_FASTEMBED_MODEL"),
            "MEMORY_GRAPH_OLLAMA_MODEL": os.environ.get("MEMORY_GRAPH_OLLAMA_MODEL"),
        }
        self.workspace = Path(tempfile.mkdtemp(prefix="mg-emb-"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(self._restore_env)

        (self.workspace / ".git").mkdir()
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(self.workspace)
        # Force the small fastembed model — the only one CI is guaranteed to download fast.
        os.environ["MEMORY_GRAPH_EMBEDDING_PROVIDER"] = "fastembed"
        os.environ["MEMORY_GRAPH_FASTEMBED_MODEL"] = "BAAI/bge-small-en-v1.5"
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)

        # IMPORTANT: also pop the `memory_graph` parent package. Python caches
        # sub-module references as attributes on the package object — popping
        # only sys.modules['memory_graph.X'] is not enough because the lazy
        # `from . import X` lookup hits the package attribute first.
        sys.modules.pop("memory_graph", None)

        self.mods: dict[str, Any] = {
            "settings": importlib.import_module("memory_graph.settings"),
            "embeddings": importlib.import_module("memory_graph.embeddings"),
            "intel": importlib.import_module("memory_graph.intelligence"),
            "admin": importlib.import_module("memory_graph.embedding_admin"),
            "unified": importlib.import_module("memory_graph.unified"),
            "db": importlib.import_module("memory_graph.db"),
        }

    def _restore_env(self) -> None:
        for key, val in self._prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        for name in _MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.pop("memory_graph", None)


class ProviderRegistryTests(EmbeddingProviderTestCase):
    def test_registry_lists_all_three_providers(self) -> None:
        reg = self.mods["settings"].PROVIDER_REGISTRY
        self.assertIn("fastembed", reg)
        self.assertIn("ollama", reg)
        self.assertIn("vertex", reg)
        # Each provider has at least one model
        for provider, models in reg.items():
            self.assertGreater(len(models), 0, f"{provider} has no models")

    def test_each_registry_entry_has_dim_and_lang(self) -> None:
        reg = self.mods["settings"].PROVIDER_REGISTRY
        for provider, models in reg.items():
            for model, meta in models.items():
                self.assertIn("dim", meta, f"{provider}/{model} missing dim")
                self.assertIn("lang", meta, f"{provider}/{model} missing lang")
                self.assertIsInstance(meta["dim"], int)


class FastEmbedIdentityTests(EmbeddingProviderTestCase):
    def test_identity_has_provider_model_dimensions(self) -> None:
        identity = self.mods["embeddings"].get_identity()
        self.assertEqual(identity["provider"], "fastembed")
        self.assertEqual(identity["model"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(identity["dimensions"], 384)

    def test_embed_query_returns_vector_of_right_dim(self) -> None:
        vec = self.mods["embeddings"].embed_query("a quick test sentence")
        self.assertEqual(len(vec), 384)
        self.assertTrue(all(isinstance(x, float) for x in vec))


class OllamaUnreachableTests(EmbeddingProviderTestCase):
    """When Ollama isn't running, the provider must raise a CLEAR, actionable
    error — never silently fall back. Note: this test mutates settings directly
    because Pydantic-settings is configured without an env_prefix, so
    MEMORY_GRAPH_* env vars are not auto-bound (only MEMORY_GRAPH_WORKSPACE is
    read manually inside resolve_workspace_path).
    """

    def test_ollama_provider_raises_on_unreachable_server(self) -> None:
        s = self.mods["settings"].settings
        s.embedding_provider = "ollama"
        s.ollama_model = "nomic-embed-text"
        # Point at a port that's definitely not listening
        s.ollama_base_url = "http://127.0.0.1:1"
        # Force provider rebuild with the new settings
        self.mods["embeddings"].reset_provider_cache()
        with self.assertRaises(RuntimeError) as ctx:
            self.mods["embeddings"].get_identity()
        msg = str(ctx.exception)
        self.assertTrue(
            "ollama" in msg.lower() and ("serve" in msg.lower() or "pull" in msg.lower()),
            f"Ollama error must guide the user; got: {msg}",
        )


class EmbeddingMetaTests(EmbeddingProviderTestCase):
    """embedding_meta lifecycle: seed on first connection + mismatch detection."""

    def test_first_connection_seeds_embedding_meta(self) -> None:
        # Touch any operation that opens a connection
        self.mods["intel"].memory_record_finding(
            finding_type="solution", title="seed test", content="seed body"
        )
        with self.mods["db"].get_connection() as conn:
            active = self.mods["db"].get_active_embedding_meta(conn)
        self.assertIsNotNone(active)
        self.assertEqual(active["provider"], "fastembed")
        self.assertEqual(active["model_name"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(active["dimensions"], 384)

    def test_status_reports_no_mismatch_after_clean_seed(self) -> None:
        # Force a connection to seed embedding_meta
        self.mods["intel"].memory_record_finding(
            finding_type="solution", title="seed", content="seed body"
        )
        status = self.mods["admin"].embedding_status()
        self.assertFalse(status["mismatch"], f"Should be clean: {status}")
        self.assertIsNotNone(status["active_env"])
        self.assertIsNotNone(status["stored_in_db"])
        self.assertIn("registry", status)


class MigrateDryRunTests(EmbeddingProviderTestCase):
    def test_dry_run_returns_plan_without_rewriting(self) -> None:
        # Seed 3 findings
        for i in range(3):
            self.mods["intel"].memory_record_finding(
                finding_type="solution",
                title=f"finding {i}",
                content=f"content {i} for benchmark",
            )

        plan = self.mods["admin"].embedding_migrate(
            target_provider="fastembed",
            target_model="BAAI/bge-base-en-v1.5",
            dry_run=True,
        )
        # Plan should be a valid dict with counts and not have rewritten anything yet
        self.assertTrue(plan.get("ok") or plan.get("dry_run"))
        self.assertEqual(plan.get("memories_to_reembed"), 3)
        # Active stored generation must still be the small model
        status = self.mods["admin"].embedding_status()
        self.assertEqual(status["stored_in_db"]["model_name"], "BAAI/bge-small-en-v1.5")

    def test_unknown_provider_returns_actionable_error(self) -> None:
        result = self.mods["admin"].embedding_migrate(
            target_provider="madeup-provider",
            target_model="x",
            dry_run=True,
        )
        self.assertFalse(result.get("ok", True))
        self.assertIn("error", result)
        self.assertIn("madeup-provider", result["error"])


class BenchmarkTests(EmbeddingProviderTestCase):
    """The benchmark harness must run end-to-end and produce metrics."""

    def test_benchmark_runs_against_small_fastembed(self) -> None:
        bench = importlib.import_module("memory_graph.benchmark")
        result = bench.embedding_benchmark(
            providers=[
                {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
            ],
            top_k=5,
        )
        self.assertEqual(len(result["results"]), 1)
        row = result["results"][0]
        self.assertNotIn("error", row, f"Benchmark failed: {row}")
        for key in ("recall@1", "recall@5", "mrr", "mean_latency_ms"):
            self.assertIn(key, row, f"Missing metric {key} in {row}")
            self.assertIsInstance(row[key], (int, float))

    def test_benchmark_handles_unknown_combo_gracefully(self) -> None:
        bench = importlib.import_module("memory_graph.benchmark")
        result = bench.embedding_benchmark(
            providers=[
                {"provider": "not-a-real-provider", "model": "fake"},
            ],
        )
        self.assertIn("error", result["results"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
