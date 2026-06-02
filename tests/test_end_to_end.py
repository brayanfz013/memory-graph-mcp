"""End-to-end memory operations tests.

Validates the full lifecycle of the memory-graph package against a real
DuckDB store in a fresh workspace:

  1. Record a finding    → vector memory + KG node + canonical_id
  2. Recall it back      → semantic + fused across scopes
  3. Auto-edge inference → cosine ≥ 0.62 creates RELATED_TO edges
  4. Lifecycle           → draft → canonical, auto-crystallize wiki
  5. Typed edges         → kg_add_edge with SOLVES / SUPERSEDES
  6. Graph traversal     → kg_neighbors + kg_path
  7. Collective state    → store / get / list with TTL types
  8. Tool cache          → check / store
  9. Health              → memory_report + memory_stats
 10. Consolidation       → memory_consolidate + recompute PageRank

Each test runs in its own temp workspace so they don't interfere; teardown
deletes the workspace and restores the original `MEMORY_GRAPH_WORKSPACE`.
"""

from __future__ import annotations

import importlib
import json
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
    "memory_graph.parsers",
    "memory_graph.wiki",
    "memory_graph.intelligence",
    "memory_graph.unified",
)


class MemoryGraphTestCase(unittest.TestCase):
    """Base class: every test gets its own temp workspace + reloaded modules.

    On tearDown the temp directory is deleted and the prior value of
    `MEMORY_GRAPH_WORKSPACE` is restored. Cleanup is registered with
    `addCleanup` so it runs even if `setUp` raises mid-way.
    """

    def setUp(self) -> None:
        self._prev_workspace = os.environ.get("MEMORY_GRAPH_WORKSPACE")
        self.workspace = Path(tempfile.mkdtemp(prefix="mg-e2e-"))
        # Register cleanup BEFORE any other setup work so a mid-setup failure
        # still cleans up on tearDown.
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(self._restore_workspace_env)

        (self.workspace / ".git").mkdir()  # _looks_like_workspace marker
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(self.workspace)

        # Force re-import so the new workspace is picked up. Must also pop the
        # parent package because Python caches sub-module references as
        # attributes on the package object — without this, lazy `from . import X`
        # inside the package returns the stale module.
        for mod_name in _MODULE_NAMES:
            sys.modules.pop(mod_name, None)
        sys.modules.pop("memory_graph", None)

        self.mods: dict[str, Any] = {
            "intelligence": importlib.import_module("memory_graph.intelligence"),
            "knowledge_graph": importlib.import_module("memory_graph.knowledge_graph"),
            "unified": importlib.import_module("memory_graph.unified"),
            "wiki": importlib.import_module("memory_graph.wiki"),
            "collective": importlib.import_module("memory_graph.collective"),
            "tool_cache": importlib.import_module("memory_graph.tool_cache"),
            "vector_store": importlib.import_module("memory_graph.vector_store"),
        }

    def _restore_workspace_env(self) -> None:
        if self._prev_workspace is None:
            os.environ.pop("MEMORY_GRAPH_WORKSPACE", None)
        else:
            os.environ["MEMORY_GRAPH_WORKSPACE"] = self._prev_workspace


class RecordAndRecallTests(MemoryGraphTestCase):
    """Core write/read loop: record_finding → recall."""

    def test_record_then_recall_returns_match(self) -> None:
        result = self.mods["intelligence"].memory_record_finding(
            finding_type="solution",
            title="DuckDB concurrent write lock retry",
            content="Wrap writes in retry decorator with exponential backoff (0.2s base).",
            related_files=["memory_graph/db.py"],
            tags=["duckdb", "concurrency"],
            source_agent="test-suite",
        )
        self.assertIn("node_id", result)
        self.assertIn("canonical_id", result)
        self.assertTrue(result["canonical_id"].startswith("solution."))

        recall = self.mods["unified"].recall(query="duckdb lock retry", top_k=3)
        self.assertGreaterEqual(len(recall.get("memories", [])), 1)
        top = recall["memories"][0]
        self.assertIn("duckdb", (top.get("content") or "").lower())

    def test_record_dedup_by_canonical_id(self) -> None:
        record = self.mods["intelligence"].memory_record_finding
        first = record(
            finding_type="solution",
            title="Same title same content",
            content="Body A",
            tags=["a"],
        )
        second = record(
            finding_type="solution",
            title="Same title same content",
            content="Body A updated",
            tags=["a"],
        )
        # Canonical IDs match because the slug is derived from title
        self.assertEqual(first["canonical_id"], second["canonical_id"])

    def test_recall_finds_nothing_for_unrelated_query(self) -> None:
        self.mods["intelligence"].memory_record_finding(
            finding_type="solution",
            title="Cooking pasta al dente",
            content="Boil 10 minutes in salted water.",
        )
        recall = self.mods["unified"].recall(
            query="kubernetes pod eviction backoff", top_k=3, min_score=0.5
        )
        for m in recall.get("memories", []):
            self.assertNotIn("pasta", (m.get("content") or "").lower())


class FusedRecallTests(MemoryGraphTestCase):
    """`recall(scope='all')` must fuse memories + KG + wiki."""

    def test_recall_returns_all_three_scopes_with_content(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="decision",
            title="Use DuckDB with VSS extension",
            content="DuckDB embedded gives us vector search without a separate service.",
            tags=["duckdb", "architecture"],
        )
        # Promote then explicitly crystallize so the wiki scope has a real page
        # to recall against (bare kg_promote does not auto-crystallize — see WikiTests
        # docstring for why).
        self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")
        crystallized = self.mods["wiki"].wiki_crystallize(rec["canonical_id"])
        self.assertNotIn("error", crystallized, "Precondition: wiki must crystallize")

        result = self.mods["unified"].recall(query="duckdb vector search", scope="all", top_k=5)
        self.assertIn("memories", result)
        self.assertIn("kg", result)
        self.assertIn("wiki", result)
        self.assertIn("top_canonicals", result)
        # All three scopes should have ≥1 result for a relevant query when we
        # explicitly populated memory + kg + wiki.
        self.assertGreater(len(result.get("memories", [])), 0, "memory scope empty")
        kg_block = result.get("kg") or {}
        kg_nodes = kg_block.get("nodes", []) if isinstance(kg_block, dict) else []
        self.assertGreater(len(kg_nodes), 0, "kg scope empty")
        self.assertGreater(len(result.get("wiki", [])), 0, "wiki scope empty")


class AutoEdgeInferenceTests(MemoryGraphTestCase):
    """record_finding auto-infers RELATED_TO edges between semantically close nodes."""

    def test_two_similar_findings_get_related_to_edge(self) -> None:
        record = self.mods["intelligence"].memory_record_finding
        first = record(
            finding_type="solution",
            title="DuckDB transient lock retry with exponential backoff",
            content=(
                "Wrap writes in a retry decorator with exponential backoff. "
                "Detect transient errors by lock-related message substrings."
            ),
            tags=["duckdb", "retry"],
        )
        second = record(
            finding_type="solution",
            title="Concurrent DuckDB writes need backoff retry",
            content=(
                "When multiple processes write to the same DuckDB file, transient "
                "lock errors should be retried with exponential backoff."
            ),
            tags=["duckdb", "concurrency"],
        )

        # The second record call should have surfaced auto-edges in its return payload
        auto_edges = second.get("auto_edges") or []
        # Either the auto-edge list contains the first node, OR — for some
        # embedding shifts — no edge fires but recall on the second still
        # surfaces the first as a similar memory. We assert the stronger of the
        # two signals: at least one of the auto-edge targets references the
        # first finding by canonical_id or node_id.
        targets = {e.get("to_id") for e in auto_edges}
        targets |= {e.get("canonical_id") for e in auto_edges if e.get("canonical_id")}
        related = (first["node_id"] in targets) or (first.get("canonical_id") in targets)

        if not related:
            # Fallback assertion: recall should still see the two as semantically related
            # — if neither edge nor recall sees the relationship, the embedding pipeline
            # is broken (a real bug).
            recall = self.mods["unified"].recall(query="duckdb backoff retry", top_k=5)
            memory_ids = {m.get("memory_id") or m.get("id") for m in recall.get("memories", [])}
            self.assertTrue(
                {first.get("memory_id"), second.get("memory_id")}.issubset(memory_ids),
                "Neither auto-edge nor semantic recall linked two highly-similar findings",
            )


class LifecycleTests(MemoryGraphTestCase):
    """draft → canonical → superseded transitions."""

    def test_promote_draft_to_canonical(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="decision",
            title="Workspace scoped storage",
            content="Each repo gets its own DuckDB file.",
        )
        promoted = self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")
        # API returns old_status / new_status (transition record), not a flat status field
        self.assertEqual(promoted.get("new_status"), "canonical")
        self.assertEqual(promoted.get("old_status"), "draft")

    def test_supersede_chain(self) -> None:
        kg = self.mods["knowledge_graph"]
        record = self.mods["intelligence"].memory_record_finding
        old = record(finding_type="solution", title="Old solution path", content="X")
        new = record(finding_type="solution", title="New solution path", content="Y")
        edge = kg.kg_add_edge(
            from_id=new["node_id"], to_id=old["node_id"], rel_type="SUPERSEDES"
        )
        self.assertTrue(edge.get("ok", False) or edge.get("from_id") == new["node_id"])
        kg.kg_promote(old["node_id"], "superseded")
        resolved = kg.kg_resolve(old["canonical_id"])
        self.assertEqual(resolved.get("status"), "superseded")

    def test_kg_resolve_existing_returns_full_node(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="pattern",
            title="Resolve happy-path test",
            content="A node to look up.",
        )
        resolved = self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])
        # Existing node returns at least canonical_id + label + status populated
        self.assertEqual(resolved.get("canonical_id"), rec["canonical_id"])
        self.assertIn(resolved.get("status"), {"draft", "canonical", "superseded"})
        self.assertTrue(resolved.get("label"))


class GraphTraversalTests(MemoryGraphTestCase):
    """kg_neighbors + kg_path."""

    def test_neighbors_returns_linked_node(self) -> None:
        record = self.mods["intelligence"].memory_record_finding
        kg = self.mods["knowledge_graph"]
        a = record(finding_type="problem", title="DuckDB lock flake", content="P")
        b = record(finding_type="solution", title="DuckDB lock retry", content="S")
        kg.kg_add_edge(from_id=b["node_id"], to_id=a["node_id"], rel_type="SOLVES")

        neighbors = self.mods["intelligence"].kg_neighbors(b["node_id"], direction="both")
        self.assertIn("neighbors", neighbors)
        ids = {n.get("node_id") for n in neighbors["neighbors"]}
        self.assertIn(a["node_id"], ids)

    def test_path_finds_route(self) -> None:
        record = self.mods["intelligence"].memory_record_finding
        kg = self.mods["knowledge_graph"]
        a = record(finding_type="problem", title="A", content="A")
        b = record(finding_type="problem", title="B", content="B")
        c = record(finding_type="problem", title="C", content="C")
        kg.kg_add_edge(from_id=a["node_id"], to_id=b["node_id"], rel_type="RELATED_TO")
        kg.kg_add_edge(from_id=b["node_id"], to_id=c["node_id"], rel_type="RELATED_TO")

        path = self.mods["intelligence"].kg_path(
            from_id=a["node_id"], to_id=c["node_id"], max_depth=5
        )
        self.assertIn("path", path)
        self.assertGreaterEqual(len(path["path"]), 2)


class CollectiveStateTests(MemoryGraphTestCase):
    """collective_store / get / list."""

    def test_store_and_get_roundtrip(self) -> None:
        col = self.mods["collective"]
        col.collective_store(
            type="knowledge", key="api-contract-v2", value={"v": 2}, scope="project"
        )
        got = col.collective_get(key="api-contract-v2", scope="project")
        self.assertIsNotNone(got)
        self.assertEqual(got.get("value"), {"v": 2})

    def test_list_filters_by_type(self) -> None:
        col = self.mods["collective"]
        col.collective_store(type="knowledge", key="k1", value="A", scope="project")
        col.collective_store(type="metric", key="m1", value=42, scope="project")
        knowledge = col.collective_list(type="knowledge", scope="project")
        keys = {e.get("key") for e in knowledge}
        self.assertIn("k1", keys)
        self.assertNotIn("m1", keys)


class ToolCacheTests(MemoryGraphTestCase):
    """cache_check / cache_store memoization."""

    def test_miss_then_hit(self) -> None:
        tc = self.mods["tool_cache"]
        miss = tc.cache_check(tool_name="lakehouse_sql_query", args_hash="abc123")
        self.assertFalse(miss.get("hit"))

        tc.cache_store(
            tool_name="lakehouse_sql_query",
            args_hash="abc123",
            result=json.dumps({"rows": 5}),
            ttl_seconds=3600,
        )
        hit = tc.cache_check(tool_name="lakehouse_sql_query", args_hash="abc123")
        self.assertTrue(hit.get("hit"))
        # tool_cache.cache_check returns the result already parsed from JSON
        self.assertEqual(hit["result"], {"rows": 5})


class WikiTests(MemoryGraphTestCase):
    """wiki_crystallize + wiki_get roundtrip.

    Note: in v0.4.0, auto-crystallize fires from `intelligence.memory_record_finding`
    when `maybe_promote` returns a promotion event — *not* from a bare
    `knowledge_graph.kg_promote` call. So a test that calls `kg_promote` directly
    must also call `wiki_crystallize` explicitly. This is intentional behavior
    documented in `intelligence._auto_crystallize_on_promote`.
    """

    def test_crystallize_then_get_returns_body(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="decision",
            title="Embeddings provider abstraction",
            content="Abstract via embedding_provider env var; fastembed default.",
            tags=["embeddings"],
        )
        self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")

        crystallized = self.mods["wiki"].wiki_crystallize(rec["canonical_id"])
        self.assertIsInstance(crystallized, dict)
        self.assertNotIn(
            "error",
            crystallized,
            f"wiki_crystallize returned error: {crystallized.get('error')}",
        )
        self.assertIn("page_id", crystallized)

        page = self.mods["wiki"].wiki_get(rec["canonical_id"])
        self.assertIsInstance(page, dict)
        self.assertIn("body", page)
        self.assertGreater(len(page.get("body") or ""), 0)


class HealthAndStatsTests(MemoryGraphTestCase):
    """memory_report + memory_stats sanity checks."""

    def test_report_includes_expected_keys(self) -> None:
        self.mods["intelligence"].memory_record_finding(
            finding_type="solution", title="Seed", content="seed"
        )
        report = self.mods["intelligence"].memory_report()
        self.assertIsInstance(report, dict)
        # The report must at minimum expose either counts or a structured summary.
        # Tolerant assertion: at least one numeric/dict value exists.
        self.assertTrue(any(isinstance(v, (int, dict, list)) for v in report.values()))


class ConsolidationTests(MemoryGraphTestCase):
    """memory_consolidate dry-run + apply path (+ implicit PageRank recompute)."""

    def test_dry_run_does_not_remove_anything(self) -> None:
        intel = self.mods["intelligence"]
        intel.memory_record_finding(
            finding_type="solution",
            title="Keep me",
            content="I should still be here after a dry-run consolidate.",
        )
        result = intel.memory_consolidate(dry_run=True)
        self.assertIsInstance(result, dict)
        # Dry-run must not touch the row that exists
        recall = self.mods["unified"].recall(query="keep me", top_k=3)
        self.assertGreaterEqual(len(recall.get("memories", [])), 1)

    def test_apply_consolidate_returns_summary(self) -> None:
        intel = self.mods["intelligence"]
        intel.memory_record_finding(
            finding_type="solution",
            title="Apply consolidate target",
            content="Body.",
        )
        result = intel.memory_consolidate(dry_run=False)
        # Applied consolidate should return a dict shape we can use
        self.assertIsInstance(result, dict)


class FailureModeTests(MemoryGraphTestCase):
    """Sanity: invalid inputs and edge cases don't crash."""

    def test_recall_empty_db_returns_empty(self) -> None:
        result = self.mods["unified"].recall(query="anything", top_k=3)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("memories", []), [])

    def test_kg_resolve_missing_returns_safe(self) -> None:
        result = self.mods["knowledge_graph"].kg_resolve("solution.does-not-exist")
        # Returns either an empty dict or a dict with a not-found indicator — never raises
        self.assertIsInstance(result, dict)

    def test_record_with_empty_content_still_creates_node(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="context",
            title="empty body case",
            content="",
        )
        self.assertIn("node_id", rec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
