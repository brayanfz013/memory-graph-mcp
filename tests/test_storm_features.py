"""Hard tests for the STORM/Co-STORM-inspired features (v0.5.0).

Covers the four additions and — crucially — asserts they deliver *real*
behavioural impact, not just new keys:

  1. Hierarchical topic clustering (topics.build_topics / memory_map)
  2. Token-saving recall (compact dedup + group_topics mind map)
  3. Outline-first wiki (wiki_get outline_only / section) + Sources grounding
  4. memory_gaps coverage critic

Each test runs in its own temp workspace with freshly reloaded modules so the
workspace-scoped DuckDB store is isolated (same pattern as test_end_to_end).
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
    "memory_graph.topics",
    "memory_graph.unified",
)

# Three texts about the same theme (DuckDB lock retry) plus one clearly
# unrelated. The similar three should cluster; the outlier should not.
_DUCKDB_FINDINGS = [
    (
        "DuckDB transient lock retry with exponential backoff",
        "Wrap writes in a retry decorator with exponential backoff. Detect "
        "transient errors by lock-related message substrings on the DuckDB file.",
    ),
    (
        "Concurrent DuckDB writes need backoff retry",
        "When multiple processes write to the same DuckDB file, transient lock "
        "errors should be retried with exponential backoff and capped delay.",
    ),
    (
        "Retry decorator for DuckDB write-ahead log lock",
        "A with_retry decorator retries DuckDB WAL lock errors using exponential "
        "backoff so concurrent writers recover from transient lock conflicts.",
    ),
]
_OUTLIER = (
    "Cooking pasta al dente",
    "Boil dry pasta for about ten minutes in heavily salted water, then drain.",
)


class StormFeatureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_workspace = os.environ.get("MEMORY_GRAPH_WORKSPACE")
        self.workspace = Path(tempfile.mkdtemp(prefix="mg-storm-"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(self._restore_workspace_env)
        (self.workspace / ".git").mkdir()
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(self.workspace)

        for mod_name in _MODULE_NAMES:
            sys.modules.pop(mod_name, None)
        sys.modules.pop("memory_graph", None)

        self.mods: dict[str, Any] = {
            name.split(".")[-1]: importlib.import_module(name)
            for name in _MODULE_NAMES
            if name not in ("memory_graph.settings", "memory_graph.db",
                            "memory_graph.embeddings", "memory_graph.parsers")
        }

    def _restore_workspace_env(self) -> None:
        if self._prev_workspace is None:
            os.environ.pop("MEMORY_GRAPH_WORKSPACE", None)
        else:
            os.environ["MEMORY_GRAPH_WORKSPACE"] = self._prev_workspace

    def _seed_duckdb_cluster(self, sources: list[str] | None = None) -> list[dict[str, Any]]:
        rec = self.mods["intelligence"].memory_record_finding
        recs = []
        for i, (title, content) in enumerate(_DUCKDB_FINDINGS):
            recs.append(rec(
                finding_type="solution", title=title, content=content,
                tags=["duckdb"], sources=sources if i == 0 else None,
            ))
        return recs


# ── Feature 1: hierarchical topic clustering ──────────────────────


class TopicClusteringTests(StormFeatureTestCase):
    def test_similar_findings_cluster_outlier_stays_isolated(self) -> None:
        self._seed_duckdb_cluster()
        self.mods["intelligence"].memory_record_finding(
            finding_type="context", title=_OUTLIER[0], content=_OUTLIER[1],
        )
        built = self.mods["topics"].build_topics()

        self.assertGreaterEqual(built["stats"]["topics"], 1, "no topic formed")
        biggest = max(built["topics"], key=lambda t: t["size"])
        self.assertGreaterEqual(biggest["size"], 2, "cluster did not group similar findings")
        # The unrelated pasta node must NOT be the cluster's representative.
        self.assertNotIn("pasta", biggest["label"].lower())
        # And it should be reported as unclustered (no qualifying affinity edge).
        self.assertGreaterEqual(built["stats"]["unclustered_nodes"], 1)
        # Hierarchy is real: the topic carries a subtopics list.
        self.assertIn("subtopics", biggest)
        self.assertIsInstance(biggest["subtopics"], list)

    def test_topics_persist_and_memory_map_reads_them_back(self) -> None:
        self._seed_duckdb_cluster()
        self.mods["topics"].build_topics()
        mp = self.mods["topics"].memory_map()
        self.assertFalse(mp.get("rebuilt"), "should read persisted topics, not rebuild")
        self.assertGreaterEqual(len(mp["topics"]), 1)
        self.assertTrue(all("topic_id" in t and "size" in t for t in mp["topics"]))

    def test_memory_map_lazy_builds_when_no_persisted_topics(self) -> None:
        self._seed_duckdb_cluster()
        mp = self.mods["topics"].memory_map()  # never built yet
        self.assertTrue(mp.get("rebuilt"), "first map call should lazily build")
        self.assertGreaterEqual(len(mp["topics"]), 1)


# ── Feature 2: token-saving recall ────────────────────────────────


class CompactRecallTests(StormFeatureTestCase):
    def test_compact_drops_duplicate_memories_and_shrinks_payload(self) -> None:
        self._seed_duckdb_cluster()
        u = self.mods["unified"]
        normal = u.recall("duckdb lock retry backoff", top_k=5)
        compact = u.recall("duckdb lock retry backoff", top_k=5, compact=True)

        self.assertTrue(compact.get("compact"))
        # Same query, fewer tokens.
        self.assertLess(
            len(json.dumps(compact)), len(json.dumps(normal)),
            "compact mode did not reduce payload size",
        )
        # Memories already represented by a returned KG node are dropped.
        self.assertLessEqual(
            len(compact.get("memories", [])), len(normal.get("memories", [])),
        )
        # The heavy raw 'content' blob is stripped from KG node properties.
        for n in compact.get("kg", {}).get("nodes", []):
            self.assertNotIn("content", n.get("properties", {}))

    def test_group_topics_adds_mind_map_block(self) -> None:
        self._seed_duckdb_cluster()
        self.mods["topics"].build_topics()
        res = self.mods["unified"].recall(
            "duckdb lock retry", top_k=5, group_topics=True,
        )
        self.assertIn("topics", res)
        self.assertIsInstance(res["topics"], list)
        # At least one topic bucket should contain ≥2 members (the cluster).
        self.assertTrue(
            any(len(b["members"]) >= 2 for b in res["topics"]),
            "group_topics did not surface the cluster",
        )

    def test_compact_is_off_by_default_preserving_legacy_shape(self) -> None:
        self._seed_duckdb_cluster()
        res = self.mods["unified"].recall("duckdb lock retry", top_k=5)
        self.assertNotIn("compact", res)
        self.assertNotIn("topics", res)
        self.assertIn("memories", res)


# ── Feature 3: outline-first wiki + Sources grounding ─────────────


class WikiOutlineAndGroundingTests(StormFeatureTestCase):
    def _crystallized_page(self, sources: list[str] | None = None) -> dict[str, Any]:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="solution",
            title="Graceful workspace fallback",
            content="Resolve storage without raising; fall back to a global per-path dir.",
            related_files=["memory_graph/settings.py"],
            sources=sources,
        )
        self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")
        cz = self.mods["wiki"].wiki_crystallize(rec["canonical_id"])
        self.assertNotIn("error", cz)
        return rec

    def test_outline_only_returns_headings_without_full_body(self) -> None:
        rec = self._crystallized_page()
        full = self.mods["wiki"].wiki_get(rec["canonical_id"])
        outline = self.mods["wiki"].wiki_get(rec["canonical_id"], outline_only=True)

        self.assertEqual(outline.get("mode"), "outline_only")
        self.assertIn("outline", outline)
        self.assertGreater(len(outline["outline"]), 0)
        # outline_only must not ship the full body and must be smaller.
        self.assertNotIn("body", outline)
        self.assertLess(len(json.dumps(outline)), len(json.dumps(full)))

    def test_section_returns_only_that_section(self) -> None:
        rec = self._crystallized_page()
        full = self.mods["wiki"].wiki_get(rec["canonical_id"])
        headings = self.mods["wiki"].wiki_get(rec["canonical_id"], outline_only=True)["outline"]
        target = next(h for h in headings if h.lower() != "contents")
        sec = self.mods["wiki"].wiki_get(rec["canonical_id"], section=target)

        self.assertEqual(sec.get("mode"), "section")
        self.assertEqual(sec.get("section"), target)
        self.assertIn(target, sec["body"])
        self.assertLess(len(sec["body"]), len(full["body"]))

    def test_unknown_section_reports_available_headings(self) -> None:
        rec = self._crystallized_page()
        sec = self.mods["wiki"].wiki_get(rec["canonical_id"], section="Nonexistent")
        self.assertEqual(sec.get("mode"), "section_not_found")
        self.assertIn("error", sec)

    def test_sources_render_as_grounding_section(self) -> None:
        srcs = ["memory_graph/settings.py:93", "https://example.com/adr-1"]
        rec = self._crystallized_page(sources=srcs)
        page = self.mods["wiki"].wiki_get(rec["canonical_id"])
        self.assertIn("Sources", page["outline"])
        body = self.mods["wiki"].wiki_get(rec["canonical_id"], section="Sources")["body"]
        for s in srcs:
            self.assertIn(s, body)

    def test_record_finding_marks_grounded_flag(self) -> None:
        rec = self.mods["intelligence"]
        grounded = rec.memory_record_finding(
            finding_type="solution", title="With proof", content="x",
            sources=["file.py:1"],
        )
        ungrounded = rec.memory_record_finding(
            finding_type="solution", title="No proof", content="y",
        )
        self.assertTrue(grounded["grounded"])
        self.assertEqual(grounded["sources"], ["file.py:1"])
        self.assertFalse(ungrounded["grounded"])


# ── Feature 4: memory_gaps coverage critic ────────────────────────


class MemoryGapsTests(StormFeatureTestCase):
    def test_isolated_node_is_flagged(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="context", title="Lonely island fact",
            content="A totally unique fact with no semantic neighbours at all.",
        )
        gaps = self.mods["intelligence"].memory_gaps()
        isolated_ids = {g["node_id"] for g in gaps["gaps"]["isolated"]}
        self.assertIn(rec["node_id"], isolated_ids)

    def test_ungrounded_and_missing_wiki_canonical_is_flagged(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="decision", title="Promoted but unproven",
            content="A canonical decision recorded without any sources or files.",
        )
        self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")
        gaps = self.mods["intelligence"].memory_gaps()
        ungrounded_ids = {g["node_id"] for g in gaps["gaps"]["ungrounded"]}
        missing_wiki_ids = {g["node_id"] for g in gaps["gaps"]["missing_wiki"]}
        self.assertIn(rec["node_id"], ungrounded_ids)
        self.assertIn(rec["node_id"], missing_wiki_ids)

    def test_grounded_canonical_not_flagged_as_ungrounded(self) -> None:
        rec = self.mods["intelligence"].memory_record_finding(
            finding_type="decision", title="Promoted with proof",
            content="A canonical decision recorded with grounding sources.",
            sources=["docs/adr-7.md"],
        )
        self.mods["knowledge_graph"].kg_promote(rec["node_id"], "canonical")
        gaps = self.mods["intelligence"].memory_gaps()
        ungrounded_ids = {g["node_id"] for g in gaps["gaps"]["ungrounded"]}
        self.assertNotIn(rec["node_id"], ungrounded_ids)

    def test_orphan_wiki_page_is_flagged(self) -> None:
        self.mods["wiki"].wiki_ingest(
            title="Dangling page", body="# Dangling\nNo node backs this.",
            canonical_id="solution.no-such-node-exists",
        )
        gaps = self.mods["intelligence"].memory_gaps()
        titles = {g["title"] for g in gaps["gaps"]["orphan_wiki"]}
        self.assertIn("Dangling page", titles)

    def test_recommendations_are_sorted_by_count(self) -> None:
        self.mods["intelligence"].memory_record_finding(
            finding_type="context", title="Solo", content="unique unconnected fact",
        )
        gaps = self.mods["intelligence"].memory_gaps()
        counts = [r["count"] for r in gaps["recommendations"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(gaps["total_gaps"], sum(len(v) for v in gaps["gaps"].values()))


# ── Consolidation rebuilds the topic map ──────────────────────────


class ConsolidateRebuildsTopicsTests(StormFeatureTestCase):
    def test_apply_consolidate_returns_topic_stats(self) -> None:
        self._seed_duckdb_cluster()
        result = self.mods["intelligence"].memory_consolidate(dry_run=False)
        self.assertIn("topics", result)
        self.assertIn("topics", result["topics"])  # nested stats dict

    def test_dry_run_does_not_build_topics(self) -> None:
        self._seed_duckdb_cluster()
        result = self.mods["intelligence"].memory_consolidate(dry_run=True)
        self.assertNotIn("topics", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
