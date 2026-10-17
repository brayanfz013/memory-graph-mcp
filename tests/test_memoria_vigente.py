"""Tests for v0.6.0 — memoria vigente (memory that stays true to the code).

Each class pins one defect measured on a real 625-node store (FashionAI repo, 2026-10-17):

  T1  reuse & promotion: recall() inflated reuse_count on every node it merely
      *returned* (a nonsense query "reused" 5 nodes), and promotion was only
      evaluated inside record_finding for the node being recorded, so 190 drafts
      with reuse >= 3 were never promoted and the wiki froze for 3.5 months.

Each test runs in its own temp workspace with freshly reloaded modules so the
workspace-scoped DuckDB store is isolated (same pattern as test_storm_features).
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
    "memory_graph.parsers",
    "memory_graph.wiki",
    "memory_graph.intelligence",
    "memory_graph.topics",
    "memory_graph.unified",
)

_FINDING = (
    "DuckDB transient lock retry with exponential backoff",
    "Wrap writes in a retry decorator with exponential backoff. Detect transient "
    "errors by lock-related message substrings on the DuckDB file.",
)


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_workspace = os.environ.get("MEMORY_GRAPH_WORKSPACE")
        self.workspace = Path(tempfile.mkdtemp(prefix="mg-vigente-"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(self._restore_workspace_env)
        (self.workspace / ".git").mkdir()
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(self.workspace)
        for mod_name in _MODULE_NAMES:
            sys.modules.pop(mod_name, None)
        sys.modules.pop("memory_graph", None)
        self.mods: dict[str, Any] = {
            name.split(".")[-1]: importlib.import_module(name) for name in _MODULE_NAMES
        }

    def _restore_workspace_env(self) -> None:
        if self._prev_workspace is None:
            os.environ.pop("MEMORY_GRAPH_WORKSPACE", None)
        else:
            os.environ["MEMORY_GRAPH_WORKSPACE"] = self._prev_workspace

    def _record(self, title: str = _FINDING[0], content: str = _FINDING[1], **kw: Any) -> dict[str, Any]:
        return self.mods["intelligence"].memory_record_finding(
            finding_type=kw.pop("finding_type", "solution"), title=title, content=content, **kw)

    def _node(self, canonical_id: str) -> dict[str, Any]:
        with self.mods["db"].get_connection() as conn:
            row = conn.execute(
                "SELECT status, reuse_count, wiki_page FROM kg_nodes WHERE canonical_id = ?",
                [canonical_id]).fetchone()
        return {"status": row[0], "reuse_count": row[1] or 0, "wiki_page": row[2]}


# ── T1: reuse counts real use, and promotion fires where reuse happens ────────


class ReuseAndPromotionTests(WorkspaceTestCase):
    def test_recall_does_not_change_reuse_count(self) -> None:
        rec = self._record()
        before = self._node(rec["canonical_id"])["reuse_count"]
        for _ in range(4):
            out = self.mods["unified"].recall("DuckDB lock retry backoff", min_score=0.0)
            self.assertTrue(out["kg"]["nodes"], "fixture must actually be returned by recall")
        self.assertEqual(self._node(rec["canonical_id"])["reuse_count"], before,
                         "recall is a read: returning a node is not reusing it")

    def test_kg_resolve_counts_as_use(self) -> None:
        rec = self._record()
        before = self._node(rec["canonical_id"])["reuse_count"]
        self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])
        self.assertEqual(self._node(rec["canonical_id"])["reuse_count"], before + 1)

    def test_kg_resolve_missing_does_not_raise_or_count(self) -> None:
        out = self.mods["knowledge_graph"].kg_resolve("solution.no-such-node")
        self.assertIn("error", out)

    def test_reaching_threshold_by_use_promotes_and_crystallizes(self) -> None:
        rec = self._record()
        cid = rec["canonical_id"]
        threshold = self.mods["intelligence"].PROMOTION_REUSE_THRESHOLD
        for _ in range(threshold):
            self.mods["knowledge_graph"].kg_resolve(cid)
        node = self._node(cid)
        self.assertEqual(node["status"], "canonical", "use reached the threshold but no promotion")
        page = self.mods["wiki"].wiki_get(cid)
        self.assertNotIn("error", page, "promotion must crystallize the wiki page")

    def test_below_threshold_stays_draft(self) -> None:
        rec = self._record()
        threshold = self.mods["intelligence"].PROMOTION_REUSE_THRESHOLD
        for _ in range(threshold - 1):
            self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])
        self.assertEqual(self._node(rec["canonical_id"])["status"], "draft")

    def test_wiki_get_counts_as_use_of_its_node(self) -> None:
        rec = self._record()
        cid = rec["canonical_id"]
        self.mods["knowledge_graph"].kg_promote(
            self.mods["knowledge_graph"].kg_resolve(cid)["node_id"], "canonical")
        self.mods["wiki"].wiki_crystallize(cid)
        before = self._node(cid)["reuse_count"]
        self.mods["wiki"].wiki_get(cid)
        self.assertEqual(self._node(cid)["reuse_count"], before + 1)

    def test_consolidate_backfills_drafts_already_over_threshold(self) -> None:
        rec = self._record()
        cid = rec["canonical_id"]
        with self.mods["db"].get_connection() as conn:  # simulate the legacy inflated store
            conn.execute("UPDATE kg_nodes SET reuse_count = 7 WHERE canonical_id = ?", [cid])

        preview = self.mods["intelligence"].memory_consolidate(dry_run=True)
        self.assertEqual(preview["would_promote"], 1)
        self.assertEqual(self._node(cid)["status"], "draft", "dry_run must not promote")

        done = self.mods["intelligence"].memory_consolidate(dry_run=False)
        self.assertEqual(done["promoted"], 1)
        self.assertEqual(self._node(cid)["status"], "canonical")
        self.assertNotIn("error", self.mods["wiki"].wiki_get(cid), "backfill must crystallize too")


if __name__ == "__main__":
    unittest.main()
