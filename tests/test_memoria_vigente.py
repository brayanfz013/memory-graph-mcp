"""Tests for v0.6.0 — memoria vigente (memory that stays true to the code).

Each class pins one defect measured on a real 625-node store (FashionAI repo, 2026-10-17):

  T1  reuse & promotion: recall() inflated reuse_count on every node it merely
      *returned* (a nonsense query "reused" 5 nodes), and promotion was only
      evaluated inside record_finding for the node being recorded, so 190 drafts
      with reuse >= 3 were never promoted and the wiki froze for 3.5 months.
  T2  freshness: 797 of 3,220 files cited by nodes (25%) no longer existed and
      nothing told the reader; a node now carries the commit it was recorded at,
      and recall/kg_resolve report which cited files are missing or changed since.
  T3  history vs present: a fixed Problem ranked ABOVE its own Solution (0.83 vs
      0.80) and only 10 SUPERSEDES edges existed; a Solution that cites the same
      files as an open Problem and talks about it now SOLVES it, and recall marks
      the problem resolved_by and ranks the fix first.

Each test runs in its own temp workspace with freshly reloaded modules so the
workspace-scoped DuckDB store is isolated (same pattern as test_storm_features).
"""

from __future__ import annotations

import importlib
import subprocess
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


# ── T2: freshness against the code (missing / changed since recorded) ─────────


def _git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


class FreshnessTests(WorkspaceTestCase):
    def setUp(self) -> None:
        super().setUp()
        shutil.rmtree(self.workspace / ".git")
        _git(self.workspace, "init", "-q")
        _git(self.workspace, "config", "user.email", "t@t")
        _git(self.workspace, "config", "user.name", "t")
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "here.py").write_text("x = 1\n")
        _git(self.workspace, "add", "-A")
        _git(self.workspace, "commit", "-q", "-m", "init")

    def _recalled(self, cid: str, query: str = "DuckDB lock retry backoff") -> dict[str, Any]:
        nodes = self.mods["unified"].recall(query, min_score=0.0, hops=0)["kg"]["nodes"]
        match = [n for n in nodes if n.get("canonical_id") == cid]
        self.assertTrue(match, "fixture node must be recalled")
        return match[0]

    def test_record_stores_head_commit(self) -> None:
        rec = self._record(related_files=["src/here.py"])
        node = self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])
        self.assertEqual(node["properties"].get("commit"), _git(self.workspace, "rev-parse", "HEAD"))

    def test_recall_flags_missing_cited_file(self) -> None:
        rec = self._record(related_files=["src/here.py", "src/gone.py"])
        node = self._recalled(rec["canonical_id"])
        self.assertEqual(node["freshness"]["missing"], ["src/gone.py"])
        self.assertTrue(node["freshness"]["stale"])

    def test_recall_fresh_node_is_not_stale(self) -> None:
        rec = self._record(related_files=["src/here.py"])
        fr = self._recalled(rec["canonical_id"])["freshness"]
        self.assertEqual((fr["missing"], fr["changed"], fr["stale"]), ([], [], False))

    def test_recall_flags_file_changed_since_recorded(self) -> None:
        rec = self._record(related_files=["src/here.py"])
        (self.workspace / "src" / "here.py").write_text("x = 2\n")
        _git(self.workspace, "commit", "-q", "-am", "change")
        fr = self._recalled(rec["canonical_id"])["freshness"]
        self.assertEqual(fr["changed"], ["src/here.py"])
        self.assertTrue(fr["stale"])

    def test_stale_node_ranks_below_its_raw_score(self) -> None:
        rec = self._record(related_files=["src/gone.py"])
        node = self._recalled(rec["canonical_id"])
        self.assertLess(node["rank_score"], node["semantic_score"])

    def test_fresh_node_outranks_closer_but_stale_node(self) -> None:
        stale = self._record(title="DuckDB lock retry backoff", content="DuckDB lock retry backoff.",
                             related_files=["src/gone.py"])
        fresh = self._record(title="Retry DuckDB writes on lock errors with backoff",
                             content="when the DuckDB file is locked, retry the write with exponential backoff.",
                             related_files=["src/here.py"])
        nodes = self.mods["unified"].recall("DuckDB lock retry backoff", min_score=0.0, hops=0)["kg"]["nodes"]
        by_cid = {n["canonical_id"]: n for n in nodes}
        s, f = by_cid[stale["canonical_id"]], by_cid[fresh["canonical_id"]]
        self.assertGreater(s["semantic_score"], f["semantic_score"],
                           "precondition: the stale node is the closer textual match")
        order = [n["canonical_id"] for n in nodes]
        self.assertLess(order.index(fresh["canonical_id"]), order.index(stale["canonical_id"]),
                        "freshness must outrank a slightly closer but stale match")

    def test_kg_resolve_reports_freshness(self) -> None:
        rec = self._record(related_files=["src/gone.py"])
        node = self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])
        self.assertEqual(node["freshness"]["missing"], ["src/gone.py"])

    def test_no_git_workspace_degrades_gracefully(self) -> None:
        shutil.rmtree(self.workspace / ".git")
        (self.workspace / ".git").mkdir()  # marker only, not a repo
        rec = self._record(related_files=["src/here.py"])
        fr = self._recalled(rec["canonical_id"])["freshness"]
        self.assertEqual((fr["missing"], fr["changed"]), ([], []))
        self.assertIsNone(self.mods["knowledge_graph"].kg_resolve(rec["canonical_id"])["properties"].get("commit"))


# ── T3: a recorded fix resolves its problem, and recall shows the present first ─


_PROBLEM = ("Problem: DuckDB writes fail with lock errors under concurrency",
            "Concurrent processes writing the DuckDB file fail with transient lock errors.")
_SOLUTION = ("Fix: retry DuckDB writes on lock errors with exponential backoff",
             "Wrap DuckDB writes in a retry decorator with exponential backoff so lock errors recover.")


class ProblemSolutionTests(WorkspaceTestCase):
    def _edges(self, rel: str) -> list[tuple[str, str]]:
        with self.mods["db"].get_connection() as conn:
            return conn.execute("SELECT from_id, to_id FROM kg_edges WHERE rel_type = ?", [rel]).fetchall()

    def test_solution_citing_problem_files_solves_it(self) -> None:
        prob = self._record(*_PROBLEM, finding_type="problem", related_files=["src/db.py"])
        sol = self._record(*_SOLUTION, finding_type="solution", related_files=["src/db.py", "tests/t.py"])
        self.assertIn((sol["node_id"], prob["node_id"]), self._edges("SOLVES"))
        self.assertEqual([s["canonical_id"] for s in sol.get("solves", [])], [prob["canonical_id"]])

    def test_no_shared_file_no_solves(self) -> None:
        self._record(*_PROBLEM, finding_type="problem", related_files=["src/db.py"])
        sol = self._record(*_SOLUTION, finding_type="solution", related_files=["src/other.py"])
        self.assertEqual(self._edges("SOLVES"), [])
        self.assertEqual(sol.get("solves", []), [])

    def test_shared_file_but_unrelated_text_no_solves(self) -> None:
        self._record("Problem: pasta overcooked", "Boiled the pasta too long and it went soft.",
                     finding_type="problem", related_files=["src/db.py"])
        self._record(*_SOLUTION, finding_type="solution", related_files=["src/db.py"])
        self.assertEqual(self._edges("SOLVES"), [], "same file is not enough: it must be the same subject")

    def test_solution_never_solves_another_solution(self) -> None:
        self._record("Fix: back off DuckDB writes when the file is locked",
                     "Retry DuckDB writes with exponential backoff on lock errors.",
                     finding_type="solution", related_files=["src/db.py"])
        self._record(*_SOLUTION, finding_type="solution", related_files=["src/db.py"])
        self.assertEqual(self._edges("SOLVES"), [], "only Problems can be solved")

    def test_superseded_problem_is_not_solved_again(self) -> None:
        prob = self._record(*_PROBLEM, finding_type="problem", related_files=["src/db.py"])
        self.mods["knowledge_graph"].kg_promote(prob["node_id"], "superseded")
        self._record(*_SOLUTION, finding_type="solution", related_files=["src/db.py"])
        self.assertEqual(self._edges("SOLVES"), [])

    def test_recall_marks_problem_resolved_and_ranks_fix_first(self) -> None:
        prob = self._record(*_PROBLEM, finding_type="problem", related_files=["src/db.py"])
        sol = self._record(*_SOLUTION, finding_type="solution", related_files=["src/db.py"])
        nodes = self.mods["unified"].recall(_PROBLEM[0], min_score=0.0, hops=0)["kg"]["nodes"]
        by = {n["canonical_id"]: n for n in nodes}
        p, s = by[prob["canonical_id"]], by[sol["canonical_id"]]
        self.assertGreater(p["semantic_score"], s["semantic_score"], "precondition: query names the problem")
        self.assertEqual(p.get("resolved_by"), [sol["canonical_id"]])
        order = [n["canonical_id"] for n in nodes]
        self.assertLess(order.index(sol["canonical_id"]), order.index(prob["canonical_id"]))

    def test_manual_solves_edge_also_resolves(self) -> None:
        prob = self._record(*_PROBLEM, finding_type="problem", related_files=["src/a.py"])
        sol = self._record(*_SOLUTION, finding_type="solution", related_files=["src/b.py"])
        self.mods["knowledge_graph"].kg_add_edge(sol["node_id"], prob["node_id"], "SOLVES")
        node = self.mods["knowledge_graph"].kg_resolve(prob["canonical_id"])
        self.assertEqual(node.get("resolved_by"), [sol["canonical_id"]])


if __name__ == "__main__":
    unittest.main()
