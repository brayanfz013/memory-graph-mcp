"""Tests for workspace-scoped memory-graph settings and lock behavior."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


class WorkspaceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_mod = importlib.import_module("memory_graph.settings")

    def test_explicit_workspace_with_project_markers_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / ".git").mkdir()

            with patch.dict(os.environ, {"MEMORY_GRAPH_WORKSPACE": str(workspace)}, clear=False):
                resolved = self.settings_mod.resolve_workspace_path()

            self.assertEqual(resolved, workspace.resolve())

    def test_editor_install_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "Microsoft VS Code"
            workspace.mkdir()

            with patch.dict(os.environ, {"MEMORY_GRAPH_WORKSPACE": str(workspace)}, clear=False):
                with self.assertRaises(self.settings_mod.WorkspaceResolutionError):
                    self.settings_mod.resolve_workspace_path()

    def test_workspace_without_project_markers_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            with patch.dict(os.environ, {"MEMORY_GRAPH_WORKSPACE": str(workspace)}, clear=False):
                with self.assertRaises(self.settings_mod.WorkspaceResolutionError):
                    self.settings_mod.resolve_workspace_path()


class TransientLockRetryTests(unittest.TestCase):
    """v0.4.0+ relies on DuckDB's WAL-based write lock plus a with_retry decorator
    instead of an explicit acquire_workspace_lock advisory file. This verifies
    the retry decorator classifies and recovers from transient lock messages.
    """

    def test_with_retry_swallows_transient_then_succeeds(self) -> None:
        db_mod = importlib.import_module("memory_graph.db")
        import duckdb

        calls = {"n": 0}

        @db_mod.with_retry(max_attempts=3, base_delay=0.0, max_delay=0.0)
        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise duckdb.IOException("Could not set lock on database")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(calls["n"], 2)

    def test_with_retry_raises_on_non_transient(self) -> None:
        db_mod = importlib.import_module("memory_graph.db")

        @db_mod.with_retry(max_attempts=3, base_delay=0.0)
        def bad() -> str:
            raise ValueError("not a transient lock error")

        with self.assertRaises(ValueError):
            bad()


if __name__ == "__main__":
    unittest.main()
