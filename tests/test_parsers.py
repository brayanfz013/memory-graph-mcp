"""Tests for tree-sitter code parsing infrastructure.

Validates:
  1. ParserManager extension routing
  2. Each language parser returns correct dict shape
  3. Empty files return empty dicts
  4. Unsupported extensions return empty dicts
  5. Regex fallback activates when tree-sitter unavailable
  6. Parse errors are isolated (one bad file doesn't crash others)
  7. Wiki integration: _summarize_directory returns dict, not None
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure package root is in path (same pattern as test_end_to_end.py)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

_MODULE_NAMES = (
    "memory_graph.settings",
    "memory_graph.parsers",
    "memory_graph.wiki",
    "memory_graph.db",
)


class ParsersTestCase(unittest.TestCase):
    """Base: every test gets its own temp workspace + reloaded modules."""

    def setUp(self) -> None:
        self._prev_workspace = os.environ.get("MEMORY_GRAPH_WORKSPACE")
        self.workspace = Path(tempfile.mkdtemp(prefix="mg-parser-"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        self.addCleanup(self._restore_workspace_env)
        os.environ["MEMORY_GRAPH_WORKSPACE"] = str(self.workspace)

        # Force re-import so fresh state
        for mod_name in _MODULE_NAMES:
            sys.modules.pop(mod_name, None)
        sys.modules.pop("memory_graph", None)

    def _restore_workspace_env(self) -> None:
        if self._prev_workspace is None:
            os.environ.pop("MEMORY_GRAPH_WORKSPACE", None)
        else:
            os.environ["MEMORY_GRAPH_WORKSPACE"] = self._prev_workspace


class ParserManagerTests(ParsersTestCase):
    """Test ParserManager routing and caching."""

    def test_extension_to_language_mapping(self) -> None:
        from memory_graph.parsers import ParserManager
        pm = ParserManager()

        # Python
        self.assertEqual(pm.EXTENSION_MAP[".py"], "python")
        self.assertEqual(pm.EXTENSION_MAP[".pyi"], "python")

        # TypeScript
        self.assertEqual(pm.EXTENSION_MAP[".ts"], "typescript")
        self.assertEqual(pm.EXTENSION_MAP[".tsx"], "typescript")
        self.assertEqual(pm.EXTENSION_MAP[".mts"], "typescript")
        self.assertEqual(pm.EXTENSION_MAP[".cts"], "typescript")

        # JavaScript
        self.assertEqual(pm.EXTENSION_MAP[".js"], "javascript")
        self.assertEqual(pm.EXTENSION_MAP[".jsx"], "javascript")
        self.assertEqual(pm.EXTENSION_MAP[".mjs"], "javascript")
        self.assertEqual(pm.EXTENSION_MAP[".cjs"], "javascript")

        # ObjectScript
        self.assertEqual(pm.EXTENSION_MAP[".cls"], "objectscript_class")
        self.assertEqual(pm.EXTENSION_MAP[".mac"], "objectscript_routine")
        self.assertEqual(pm.EXTENSION_MAP[".inc"], "objectscript_routine")
        self.assertEqual(pm.EXTENSION_MAP[".rtn"], "objectscript_routine")

    def test_unsupported_extension_returns_empty(self) -> None:
        from memory_graph.parsers import get_parser_manager

        tmp = Path(tempfile.mktemp(suffix=".go"))
        try:
            tmp.write_text("package main\n")
            result = get_parser_manager().parse(tmp)
            self.assertEqual(result, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_parser_caching(self) -> None:
        from memory_graph.parsers import get_parser_manager

        pm = get_parser_manager()

        # Create a temp .py file to trigger parser loading
        tmp = Path(tempfile.mktemp(suffix=".py"))
        try:
            tmp.write_text("def foo(): pass\n")
            pm.parse(tmp)  # First call loads parser
            self.assertIn("python", pm._parsers)
            pm.parse(tmp)  # Second call uses cached parser
        finally:
            tmp.unlink(missing_ok=True)


class PythonParserTests(ParsersTestCase):
    """Test Python tree-sitter parsing."""

    def test_extract_classes_and_functions(self) -> None:
        from memory_graph.parsers import PythonParser

        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text(
            '"""Module docstring."""\n'
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "def standalone():\n"
            "    pass\n"
        )
        try:
            parser = PythonParser()
            result = parser.parse(tmp)

            self.assertIn("MyClass", result["classes"])
            self.assertIn("standalone", result["functions"])
            self.assertIn("MyClass.method", result["methods"])
            self.assertGreater(len(result["docstrings"]), 0)
            self.assertIn("Module docstring", result["docstrings"][0])
        finally:
            tmp.unlink(missing_ok=True)

    def test_empty_file_returns_empty(self) -> None:
        from memory_graph.parsers import PythonParser

        tmp = Path(tempfile.mktemp(suffix=".py"))
        try:
            tmp.write_text("")
            parser = PythonParser()
            result = parser.parse(tmp)
            self.assertEqual(result["classes"], [])
            self.assertEqual(result["functions"], [])
        finally:
            tmp.unlink(missing_ok=True)

    def test_private_symbols_excluded(self) -> None:
        from memory_graph.parsers import PythonParser

        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text(
            "class _Private:\n"
            "    def _private_method(self):\n"
            "        pass\n"
            "\n"
            "def _private_func():\n"
            "    pass\n"
        )
        try:
            parser = PythonParser()
            result = parser.parse(tmp)
            self.assertNotIn("_Private", result["classes"])
            self.assertNotIn("_private_method", result["methods"])
            self.assertNotIn("_private_func", result["functions"])
        finally:
            tmp.unlink(missing_ok=True)


class TypeScriptParserTests(ParsersTestCase):
    """Test TypeScript tree-sitter parsing."""

    def test_extract_exports_and_interfaces(self) -> None:
        from memory_graph.parsers import TypeScriptParser

        tmp = Path(tempfile.mktemp(suffix=".ts"))
        tmp.write_text(
            "export function hello(): void {}\n"
            "export const world = 42\n"
            "export interface MyInterface { x: number }\n"
            "function hidden(): void {}\n"
        )
        try:
            parser = TypeScriptParser()
            result = parser.parse(tmp)
            self.assertIn("hello", result["exports"])
            self.assertIn("world", result["exports"])
            self.assertIn("MyInterface", result["interfaces"])
        finally:
            tmp.unlink(missing_ok=True)


class JavaScriptParserTests(ParsersTestCase):
    """Test JavaScript tree-sitter parsing."""

    def test_extract_exports(self) -> None:
        from memory_graph.parsers import JavaScriptParser

        tmp = Path(tempfile.mktemp(suffix=".js"))
        tmp.write_text(
            "export function foo() {}\n"
            "export const bar = 42\n"
            "function hidden() {}\n"
        )
        try:
            parser = JavaScriptParser()
            result = parser.parse(tmp)
            self.assertIn("foo", result["exports"])
            self.assertIn("bar", result["exports"])
        finally:
            tmp.unlink(missing_ok=True)


class RegexFallbackTests(ParsersTestCase):
    """Test regex fallback when tree-sitter is unavailable."""

    def test_python_regex_fallback(self) -> None:
        from memory_graph.parsers import ParserManager

        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text(
            '"""Test module."""\n'
            "class TestClass:\n"
            "    def test_method(self):\n"
            "        pass\n"
            "\n"
            "def test_func():\n"
            "    pass\n"
        )
        try:
            pm = ParserManager()
            with mock.patch.object(pm, "_get_parser", return_value=None):
                result = pm.parse(tmp)

            self.assertIn("TestClass", result["classes"])
            self.assertIn("test_func", result["functions"])
            self.assertGreater(len(result["docstrings"]), 0)
        finally:
            tmp.unlink(missing_ok=True)

    def test_js_regex_fallback(self) -> None:
        from memory_graph.parsers import ParserManager

        tmp = Path(tempfile.mktemp(suffix=".js"))
        tmp.write_text("export function hello() {}\n")
        try:
            pm = ParserManager()
            with mock.patch.object(pm, "_get_parser", return_value=None):
                result = pm.parse(tmp)

            self.assertIn("hello", result["exports"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_ts_regex_fallback(self) -> None:
        from memory_graph.parsers import ParserManager

        tmp = Path(tempfile.mktemp(suffix=".ts"))
        tmp.write_text(
            "export function hello(): void {}\n"
            "export interface World {}\n"
        )
        try:
            pm = ParserManager()
            with mock.patch.object(pm, "_get_parser", return_value=None):
                result = pm.parse(tmp)

            self.assertIn("hello", result["exports"])
            self.assertIn("World", result["interfaces"])
        finally:
            tmp.unlink(missing_ok=True)


class ErrorIsolationTests(ParsersTestCase):
    """Test that parse errors in one file don't crash the manager."""

    def test_malformed_file_does_not_crash(self) -> None:
        from memory_graph.parsers import get_parser_manager

        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_bytes(b"\x00\x01\x02\xff\xfe")
        try:
            result = get_parser_manager().parse(tmp)
            self.assertIsInstance(result, dict)
            self.assertEqual(result.get("classes", []), [])
        finally:
            tmp.unlink(missing_ok=True)

    def test_multiple_files_one_failure(self) -> None:
        from memory_graph.parsers import get_parser_manager

        pm = get_parser_manager()

        good = Path(tempfile.mktemp(suffix=".py"))
        good.write_text("def good(): pass\n")

        bad = Path(tempfile.mktemp(suffix=".py"))
        bad.write_bytes(b"\xff\xfe\x00\x01")

        try:
            good_result = pm.parse(good)
            bad_result = pm.parse(bad)
            good_result2 = pm.parse(good)

            self.assertIn("good", good_result["functions"])
            self.assertIn("good", good_result2["functions"])
            self.assertIsInstance(bad_result, dict)
        finally:
            good.unlink(missing_ok=True)
            bad.unlink(missing_ok=True)


class WikiIntegrationTests(ParsersTestCase):
    """Test that wiki.py correctly uses the new parser."""

    def test_summarize_directory_returns_dict_not_none(self) -> None:
        """Directory with code file but no docstring should still get a page."""
        import memory_graph.wiki as wiki

        subdir = self.workspace / "nodocstring"
        subdir.mkdir()
        # File with no docstring, no classes, no functions
        (subdir / "mod.py").write_text("x = 1\n")

        result = wiki._summarize_directory(subdir, self.workspace)
        self.assertIsNotNone(result)
        self.assertIn("body", result)

    def test_summarize_directory_with_no_code_files_is_none(self) -> None:
        """Empty directory without README should return None."""
        import memory_graph.wiki as wiki

        subdir = self.workspace / "empty"
        subdir.mkdir()

        result = wiki._summarize_directory(subdir, self.workspace)
        self.assertIsNone(result)

    def test_summarize_directory_with_readme_only(self) -> None:
        """Directory with only README should get a page."""
        import memory_graph.wiki as wiki

        subdir = self.workspace / "readmeonly"
        subdir.mkdir()
        (subdir / "README.md").write_text("# My Dir\nSome content.\n")

        result = wiki._summarize_directory(subdir, self.workspace)
        self.assertIsNotNone(result)
        self.assertIn("README", result["body"])

    def test_format_parsed_summary_with_data(self) -> None:
        import memory_graph.wiki as wiki

        parsed = {
            "classes": ["Foo", "Bar"],
            "functions": ["main", "init"],
            "methods": ["Foo.method1", "Bar.method2"],
            "exports": ["exported"],
            "interfaces": ["IFoo"],
            "docstrings": ["Module doc"],
            "namespaces": ["NS1"],
        }
        result = wiki._format_parsed_summary(parsed)
        self.assertIn("Foo", result)
        self.assertIn("main", result)
        self.assertIn("Foo.method1", result)
        self.assertIn("exported", result)
        self.assertIn("IFoo", result)
        self.assertIn("Module doc", result)
        self.assertIn("NS1", result)

    def test_format_parsed_summary_empty_returns_fallback(self) -> None:
        import memory_graph.wiki as wiki

        result = wiki._format_parsed_summary({})
        self.assertEqual(result, "*No public symbols detected.*")


class ParseCodeFileTests(ParsersTestCase):
    """Test the public parse_code_file convenience function."""

    def test_parse_code_file_python(self) -> None:
        from memory_graph.parsers import parse_code_file

        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text("class Foo: pass\n")
        try:
            result = parse_code_file(tmp)
            self.assertIn("Foo", result["classes"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_parse_code_file_unsupported(self) -> None:
        from memory_graph.parsers import parse_code_file

        tmp = Path(tempfile.mktemp(suffix=".zig"))
        tmp.write_text("pub fn main() void {}\n")
        try:
            result = parse_code_file(tmp)
            self.assertEqual(result, {})
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
