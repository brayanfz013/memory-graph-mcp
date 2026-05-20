"""Validate the plugin's distribution artifacts.

These tests guarantee the plugin is installable in a clean clone:
  - All required files exist
  - JSON artifacts are syntactically valid
  - JSON artifacts have required schema fields
  - pyproject.toml metadata is internally consistent
  - README links point at files that actually exist
  - License notice matches license declared in pyproject

This is the test you run BEFORE publishing/tagging a release.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class RequiredFilesTests(unittest.TestCase):
    """Every file a downstream consumer needs must exist."""

    REQUIRED = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "PROTOCOL.md",
        ".gitignore",
        ".mcp.json",
        ".claude-plugin/plugin.json",
        ".claude-plugin/marketplace.json",
        "skills/memory-graph/SKILL.md",
        "commands/memory-recall.md",
        "examples/quickstart.py",
        "examples/.mcp.json",
        "memory_graph/__init__.py",
        "memory_graph/server.py",
        "memory_graph/db.py",
        "memory_graph/settings.py",
        "memory_graph/embeddings.py",
        "memory_graph/embedding_admin.py",
        "memory_graph/benchmark.py",
        "eval/eval_set.json",
    ]

    def test_all_required_files_exist(self) -> None:
        missing = [p for p in self.REQUIRED if not (PACKAGE_ROOT / p).is_file()]
        self.assertEqual(missing, [], f"Missing required files: {missing}")


class PluginManifestTests(unittest.TestCase):
    """`.claude-plugin/plugin.json` must declare a usable plugin."""

    def setUp(self) -> None:
        with open(PACKAGE_ROOT / ".claude-plugin" / "plugin.json") as fh:
            self.manifest = json.load(fh)

    def test_has_core_fields(self) -> None:
        # Official Claude Code plugin manifest schema:
        # https://code.claude.com/docs/en/plugins-reference#plugin-manifest-schema
        # `name` is the only required field; the rest are metadata.
        # We assert the metadata we always want present for this plugin.
        for field in ("name", "version", "description", "author", "license"):
            self.assertIn(field, self.manifest, f"Missing field: {field}")

    def test_no_unknown_top_level_fields(self) -> None:
        # Validates against the official plugin manifest schema. Catches the
        # v0.4.2 regression where `components`, `requirements`, and `installNotes`
        # were used — none of those are in the spec, and `/plugin install` rejected
        # the manifest.
        known = {
            "$schema", "name", "displayName", "version", "description",
            "author", "homepage", "repository", "license", "keywords",
            "skills", "commands", "agents", "hooks", "mcpServers",
            "outputStyles", "lspServers", "experimental", "userConfig",
            "channels", "dependencies",
        }
        unknown = set(self.manifest.keys()) - known
        self.assertEqual(
            unknown, set(),
            f"plugin.json has fields not in the official schema: {sorted(unknown)}. "
            f"See https://code.claude.com/docs/en/plugins-reference"
        )

    def test_version_matches_pyproject(self) -> None:
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
        self.assertIsNotNone(match)
        self.assertEqual(self.manifest["version"], match.group(1))

    def test_default_component_paths_exist(self) -> None:
        # With no explicit `components` field, Claude Code auto-discovers
        # components from these default paths. Verify each path the plugin
        # relies on is actually present.
        defaults = [
            (".mcp.json", "file"),
            ("skills/memory-graph/SKILL.md", "file"),
            ("commands/memory-recall.md", "file"),
        ]
        for path, kind in defaults:
            full = PACKAGE_ROOT / path
            if kind == "file":
                self.assertTrue(full.is_file(), f"Auto-discovered component missing: {path}")


class MarketplaceManifestTests(unittest.TestCase):
    """`.claude-plugin/marketplace.json` must be valid and list the plugin."""

    def setUp(self) -> None:
        with open(PACKAGE_ROOT / ".claude-plugin" / "marketplace.json") as fh:
            self.market = json.load(fh)

    def test_lists_plugin(self) -> None:
        plugins = self.market.get("plugins", [])
        self.assertGreaterEqual(len(plugins), 1)
        names = [p.get("name") for p in plugins]
        self.assertIn("memory-graph", names)

    def test_plugin_source_starts_with_dotslash(self) -> None:
        # Per the marketplace spec: "Must start with `./`. Resolved relative to
        # the marketplace root." A bare "." silently fails install in v0.4.2.
        for plugin in self.market.get("plugins", []):
            source = plugin.get("source")
            if isinstance(source, str):
                self.assertTrue(
                    source.startswith("./"),
                    f"plugin {plugin.get('name')!r} source must start with './', got {source!r}"
                )

    def test_owner_has_no_unknown_fields(self) -> None:
        # owner schema only supports `name` (required) and `email` (optional).
        # Anything else (url, etc.) is invalid per the spec.
        owner = self.market.get("owner", {})
        unknown = set(owner.keys()) - {"name", "email"}
        self.assertEqual(unknown, set(), f"owner has invalid fields: {sorted(unknown)}")


class McpJsonTests(unittest.TestCase):
    """`.mcp.json` shapes that any MCP client can copy/paste."""

    def test_root_mcp_json_declares_memory_graph_server(self) -> None:
        with open(PACKAGE_ROOT / ".mcp.json") as fh:
            data = json.load(fh)
        self.assertIn("mcpServers", data)
        self.assertIn("memory-graph", data["mcpServers"])
        srv = data["mcpServers"]["memory-graph"]
        self.assertEqual(srv["type"], "stdio")
        self.assertIn("command", srv)

    def test_example_mcp_json_is_valid(self) -> None:
        with open(PACKAGE_ROOT / "examples" / ".mcp.json") as fh:
            data = json.load(fh)
        # The example uses VS Code's "servers" key; either shape is fine
        self.assertTrue("servers" in data or "mcpServers" in data)


class PyProjectTests(unittest.TestCase):
    """Verify pyproject.toml is consistent and PyPI-publishable."""

    def setUp(self) -> None:
        self.text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def test_has_required_metadata(self) -> None:
        for needle in (
            'name = "memory-graph-mcp"',
            'license = { text = "Apache-2.0" }',
            'readme = "README.md"',
            "authors =",
            "[project.scripts]",
            'memory-graph = "memory_graph.server:main"',
        ):
            self.assertIn(needle, self.text, f"pyproject missing: {needle!r}")

    def test_declares_minimum_dependencies(self) -> None:
        for dep in ("mcp>=1.0", "duckdb>=1.1", "pydantic-settings>=2.0", "fastembed>=0.4"):
            self.assertIn(dep, self.text, f"Missing dependency: {dep}")


class ReadmeLinkTests(unittest.TestCase):
    """Internal markdown links in README must resolve to real files."""

    def test_referenced_files_exist(self) -> None:
        readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        # Pull out relative-path links (skip http(s), mailto, anchors)
        links = re.findall(r"\]\(([^)\s]+)\)", readme)
        relative = [
            link for link in links
            if not link.startswith(("http://", "https://", "mailto:", "#"))
        ]
        missing = [link for link in relative if not (PACKAGE_ROOT / link).exists()]
        self.assertEqual(missing, [], f"README references missing files: {missing}")


class LicenseConsistencyTests(unittest.TestCase):
    """LICENSE file must be Apache 2.0 to match pyproject declaration."""

    def test_license_is_apache_2(self) -> None:
        license_text = (PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)


class SkillMetadataTests(unittest.TestCase):
    """Bundled skill must have a name+description frontmatter for skill loader."""

    def test_skill_has_frontmatter(self) -> None:
        skill = (PACKAGE_ROOT / "skills" / "memory-graph" / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill.startswith("---"))
        head = skill.split("---", 2)[1]
        self.assertIn("name:", head)
        self.assertIn("description:", head)


if __name__ == "__main__":
    unittest.main()
