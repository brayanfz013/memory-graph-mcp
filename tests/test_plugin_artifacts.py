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
        for field in ("name", "version", "description", "author", "license", "components"):
            self.assertIn(field, self.manifest, f"Missing field: {field}")

    def test_version_matches_pyproject(self) -> None:
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
        self.assertIsNotNone(match)
        self.assertEqual(self.manifest["version"], match.group(1))

    def test_referenced_components_exist(self) -> None:
        components = self.manifest.get("components", {})
        for path in components.get("mcpServers", []):
            self.assertTrue((PACKAGE_ROOT / path).exists(), f"Missing component: {path}")
        for path in components.get("skills", []):
            full = PACKAGE_ROOT / path
            self.assertTrue(full.exists() and full.is_dir(), f"Missing skill dir: {path}")
            self.assertTrue((full / "SKILL.md").is_file(), f"Skill missing SKILL.md: {path}")
        for path in components.get("commands", []):
            self.assertTrue((PACKAGE_ROOT / path).is_file(), f"Missing command: {path}")


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
