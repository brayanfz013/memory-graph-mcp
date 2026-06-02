"""Tree-sitter code parsing with regex fallback for wiki bootstrap.

Provides AST-based extraction of classes, functions, methods, exports,
interfaces, and docstrings from Python, TypeScript, JavaScript, and
InterSystems ObjectScript (.cls, .mac) files.

When tree-sitter grammars are unavailable or fail, falls back to
regex-based extraction so wiki bootstrap never silently skips directories.
"""

from __future__ import annotations

import abc
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Unified return shape ──────────────────────────────────────────

ParsedSummary = dict[str, Any]  # {classes, functions, methods, exports, interfaces, docstrings, namespaces}

# ── Regex fallback functions ──────────────────────────────────────

def _regex_python(text: bytes) -> ParsedSummary:
    content = text.decode(errors="replace")
    s: ParsedSummary = {
        "classes": [], "functions": [], "methods": [],
        "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
    }
    docstring_match = re.match(r'^(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', content, re.DOTALL)
    if docstring_match:
        doc = (docstring_match.group(1) or docstring_match.group(2) or "").strip()
        if doc:
            s["docstrings"].append(doc[:300])
    classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
    functions = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
    s["classes"] = [c for c in classes if not c.startswith("_")][:15]
    s["functions"] = [f for f in functions if not f.startswith("_")][:15]
    return s


def _regex_javascript(text: bytes) -> ParsedSummary:
    content = text.decode(errors="replace")
    s: ParsedSummary = {
        "classes": [], "functions": [], "methods": [],
        "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
    }
    jsdoc_match = re.search(r"/\*\*\s*\n\s*\*\s*(.+?)(?:\n\s*\*\s*@|\s*\*/)", content, re.DOTALL)
    if jsdoc_match:
        doc = jsdoc_match.group(1).strip().replace("\n * ", " ")[:300]
        s["docstrings"].append(doc)
    exports = re.findall(r"export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)", content)
    s["exports"] = exports[:15]
    return s


def _regex_typescript(text: bytes) -> ParsedSummary:
    content = text.decode(errors="replace")
    s: ParsedSummary = {
        "classes": [], "functions": [], "methods": [],
        "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
    }
    jsdoc_match = re.search(r"/\*\*\s*\n\s*\*\s*(.+?)(?:\n\s*\*\s*@|\s*\*/)", content, re.DOTALL)
    if jsdoc_match:
        doc = jsdoc_match.group(1).strip().replace("\n * ", " ")[:300]
        s["docstrings"].append(doc)
    exports = re.findall(r"export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)", content)
    s["exports"] = exports[:15]
    interfaces = re.findall(r"(?:^|\s)interface\s+(\w+)", content)
    s["interfaces"] = interfaces[:15]
    return s


def _regex_objectscript_class(text: bytes) -> ParsedSummary:
    content = text.decode(errors="replace")
    s: ParsedSummary = {
        "classes": [], "functions": [], "methods": [],
        "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
    }
    classes = re.findall(r"Class\s+(\w+)(?:\s+extends\s+\w+)?\s", content)
    s["classes"] = classes[:15]
    methods = re.findall(r"^\s+(?:ClassMethod|Method)\s+(\w+)\s*\(", content, re.MULTILINE)
    if classes and methods:
        s["methods"] = [f"{c}.{m}" for c, m in zip(classes[:15], methods[:15])]
    return s


def _regex_objectscript_routine(text: bytes) -> ParsedSummary:
    content = text.decode(errors="replace")
    s: ParsedSummary = {
        "classes": [], "functions": [], "methods": [],
        "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
    }
    routines = re.findall(r"^(?!\s)\w+:", content, re.MULTILINE)
    s["functions"] = routines[:15]
    return s


# ── Language parsers ─────────────────────────────────────────────

class BaseParser(abc.ABC):
    """Abstract base for tree-sitter language parsers."""

    @abc.abstractmethod
    def parse(self, path: Path) -> ParsedSummary:
        ...

    def _read_content(self, path: Path) -> bytes | None:
        """Read file as bytes. Returns None on error or empty."""
        try:
            content = path.read_bytes()
        except OSError:
            return None
        return content if content else None

    def _build_parser(self, path: Path) -> Any | None:
        """Load tree-sitter Language. Subclass implements _load_language."""
        language = self._load_language(path)
        if language is None:
            return None
        try:
            import tree_sitter
            return tree_sitter.Parser(language)
        except Exception:
            return None


class PythonParser(BaseParser):
    """Extract classes, functions, methods, docstrings from Python files."""

    def _load_language(self, path: Path) -> Any | None:
        try:
            from tree_sitter_python import language
            import tree_sitter
            return tree_sitter.Language(language())
        except (ImportError, AttributeError, ValueError, KeyError, TypeError):
            return None

    def parse(self, path: Path) -> ParsedSummary:
        content = self._read_content(path)
        if content is None:
            return _regex_python(content if content is not None else b"")

        parser = self._build_parser(path)
        if parser is None:
            return _regex_python(content)

        try:
            tree = parser.parse(content)
        except Exception:
            return _regex_python(content)

        summary: ParsedSummary = {
            "classes": [], "functions": [], "methods": [],
            "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
        }

        # Module docstring: first statement in the module
        root = tree.root_node
        for child in root.children:
            if child.type in ("expression_statement",):
                # Could be a docstring (string literal)
                for gc in child.children:
                    if gc.type in ("string",):
                        text = gc.text.decode(errors="replace").strip()
                        if text:
                            summary["docstrings"].append(text[:300])
                        break

        # Classes, functions, methods via recursive walk
        self._walk(root, summary, class_name=None)
        return summary

    def _walk(self, node: Any, summary: ParsedSummary, class_name: str | None) -> None:
        for child in node.children:
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode(errors="replace")
                    if not name.startswith("_"):
                        summary["classes"].append(name)
                    self._walk(child, summary, class_name=name)
                else:
                    self._walk(child, summary, class_name=class_name)
            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode(errors="replace")
                    if not name.startswith("_"):
                        if class_name:
                            summary["methods"].append(f"{class_name}.{name}")
                        else:
                            summary["functions"].append(name)
                self._walk(child, summary, class_name=class_name)
            else:
                self._walk(child, summary, class_name=class_name)


class TypeScriptParser(BaseParser):
    """Extract classes, functions, interfaces, exports from TypeScript files."""

    def _load_language(self, path: Path) -> Any | None:
        try:
            import tree_sitter
            from tree_sitter_typescript import language_tsx, language_typescript
            # TSX grammar for .tsx files, TypeScript for .ts/.mts/.cts
            lang_fn = language_tsx if path.suffix == ".tsx" else language_typescript
            return tree_sitter.Language(lang_fn())
        except (ImportError, AttributeError, ValueError, KeyError, TypeError):
            return None

    def parse(self, path: Path) -> ParsedSummary:
        content = self._read_content(path)
        if content is None:
            return _regex_typescript(content if content is not None else b"")

        parser = self._build_parser(path)
        if parser is None:
            return _regex_typescript(content)

        try:
            tree = parser.parse(content)
        except Exception:
            return _regex_typescript(content)

        summary: ParsedSummary = {
            "classes": [], "functions": [], "methods": [],
            "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
        }

        self._walk_exports(tree.root_node, summary)
        self._walk_declarations(tree.root_node, summary)
        return summary

    def _walk_exports(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "export_statement":
                decl = child.child_by_field_name("declaration")
                if decl:
                    name = self._get_name(decl)
                    if name:
                        summary["exports"].append(name)
            self._walk_exports(child, summary)

    def _walk_declarations(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "class_declaration":
                name = self._get_name(child)
                if name and not name.startswith("_"):
                    summary["classes"].append(name)
            elif child.type in ("interface_declaration", "type_alias_declaration"):
                name = self._get_name(child)
                if name and not name.startswith("_"):
                    summary["interfaces"].append(name)
            elif child.type == "function_declaration":
                name = self._get_name(child)
                if name and not name.startswith("_"):
                    summary["functions"].append(name)
            elif child.type == "lexical_declaration":
                # const/let/var declarations — treat as exports if they have a name
                for gc in child.children:
                    if gc.type == "variable_declarator":
                        name = self._get_name(gc)
                        if name and not name.startswith("_"):
                            summary["functions"].append(name)
            self._walk_declarations(child, summary)

    def _get_name(self, node: Any) -> str | None:
        name_node = node.child_by_field_name("name") or node.child_by_field_name("left")
        if name_node:
            return name_node.text.decode(errors="replace")
        # Handle variable_declarator (const/let/var bindings)
        for child in node.children:
            if child.type == "variable_declarator":
                n = child.child_by_field_name("name")
                if n:
                    return n.text.decode(errors="replace")
        # Fallback: first identifier child
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return child.text.decode(errors="replace")
        return None


class JavaScriptParser(BaseParser):
    """Extract classes, functions, exports from JavaScript files."""

    def _load_language(self, path: Path) -> Any | None:
        try:
            import tree_sitter
            from tree_sitter_javascript import language
            return tree_sitter.Language(language())
        except (ImportError, AttributeError, ValueError, KeyError, TypeError):
            return None

    def parse(self, path: Path) -> ParsedSummary:
        content = self._read_content(path)
        if content is None:
            return _regex_javascript(content if content is not None else b"")

        parser = self._build_parser(path)
        if parser is None:
            return _regex_javascript(content)

        try:
            tree = parser.parse(content)
        except Exception:
            return _regex_javascript(content)

        summary: ParsedSummary = {
            "classes": [], "functions": [], "methods": [],
            "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
        }

        self._walk_exports(tree.root_node, summary)
        self._walk_declarations(tree.root_node, summary)
        return summary

    def _walk_exports(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "export_statement":
                decl = child.child_by_field_name("declaration")
                if decl:
                    name = self._get_name(decl)
                    if name:
                        summary["exports"].append(name)
            self._walk_exports(child, summary)

    def _walk_declarations(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "class_declaration":
                name = self._get_name(child)
                if name and not name.startswith("_"):
                    summary["classes"].append(name)
            elif child.type == "function_declaration":
                name = self._get_name(child)
                if name and not name.startswith("_"):
                    summary["functions"].append(name)
            elif child.type == "lexical_declaration":
                for gc in child.children:
                    if gc.type == "variable_declarator":
                        name = self._get_name(gc)
                        if name and not name.startswith("_"):
                            summary["functions"].append(name)
            self._walk_declarations(child, summary)

    def _get_name(self, node: Any) -> str | None:
        name_node = node.child_by_field_name("name") or node.child_by_field_name("left")
        if name_node:
            return name_node.text.decode(errors="replace")
        for child in node.children:
            if child.type == "variable_declarator":
                n = child.child_by_field_name("name")
                if n:
                    return n.text.decode(errors="replace")
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return child.text.decode(errors="replace")
        return None

class ObjectScriptClassParser(BaseParser):
    """Parse .cls files (InterSystems ObjectScript class definitions)."""

    def _load_language(self, path: Path) -> Any | None:
        try:
            import tree_sitter
            from tree_sitter_objectscript_udl import language as _lang
            return tree_sitter.Language(_lang())
        except ImportError:
            logger.debug("tree-sitter-objectscript not installed. Install with: pip install tree-sitter-objectscript")
            return None

    def parse(self, path: Path) -> ParsedSummary:
        content = self._read_content(path)
        if content is None:
            return _regex_objectscript_class(content if content is not None else b"")

        parser = self._build_parser(path)
        if parser is None:
            return _regex_objectscript_class(content)

        try:
            tree = parser.parse(content)
        except Exception:
            return _regex_objectscript_class(content)

        summary: ParsedSummary = {
            "classes": [], "functions": [], "methods": [],
            "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
        }

        self._walk_class(tree.root_node, summary)
        return summary

    def _walk_class(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode(errors="replace")
                    if not name.startswith("_"):
                        summary["classes"].append(name)
                    self._walk_methods(child, summary, name)
            self._walk_class(child, summary)

    def _walk_methods(self, node: Any, summary: ParsedSummary, class_name: str) -> None:
        for child in node.children:
            if child.type in ("method_definition", "method", "class_method_definition"):
                name_node = child.child_by_field_name("name")
                if name_node:
                    method_name = name_node.text.decode(errors="replace")
                    if not method_name.startswith("_"):
                        summary["methods"].append(f"{class_name}.{method_name}")
            self._walk_methods(child, summary, class_name)


class ObjectScriptRoutineParser(BaseParser):
    """Parse .mac/.rtn/.inc files (InterSystems ObjectScript routines)."""

    def _load_language(self, path: Path) -> Any | None:
        try:
            import tree_sitter
            from tree_sitter_objectscript_routine import language as _lang
            return tree_sitter.Language(_lang())
        except ImportError:
            logger.debug("tree-sitter-objectscript_routine not installed. Install with: pip install tree-sitter-objectscript")
            return None

    def parse(self, path: Path) -> ParsedSummary:
        content = self._read_content(path)
        if content is None:
            return _regex_objectscript_routine(content if content is not None else b"")

        parser = self._build_parser(path)
        if parser is None:
            return _regex_objectscript_routine(content)

        try:
            tree = parser.parse(content)
        except Exception:
            return _regex_objectscript_routine(content)

        summary: ParsedSummary = {
            "classes": [], "functions": [], "methods": [],
            "exports": [], "interfaces": [], "docstrings": [], "namespaces": [],
        }

        self._walk_routine(tree.root_node, summary)
        return summary

    def _walk_routine(self, node: Any, summary: ParsedSummary) -> None:
        for child in node.children:
            if child.type == "routine":
                name_node = child.child_by_field_name("name")
                if name_node:
                    summary["functions"].append(name_node.text.decode(errors="replace"))
            # Labels that serve as function boundaries
            if child.type in ("label", "routine_label"):
                name_node = child.child_by_field_name("name") or child.child_by_field_name("identifier")
                if name_node:
                    name = name_node.text.decode(errors="replace")
                    if not name.startswith("_"):
                        summary["functions"].append(name)
            self._walk_routine(child, summary)


# ── ParserManager ──────────────────────────────────────────────────

class ParserManager:
    """Central coordinator for language parsing with lazy loading."""

    EXTENSION_MAP: dict[str, str] = {
        # Python
        ".py": "python",
        ".pyi": "python",
        # TypeScript
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mts": "typescript",
        ".cts": "typescript",
        # JavaScript
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        # ObjectScript
        ".cls": "objectscript_class",
        ".mac": "objectscript_routine",
        ".inc": "objectscript_routine",
        ".rtn": "objectscript_routine",
    }

    PARSER_CLASS_MAP: dict[str, type[BaseParser]] = {
        "python": PythonParser,
        "typescript": TypeScriptParser,
        "javascript": JavaScriptParser,
        "objectscript_class": ObjectScriptClassParser,
        "objectscript_routine": ObjectScriptRoutineParser,
    }

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}
        self._load_failures: set[str] = set()

    def parse(self, path: Path) -> ParsedSummary:
        """Parse a code file. Never raises. Falls back to regex.

        Returns {} for unsupported extensions.
        """
        ext = path.suffix.lower()
        if ext not in self.EXTENSION_MAP:
            return {}

        lang_name = self.EXTENSION_MAP[ext]
        parser = self._get_parser(lang_name)
        if parser is None:
            return self._fallback_regex_parse(path, ext)

        try:
            return parser.parse(path)
        except Exception as exc:
            logger.warning("parse failed for %s: %s", path, exc)
            return self._fallback_regex_parse(path, ext)

    def _get_parser(self, lang_name: str) -> BaseParser | None:
        if lang_name in self._parsers:
            return self._parsers[lang_name]
        if lang_name in self._load_failures:
            return None

        parser_class = self.PARSER_CLASS_MAP.get(lang_name)
        if parser_class is None:
            self._load_failures.add(lang_name)
            return None

        try:
            parser = parser_class()
            self._parsers[lang_name] = parser
            return parser
        except ImportError as exc:
            logger.warning("Parser %s unavailable (grammar not installed): %s", lang_name, exc)
            self._load_failures.add(lang_name)
            return None
        except Exception as exc:
            logger.warning("Parser %s failed to initialize: %s", lang_name, exc)
            self._load_failures.add(lang_name)
            return None

    def _fallback_regex_parse(self, path: Path, ext: str) -> ParsedSummary:
        try:
            content = path.read_bytes()
        except OSError:
            return {}
        if not content:
            return {}

        if ext == ".py":
            return _regex_python(content)
        if ext in {".js", ".jsx", ".mjs", ".cjs"}:
            return _regex_javascript(content)
        if ext in {".ts", ".tsx", ".mts", ".cts"}:
            return _regex_typescript(content)
        if ext == ".cls":
            return _regex_objectscript_class(content)
        if ext in {".mac", ".inc", ".rtn"}:
            return _regex_objectscript_routine(content)
        return {}


# ── Public API ─────────────────────────────────────────────────────

_parser_manager: ParserManager | None = None


def get_parser_manager() -> ParserManager:
    """Get (or create) the module-level ParserManager singleton.

    Lazy-loaded: tree-sitter grammars are not loaded until the first
    parse() call, so server startup remains fast.
    """
    global _parser_manager
    if _parser_manager is None:
        _parser_manager = ParserManager()
    return _parser_manager


def parse_code_file(path: Path) -> ParsedSummary:
    """Convenience function: parse a single code file.

    Never raises. Returns {} for unsupported extensions.
    Falls back to regex if tree-sitter is unavailable.
    """
    return get_parser_manager().parse(path)
