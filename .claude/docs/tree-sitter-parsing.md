# Tree-Sitter Code Parsing

**Added in v0.4.3** — Replaced fragile regex-based code extraction with AST-based parsing using tree-sitter.

## The Problem

The old `wiki.py` used regex patterns (`_extract_python_summary`, `_extract_js_summary`) to extract code symbols from files. This had two major issues:

1. **Files without docstrings returned `None`** — directories containing code without docstrings/JSDoc were skipped entirely from the wiki bootstrap, leaving gaps in the documentation.
2. **Limited language support** — only Python (.py) and JS/TS (.js, .ts, .tsx) were handled. No support for InterSystems ObjectScript (.cls, .mac, .inc, .rtn).

## The Solution

A new `memory_graph.parsers` module with tree-sitter AST-based parsing:

- **Accurate extraction** — parses actual AST nodes, not regex patterns
- **Graceful degradation** — malformed files return empty dicts instead of crashing
- **Multiple languages** — Python, TypeScript, JavaScript, ObjectScript (CLS/Routine)
- **Lazy loading** — grammars loaded on first parse call, never at server startup
- **Regex fallback** — when tree-sitter is unavailable, regex patterns produce the same output shape

## Files Added

### `memory_graph/parsers.py`

Core module with:

| Component | Purpose |
|---|---|
| `ParsedSummary` | Unified dict shape: `{classes, functions, methods, exports, interfaces, docstrings, namespaces}` |
| `BaseParser` (ABC) | Common interface with `_read_content()` and `_build_parser()` helpers |
| `PythonParser` | Extracts classes, functions, methods, module docstrings |
| `TypeScriptParser` | Extracts classes, functions, interfaces, exports |
| `JavaScriptParser` | Extracts classes, functions, exports |
| `ObjectScriptClassParser` | Parses `.cls` files — class definitions and methods |
| `ObjectScriptRoutineParser` | Parses `.mac`, `.inc`, `.rtn` — routine labels and namespaces |
| `ParserManager` | Singleton with lazy grammar loading, caching, and regex fallback |

### `tests/test_parsers.py`

8 test classes covering:

- Extension-to-language mapping and unsupported extensions
- Parser caching behavior
- Class/function/method extraction (Python, TS, JS)
- Private symbol exclusion
- Empty file handling
- Regex fallback activation (mocked tree-sitter unavailability)
- Error isolation (malformed binary files don't crash)
- Wiki integration (directories without docstrings now get pages)

## Files Modified

### `memory_graph/wiki.py`

- Removed `import re`, `_extract_python_summary()`, `_extract_js_summary()`
- Added `from .parsers import parse_code_file`
- Added `_SUPPORTED_EXTS` constant: `.py .pyi .js .jsx .mjs .cjs .ts .tsx .mts .cts .cls .mac .inc .rtn`
- Added `_format_parsed_summary(parsed: dict) -> str` — formats ParsedSummary into markdown
- Updated `_summarize_directory()`:
  - Extension filter uses `_SUPPORTED_EXTS`
  - Replaced regex dispatch with `parsed = parse_code_file(code_file)`
  - **Removed `return None` early-exit** — directories always get a page even without code summaries

### `pyproject.toml`

Dependencies added:

```toml
"tree-sitter>=0.25",
"tree-sitter-languages>=1.10",
```

Optional extra:

```toml
[project.optional-dependencies]
objectscript = ["tree-sitter-objectscript>=0.7"]
```

### `tests/test_end_to_end.py`

Added `"memory_graph.parsers"` to `_MODULE_NAMES` cleanup tuple.

## Supported Languages & Extensions

| Language | Extensions | Extracted |
|---|---|---|
| Python | `.py`, `.pyi` | classes, functions, methods, docstrings |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | classes, functions, interfaces, exports |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | classes, functions, exports |
| ObjectScript Class | `.cls` | class definitions, class methods |
| ObjectScript Routine | `.mac`, `.inc`, `.rtn` | routine labels, namespaces |

## Installation

```bash
# Core (Python, TypeScript, JavaScript)
uv pip install --system -e ".[dev]"

# Plus ObjectScript (.cls, .mac, .inc, .rtn)
uv pip install --system -e ".[dev,objectscript]"
```

Requires Python 3.11 or 3.12 (tree-sitter-languages wheels only available for those versions).

## Design Decisions

1. **tree-sitter-objectscript as optional extra** — not bundled; users who need ObjectScript explicitly install `[objectscript]`. Keeps default install lightweight.

2. **Regex fallback preserved** — when tree-sitter grammar is missing or fails to load, regex patterns produce the same ParsedSummary dict. Old behavior is preserved for unsupported configurations.

3. **Lazy loading via singleton** — `ParserManager` created on first `parse_code_file()` call. No grammar loading at server startup.

4. **`return None` removed from `_summarize_directory`** — previously, a directory with code files but no docstrings returned `None` and was skipped. Now it returns a wiki page with whatever content is available (README, config files, or empty symbol list).

5. **Unified ParsedSummary shape** — all parsers return the same dict keys regardless of language. Languages without a feature simply have an empty list for that key (e.g., JavaScript has no `interfaces`).

## Edge Cases Handled

| Case | Behavior |
|---|---|
| Malformed binary file | Returns `{}`, no crash |
| File with no public symbols | Returns `{}`, directory still gets wiki page |
| tree-sitter API variation | `_load_language()` tries both `get_language()` and `.python()` patterns |
| Missing optional grammar (.cls without objectscript) | Falls back to regex |
| Syntax errors in code | Tree-sitter returns partial AST; acceptable |
| Large files | 4KB cap preserved from `_read_file_safe` |
| Case-insensitive extensions | `f.suffix.lower()` in extension check |
