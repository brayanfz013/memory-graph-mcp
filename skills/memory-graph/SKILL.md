---
name: memory-graph
description: Operating protocol for the memory-graph MCP server. Load at session start so the agent runs `recall` before any task, persists findings with `record_finding` after solving, and uses the knowledge graph for relationships. Required reading for any agent paired with this plugin.
version: 0.4.3
---

# memory-graph — usage protocol

> The MCP server is the engine; this skill is the operating manual. Three pillars: **recall, record, link**. Anything else is sugar.

## When to load this skill

- Every session start.
- Any time an agent says "let me figure out how to use memory" — load this instead.
- Before any non-trivial task (3+ steps, debugging, architecture decisions).

## Pillar 1 — Recall *before* you work (MANDATORY)

```python
recall(query="<the task in your own words>", top_k=5)
```

This single call searches **across memories + KG + wiki** by semantic similarity. If a result scores > 0.6, read it before doing the work. If `top_canonicals` is non-empty, expand the most relevant with `wiki_get(canonical_id)`.

| Need | Call |
|---|---|
| Just past solutions | `recall(query=..., scope="memories")` |
| Just decisions/patterns | `recall(query=..., scope="kg", node_type="Decision")` |
| Just curated docs | `recall(query=..., scope="wiki")` |
| Expand a hit into full text | `wiki_get(canonical_id)` |

> **Rule:** if `recall` returns a relevant result, USE it. Don't re-derive.

## Pillar 2 — Record *after* you solve

```python
record_finding(
    finding_type="solution",        # solution | decision | insight | problem | pattern | context
    title="Short, specific, searchable",
    content="Symptom → cause → fix → validation. Terse but complete.",
    related_files=["path/to/touched.py"],
    tags=["domain", "subsystem"],
    source_agent="<your-agent-name>",
)
```

One call does: vector memory + KG node + canonical_id dedup + tldr_32/brief_96 compression + auto-edges (cosine ≥ 0.62) + auto-promote at reuse ≥ 3 + auto-wiki on canonical.

### Type cheatsheet

| Type | When | Example |
|---|---|---|
| `solution` | Bug fixed | "Fix: DuckDB lock → retry with backoff" |
| `decision` | Design choice | "Decision: workspace-scoped DuckDB, never global" |
| `insight` | Non-obvious pattern | "Insight: VS Code MCP customInstructions only fire on cold start" |
| `problem` | Issue identified | "Problem: wiki_lint flags 12 orphaned pages" |
| `pattern` | Reusable approach | "Pattern: recall before file-search saves 5+ tool calls" |
| `context` | Resumable state | "Context: mid-migration v3→v4; halt at db.py:142" |

### Never record

- Secrets, tokens, PII.
- Raw data dumps — reference files via `related_files`.
- Trivial facts (`git log` already captures them).

## Pillar 3 — Link relationships when they matter

Auto-edges (`RELATED_TO`) are inferred by `record_finding`. Add **typed** edges manually when meaningful:

```python
kg_add_edge(from_id="solution.new", to_id="solution.old", rel_type="SUPERSEDES")
kg_add_edge(from_id="solution.x",   to_id="problem.y",    rel_type="SOLVES")
kg_add_edge(from_id="decision.a",   to_id="decision.b",   rel_type="DEPENDS_ON")
```

Valid types: `SOLVES, CAUSED_BY, DEPENDS_ON, RELATED_TO, USES_TOOL, SUPERSEDES`.

Traverse when you need lineage:

```python
kg_neighbors(node_id="...", direction="both")
kg_path(from_id="problem.y", to_id="solution.x")
kg_influential(top_k=10, node_type="Decision")
```

## Cross-agent coordination

```python
collective_store(type="knowledge", key="api-contract-v2", value={...}, scope="project")
collective_get(key="api-contract-v2", scope="project")
```

TTLs: `knowledge|result|consensus|system` = permanent · `context` = 1h · `task` = 30m · `error` = 24h · `metric` = 1h.

Memoize expensive idempotent tool calls:

```python
hit = cache_check(tool_name="lakehouse_sql_query", args_hash="ventas_q1")
if hit["hit"]:
    return hit["result"]
# ... run ...
cache_store(tool_name="lakehouse_sql_query", args_hash="ventas_q1", result=json.dumps(r), ttl_seconds=3600)
```

## Composite workflows

### "I solved a bug"
1. `record_finding(finding_type="solution", title=..., content=..., related_files=[...])`
2. `kg_add_edge(from_id=<solution>, to_id=<problem>, rel_type="SOLVES")` (optional)

### "Architecture decision"
1. `recall(query="prior decisions about <topic>")`
2. `kg_influential(top_k=5, node_type="Decision")`
3. `record_finding(finding_type="decision", title=..., content=...)`
4. `kg_add_edge(from_id=<new>, to_id=<superseded>, rel_type="SUPERSEDES")` if it replaces something

### "Debug an error"
1. `recall(query="<error message>", scope="memories")` — past fixes
2. `recall(query="<error keywords>", scope="kg")` — related nodes
3. ... fix ...
4. `record_finding(finding_type="solution", title="Fix: <error>", content="<root cause + fix>")`

### "Session start"
1. `memory_report()` — health
2. `recall(query="recent context", top_k=3)` — where we left off
3. `kg_influential(top_k=5)` — most important decisions

## Embedding provider hygiene (v0.4.1)

The retrieval quality of every other operation in this protocol depends on the embedding model. To change models safely:

```python
embedding_status()                              # see active vs DB-stored + mismatch
embedding_benchmark(providers=[...])            # measure Recall@1/@5/MRR before swapping
embedding_migrate(target_provider="fastembed",
                  target_model="...",
                  dry_run=False)                # re-embed everything under the new model
```

> **Never swap providers without migrating.** Mixing vectors from different models silently breaks recall — cosine distances stop being comparable across models.

See `PROTOCOL.md` for the full operator manual.

## Maintenance

- `memory_report()` — counts + lint + top influential
- `memory_consolidate(dry_run=False)` — purge expired + recompute PageRank (weekly)
- `wiki_bootstrap()` — crystallize new canonicals (after large batches)

## Anti-patterns

- Skipping `recall` because "I know what to do" — you don't know what the other agent decided last sprint.
- Recording every micro-detail — capture the *why* and *non-obvious*, not the obvious *what*.
- Manual `kg_add_node` / `wiki_crystallize` for every node — `record_finding` handles both.
- Calling legacy `memory_recall` / `kg_query` as default — prefer `recall` (fused).

## Quick reference

```
Start of task         → recall(query, top_k=5)
After solving         → record_finding(type, title, content, files, tags)
Worth linking         → kg_add_edge(from, to, rel_type)
Need full doc         → wiki_get(canonical_id)
Share with agents     → collective_store(type, key, value, scope)
Skip expensive call   → cache_check → cache_store
Weekly cleanup        → memory_consolidate(dry_run=False)
Health check          → memory_report()
```
