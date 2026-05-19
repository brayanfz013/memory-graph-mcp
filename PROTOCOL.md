# Memory Protocol — how to actually use memory-graph

> This is the **usage protocol**. The MCP server is the engine; this document is the operating manual every agent (and every human pairing with one) should follow.

The protocol has **three pillars** and **one anti-pattern list**. If you do nothing else, do these three things.

---

## Pillar 1 — Recall *before* you work

Before starting **any task** (code, conversation, research — no exceptions), run:

```python
recall(query="<the task in your own words>", top_k=5)
```

That single call searches across `memories + KG + wiki` semantically and returns ranked results plus `top_canonicals` (the most-relevant canonical IDs). If anything comes back with score > 0.6, **read it before doing the work**.

Cost: one tool call. Benefit: stop re-discovering solutions you already wrote.

> **Rule of thumb:** if `recall` returns a relevant result, *use it*. Don't re-derive the answer just because you didn't see the original.

Variants you'll want:

| Need | Call |
|---|---|
| Just past solutions | `recall(query=..., scope="memories")` |
| Just decisions / patterns | `recall(query=..., scope="kg", node_type="Decision")` |
| Just curated docs | `recall(query=..., scope="wiki")` |
| Expand a hit into full text | `wiki_get(canonical_id)` |

---

## Pillar 2 — Record *after* you solve

Every time you fix something, decide something, or notice a pattern, persist it:

```python
record_finding(
    finding_type="solution",        # solution | decision | insight | problem | pattern | context
    title="Short, specific, searchable",
    content="Symptom → cause → fix → validation. Be terse but complete.",
    related_files=["path/to/touched.py"],
    tags=["domain", "subsystem"],
    source_agent="backend-engineer",
)
```

What this one call does behind the scenes:
1. Stores a vector memory (embedded for semantic recall).
2. Creates a KG node with a stable `canonical_id` (deduplicates if you record the same thing twice).
3. Generates `tldr_32 / brief_96 / summary_256` compression levels.
4. Auto-infers up to 3 `RELATED_TO` edges to semantically similar existing nodes (cosine ≥ 0.62).
5. Auto-promotes to `canonical` when `reuse_count ≥ 3`.
6. Auto-crystallizes a wiki page when the node becomes canonical.

> **Rule of thumb:** if a future you (or future agent) would benefit from knowing this, record it. The cost is one call; the cost of *not* recording is paying the discovery cost again.

### What to record

| Type | When | Example |
|---|---|---|
| `solution` | A bug got fixed | "Fix: DuckDB lock → retry with backoff" |
| `decision` | An architecture/design choice | "Decision: one DuckDB file per workspace, never global" |
| `insight` | A non-obvious pattern noticed | "Insight: VS Code MCP customInstructions only fire on cold start" |
| `problem` | Issue identified but not yet fixed | "Problem: wiki_lint flags 12 orphaned pages after refactor" |
| `pattern` | Reusable approach across cases | "Pattern: always `recall` before file-search; saves 5+ tool calls" |
| `context` | Working state worth resuming | "Context: mid-migration v3→v4 schema; halt point is db.py:142" |

### What NOT to record

- Secrets, tokens, PII, passwords.
- Raw data dumps (use files; reference path in `related_files`).
- Trivial facts that won't be reused.
- Anything already captured by `git log` / `git blame`.

---

## Pillar 3 — Link relationships when they matter

Vector recall finds *similar* things. The knowledge graph finds *related* things and surfaces what's structurally important. The auto-edges from `record_finding` give you a baseline graph for free, but typed edges are still worth adding by hand for important links:

```python
# A new fix supersedes an older approach
kg_add_edge(from_id="solution.new-approach", to_id="solution.old-approach", rel_type="SUPERSEDES")

# A solution solves a known problem
kg_add_edge(from_id="solution.duckdb-retry", to_id="problem.duckdb-lock-flake", rel_type="SOLVES")

# A decision depends on another
kg_add_edge(from_id="decision.use-fastembed", to_id="decision.no-vendor-lockin", rel_type="DEPENDS_ON")
```

Valid edge types: `SOLVES`, `CAUSED_BY`, `DEPENDS_ON`, `RELATED_TO`, `USES_TOOL`, `SUPERSEDES`.

Traverse when you need context:

```python
kg_neighbors(node_id="decision.use-fastembed", direction="both")
kg_path(from_id="problem.duckdb-lock-flake", to_id="solution.duckdb-retry")
```

PageRank runs automatically inside `memory_consolidate(dry_run=False)`. To see the most influential nodes:

```python
kg_influential(top_k=10, node_type="Decision")
```

---

## Cross-agent coordination

When multiple agents work on the same task, share state through `collective_*`:

```python
collective_store(type="knowledge", key="api-contract-v2", value={...}, scope="project")
collective_get(key="api-contract-v2", scope="project")
collective_list(type="knowledge", scope="project")
```

| Type | TTL |
|---|---|
| `knowledge`, `result`, `consensus`, `system` | permanent |
| `context` | 1h |
| `task` | 30m |
| `error` | 24h |
| `metric` | 1h |

For expensive idempotent tool calls, memoize with `cache_check / cache_store`:

```python
hit = cache_check(tool_name="lakehouse_sql_query", args_hash="ventas_2024_q1")
if hit["hit"]:
    return hit["result"]
# ... run the expensive call ...
cache_store(tool_name="lakehouse_sql_query", args_hash="ventas_2024_q1", result=json.dumps(result), ttl_seconds=3600)
```

---

## Embedding provider hygiene

Different embedding models give different recall quality. `memory-graph` ships pluggable providers; the protocol for swapping safely is:

1. **Status first** — `embedding_status()` tells you what's active vs what's stored. A mismatch warning means the current env points at a different model than the DB was last written with. Recall still works, but new writes will mix vector spaces if you don't migrate.
2. **Benchmark before swapping** — `embedding_benchmark(providers=[...])` scores Recall@1, Recall@5, MRR against the bundled eval set. Don't choose a model on reputation alone; let the numbers speak for your corpus.
3. **Migrate when you've decided** — `embedding_migrate(target_provider, target_model, dry_run=False)` re-embeds every memory + wiki page in place. The original text is untouched; only vectors change. A new generation is recorded in `embedding_meta`.

> **Rule:** never swap providers without migrating. Mixing vectors from different models is silently broken — cosine distances stop being comparable, and recall quality degrades unpredictably.

## Maintenance cadence

| Frequency | Action | Why |
|---|---|---|
| Per session start | `recall(query="recent context", top_k=3)` | Pick up where you left off |
| Per session end | `record_finding(...)` for anything notable | Don't lose what you learned |
| Weekly (or after large batch of new canonicals) | `wiki_bootstrap()` | Crystallize new canonicals into wiki |
| Weekly | `memory_consolidate(dry_run=False)` | Purge expired + recompute PageRank |
| As needed | `memory_report()` | Health: counts, lint, top influential |

---

## Anti-patterns

- **Skipping `recall` because "I know what to do"** — you don't know what was decided last sprint by another agent.
- **Recording every micro-detail** — record the *non-obvious* part. The code shows *what*; memory should capture *why*.
- **Storing secrets, tokens, raw payloads** — never. Reference files instead.
- **Manually building edges before recording** — `record_finding` auto-infers `RELATED_TO`; add typed edges (`SOLVES`, `SUPERSEDES`) only when meaningful.
- **Treating `memory_recall` / `kg_query` as the default** — they're legacy. Prefer `recall` (fused).
- **Manual `wiki_crystallize` for every node** — canonical promotion already triggers it.

---

## Quick reference card

```
Start of any task     → recall(query, top_k=5)
After solving         → record_finding(type, title, content, files, tags)
Worth linking         → kg_add_edge(from, to, rel_type)
Need full doc         → wiki_get(canonical_id)
Share with other agent → collective_store(type, key, value, scope)
Skip expensive call   → cache_check(tool_name, args_hash) → cache_store(...)
Weekly cleanup        → memory_consolidate(dry_run=False)
Sanity check          → memory_report()
```

That's the protocol. Three pillars: recall, record, link. Everything else is sugar on top.
