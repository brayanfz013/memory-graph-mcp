# Inspiration: what we took from STORM / Co-STORM (and what we didn't)

`memory-graph-mcp` v0.5.0 adopts several **knowledge-organization ideas** from
Stanford OVAL's [STORM and Co-STORM](https://github.com/stanford-oval/storm)
(MIT-licensed). We ported *concepts*, not code — this repo has **no dependency
on `storm`/`dspy`** and makes **no LLM calls**. This document records what was
adopted, how it was adapted to a local/zero-config server, and what was
deliberately left out, so the boundary stays clear.

## What STORM is

STORM is a research system that writes Wikipedia-style, cited articles by
(a) researching a topic through multi-perspective question-asking, (b) generating
an outline, and (c) writing the article section-by-section with grounded
citations. **Co-STORM** adds a collaborative discourse protocol and a *dynamic
mind map* — a hierarchical concept structure that organises gathered information
to reduce redundancy and guide retrieval.

## Why we did NOT adopt it wholesale

STORM's core is **LLM generation orchestrated with DSPy + online retrieval**.
`memory-graph-mcp` is intentionally the opposite: a lightweight, local,
deterministic MCP server (DuckDB + local `fastembed` embeddings + a PageRank
knowledge graph + a wiki layer), with **no API keys and no per-call LLM cost**.
Pulling in DSPy or article-generation would break that contract. So we took the
*organizational algorithms* and re-expressed them with the signals we already
have.

## What we adopted, and how we adapted it

| STORM / Co-STORM idea | Our adaptation (local, no LLM) | Where |
| --- | --- | --- |
| **Dynamic mind map** — hierarchical concept structure over gathered info | Community detection over the `RELATED_TO` affinity graph (`weight` = cosine sim from `record_finding`): weighted **label propagation** for coarse topics, then connected components at a tighter threshold for subtopics. No re-embedding, near-linear. (We started with single-linkage union-find but it chained the dense graph into one giant blob on real data — LPA fixed it; see CHANGELOG v0.5.1.) | `memory_graph/topics.py`, `memory_map` tool, schema v6 |
| **Outline-first article writing** | Crystallized wiki pages carry a `## Contents` outline; `wiki_get(outline_only=…/section=…)` returns just the headings or one section. | `memory_graph/wiki.py` |
| **Per-sentence source grounding / citations** | `record_finding(sources=[...])` stores grounding anchors and renders a numbered `## Sources` section; findings expose a `grounded` flag. | `memory_graph/intelligence.py`, `wiki.py` |
| **Multi-perspective "what's missing?" questioning** | Deterministic structural coverage critic: isolated nodes, ungrounded canonicals, missing wikis, promote-ready drafts, stale canonicals, orphan pages — each with a recommended action. | `memory_gaps` in `memory_graph/intelligence.py` |

## What we left out (on purpose)

- **DSPy / prompt-programming pipeline** — not needed; we cluster over existing
  embeddings.
- **LLM-driven question generation, outline writing, and prose synthesis** —
  would add API cost and non-determinism. Our outlines/clusters are computed
  from structure.
- **Co-STORM's multi-agent live discourse** — out of scope for a memory store;
  cross-agent coordination here is the existing `collective_*` primitives.

## A token-quality note

Beyond fidelity to STORM, these features target a concrete problem observed in
practice: `recall(scope="all")` could return the *same fact three times* (as a
memory, a KG node carrying the full content, and a wiki page), inflating context
and diluting answers. `compact` dedup + outline-first wiki + topic grouping
directly cut that redundancy.

## License / attribution

STORM is MIT-licensed. Because we ported ideas rather than source, attribution
(this file + the `CHANGELOG`/`README` credit) is sufficient. If any STORM source
is vendored in the future, add its MIT license text under `licenses/`.
