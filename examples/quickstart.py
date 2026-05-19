"""Quickstart for memory-graph: end-to-end demo of recall → record → link → wiki.

Run with:
    MEMORY_GRAPH_WORKSPACE=/tmp/mg-demo python examples/quickstart.py

The script:
  1. Spins up a fresh workspace.
  2. Records two findings (a problem + a solution).
  3. Links them with a typed SOLVES edge.
  4. Promotes the solution to canonical (which auto-crystallizes a wiki page).
  5. Recalls the topic across all scopes.
  6. Fetches the wiki page in full.

No MCP client needed — this calls the library directly.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    workspace = Path(os.environ.get("MEMORY_GRAPH_WORKSPACE", tempfile.mkdtemp(prefix="mg-demo-")))
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".git").mkdir(exist_ok=True)  # makes the workspace look like a project root
    os.environ["MEMORY_GRAPH_WORKSPACE"] = str(workspace)

    print(f"[demo] workspace = {workspace}")

    # Imports must happen AFTER MEMORY_GRAPH_WORKSPACE is set
    # (settings are loaded eagerly when the package is first imported).
    from memory_graph import intelligence, knowledge_graph, unified, wiki

    # 1. Record a problem
    problem = intelligence.memory_record_finding(
        finding_type="problem",
        title="DuckDB lock under concurrent writes",
        content=(
            "Two agents writing to the same .memory-graph/memory.duckdb "
            "occasionally raise 'Could not set lock'. Reproducible with 3 "
            "parallel record_finding calls."
        ),
        related_files=["memory_graph/db.py"],
        tags=["duckdb", "concurrency"],
        source_agent="demo",
    )
    print(f"[demo] recorded problem → {problem.get('canonical_id')}")

    # 2. Record a solution
    solution = intelligence.memory_record_finding(
        finding_type="solution",
        title="DuckDB lock retry with exponential backoff",
        content=(
            "Wrap writes in @with_retry(max_attempts=3, base_delay=0.2). "
            "Detect transient errors by message ('could not set lock', "
            "'database is locked', 'write-ahead log'). Validated with 3 "
            "concurrent agents writing 100 findings each — zero failures."
        ),
        related_files=["memory_graph/db.py"],
        tags=["duckdb", "concurrency", "retry"],
        source_agent="demo",
    )
    print(f"[demo] recorded solution → {solution.get('canonical_id')}")

    # 3. Link them with a typed edge
    knowledge_graph.kg_add_edge(
        from_id=solution["node_id"],
        to_id=problem["node_id"],
        rel_type="SOLVES",
        weight=1.0,
    )
    print("[demo] linked solution --SOLVES--> problem")

    # 4. Promote the solution to canonical (auto-crystallizes wiki)
    knowledge_graph.kg_promote(solution["node_id"], "canonical")
    try:
        wiki.wiki_crystallize(solution["canonical_id"])
    except Exception as exc:
        print(f"[demo] wiki_crystallize skipped: {exc}")
    print("[demo] promoted solution → canonical (wiki auto-crystallized)")

    # 5. Recall across all scopes
    results = unified.recall(query="duckdb concurrent write lock", top_k=5)
    print("[demo] recall results:")
    print(json.dumps(
        {
            "memories": len(results.get("memories", [])),
            "kg_nodes": len(results.get("kg", {}).get("nodes", [])),
            "wiki_pages": len(results.get("wiki", [])),
            "top_canonicals": results.get("top_canonicals", [])[:3],
        },
        indent=2,
    ))

    # 6. Fetch the wiki page in full
    page = wiki.wiki_get(solution["canonical_id"])
    body_preview = (page.get("body") or "")[:240]
    print(f"[demo] wiki_get body preview:\n{body_preview}…")

    print("\n[demo] done. To clean up:  rm -rf", workspace)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[demo] FAILED: {exc!r}")
        raise
