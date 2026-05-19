---
name: memory-recall
description: Run a semantic recall across memories, KG, and wiki for the current task. Use at the start of any non-trivial work.
allowed-tools: ["mcp__memory-graph__recall", "mcp__memory-graph__wiki_get"]
---

You are about to start a task. Before doing anything else, run the memory protocol:

1. Call `mcp__memory-graph__recall` with:
   - `query` = the user's request (in your own words if needed for better semantic match)
   - `top_k` = 5
   - `scope` = "all"

2. If `top_canonicals` has any entries with score > 0.6, expand the top 1-2 with `mcp__memory-graph__wiki_get(canonical_id)`.

3. Report back in ≤ 5 lines:
   - **Found:** N memories / M KG nodes / K wiki pages.
   - **Most relevant:** title + 1-line summary of the highest-score hit (or "nothing relevant" if all scores < 0.5).
   - **Recommendation:** "use existing finding X" / "extend existing pattern Y" / "no prior art — proceed fresh".

User request:

$ARGUMENTS
