"""Freshness of recorded knowledge against the code it cites.

A finding cites files (`files`) and, since v0.6.0, the commit it was recorded at
(`commit`). Knowledge about code goes stale silently: measured on a real store,
25% of cited files no longer existed and nothing told the reader. This module
answers, per node, which cited files are now missing and which changed since the
node was recorded, so recall can flag and down-rank stale knowledge.

Everything degrades to "unknown = not stale" when the workspace is not a git
repository or git is unavailable: freshness must never break a read.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from .settings import settings

logger = logging.getLogger(__name__)

#: Rank multiplier applied to a node whose cited code is missing or changed.
STALE_PENALTY = 0.85
_GIT_TIMEOUT_S = 5


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", args, exc)
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _workspace() -> Path:
    return Path(settings.workspace_path)


def head_commit(root: Path | None = None) -> str | None:
    """HEAD sha of the workspace, or None when it is not a git repo."""
    return _git(root or _workspace(), "rev-parse", "HEAD")


_DIFF_CACHE: dict[tuple[str, str, str], frozenset[str]] = {}


def _changed_since(root: str, commit: str, head: str) -> frozenset[str] | None:
    """Files changed between `commit` and `head`, relative to `root`.

    `--relative` makes paths relative to the workspace even when it is a
    sub-directory of a bigger repo. Only successes are cached (keyed by HEAD, so
    the cache invalidates when HEAD moves); a failure — e.g. a commit no longer
    reachable after a rebase — returns None, which callers report as unknown.
    """
    key = (root, commit, head)
    if key in _DIFF_CACHE:
        return _DIFF_CACHE[key]
    out = _git(Path(root), "diff", "--name-only", "--relative", commit, head)
    if out is None:
        return None
    names = frozenset(os.path.normpath(n) for n in out.splitlines() if n)
    if len(_DIFF_CACHE) > 512:
        _DIFF_CACHE.clear()
    _DIFF_CACHE[key] = names
    return names


def _relative(root: Path, f: str) -> str | None:
    p = Path(f)
    if p.is_absolute():
        try:
            return os.path.normpath(str(p.resolve().relative_to(root.resolve())))
        except ValueError:
            return None  # outside this workspace: cannot judge it
    return os.path.normpath(f)


_NO_HEAD = object()


def file_freshness(
    properties: dict[str, Any], root: Path | None = None, head: Any = _NO_HEAD,
) -> dict[str, Any]:
    """{'missing', 'changed', 'stale', 'unknown'} for one node's cited files.

    `head` lets a caller that annotates many nodes resolve HEAD once. `unknown`
    is True when the recorded commit can no longer be compared (e.g. rewritten
    history): the node is then neither claimed fresh nor stale on that ground.
    Uncommitted edits are not seen (the comparison is commit..HEAD).
    """
    root = root or _workspace()
    files = [f for f in (properties.get("files") or []) if isinstance(f, str) and f]
    missing: list[str] = []
    for f in files:
        p = Path(f) if Path(f).is_absolute() else root / f
        if not p.exists():
            missing.append(f)

    changed: list[str] = []
    unknown = False
    commit = properties.get("commit")
    if commit and files:
        if head is _NO_HEAD:
            head = head_commit(root)
        if head and head != commit:
            diff = _changed_since(str(root), commit, head)
            if diff is None:
                unknown = True
            else:
                for f in files:
                    rel = _relative(root, f)
                    if rel and rel in diff and f not in missing:
                        changed.append(f)

    return {"missing": missing, "changed": changed, "stale": bool(missing or changed),
            "unknown": unknown}


def annotate(node: dict[str, Any], root: Path | None = None, head: Any = _NO_HEAD) -> dict[str, Any]:
    """Attach `freshness` and `rank_score` (semantic_score down-weighted when stale)."""
    fr = file_freshness(node.get("properties") or {}, root, head)
    node["freshness"] = fr
    base = float(node.get("semantic_score", 0.0) or 0.0)
    node["rank_score"] = round(base * STALE_PENALTY, 4) if fr["stale"] else base
    return node
