#!/usr/bin/env bash
# One-shot script to create the GitHub repo, push, and verify CI fires.
#
# Prerequisites you must have BEFORE running:
#   1. gh CLI installed:           https://cli.github.com/   (or: sudo apt install gh)
#   2. Authenticated:              gh auth login   (opens browser; do not paste token)
#   3. This repo cloned locally with .git initialized.
#
# What this script does:
#   1. Verifies gh + git are present + you're authenticated.
#   2. Creates the public repo brayanfz013/memory-graph-mcp on GitHub.
#   3. Adds it as origin and pushes main.
#   4. Reports CI run URL.
#
# Run with:
#   ./scripts/setup-github.sh
#
# This is safe to re-run (it skips create + remote-add if they already exist).
set -euo pipefail

REPO_OWNER="brayanfz013"
REPO_NAME="memory-graph-mcp"
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Checking prerequisites..."
if ! command -v gh &>/dev/null; then
  echo "ERROR: gh CLI not installed." >&2
  echo "  Install with: sudo apt install gh   (or see https://cli.github.com/)" >&2
  exit 1
fi
if ! command -v git &>/dev/null; then
  echo "ERROR: git not installed." >&2
  exit 1
fi
if ! gh auth status &>/dev/null; then
  echo "ERROR: not authenticated with gh." >&2
  echo "  Run: gh auth login   (opens browser, no token paste needed)" >&2
  exit 1
fi
echo "  gh auth OK"

echo "==> Checking git state..."
if [ ! -d .git ]; then
  echo "ERROR: not a git repo. Run from inside memory-graph-mcp/." >&2
  exit 1
fi
if ! git rev-parse HEAD &>/dev/null; then
  echo "ERROR: no commits yet. Run: git add -A && git commit -m 'feat: initial commit'" >&2
  exit 1
fi
echo "  git OK (HEAD: $(git rev-parse --short HEAD))"

echo "==> Checking if repo already exists on GitHub..."
if gh repo view "$REPO_FULL" &>/dev/null; then
  echo "  $REPO_FULL already exists on GitHub. Will reuse."
  REPO_EXISTS=1
else
  echo "  $REPO_FULL does not exist yet. Will create."
  REPO_EXISTS=0
fi

echo "==> Checking origin remote..."
if git remote get-url origin &>/dev/null; then
  current_origin=$(git remote get-url origin)
  echo "  origin already set: $current_origin"
else
  echo "  origin not set yet."
fi

if [ "$REPO_EXISTS" -eq 0 ]; then
  echo "==> Creating repo $REPO_FULL (public)..."
  gh repo create "$REPO_FULL" \
    --public \
    --description "Unified semantic memory MCP server with pluggable embedding providers (fastembed/ollama/vertex), knowledge graph, wiki crystallization, and benchmark + migration tooling." \
    --homepage "https://github.com/$REPO_FULL" \
    --source=. \
    --remote=origin \
    --push
else
  if ! git remote get-url origin &>/dev/null; then
    echo "==> Adding origin remote..."
    git remote add origin "https://github.com/$REPO_FULL.git"
  fi
  echo "==> Pushing main to origin..."
  git push -u origin main
fi

echo ""
echo "==> Done."
echo "  Repo:   https://github.com/$REPO_FULL"
echo "  CI:     https://github.com/$REPO_FULL/actions"
echo "  Issues: https://github.com/$REPO_FULL/issues"
echo ""
echo "Watch the first CI run with:"
echo "  gh run watch --repo $REPO_FULL"
