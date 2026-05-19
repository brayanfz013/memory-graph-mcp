#!/usr/bin/env bash
# Tag and push a release.
#
# Usage:
#   ./scripts/release.sh 0.4.2
#
# Steps:
#   1. Verifies the working tree is clean.
#   2. Verifies pyproject + plugin.json + marketplace.json + server.py all
#      reference the target version (run scripts/bump-version.sh first if not).
#   3. Verifies CHANGELOG has a section for the new version.
#   4. Runs pytest + ruff one more time.
#   5. Creates a git tag vX.Y.Z and pushes main + the tag.
#
# After this script:
#   - GitHub Actions CI runs against the tag.
#   - To publish to PyPI: python -m build && twine upload dist/*
set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  echo "usage: ./scripts/release.sh X.Y.Z" >&2
  exit 1
fi
TAG="v${VERSION}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Checking working tree is clean..."
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

echo "==> Verifying version references match $VERSION..."
fails=()
grep -q "version = \"$VERSION\"" pyproject.toml || fails+=("pyproject.toml")
grep -q "\"version\": \"$VERSION\"" .claude-plugin/plugin.json || fails+=(".claude-plugin/plugin.json")
grep -q "\"version\": \"$VERSION\"" .claude-plugin/marketplace.json || fails+=(".claude-plugin/marketplace.json")
grep -q "v$VERSION" memory_graph/server.py || fails+=("memory_graph/server.py (log line)")
if [ "${#fails[@]}" -gt 0 ]; then
  echo "ERROR: these files do not reference $VERSION:" >&2
  printf '  - %s\n' "${fails[@]}" >&2
  echo "Update them first or run scripts/bump-version.sh $VERSION" >&2
  exit 1
fi

echo "==> Verifying CHANGELOG has section for $VERSION..."
if ! grep -q "## \[$VERSION\]" CHANGELOG.md; then
  echo "ERROR: CHANGELOG.md has no '## [$VERSION]' section." >&2
  exit 1
fi

echo "==> Running tests..."
if [ -d .venv ]; then
  source .venv/bin/activate
fi
pytest -q

echo "==> Running ruff..."
uvx ruff check memory_graph tests examples

echo "==> Checking tag $TAG does not already exist..."
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "ERROR: tag $TAG already exists." >&2
  exit 1
fi

echo "==> Creating tag $TAG..."
git tag -a "$TAG" -m "Release $TAG"

echo "==> Pushing main + tag..."
git push origin main
git push origin "$TAG"

echo ""
echo "✅ Released $TAG."
echo "Next: GitHub Actions runs CI against the tag."
echo "      To publish to PyPI manually:  python -m build && twine upload dist/*"
