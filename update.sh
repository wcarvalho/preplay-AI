#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./update.sh \"commit message\""
  exit 1
fi

MSG="$1"

# Push each submodule
git submodule foreach '
  echo "==> Processing $name"

  # If in detached HEAD, checkout main and merge the detached commit
  if ! git symbolic-ref -q HEAD > /dev/null 2>&1; then
    DETACHED_SHA=$(git rev-parse HEAD)
    echo "  Detached HEAD at $DETACHED_SHA, checking out main and merging..."
    git checkout main
    git merge --no-edit "$DETACHED_SHA"
  fi

  git add -A
  git diff --cached --quiet && echo "  Nothing to commit" || {
    git commit -m "'"$MSG"'"
  }
  git push
'

# Push main repo
echo "==> Pushing main repo"
git add -A
git diff --cached --quiet && echo "Nothing to commit in main repo" || {
  git commit -m "$MSG"
}
git push
