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

  # Commit any changes first
  git add -u
  git diff --cached --quiet && echo "  Nothing to commit" || {
    git commit -m "'"$MSG"'"
  }

  # If in detached HEAD, merge into main via temp branch
  if ! git symbolic-ref -q HEAD > /dev/null 2>&1; then
    TEMP_BRANCH="temp-update-$(date +%s)"
    echo "  Detached HEAD, creating temp branch and merging into main..."
    git checkout -b "$TEMP_BRANCH"
    git checkout main
    git merge --no-edit "$TEMP_BRANCH"
    git branch -d "$TEMP_BRANCH"
  fi
  git push
'

# Push main repo
echo "==> Pushing main repo"
git add -A
git diff --cached --quiet && echo "Nothing to commit in main repo" || {
  git commit -m "$MSG"
}
git push
