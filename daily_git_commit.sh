#!/bin/bash
# Daily Git Commit & Tag Script
# Automatically commits all changes and creates a tag with date/time

set -e  # Exit on error

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Generate tag name with date and time (YYYY-MM-DD-HHMM)
TAG_NAME="daily-$(date +%Y-%m-%d-%H%M)"
COMMIT_MESSAGE="Daily sync: $(date +'%Y-%m-%d %H:%M:%S')"

# Check if there are any changes to commit
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "No changes to commit. Skipping daily commit."
    exit 0
fi

# Stage all changes (including untracked files)
git add -A

# Commit all changes
git commit -m "$COMMIT_MESSAGE" || {
    echo "Warning: Commit failed (might be no changes after staging)"
    exit 0
}

# Create tag with date/time
git tag -a "$TAG_NAME" -m "Daily tag: $(date +'%Y-%m-%d %H:%M:%S')"

# Push commits and tags to GitHub
echo "Pushing commits and tag to GitHub..."
git push origin main || {
    echo "Error: Failed to push commits"
    exit 1
}

git push origin "$TAG_NAME" || {
    echo "Error: Failed to push tag"
    exit 1
}

echo "✅ Daily commit and tag created successfully!"
echo "   Tag: $TAG_NAME"
echo "   Commit: $COMMIT_MESSAGE"

