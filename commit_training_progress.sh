#!/bin/bash
#
# Auto-commit Training Progress
#
# Run this periodically (via cron) to commit training logs and checkpoints to GitHub
# Useful for monitoring training progress from another machine
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if git repo
if [ ! -d ".git" ]; then
    echo "Not a git repository. Exiting."
    exit 0
fi

# Check for changes
CHANGES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

if [ "$CHANGES" -eq 0 ]; then
    # No changes, exit silently
    exit 0
fi

# Add training outputs
echo "📦 Committing training progress..."

# Add logs
git add training*.log 2>/dev/null || true

# Add checkpoints (only new ones to avoid re-uploading)
git add models/*.zip 2>/dev/null || true

# Commit
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "Training progress: $TIMESTAMP" 2>/dev/null || {
    echo "No changes to commit"
    exit 0
}

# Push
BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
if git push origin "$BRANCH" 2>/dev/null; then
    echo "✅ Training progress pushed to GitHub ($TIMESTAMP)"
else
    echo "⚠️  Push failed (will retry next time)"
fi

