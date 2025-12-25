#!/bin/bash
# Git post-commit hook to automatically deploy to Fly.io
# Install: ln -s ../../git_hook_deploy.sh .git/hooks/post-commit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Only deploy if FLY_AUTO_DEPLOY is set
if [ "$FLY_AUTO_DEPLOY" != "true" ]; then
    exit 0
fi

echo ""
echo "🔄 Auto-deploying to Fly.io after commit..."
echo ""

# Run deployment script in auto mode
./deploy_to_fly.sh --auto


