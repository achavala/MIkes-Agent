#!/bin/bash
# Watch for file changes and automatically deploy to Fly.io
# Usage: ./watch_and_deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================================="
echo "FILE WATCHER - AUTO DEPLOY TO FLY.IO"
echo "=================================================================================="
echo ""
echo "This script will watch for file changes and automatically deploy to Fly.io"
echo "Press Ctrl+C to stop"
echo ""

# Check if fswatch is installed (macOS file watcher)
if ! command -v fswatch &> /dev/null; then
    echo "⚠️  fswatch not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install fswatch
    else
        echo "❌ Homebrew not found. Please install fswatch manually:"
        echo "   brew install fswatch"
        echo ""
        echo "Or use the manual deploy script: ./deploy_to_fly.sh"
        exit 1
    fi
fi

# Files/directories to watch
WATCH_PATTERNS=(
    "*.py"
    "*.sh"
    "Dockerfile"
    "fly.toml"
    "requirements.txt"
    "config.py"
    "start_cloud.sh"
    ".dockerignore"
)

# Build watch command
WATCH_CMD="fswatch -1"
for pattern in "${WATCH_PATTERNS[@]}"; do
    WATCH_CMD="$WATCH_CMD $pattern"
done

# Also watch directories
WATCH_CMD="$WATCH_CMD core/ utils/ phase0_backtest/"

echo "👀 Watching for changes in:"
for pattern in "${WATCH_PATTERNS[@]}"; do
    echo "   - $pattern"
done
echo "   - core/"
echo "   - utils/"
echo "   - phase0_backtest/"
echo ""
echo "Waiting for file changes..."
echo ""

# Watch loop
LAST_DEPLOY=0
MIN_DEPLOY_INTERVAL=30  # Minimum 30 seconds between deployments

while true; do
    # Wait for file change
    eval "$WATCH_CMD" > /dev/null 2>&1
    
    CURRENT_TIME=$(date +%s)
    TIME_SINCE_LAST_DEPLOY=$((CURRENT_TIME - LAST_DEPLOY))
    
    if [ $TIME_SINCE_LAST_DEPLOY -lt $MIN_DEPLOY_INTERVAL ]; then
        REMAINING=$((MIN_DEPLOY_INTERVAL - TIME_SINCE_LAST_DEPLOY))
        echo "⏳ Waiting ${REMAINING}s before next deployment (rate limiting)..."
        sleep $REMAINING
    fi
    
    echo ""
    echo "📝 File change detected! Deploying to Fly.io..."
    echo ""
    
    # Run deployment
    if ./deploy_to_fly.sh --auto; then
        LAST_DEPLOY=$(date +%s)
        echo ""
        echo "✅ Deployment complete. Watching for more changes..."
        echo ""
    else
        echo ""
        echo "❌ Deployment failed. Still watching for changes..."
        echo ""
    fi
    
    # Small delay before watching again
    sleep 2
done


