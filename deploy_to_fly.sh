#!/bin/bash
# Automatic deployment script for Fly.io
# Deploys changes immediately to Fly.io

set -e

# Check if --auto flag is set (for watch script)
AUTO_MODE=false
if [ "$1" == "--auto" ]; then
    AUTO_MODE=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================================="
echo "FLY.IO AUTOMATIC DEPLOYMENT"
echo "=================================================================================="
echo ""

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI not found. Please install it first:"
    echo "   https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "fly.toml" ]; then
    echo "❌ fly.toml not found. Are you in the project root?"
    exit 1
fi

# Get app name from fly.toml
APP_NAME=$(grep "^app = " fly.toml | sed 's/app = "\(.*\)"/\1/' | tr -d ' ')

if [ -z "$APP_NAME" ]; then
    echo "⚠️  Could not determine app name from fly.toml, using default"
    APP_NAME="mike-agent-project"
fi

echo "📦 App Name: $APP_NAME"
echo ""

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  WARNING: You have uncommitted changes"
    echo ""
    git status --short
    echo ""
    read -p "Continue with deployment anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
fi

# Show what will be deployed
echo "📋 Files to be deployed:"
echo "   - Dockerfile"
echo "   - fly.toml"
echo "   - All Python files"
echo "   - Configuration files"
echo ""

# Ask for confirmation (skip in auto mode)
if [ "$AUTO_MODE" = false ]; then
    read -p "🚀 Deploy to Fly.io now? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
else
    echo "🚀 Auto-deploying to Fly.io..."
fi

echo ""
echo "🚀 Starting deployment to Fly.io..."
echo ""

# Deploy to Fly.io
fly deploy --app "$APP_NAME"

DEPLOY_EXIT_CODE=$?

if [ $DEPLOY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "✅ DEPLOYMENT SUCCESSFUL"
    echo "=================================================================================="
    echo ""
    echo "Your changes are now live on Fly.io!"
    echo ""
    echo "To verify deployment:"
    echo "  fly status --app $APP_NAME"
    echo "  fly logs --app $APP_NAME"
    echo ""
    echo "To view app:"
    echo "  fly open --app $APP_NAME"
    echo ""
else
    echo ""
    echo "=================================================================================="
    echo "❌ DEPLOYMENT FAILED"
    echo "=================================================================================="
    echo ""
    echo "Check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "  - Build errors (check Dockerfile)"
    echo "  - Missing files (check .dockerignore)"
    echo "  - API key issues (check fly secrets)"
    echo ""
    exit $DEPLOY_EXIT_CODE
fi

