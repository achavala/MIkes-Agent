#!/bin/bash
# Setup Fly.io secrets for Mike Agent
# This script sets all required environment variables in Fly.io

echo "="*80
echo "FLY.IO SECRETS SETUP"
echo "="*80
echo ""

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI not found. Please install it first:"
    echo "   https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    echo "   Please add your API keys to .env file first"
    exit 1
fi

echo "📋 Reading secrets from .env file..."
echo ""

# Read secrets from .env file
ALPACA_KEY=$(grep "^ALPACA_API_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
ALPACA_SECRET=$(grep "^ALPACA_SECRET_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
MASSIVE_API_KEY=$(grep "^MASSIVE_API_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
POLYGON_API_KEY=$(grep "^POLYGON_API_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")

# Check which secrets are available
SECRETS_TO_SET=""

if [ -n "$ALPACA_KEY" ]; then
    echo "✅ Found ALPACA_KEY"
    SECRETS_TO_SET="$SECRETS_TO_SET ALPACA_KEY=$ALPACA_KEY"
else
    echo "⚠️  ALPACA_KEY not found in .env"
fi

if [ -n "$ALPACA_SECRET" ]; then
    echo "✅ Found ALPACA_SECRET"
    SECRETS_TO_SET="$SECRETS_TO_SET ALPACA_SECRET=$ALPACA_SECRET"
else
    echo "⚠️  ALPACA_SECRET not found in .env"
fi

# Use MASSIVE_API_KEY if available, otherwise try POLYGON_API_KEY
if [ -n "$MASSIVE_API_KEY" ]; then
    echo "✅ Found MASSIVE_API_KEY"
    SECRETS_TO_SET="$SECRETS_TO_SET MASSIVE_API_KEY=$MASSIVE_API_KEY"
elif [ -n "$POLYGON_API_KEY" ]; then
    echo "✅ Found POLYGON_API_KEY (using as MASSIVE_API_KEY)"
    SECRETS_TO_SET="$SECRETS_TO_SET MASSIVE_API_KEY=$POLYGON_API_KEY"
else
    echo "⚠️  MASSIVE_API_KEY not found in .env (optional but recommended)"
fi

echo ""
echo "🚀 Setting secrets in Fly.io..."
echo ""

# Set secrets in Fly.io
if [ -n "$SECRETS_TO_SET" ]; then
    fly secrets set $SECRETS_TO_SET
    echo ""
    echo "✅ Secrets set successfully!"
    echo ""
    echo "📋 To verify secrets are set:"
    echo "   fly secrets list"
    echo ""
    echo "🔄 To restart the app with new secrets:"
    echo "   fly apps restart mike-agent-project"
else
    echo "❌ No secrets found to set. Please check your .env file."
    exit 1
fi

echo "="*80


