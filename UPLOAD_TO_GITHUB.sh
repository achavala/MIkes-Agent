#!/bin/bash
# Script to upload Mike Agent v3 to GitHub

echo "=========================================="
echo "Mike Agent v3 - GitHub Upload Helper"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git repository not initialized"
    echo "Run: git init"
    exit 1
fi

# Check for sensitive files
echo "🔍 Checking for sensitive files..."
if [ -f config.py ]; then
    if git check-ignore config.py > /dev/null 2>&1; then
        echo "✅ config.py is in .gitignore (safe)"
    else
        echo "⚠️  WARNING: config.py is NOT in .gitignore!"
        echo "   This file contains API keys and should NOT be committed"
        exit 1
    fi
fi

# Show what will be committed
echo ""
echo "📋 Files that will be committed:"
git status --short | grep -v "^??" | head -20

echo ""
echo "📋 Untracked files (will be added):"
git status --short | grep "^??" | head -20

echo ""
read -p "Continue with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Add files
echo ""
echo "📦 Adding files..."
git add .

# Commit
echo ""
echo "💾 Committing changes..."
git commit -m "Mike Agent v3: RL trading agent with 5-tier TP, volatility regimes, and comprehensive risk management

Features:
- Reinforcement Learning (PPO) model for entry signals
- 5-tier take-profit system (TP1-TP5) with partial exits
- Fixed -15% stop-loss (always enforced)
- Volatility regime engine (4 regimes: Calm, Normal, Storm, Crash)
- 13-layer risk safeguards
- Multi-symbol support (SPY, QQQ, SPX)
- Comprehensive backtesting engine
- Real-time Streamlit dashboard
- Trade logging and database
- Full Alpaca API integration"

# Check if remote exists
if git remote | grep -q origin; then
    echo ""
    echo "🌐 Remote 'origin' already exists"
    git remote -v
    echo ""
    read -p "Push to existing remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🚀 Pushing to GitHub..."
        git push -u origin main
        echo ""
        echo "✅ Done! Your code is on GitHub"
    else
        echo ""
        echo "To push manually, run:"
        echo "  git push -u origin main"
    fi
else
    echo ""
    echo "⚠️  No remote repository configured"
    echo ""
    echo "To add a GitHub remote, run:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
    echo ""
    echo "Or create a new repo at: https://github.com/new"
fi

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="

