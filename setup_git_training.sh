#!/bin/bash
#
# Setup Git for Training - Push training files to GitHub
#
# Run this on M2 Mac to prepare repository for GitHub-based training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=" | tr '=' '═'
echo "🔄 SETTING UP GIT FOR TRAINING"
echo "=" | tr '=' '═'
echo ""

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository!"
    echo ""
    echo "Initialize git repository first:"
    echo "  git init"
    echo "  git remote add origin YOUR_GITHUB_REPO_URL"
    echo "  git add ."
    echo "  git commit -m 'Initial commit'"
    echo "  git push -u origin main"
    exit 1
fi

# Check remote
REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE" ]; then
    echo "⚠️  No remote repository configured!"
    echo ""
    echo "Add remote repository:"
    echo "  git remote add origin YOUR_GITHUB_REPO_URL"
    exit 1
fi

echo "✅ Git repository found"
echo "   Remote: $REMOTE"
echo ""

# Check .gitignore
echo "📋 Checking .gitignore..."

if ! grep -q "data/historical/enriched" .gitignore 2>/dev/null; then
    echo "   ⚠️  Adding data files to .gitignore..."
    cat >> .gitignore << 'EOF'

# Training data files (too large for GitHub)
data/historical/enriched/*.pkl
*.pkl

# But track training outputs
!models/*.zip
!training*.log
EOF
    echo "   ✅ Updated .gitignore"
else
    echo "   ✅ .gitignore already configured"
fi

# Show files to be committed
echo ""
echo "📁 Files ready to commit:"
echo ""

# Training scripts
TRAINING_FILES=(
    "historical_training_system.py"
    "train_historical_model.py"
    "quant_features_collector.py"
    "institutional_features.py"
    "greeks_calculator.py"
    "latency_monitor.py"
    "prevent_sleep.sh"
    "start_training.sh"
    "comprehensive_pre_training_validation.py"
    "TRAINING_TIME_ESTIMATE.py"
)

echo "Training Scripts:"
for file in "${TRAINING_FILES[@]}"; do
    if [ -f "$file" ]; then
        STATUS=$(git status --porcelain "$file" 2>/dev/null | cut -c1-2 || echo "  ")
        if [[ "$STATUS" == "??" ]] || [[ "$STATUS" == " M" ]] || [[ "$STATUS" == "A " ]]; then
            echo "   ✅ $file (new/modified)"
        else
            echo "   ✓  $file (already tracked)"
        fi
    fi
done

# Documentation
echo ""
echo "Documentation:"
DOC_FILES=(
    "COMPLETE_TRAINING_PLAN.md"
    "PARALLEL_TRAINING_SETUP.md"
    "GITHUB_TRAINING_SETUP.md"
    "QUICK_PARALLEL_SETUP.md"
    "FINAL_PRE_TRAINING_VALIDATION_REPORT.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        STATUS=$(git status --porcelain "$file" 2>/dev/null | cut -c1-2 || echo "  ")
        if [[ "$STATUS" == "??" ]] || [[ "$STATUS" == " M" ]] || [[ "$STATUS" == "A " ]]; then
            echo "   ✅ $file (new/modified)"
        else
            echo "   ✓  $file (already tracked)"
        fi
    fi
done

# Check current branch
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo ""
echo "Current branch: $BRANCH"
echo ""

# Ask for confirmation
read -p "Do you want to add, commit, and push these files? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Add files
echo ""
echo "📦 Adding files to git..."

# Add all training-related files
git add historical_training_system.py train_historical_model.py 2>/dev/null || true
git add quant_features_collector.py institutional_features.py 2>/dev/null || true
git add greeks_calculator.py latency_monitor.py 2>/dev/null || true
git add prevent_sleep.sh start_training.sh 2>/dev/null || true
git add comprehensive_pre_training_validation.py TRAINING_TIME_ESTIMATE.py 2>/dev/null || true
git add commit_training_progress.sh setup_git_training.sh 2>/dev/null || true

# Add all new training scripts
git add collect_historical_data.py collect_quant_features.py 2>/dev/null || true
git add historical_data_collector.py 2>/dev/null || true
git add validate_quant_features.py validate_training_status.py 2>/dev/null || true
git add validate_institutional_upgrade.py 2>/dev/null || true
git add run_data_collection.sh 2>/dev/null || true

# Add documentation files
git add *.md 2>/dev/null || true

# Add .gitignore
git add .gitignore 2>/dev/null || true

# Add modified files if any
if git diff --name-only | grep -q "."; then
    echo "   Adding modified files..."
    git add -u
fi

# Check what will be committed
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
if [ "$STAGED" -gt 0 ]; then
    echo "   ✅ $STAGED files staged for commit"
else
    echo "   ⚠️  No new files to add (they may already be committed)"
fi

# Commit
echo "💾 Committing changes..."
git commit -m "Add complete training system for GitHub-based parallel training

- Training scripts and infrastructure
- Comprehensive validation and planning docs
- GitHub-based sync setup
- Ready for desktop Mac training setup" || echo "⚠️  No changes to commit"

# Push
echo "🚀 Pushing to GitHub..."
if git push origin "$BRANCH" 2>/dev/null; then
    echo ""
    echo "=" | tr '=' '═'
    echo "✅ SUCCESS!"
    echo "=" | tr '=' '═'
    echo ""
    echo "Repository is ready for GitHub-based training!"
    echo ""
    echo "Next steps on Desktop Mac:"
    echo ""
    echo "1. Clone repository:"
    echo "   git clone $REMOTE ~/Desktop/mike-agent-training"
    echo ""
    echo "2. Install Cursor (if needed):"
    echo "   Download from: https://cursor.sh"
    echo ""
    echo "3. Setup environment:"
    echo "   cd ~/Desktop/mike-agent-training"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo "   pip install 'stable-baselines3[extra]' gymnasium"
    echo ""
    echo "4. Download data files (see GITHUB_TRAINING_SETUP.md)"
    echo ""
    echo "5. Open in Cursor:"
    echo "   cursor ."
    echo ""
    echo "6. Start training:"
    echo "   ./prevent_sleep.sh start"
    echo "   ./start_training.sh"
    echo ""
else
    echo ""
    echo "⚠️  Push failed. Check:"
    echo "   - GitHub credentials"
    echo "   - Remote URL: $REMOTE"
    echo "   - Branch: $BRANCH"
    echo ""
    echo "You can push manually:"
    echo "   git push origin $BRANCH"
fi

