#!/bin/bash
# Daily GitHub Push Script
# Automatically commits and pushes changes to GitHub once per day

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}🔄 Starting daily GitHub sync...${NC}"

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${RED}❌ Error: Not a git repository. Run 'git init' first.${NC}"
    exit 1
fi

# Check if remote is configured
if ! git remote | grep -q origin; then
    echo -e "${YELLOW}⚠️  Warning: No 'origin' remote configured.${NC}"
    echo "   To add: git remote add origin <your-github-url>"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${GREEN} branch: ${CURRENT_BRANCH}${NC}"

# Check if there are any changes
if git diff --quiet && git diff --cached --quiet; then
    echo -e "${YELLOW}ℹ️  No changes to commit.${NC}"
    
    # Still try to pull latest changes
    echo -e "${GREEN}⬇️  Pulling latest changes from GitHub...${NC}"
    git pull origin "$CURRENT_BRANCH" || echo -e "${YELLOW}⚠️  Pull failed (may be expected if no remote changes)${NC}"
    exit 0
fi

# Show status
echo -e "${GREEN}📊 Current status:${NC}"
git status --short

# Add all changes (respects .gitignore)
echo -e "${GREEN}➕ Staging changes...${NC}"
git add -A

# Create commit with timestamp
COMMIT_MSG="Daily sync: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${GREEN}💾 Committing changes...${NC}"
git commit -m "$COMMIT_MSG" || {
    echo -e "${YELLOW}⚠️  Nothing to commit (all changes may be ignored)${NC}"
    exit 0
}

# Pull latest changes first (to avoid conflicts)
echo -e "${GREEN}⬇️  Pulling latest changes from GitHub...${NC}"
git pull origin "$CURRENT_BRANCH" --no-rebase || {
    echo -e "${YELLOW}⚠️  Pull had conflicts or failed. Please resolve manually.${NC}"
    exit 1
}

# Push to GitHub
echo -e "${GREEN}⬆️  Pushing to GitHub...${NC}"
git push origin "$CURRENT_BRANCH" || {
    echo -e "${RED}❌ Push failed. Please check your GitHub credentials and network.${NC}"
    exit 1
}

echo -e "${GREEN}✅ Daily sync completed successfully!${NC}"
echo -e "${GREEN}📅 Last sync: $(date)${NC}"





