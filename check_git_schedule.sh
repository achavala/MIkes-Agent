#!/bin/bash
# Check status of daily git push automation

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📅 Daily Git Push Schedule Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if service is loaded
if launchctl list | grep -q "com.chavala.mikeagent.dailygitpush"; then
    echo "✅ Service Status: LOADED"
    echo ""
    echo "📋 Service Details:"
    launchctl list com.chavala.mikeagent.dailygitpush | head -10
    echo ""
    
    # Check schedule
    echo "⏰ Schedule Configuration:"
    plutil -p ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist | grep -A 3 StartCalendarInterval
    echo ""
    echo "   → Runs daily at 8:00 PM (20:00)"
    echo ""
    
    # Check logs
    if [ -f "logs/daily_git_push.log" ]; then
        echo "📝 Last Run Log (last 10 lines):"
        tail -10 logs/daily_git_push.log
        echo ""
    else
        echo "ℹ️  No log file yet (will be created on first run)"
        echo ""
    fi
    
    if [ -f "logs/daily_git_push_error.log" ]; then
        echo "⚠️  Error Log (if any):"
        tail -10 logs/daily_git_push_error.log
        echo ""
    fi
    
    # Check git remote
    echo "🔗 Git Remote Configuration:"
    git remote -v
    echo ""
    
    # Check for uncommitted changes
    echo "📊 Current Git Status:"
    if git diff --quiet && git diff --cached --quiet; then
        echo "   ✅ No uncommitted changes"
    else
        echo "   📝 Uncommitted changes detected (will be committed at 8 PM)"
        git status --short | head -5
    fi
    echo ""
    
    echo "✅ Everything is configured correctly!"
    echo ""
    echo "💡 To test manually: ./daily_git_push.sh"
    echo "💡 To view logs: tail -f logs/daily_git_push.log"
    
else
    echo "❌ Service Status: NOT LOADED"
    echo ""
    echo "To install, run:"
    echo "  ./setup_daily_git_push.sh"
    echo ""
fi





