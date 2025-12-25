#!/bin/bash
# Collect paper mode logs after trading session

DATE=$(date +%Y%m%d)
LOG_DIR="paper_mode_logs_${DATE}"

echo "📂 Collecting paper mode logs for ${DATE}..."
echo ""

# Create directory
mkdir -p "$LOG_DIR"

# Collect agent logs
echo "📋 Collecting agent logs..."
if ls logs/live/agent_*.log 1> /dev/null 2>&1; then
    cp logs/live/agent_*.log "$LOG_DIR/" 2>/dev/null
    echo "   ✅ Agent logs copied"
else
    echo "   ⚠️  No agent logs found"
fi

# Collect action logs
echo "📋 Collecting action logs..."
if ls logs/live/actions_*.json 1> /dev/null 2>&1; then
    cp logs/live/actions_*.json "$LOG_DIR/" 2>/dev/null
    echo "   ✅ Action logs copied"
else
    echo "   ⚠️  No action logs found"
fi

# Collect trade logs
echo "📋 Collecting trade logs..."
if ls logs/live/trades_*.json 1> /dev/null 2>&1; then
    cp logs/live/trades_*.json "$LOG_DIR/" 2>/dev/null
    echo "   ✅ Trade logs copied"
else
    echo "   ⚠️  No trade logs found"
fi

# Create summary
echo ""
echo "📊 Creating summary..."
cat > "$LOG_DIR/SUMMARY.txt" << EOF
PAPER MODE SESSION SUMMARY
==========================
Date: ${DATE}
Collection Time: $(date)

LOGS COLLECTED:
- Agent logs: $(ls -1 "$LOG_DIR"/agent_*.log 2>/dev/null | wc -l | tr -d ' ') files
- Action logs: $(ls -1 "$LOG_DIR"/actions_*.json 2>/dev/null | wc -l | tr -d ' ') files
- Trade logs: $(ls -1 "$LOG_DIR"/trades_*.json 2>/dev/null | wc -l | tr -d ' ') files

NEXT STEPS:
1. Review logs for:
   - Top 10 profitable trades
   - Top 10 losing trades
   - Any -15% stops triggered
   - Action probability snapshots
   - BUY/HOLD patterns

2. Send logs for analysis and tuning
EOF

echo "   ✅ Summary created"

echo ""
echo "✅ Logs collected in: $LOG_DIR"
echo ""
echo "📋 Files collected:"
ls -lh "$LOG_DIR" | tail -n +2

echo ""
echo "💡 Next: Review logs and send for analysis"





