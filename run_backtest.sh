#!/bin/bash
# Simple wrapper to run backtest for last 3 months

echo "Running Mike Agent v3 Backtest (Last 3 Months)..."
echo ""

# Calculate dates (3 months ago to today)
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -v-3m +%Y-%m-%d 2>/dev/null || date -d "3 months ago" +%Y-%m-%d 2>/dev/null || python3 -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))")

echo "Period: $START_DATE to $END_DATE"
echo ""

# Activate venv and run
source venv/bin/activate
python3 backtest_mike_agent_v3.py --start "$START_DATE" --end "$END_DATE" --interval 1d

