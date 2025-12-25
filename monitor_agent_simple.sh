#!/bin/bash
# Simplified Real-time Agent Monitor
# Shows only key events: RL signals, selections, trades, blocks

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🔍 MIKE AGENT ACTIVITY MONITOR (Simplified)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Showing: RL Signals | Symbol Selection | Trades | Blocks"
echo "Press Ctrl+C to stop"
echo ""

# Monitor and filter for key events only
fly logs --app mike-agent-project 2>/dev/null | grep --line-buffered -E \
    "(RL Action|RL Probs|Symbol selected|SYMBOL SELECTION|TRADE_OPENED|NEW ENTRY|EXECUTED.*BUY|BLOCKED|⛔|Ensemble|STOP-LOSS|TAKE-PROFIT|SAFEGUARD.*TRIGGERED)" \
    | while IFS= read -r line; do
    # Add timestamp
    timestamp=$(date '+%H:%M:%S')
    
    # Color code by type
    if echo "$line" | grep -qiE "(TRADE_OPENED|NEW ENTRY|EXECUTED.*BUY)"; then
        echo -e "\033[0;32m[$timestamp] ✅ TRADE\033[0m $line"
    elif echo "$line" | grep -qiE "(BLOCKED|⛔)"; then
        echo -e "\033[0;31m[$timestamp] ⛔ BLOCKED\033[0m $line"
    elif echo "$line" | grep -qiE "(RL Action|RL Probs)"; then
        echo -e "\033[0;36m[$timestamp] 🤖 RL SIGNAL\033[0m $line"
    elif echo "$line" | grep -qiE "(Symbol selected|SYMBOL SELECTION)"; then
        echo -e "\033[0;35m[$timestamp] 🎯 SELECTED\033[0m $line"
    elif echo "$line" | grep -qiE "(Ensemble)"; then
        echo -e "\033[0;34m[$timestamp] 🎯 ENSEMBLE\033[0m $line"
    elif echo "$line" | grep -qiE "(STOP-LOSS|TAKE-PROFIT)"; then
        echo -e "\033[0;33m[$timestamp] 📊 EXIT\033[0m $line"
    elif echo "$line" | grep -qiE "(SAFEGUARD.*TRIGGERED)"; then
        echo -e "\033[1;31m[$timestamp] 🚨 SAFEGUARD\033[0m $line"
    else
        echo -e "[$timestamp] $line"
    fi
done





