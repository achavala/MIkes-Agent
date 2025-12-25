#!/bin/bash
# Paper Mode Monitoring Script
# Monitors 6 critical health metrics for LSTM model validation

echo "📊 LSTM Paper Mode Health Monitor"
echo "=================================="
echo ""

# Find latest agent log
LATEST_LOG=$(ls -t logs/live/agent_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No agent logs found"
    echo "   Start agent first: python3 mike_agent_live_safe.py"
    exit 1
fi

echo "📁 Monitoring: $LATEST_LOG"
echo ""

# 1. HOLD Rate
echo "1️⃣  HOLD RATE (Target: < 40%)"
echo "-----------------------------------"
HOLD_COUNT=$(grep -c "action=0 (HOLD)" "$LATEST_LOG" 2>/dev/null || echo "0")
TOTAL_ACTIONS=$(grep -c "RL Inference" "$LATEST_LOG" 2>/dev/null || echo "0")
if [ "$TOTAL_ACTIONS" -gt 0 ]; then
    HOLD_PCT=$(echo "scale=1; $HOLD_COUNT * 100 / $TOTAL_ACTIONS" | bc)
    echo "   HOLD: $HOLD_COUNT / $TOTAL_ACTIONS = ${HOLD_PCT}%"
    if (( $(echo "$HOLD_PCT < 40" | bc -l) )); then
        echo "   ✅ PASS (Target: < 40%)"
    else
        echo "   ⚠️  HIGH (Target: < 40%)"
    fi
else
    echo "   ⚠️  No actions logged yet"
fi
echo ""

# 2. BUY Rate
echo "2️⃣  BUY RATE (Target: 40-60% in strong conditions)"
echo "-----------------------------------"
BUY_CALL=$(grep -c "action=1 (BUY CALL)" "$LATEST_LOG" 2>/dev/null || echo "0")
BUY_PUT=$(grep -c "action=2 (BUY PUT)" "$LATEST_LOG" 2>/dev/null || echo "0")
BUY_TOTAL=$((BUY_CALL + BUY_PUT))
if [ "$TOTAL_ACTIONS" -gt 0 ]; then
    BUY_PCT=$(echo "scale=1; $BUY_TOTAL * 100 / $TOTAL_ACTIONS" | bc)
    echo "   BUY_CALL: $BUY_CALL"
    echo "   BUY_PUT: $BUY_PUT"
    echo "   BUY Total: $BUY_TOTAL / $TOTAL_ACTIONS = ${BUY_PCT}%"
    if (( $(echo "$BUY_PCT >= 20 && $BUY_PCT <= 70" | bc -l) )); then
        echo "   ✅ PASS (Target: 20-70%)"
    elif (( $(echo "$BUY_PCT < 20" | bc -l) )); then
        echo "   ⚠️  LOW (Underfitting - Target: 20-70%)"
    else
        echo "   ⚠️  HIGH (Reckless bias - Target: 20-70%)"
    fi
else
    echo "   ⚠️  No actions logged yet"
fi
echo ""

# 3. EXIT Rate
echo "3️⃣  EXIT RATE (Target: > 30%)"
echo "-----------------------------------"
EXIT_COUNT=$(grep -c "action=5 (FULL EXIT)" "$LATEST_LOG" 2>/dev/null || echo "0")
if [ "$TOTAL_ACTIONS" -gt 0 ]; then
    EXIT_PCT=$(echo "scale=1; $EXIT_COUNT * 100 / $TOTAL_ACTIONS" | bc)
    echo "   EXIT: $EXIT_COUNT / $TOTAL_ACTIONS = ${EXIT_PCT}%"
    if (( $(echo "$EXIT_PCT > 30" | bc -l) )); then
        echo "   ✅ PASS (Target: > 30%)"
    elif (( $(echo "$EXIT_PCT > 60" | bc -l) )); then
        echo "   ⚠️  TOO HIGH (Fear behavior - Target: 30-60%)"
    else
        echo "   ⚠️  LOW (Target: > 30%)"
    fi
else
    echo "   ⚠️  No actions logged yet"
fi
echo ""

# 4. Confidence Scores
echo "4️⃣  CONFIDENCE SCORES (Target: 0.55-0.85)"
echo "-----------------------------------"
STRENGTHS=$(grep "Strength:" "$LATEST_LOG" 2>/dev/null | grep -oP "Strength: \K[0-9.]+" | head -20)
if [ -n "$STRENGTHS" ]; then
    AVG_STRENGTH=$(echo "$STRENGTHS" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    MIN_STRENGTH=$(echo "$STRENGTHS" | sort -n | head -1)
    MAX_STRENGTH=$(echo "$STRENGTHS" | sort -n | tail -1)
    echo "   Average: ${AVG_STRENGTH}"
    echo "   Range: ${MIN_STRENGTH} - ${MAX_STRENGTH}"
    if (( $(echo "$AVG_STRENGTH >= 0.55 && $AVG_STRENGTH <= 0.85" | bc -l) )); then
        echo "   ✅ PASS (Target: 0.55-0.85)"
    else
        echo "   ⚠️  OUT OF RANGE (Target: 0.55-0.85)"
    fi
else
    echo "   ⚠️  No strength scores logged yet"
fi
echo ""

# 5. Signal Timing (Trend Response)
echo "5️⃣  SIGNAL TIMING (Target: Responds within 1-3 bars)"
echo "-----------------------------------"
echo "   ⚠️  Manual check required"
echo "   Look for: Action changes within 1-3 bars of price trend shift"
echo "   Check logs for: Trend shift → Action change pattern"
echo ""

# 6. Action Distribution
echo "6️⃣  ACTION DISTRIBUTION (Target: Diverse)"
echo "-----------------------------------"
echo "   HOLD: $HOLD_COUNT"
echo "   BUY_CALL: $BUY_CALL"
echo "   BUY_PUT: $BUY_PUT"
echo "   EXIT: $EXIT_COUNT"
TRIM_50=$(grep -c "action=3 (TRIM 50%)" "$LATEST_LOG" 2>/dev/null || echo "0")
TRIM_70=$(grep -c "action=4 (TRIM 70%)" "$LATEST_LOG" 2>/dev/null || echo "0")
echo "   TRIM_50: $TRIM_50"
echo "   TRIM_70: $TRIM_70"

# Check diversity
UNIQUE_ACTIONS=0
[ "$HOLD_COUNT" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))
[ "$BUY_CALL" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))
[ "$BUY_PUT" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))
[ "$EXIT_COUNT" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))
[ "$TRIM_50" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))
[ "$TRIM_70" -gt 0 ] && UNIQUE_ACTIONS=$((UNIQUE_ACTIONS + 1))

if [ "$UNIQUE_ACTIONS" -ge 3 ]; then
    echo "   ✅ PASS (Diverse: $UNIQUE_ACTIONS unique actions)"
else
    echo "   ⚠️  LOW DIVERSITY (Only $UNIQUE_ACTIONS unique actions)"
fi
echo ""

# Summary
echo "📋 SUMMARY"
echo "=================================="
echo "Total Actions Logged: $TOTAL_ACTIONS"
echo "Latest Log: $LATEST_LOG"
echo ""
echo "💡 To watch live:"
echo "   tail -f $LATEST_LOG | grep 'RL Inference'"
echo ""
echo "💡 To check for errors:"
echo "   grep -i error $LATEST_LOG | tail -10"





