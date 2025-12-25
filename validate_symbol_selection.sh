#!/bin/bash
# Quick validation script for symbol selection at market open

echo "================================================================================"
echo "SYMBOL SELECTION VALIDATION - Market Open Check"
echo "================================================================================"
echo ""

LOG_FILE="logs/agent_*.log"

echo "1️⃣  RL INFERENCE WITH STRENGTH VALUES"
echo "────────────────────────────────────────────────────────────────────────────────"
grep "RL Inference.*Strength" $LOG_FILE 2>/dev/null | tail -30 | head -15
echo ""

echo "2️⃣  SYMBOL SELECTION WITH CANDIDATES"
echo "────────────────────────────────────────────────────────────────────────────────"
grep "Symbol selected" $LOG_FILE 2>/dev/null | tail -20
echo ""

echo "3️⃣  POSITION FILTERING (Should block when positions exist)"
echo "────────────────────────────────────────────────────────────────────────────────"
grep "No eligible symbols" $LOG_FILE 2>/dev/null | tail -10
echo ""

echo "4️⃣  TRADES EXECUTED (Should see SPY, QQQ, AND SPX)"
echo "────────────────────────────────────────────────────────────────────────────────"
grep "TRADE EXECUTED" $LOG_FILE 2>/dev/null | tail -20
echo ""

echo "5️⃣  QQQ TRADING CHECK"
echo "────────────────────────────────────────────────────────────────────────────────"
QQQ_TRADES=$(grep "TRADE EXECUTED.*QQQ" $LOG_FILE 2>/dev/null | wc -l | tr -d ' ')
if [ "$QQQ_TRADES" -gt 0 ]; then
    echo "✅ QQQ IS TRADING! ($QQQ_TRADES trades found)"
    grep "TRADE EXECUTED.*QQQ" $LOG_FILE 2>/dev/null | tail -5
else
    echo "❌ QQQ NOT TRADING YET (0 trades found)"
fi
echo ""

echo "6️⃣  SPX TRADING CHECK"
echo "────────────────────────────────────────────────────────────────────────────────"
SPX_TRADES=$(grep "TRADE EXECUTED.*SPX" $LOG_FILE 2>/dev/null | wc -l | tr -d ' ')
if [ "$SPX_TRADES" -gt 0 ]; then
    echo "✅ SPX IS TRADING! ($SPX_TRADES trades found)"
    grep "TRADE EXECUTED.*SPX" $LOG_FILE 2>/dev/null | tail -5
else
    echo "❌ SPX NOT TRADING YET (0 trades found)"
fi
echo ""

echo "7️⃣  STRENGTH-BASED SELECTION CHECK"
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Looking for cases where strongest signal was picked..."
grep "Symbol selected.*strength=" $LOG_FILE 2>/dev/null | tail -10
echo ""

echo "================================================================================"
echo "VALIDATION SUMMARY"
echo "================================================================================"

SPY_TRADES=$(grep "TRADE EXECUTED.*SPY" $LOG_FILE 2>/dev/null | wc -l | tr -d ' ')

echo "SPY Trades: $SPY_TRADES"
echo "QQQ Trades: $QQQ_TRADES"
echo "SPX Trades: $SPX_TRADES"
echo ""

if [ "$QQQ_TRADES" -gt 0 ] && [ "$SPX_TRADES" -gt 0 ]; then
    echo "✅ SUCCESS! All three symbols are trading!"
    echo "   Symbol selection upgrade is working correctly."
elif [ "$QQQ_TRADES" -gt 0 ] || [ "$SPX_TRADES" -gt 0 ]; then
    echo "⚠️  PARTIAL: Some symbols trading, continue monitoring"
else
    echo "⏰ WAITING: No QQQ/SPX trades yet (may need more time)"
fi

echo ""
echo "Run this script periodically during trading hours:"
echo "  bash validate_symbol_selection.sh"
echo ""
echo "================================================================================"





