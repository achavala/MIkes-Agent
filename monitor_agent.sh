#!/bin/bash
# Real-time Agent Activity Monitor
# Shows what the agent is checking, selecting, and executing

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🔍 MIKE AGENT REAL-TIME ACTIVITY MONITOR                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Monitoring: RL Actions | Symbol Selection | Safeguards | Trades"
echo "Press Ctrl+C to stop"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor Fly.io logs with filtering and color coding
fly logs --app mike-agent-project 2>/dev/null | while IFS= read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    # Extract timestamp if present
    timestamp=$(echo "$line" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    
    # RL INFERENCE & SIGNALS
    if echo "$line" | grep -qiE "(RL Action|RL Probs|RL Inference|🔍.*RL)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)" | head -1)
        action=$(echo "$line" | grep -oE "action=[0-9]+" | grep -oE "[0-9]+")
        strength=$(echo "$line" | grep -oE "strength=[0-9.]+" | grep -oE "[0-9.]+")
        if [ ! -z "$symbol" ] && [ ! -z "$action" ] && [ ! -z "$strength" ]; then
            action_name=""
            case "$action" in
                0) action_name="HOLD" ;;
                1) action_name="BUY CALL" ;;
                2) action_name="BUY PUT" ;;
                3) action_name="TRIM 50%" ;;
                4) action_name="TRIM 70%" ;;
                5) action_name="EXIT" ;;
            esac
            strength_pct=$(echo "$strength * 100" | bc -l | xargs printf "%.1f")
            if (( $(echo "$strength >= 0.65" | bc -l) )); then
                echo -e "${GREEN}[RL]${NC} ${CYAN}$symbol${NC} → ${GREEN}$action_name${NC} (${GREEN}${strength_pct}%${NC}) ${timestamp:+[$timestamp]}"
            else
                echo -e "${YELLOW}[RL]${NC} ${CYAN}$symbol${NC} → ${YELLOW}$action_name${NC} (${YELLOW}${strength_pct}%${NC}) ${RED}[BLOCKED: Low Confidence]${NC} ${timestamp:+[$timestamp]}"
            fi
        else
            echo -e "${CYAN}[RL]${NC} $line"
        fi
        continue
    fi
    
    # SYMBOL SELECTION
    if echo "$line" | grep -qiE "(SYMBOL SELECTION|Symbol selected|choose_best_symbol)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)" | head -1)
        action=$(echo "$line" | grep -oE "(BUY CALL|BUY PUT|CALL|PUT)" | head -1)
        if [ ! -z "$symbol" ]; then
            echo -e "${MAGENTA}[SELECT]${NC} ${CYAN}$symbol${NC} selected for ${GREEN}$action${NC} ${timestamp:+[$timestamp]}"
        else
            echo -e "${MAGENTA}[SELECT]${NC} $line"
        fi
        continue
    fi
    
    # ENSEMBLE SIGNALS
    if echo "$line" | grep -qiE "(Ensemble|🎯.*Ensemble)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)" | head -1)
        confidence=$(echo "$line" | grep -oE "confidence=[0-9.]+" | grep -oE "[0-9.]+")
        if [ ! -z "$symbol" ] && [ ! -z "$confidence" ]; then
            conf_pct=$(echo "$confidence * 100" | bc -l | xargs printf "%.1f")
            echo -e "${BLUE}[ENSEMBLE]${NC} ${CYAN}$symbol${NC} confidence: ${BLUE}${conf_pct}%${NC} ${timestamp:+[$timestamp]}"
        else
            echo -e "${BLUE}[ENSEMBLE]${NC} $line"
        fi
        continue
    fi
    
    # TRADE EXECUTION
    if echo "$line" | grep -qiE "(TRADE_OPENED|NEW ENTRY|EXECUTED.*BUY|✓ EXECUTED)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)[0-9]+[CP][0-9]+" | head -1)
        qty=$(echo "$line" | grep -oE "qty=[0-9]+" | grep -oE "[0-9]+")
        premium=$(echo "$line" | grep -oE "premium=\$[0-9.]+" | grep -oE "[0-9.]+")
        strike=$(echo "$line" | grep -oE "strike=\$[0-9.]+" | grep -oE "[0-9.]+")
        if [ ! -z "$symbol" ]; then
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${GREEN}✅ TRADE EXECUTED${NC} ${CYAN}$symbol${NC}"
            [ ! -z "$qty" ] && echo -e "   ${WHITE}Qty:${NC} $qty contracts"
            [ ! -z "$premium" ] && echo -e "   ${WHITE}Premium:${NC} \$$premium"
            [ ! -z "$strike" ] && echo -e "   ${WHITE}Strike:${NC} \$$strike"
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        else
            echo -e "${GREEN}[TRADE]${NC} $line"
        fi
        continue
    fi
    
    # TRADE BLOCKED
    if echo "$line" | grep -qiE "(BLOCKED|⛔)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)" | head -1)
        reason=$(echo "$line" | sed 's/.*BLOCKED: //' | sed 's/ |.*//')
        if [ ! -z "$symbol" ]; then
            echo -e "${RED}⛔ BLOCKED${NC} ${CYAN}$symbol${NC}: ${YELLOW}$reason${NC} ${timestamp:+[$timestamp]}"
        else
            echo -e "${RED}[BLOCK]${NC} $line"
        fi
        continue
    fi
    
    # SAFEGUARD CHECKS
    if echo "$line" | grep -qiE "(SAFEGUARD|check_safeguards|Daily loss|VIX|Max concurrent)"; then
        if echo "$line" | grep -qiE "(TRIGGERED|CRITICAL)"; then
            echo -e "${RED}🚨 SAFEGUARD${NC} $line"
        else
            echo -e "${YELLOW}[SAFEGUARD]${NC} $line"
        fi
        continue
    fi
    
    # OBSERVATION PREPARATION
    if echo "$line" | grep -qiE "(Observation|prepare_observation|🔍.*Observation)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)" | head -1)
        if [ ! -z "$symbol" ]; then
            echo -e "${CYAN}[DATA]${NC} Preparing observation for ${CYAN}$symbol${NC} ${timestamp:+[$timestamp]}"
        fi
        continue
    fi
    
    # POSITION MONITORING
    if echo "$line" | grep -qiE "(STOP-LOSS|TAKE-PROFIT|TRAILING|Position closed|EXIT)"; then
        symbol=$(echo "$line" | grep -oE "(SPY|QQQ|SPX)[0-9]+[CP][0-9]+" | head -1)
        if [ ! -z "$symbol" ]; then
            if echo "$line" | grep -qiE "(STOP-LOSS|EXIT)"; then
                echo -e "${RED}[EXIT]${NC} ${CYAN}$symbol${NC} $line"
            else
                echo -e "${GREEN}[EXIT]${NC} ${CYAN}$symbol${NC} $line"
            fi
        else
            echo -e "${YELLOW}[POSITION]${NC} $line"
        fi
        continue
    fi
    
    # MARKET DATA
    if echo "$line" | grep -qiE "(get_market_data|Insufficient data|Market closed)"; then
        echo -e "${BLUE}[MARKET]${NC} $line"
        continue
    fi
    
    # ERRORS
    if echo "$line" | grep -qiE "(ERROR|Error|Exception|Failed)"; then
        echo -e "${RED}[ERROR]${NC} $line"
        continue
    fi
    
    # WARNINGS
    if echo "$line" | grep -qiE "(WARNING|Warning|⚠️)"; then
        echo -e "${YELLOW}[WARN]${NC} $line"
        continue
    fi
    
    # INFO (less verbose, only important)
    if echo "$line" | grep -qiE "(INFO|Starting|Connected|Model loaded)"; then
        # Only show important info messages
        if echo "$line" | grep -qiE "(Starting|Connected|Model loaded|Agent started)"; then
            echo -e "${WHITE}[INFO]${NC} $line"
        fi
        continue
    fi
    
    # Default: show all other lines (can be commented out for less verbose)
    # echo "$line"
done





