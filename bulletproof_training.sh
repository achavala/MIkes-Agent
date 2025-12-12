#!/bin/bash
#
# 🛡️ BULLETPROOF TRAINING SCRIPT
# 
# Ensures training continues even if:
# - Internet disconnects
# - Lid closes
# - System tries to sleep
# - Process gets interrupted
#
# This script makes training completely resilient to interruptions.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
SYMBOLS="SPY,QQQ,SPX"
START_DATE="2002-01-01"
TIMESTEPS=5000000
# Note: save_freq is hardcoded to 50000 in train_historical_model.py (CheckpointCallback)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"
PID_FILE=".training.pid"
CAFFEINATE_PID_FILE=".caffeinate.pid"
LOCK_FILE=".training.lock"
MAX_RESTARTS=10
RESTART_COUNT_FILE=".restart_count"

echo "═══════════════════════════════════════════════════════════"
echo "🛡️  BULLETPROOF TRAINING LAUNCHER"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Function to prevent sleep (bulletproof)
prevent_sleep_bulletproof() {
    echo "☕ Setting up bulletproof sleep prevention..."
    
    # Kill any existing caffeinate
    if [ -f "$CAFFEINATE_PID_FILE" ]; then
        OLD_PID=$(cat "$CAFFEINATE_PID_FILE" 2>/dev/null || echo "")
        if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill "$OLD_PID" 2>/dev/null || true
        fi
        rm -f "$CAFFEINATE_PID_FILE"
    fi
    
    # Start caffeinate with ALL flags for maximum protection
    # -d: Prevent display from sleeping
    # -i: Prevent system from idle sleeping
    # -m: Prevent disk from idle sleeping
    # -s: Prevent system from sleeping (even on AC power)
    # -u: Prevent system from sleeping when lid is closed (CRITICAL)
    # -w: Wait for process (we'll use nohup, so this ensures it stays active)
    nohup caffeinate -d -i -m -s -u > /dev/null 2>&1 &
    CAFFEINATE_PID=$!
    echo "$CAFFEINATE_PID" > "$CAFFEINATE_PID_FILE"
    
    # Also set system-level sleep prevention (optional, requires sudo)
    # These are optional - caffeinate should be sufficient
    if sudo -n true 2>/dev/null; then
        # Sudo available without password prompt
        sudo pmset -a sleep 0 2>/dev/null && echo "   ✅ System sleep disabled"
        sudo pmset -a disablesleep 1 2>/dev/null && echo "   ✅ Idle sleep disabled"
    else
        # No sudo or requires password - that's OK, caffeinate is sufficient
        echo "   ℹ️  System-level settings skipped (sudo not available)"
        echo "   ✅ Caffeinate is active (sufficient for lid-closed operation)"
    fi
    
    echo "   ✅ Sleep prevention active (PID: $CAFFEINATE_PID)"
}

# Function to check if training is running
is_training_running() {
    if [ -f "$PID_FILE" ]; then
        TRAINING_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [ -n "$TRAINING_PID" ] && ps -p "$TRAINING_PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to start training
start_training() {
    echo "🚀 Starting training process..."
    
    # Activate virtual environment
    if [ ! -d "venv" ]; then
        echo "❌ Virtual environment not found!"
        exit 1
    fi
    source venv/bin/activate
    
    # Verify data files exist (local files - no internet needed)
    echo "📊 Verifying local data files..."
    DATA_FILES=(
        "data/historical/enriched/SPY_enriched_2002-01-01_latest.pkl"
        "data/historical/enriched/QQQ_enriched_2002-01-01_latest.pkl"
        "data/historical/enriched/SPX_enriched_2002-01-01_latest.pkl"
    )
    
    for file in "${DATA_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "❌ Missing data file: $file"
            echo "   Training requires local data files (no internet needed during training)"
            exit 1
        fi
    done
    echo "   ✅ All data files found (training will work offline)"
    
    # Check for existing checkpoint to resume
    LATEST_CHECKPOINT=$(ls -t models/mike_rl_model_*.zip 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "   📁 Found checkpoint: $LATEST_CHECKPOINT"
        echo "   ℹ️  Training will continue from checkpoint if available"
    fi
    
    # Start training with nohup (survives terminal closure)
    # Use nohup to ensure process continues even if terminal closes
    # Note: CheckpointCallback uses save_freq=50000 internally (hardcoded in script)
    nohup python train_historical_model.py \
        --symbols "$SYMBOLS" \
        --start-date "$START_DATE" \
        --timesteps "$TIMESTEPS" \
        --use-greeks \
        --regime-balanced \
        >> "$LOG_FILE" 2>&1 &
    
    TRAINING_PID=$!
    echo "$TRAINING_PID" > "$PID_FILE"
    touch "$LOCK_FILE"
    
    echo "   ✅ Training started (PID: $TRAINING_PID)"
    echo "   📝 Log: $LOG_FILE"
    
    # Wait longer to ensure process started and check for immediate errors
    sleep 5
    
    # Check if process is still running
    if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "   ✅ Training process confirmed running"
        
        # Check for immediate errors in log (python syntax errors, import errors, etc.)
        if [ -f "$LOG_FILE" ]; then
            ERROR_LINES=$(tail -20 "$LOG_FILE" | grep -i "error\|exception\|traceback\|failed" || true)
            if [ -n "$ERROR_LINES" ]; then
                echo "   ⚠️  Warning: Errors detected in log:"
                echo "$ERROR_LINES" | head -5 | sed 's/^/      /'
                echo "   💡 Check full log: tail -50 $LOG_FILE"
            fi
        fi
        
        return 0
    else
        echo "   ❌ Training process failed to start"
        
        # Show error from log file
        if [ -f "$LOG_FILE" ]; then
            echo "   📋 Last 10 lines from log:"
            tail -10 "$LOG_FILE" | sed 's/^/      /'
        fi
        
        # Clean up PID file
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function to monitor and auto-restart
monitor_training() {
    echo ""
    echo "🔄 Starting monitoring loop (auto-restart enabled)..."
    echo "   Training will auto-restart if it stops (max $MAX_RESTARTS times)"
    echo ""
    
    RESTART_COUNT=0
    if [ -f "$RESTART_COUNT_FILE" ]; then
        RESTART_COUNT=$(cat "$RESTART_COUNT_FILE" 2>/dev/null || echo "0")
    fi
    
    while true; do
        sleep 60  # Check every minute
        
        # Check if training is running
        if ! is_training_running; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Training process stopped!"
            
            if [ "$RESTART_COUNT" -lt "$MAX_RESTARTS" ]; then
                RESTART_COUNT=$((RESTART_COUNT + 1))
                echo "$RESTART_COUNT" > "$RESTART_COUNT_FILE"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Restarting training (attempt $RESTART_COUNT/$MAX_RESTARTS)..."
                
                # Restart training
                if start_training; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Training restarted successfully"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed to restart training"
                fi
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Max restart attempts reached ($MAX_RESTARTS)"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛑 Stopping monitoring"
                break
            fi
        fi
        
        # Check if caffeinate is running
        if [ -f "$CAFFEINATE_PID_FILE" ]; then
            CAFFEINATE_PID=$(cat "$CAFFEINATE_PID_FILE" 2>/dev/null || echo "")
            if [ -z "$CAFFEINATE_PID" ] || ! ps -p "$CAFFEINATE_PID" > /dev/null 2>&1; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Caffeinate stopped, restarting..."
                prevent_sleep_bulletproof
            fi
        fi
    done
}

# Main execution
main() {
    # Check power status
    if ! pmset -g batt | grep -q "AC Power"; then
        echo "⚠️  WARNING: MacBook is on battery!"
        echo "   Training may stop when lid closes if not plugged in."
        echo "   For bulletproof operation, plug in power adapter."
        echo ""
        read -p "   Continue anyway? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   Exiting. Please plug in power adapter first."
            exit 1
        fi
    else
        echo "✅ MacBook is plugged into power (required for lid-closed)"
    fi
    
    # Check if training already running
    if is_training_running; then
        TRAINING_PID=$(cat "$PID_FILE")
        echo "⚠️  Training is already running (PID: $TRAINING_PID)"
        echo ""
        echo "   To monitor: tail -f training_*.log"
        echo "   To stop: kill $TRAINING_PID && rm $PID_FILE"
        exit 1
    fi
    
    # Prevent sleep
    prevent_sleep_bulletproof
    
    # Start training
    if start_training; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "✅ TRAINING STARTED SUCCESSFULLY"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        echo "📊 Process ID: $(cat $PID_FILE)"
        echo "📝 Log file: $LOG_FILE"
        echo ""
        echo "🛡️  BULLETPROOF FEATURES ACTIVE:"
        echo "   ✅ Sleep prevention (lid-closed compatible)"
        echo "   ✅ Offline operation (no internet required)"
        echo "   ✅ Auto-restart on failure"
        echo "   ✅ Process survives terminal closure"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Wait 2-3 minutes"
        echo "   2. Check log: tail -20 $LOG_FILE"
        echo "   3. Close laptop lid - training will continue!"
        echo "   4. Training works offline (no internet needed)"
        echo ""
        echo "📊 Monitor training:"
        echo "   tail -f $LOG_FILE"
        echo ""
        echo "🔍 Check status:"
        echo "   ./check_training_status.sh"
        echo ""
        echo "🛑 Stop training:"
        echo "   kill \$(cat $PID_FILE)"
        echo "   ./prevent_sleep.sh stop"
        echo ""
        
        # Start monitoring in background
        monitor_training &
        MONITOR_PID=$!
        echo "$MONITOR_PID" > .monitor.pid
        echo "   🔄 Monitoring started (PID: $MONITOR_PID)"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
    else
        echo "❌ Failed to start training"
        exit 1
    fi
}

# Run main function
main

