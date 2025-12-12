#!/bin/bash
#
# Prevent Mac from Sleeping During Training
#
# This script prevents your Mac from sleeping while training runs.
# Use it before starting training, and restore normal sleep after training.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.caffeinate.pid"
LOCK_FILE="${SCRIPT_DIR}/.training.lock"

function start_prevent_sleep() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "⚠️  caffeinate is already running (PID: $OLD_PID)"
            echo "   To stop it, run: ./prevent_sleep.sh stop"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    echo "☕ Starting caffeinate to prevent sleep..."
    echo "   Your Mac will stay awake during training."
    echo ""
    
    # Start caffeinate in the background
    # -d: Prevent display from sleeping
    # -i: Prevent system from idle sleeping
    # -m: Prevent disk from idle sleeping
    # -s: Prevent system from sleeping (even on AC power)
    # -u: Prevent system from sleeping when lid is closed (CRITICAL for MacBook)
    caffeinate -d -i -m -s -u &
    CAFFEINATE_PID=$!
    
    echo "$CAFFEINATE_PID" > "$PID_FILE"
    echo "✅ caffeinate started (PID: $CAFFEINATE_PID)"
    echo ""
    echo "📝 To stop caffeinate, run:"
    echo "   ./prevent_sleep.sh stop"
    echo ""
    
    # Create lock file
    touch "$LOCK_FILE"
}

function stop_prevent_sleep() {
    if [ ! -f "$PID_FILE" ]; then
        echo "ℹ️  caffeinate is not running (no PID file found)"
        return 0
    fi
    
    CAFFEINATE_PID=$(cat "$PID_FILE")
    
    if ps -p "$CAFFEINATE_PID" > /dev/null 2>&1; then
        echo "🛑 Stopping caffeinate (PID: $CAFFEINATE_PII)..."
        kill "$CAFFEINATE_PID"
        rm -f "$PID_FILE"
        rm -f "$LOCK_FILE"
        echo "✅ caffeinate stopped. Mac will now sleep normally."
    else
        echo "⚠️  caffeinate process not found (PID: $CAFFEINATE_PID)"
        rm -f "$PID_FILE"
        rm -f "$LOCK_FILE"
    fi
}

function status() {
    if [ -f "$PID_FILE" ]; then
        CAFFEINATE_PID=$(cat "$PID_FILE")
        if ps -p "$CAFFEINATE_PID" > /dev/null 2>&1; then
            echo "✅ caffeinate is RUNNING (PID: $CAFFEINATE_PID)"
            echo "   Your Mac will not sleep."
            return 0
        else
            echo "❌ caffeinate is NOT running (stale PID file)"
            rm -f "$PID_FILE"
            rm -f "$LOCK_FILE"
            return 1
        fi
    else
        echo "❌ caffeinate is NOT running"
        return 1
    fi
}

function check_system_preferences() {
    echo "🔧 System Preferences Check:"
    echo ""
    echo "To maximize training efficiency, also check:"
    echo ""
    echo "1. Energy Saver Settings:"
    echo "   System Preferences → Battery → Options:"
    echo "   • Set 'Turn display off after' to 'Never'"
    echo "   • Uncheck 'Prevent automatic sleeping on power adapter when display is off'"
    echo ""
    echo "2. Automatic Graphics Switching:"
    echo "   System Preferences → Battery → Options:"
    echo "   • Uncheck 'Automatic graphics switching' (if available)"
    echo ""
    echo "3. Power Nap:"
    echo "   System Preferences → Battery → Options:"
    echo "   • Uncheck 'Enable Power Nap' (if available)"
    echo ""
}

case "$1" in
    start)
        start_prevent_sleep
        ;;
    stop)
        stop_prevent_sleep
        ;;
    status)
        status
        ;;
    check)
        check_system_preferences
        ;;
    *)
        echo "Usage: $0 {start|stop|status|check}"
        echo ""
        echo "Commands:"
        echo "  start  - Start preventing sleep (run before training)"
        echo "  stop   - Stop preventing sleep (run after training)"
        echo "  status - Check if caffeinate is running"
        echo "  check  - Show system preferences recommendations"
        exit 1
        ;;
esac

