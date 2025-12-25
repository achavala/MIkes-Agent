#!/bin/bash
# Setup script for daily GitHub push automation
# This installs a launchd service on macOS to run daily at 8 PM

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLIST_NAME="com.chavala.mikeagent.dailygitpush.plist"
PLIST_FILE="$SCRIPT_DIR/$PLIST_NAME"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
LAUNCHD_FILE="$LAUNCHD_DIR/$PLIST_NAME"

echo "🔧 Setting up daily GitHub push automation..."

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Make sure the push script is executable
chmod +x "$SCRIPT_DIR/daily_git_push.sh"

# Copy plist to LaunchAgents directory
echo "📋 Installing launchd service..."
mkdir -p "$LAUNCHD_DIR"
cp "$PLIST_FILE" "$LAUNCHD_FILE"

# Load the service
echo "🚀 Loading launchd service..."
launchctl unload "$LAUNCHD_FILE" 2>/dev/null || true  # Unload if already exists
launchctl load "$LAUNCHD_FILE"

echo "✅ Daily GitHub push automation installed!"
echo ""
echo "📅 Schedule: Daily at 8:00 PM"
echo "📝 Logs: $SCRIPT_DIR/logs/daily_git_push.log"
echo ""
echo "To check status:"
echo "  launchctl list | grep mikeagent"
echo ""
echo "To uninstall:"
echo "  launchctl unload $LAUNCHD_FILE"
echo "  rm $LAUNCHD_FILE"





