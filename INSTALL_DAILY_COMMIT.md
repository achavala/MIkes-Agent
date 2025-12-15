# Daily Git Commit & Tag Automation

This script automatically commits all changes and creates a tag with date/time every day.

## Setup Instructions

### Option 1: macOS Launch Agent (Recommended)

1. **Install the launch agent:**
   ```bash
   # Copy plist to LaunchAgents directory
   cp com.mikeagent.dailycommit.plist ~/Library/LaunchAgents/
   
   # Load the agent (use bootstrap for newer macOS)
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mikeagent.dailycommit.plist
   ```

2. **Verify it's loaded:**
   ```bash
   launchctl list | grep mikeagent
   ```

3. **Check logs:**
   ```bash
   tail -f logs/daily_commit.log
   ```

4. **To uninstall:**
   ```bash
   launchctl bootout gui/$(id -u)/com.mikeagent.dailycommit
   rm ~/Library/LaunchAgents/com.mikeagent.dailycommit.plist
   ```

### Option 2: Cron Job (Alternative)

1. **Edit crontab:**
   ```bash
   crontab -e
   ```

2. **Add this line (runs at 8 PM daily):**
   ```
   0 20 * * * /Users/chavala/Mike-agent-project/daily_git_commit.sh >> /Users/chavala/Mike-agent-project/logs/daily_commit.log 2>&1
   ```

### Option 3: Manual Run

```bash
./daily_git_commit.sh
```

## Configuration

### Change Daily Commit Time

Edit `com.mikeagent.dailycommit.plist`:
- `Hour`: 20 = 8 PM (change to desired hour, 0-23)
- `Minute`: 0 = top of the hour (change to desired minute, 0-59)

Then reload:
```bash
launchctl bootout gui/$(id -u)/com.mikeagent.dailycommit
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mikeagent.dailycommit.plist
```

## Tag Format

Tags are created as: `daily-YYYY-MM-DD-HHMM`

Example: `daily-2025-12-15-2000` (Dec 15, 2025 at 8:00 PM)

## Notes

- Script only commits if there are changes
- Tag is created only if commit succeeds
- Logs are saved to `logs/daily_commit.log`
- Errors are saved to `logs/daily_commit_error.log`

