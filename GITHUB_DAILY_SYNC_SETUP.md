# 📅 **Daily GitHub Sync Setup**

**Date**: 2025-12-12  
**Status**: ✅ **Automation Scripts Created**

---

## 🎯 **OVERVIEW**

This setup automatically commits and pushes your project changes to GitHub once per day at 8:00 PM.

---

## 📋 **FILES CREATED**

### **1. `daily_git_push.sh`**
- Main script that performs git add, commit, and push
- Handles conflicts gracefully
- Shows status and progress
- Respects `.gitignore` rules

### **2. `com.chavala.mikeagent.dailygitpush.plist`**
- macOS launchd configuration file
- Schedules daily execution at 8:00 PM
- Logs output to `logs/daily_git_push.log`

### **3. `setup_daily_git_push.sh`**
- Installation script for macOS
- Sets up launchd service
- Makes scripts executable

---

## 🚀 **SETUP INSTRUCTIONS**

### **Step 1: Ensure Git is Configured**

```bash
cd /Users/chavala/Mike-agent-project

# Check if git remote is set
git remote -v

# If not set, add your GitHub remote:
# git remote add origin https://github.com/yourusername/your-repo.git
```

### **Step 2: Run Setup Script**

```bash
./setup_daily_git_push.sh
```

This will:
- Install the launchd service
- Schedule daily runs at 8:00 PM
- Create log files

### **Step 3: Verify Installation**

```bash
# Check if service is loaded
launchctl list | grep mikeagent

# Check logs
tail -f logs/daily_git_push.log
```

---

## 🔧 **MANUAL USAGE**

You can also run the script manually anytime:

```bash
./daily_git_push.sh
```

---

## ⚙️ **CUSTOMIZATION**

### **Change Schedule Time**

Edit `com.chavala.mikeagent.dailygitpush.plist`:

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>20</integer>  <!-- Change this (0-23) -->
    <key>Minute</key>
    <integer>0</integer>   <!-- Change this (0-59) -->
</dict>
```

Then reload:
```bash
launchctl unload ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
launchctl load ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
```

### **Change Commit Message**

Edit `daily_git_push.sh`:

```bash
COMMIT_MSG="Your custom message: $(date '+%Y-%m-%d %H:%M:%S')"
```

---

## 📊 **MONITORING**

### **Check Service Status**

```bash
launchctl list | grep mikeagent
```

### **View Logs**

```bash
# Success logs
tail -f logs/daily_git_push.log

# Error logs
tail -f logs/daily_git_push_error.log
```

### **Test Run**

```bash
# Run manually to test
./daily_git_push.sh
```

---

## 🛠 **TROUBLESHOOTING**

### **Service Not Running**

```bash
# Reload service
launchctl unload ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
launchctl load ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
```

### **Git Credentials**

If you get authentication errors:

```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:yourusername/your-repo.git

# Or configure GitHub CLI
gh auth login
```

### **Permission Errors**

```bash
# Make scripts executable
chmod +x daily_git_push.sh
chmod +x setup_daily_git_push.sh
```

---

## 🗑️ **UNINSTALL**

To remove the daily automation:

```bash
launchctl unload ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
rm ~/Library/LaunchAgents/com.chavala.mikeagent.dailygitpush.plist
```

---

## ✅ **WHAT GETS COMMITTED**

The script respects your `.gitignore` file, so it will:
- ✅ Commit code changes
- ✅ Commit documentation
- ✅ Commit configuration (non-sensitive)
- ❌ Skip `.env` files
- ❌ Skip `logs/` directory
- ❌ Skip other ignored files

---

## 🔒 **SECURITY NOTES**

- The script will **NOT** commit sensitive files (they're in `.gitignore`)
- Make sure `.env` and API keys are in `.gitignore`
- Review what gets committed before first run
- Use SSH keys or GitHub CLI for authentication

---

## 📝 **EXAMPLE OUTPUT**

```
🔄 Starting daily GitHub sync...
📊 Current status:
 M mike_agent_live_safe.py
 A new_feature.py
➕ Staging changes...
💾 Committing changes...
⬇️  Pulling latest changes from GitHub...
⬆️  Pushing to GitHub...
✅ Daily sync completed successfully!
📅 Last sync: Thu Dec 12 20:00:00 EST 2025
```

---

**Last Updated**: 2025-12-12





