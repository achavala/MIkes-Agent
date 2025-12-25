# 🚀 FLY.IO AUTOMATIC DEPLOYMENT SETUP

**Date:** December 23, 2025  
**Purpose:** Deploy changes to Fly.io immediately

---

## 🎯 QUICK START

### **Option 1: Manual Deployment (Recommended for Testing)**

```bash
./deploy_to_fly.sh
```

This will:
- Check for uncommitted changes
- Show what will be deployed
- Ask for confirmation
- Deploy to Fly.io
- Verify deployment

### **Option 2: Watch for Changes (Auto-Deploy)**

```bash
./watch_and_deploy.sh
```

This will:
- Watch for file changes
- Automatically deploy when files change
- Rate limit deployments (30 seconds minimum)

### **Option 3: Git Hook (Auto-Deploy on Commit)**

```bash
# Enable auto-deploy on commit
export FLY_AUTO_DEPLOY=true
ln -s ../../git_hook_deploy.sh .git/hooks/post-commit

# Now every commit will auto-deploy
git commit -m "Your changes"
```

---

## 📋 DEPLOYMENT METHODS

### **1. Manual Deployment Script**

**File:** `deploy_to_fly.sh`

**Usage:**
```bash
./deploy_to_fly.sh
```

**Features:**
- ✅ Checks for uncommitted changes
- ✅ Shows what will be deployed
- ✅ Asks for confirmation
- ✅ Deploys to Fly.io
- ✅ Verifies deployment

### **2. File Watcher (Auto-Deploy)**

**File:** `watch_and_deploy.sh`

**Usage:**
```bash
./watch_and_deploy.sh
```

**Features:**
- ✅ Watches for file changes
- ✅ Automatically deploys on change
- ✅ Rate limiting (30s minimum between deployments)
- ✅ Watches: `*.py`, `*.sh`, `Dockerfile`, `fly.toml`, etc.

**Requirements:**
- `fswatch` (install with: `brew install fswatch`)

### **3. Git Hook (Auto-Deploy on Commit)**

**File:** `git_hook_deploy.sh`

**Setup:**
```bash
# Enable auto-deploy
export FLY_AUTO_DEPLOY=true

# Install hook
ln -s ../../git_hook_deploy.sh .git/hooks/post-commit
```

**Usage:**
```bash
# Just commit normally
git commit -m "Your changes"
# → Automatically deploys to Fly.io
```

**Disable:**
```bash
unset FLY_AUTO_DEPLOY
# Or remove the hook
rm .git/hooks/post-commit
```

---

## 🔍 VERIFICATION

### **Check Deployment Status**

```bash
fly status --app mike-agent-project
```

### **View Deployment Logs**

```bash
fly logs --app mike-agent-project
```

### **View App**

```bash
fly open --app mike-agent-project
```

### **Check Recent Deployments**

```bash
fly releases --app mike-agent-project
```

---

## ⚙️ CONFIGURATION

### **App Name**

The script automatically detects app name from `fly.toml`:
```toml
app = 'mike-agent-project'
```

### **Files Deployed**

The following files are deployed:
- `Dockerfile` - Container build instructions
- `fly.toml` - Fly.io configuration
- All Python files (`.py`)
- All shell scripts (`.sh`)
- Configuration files
- `requirements.txt` - Python dependencies
- `start_cloud.sh` - Startup script

### **Files NOT Deployed**

Files excluded by `.dockerignore`:
- `*.log` files
- `__pycache__/` directories
- `.git/` directory
- Virtual environments
- Test files

---

## 🛠️ TROUBLESHOOTING

### **Deployment Fails**

1. **Check Fly CLI:**
   ```bash
   fly version
   ```

2. **Check Authentication:**
   ```bash
   fly auth whoami
   ```

3. **Check Build Logs:**
   ```bash
   fly logs --app mike-agent-project
   ```

4. **Check Secrets:**
   ```bash
   fly secrets list --app mike-agent-project
   ```

### **Watch Script Not Working**

1. **Install fswatch:**
   ```bash
   brew install fswatch
   ```

2. **Check Permissions:**
   ```bash
   chmod +x watch_and_deploy.sh
   ```

### **Git Hook Not Working**

1. **Check Hook is Installed:**
   ```bash
   ls -la .git/hooks/post-commit
   ```

2. **Check Environment Variable:**
   ```bash
   echo $FLY_AUTO_DEPLOY
   ```

3. **Test Hook Manually:**
   ```bash
   .git/hooks/post-commit
   ```

---

## 📝 BEST PRACTICES

### **1. Test Locally First**

Before deploying, test changes locally:
```bash
# Test Python code
python3 -m pytest

# Test agent startup
python3 mike_agent_live_safe.py --help
```

### **2. Commit Changes**

Always commit changes before deploying:
```bash
git add .
git commit -m "Description of changes"
./deploy_to_fly.sh
```

### **3. Monitor Deployment**

Watch logs during deployment:
```bash
# In one terminal
./deploy_to_fly.sh

# In another terminal
fly logs --app mike-agent-project
```

### **4. Verify After Deployment**

After deployment, verify it's working:
```bash
fly status --app mike-agent-project
fly logs --app mike-agent-project | tail -50
```

---

## 🎯 RECOMMENDED WORKFLOW

### **For Development:**

1. **Make changes locally**
2. **Test locally**
3. **Commit changes:**
   ```bash
   git add .
   git commit -m "Your changes"
   ```
4. **Deploy:**
   ```bash
   ./deploy_to_fly.sh
   ```
5. **Verify:**
   ```bash
   fly logs --app mike-agent-project
   ```

### **For Continuous Development:**

1. **Start file watcher:**
   ```bash
   ./watch_and_deploy.sh
   ```
2. **Make changes** - automatically deploys
3. **Monitor logs** in another terminal

---

## 📦 FILES CREATED

1. **`deploy_to_fly.sh`** - Manual deployment script
2. **`watch_and_deploy.sh`** - File watcher for auto-deploy
3. **`git_hook_deploy.sh`** - Git hook for auto-deploy on commit
4. **`FLY_IO_AUTO_DEPLOY_SETUP.md`** - This documentation

---

## ✅ SUMMARY

**Three ways to deploy:**

1. **Manual:** `./deploy_to_fly.sh` (recommended for testing)
2. **Watch:** `./watch_and_deploy.sh` (auto-deploy on file changes)
3. **Git Hook:** Install hook, commit normally (auto-deploy on commit)

**All methods:**
- ✅ Deploy immediately to Fly.io
- ✅ Verify deployment
- ✅ Show deployment status
- ✅ Handle errors gracefully

---

**Last Updated:** December 23, 2025


