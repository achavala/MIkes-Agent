# 🚀 QUICK DEPLOY TO FLY.IO

**Three ways to deploy changes immediately:**

---

## 1️⃣ MANUAL DEPLOYMENT (Recommended)

```bash
./deploy_to_fly.sh
```

**What it does:**
- ✅ Checks for uncommitted changes
- ✅ Shows what will be deployed
- ✅ Asks for confirmation
- ✅ Deploys to Fly.io
- ✅ Verifies deployment

---

## 2️⃣ AUTO-DEPLOY ON FILE CHANGES

```bash
./watch_and_deploy.sh
```

**What it does:**
- ✅ Watches for file changes
- ✅ Automatically deploys when files change
- ✅ Rate limiting (30s minimum between deployments)

**Requirements:**
```bash
brew install fswatch  # macOS only
```

---

## 3️⃣ AUTO-DEPLOY ON GIT COMMIT

```bash
# Setup (one time)
export FLY_AUTO_DEPLOY=true
ln -s ../../git_hook_deploy.sh .git/hooks/post-commit

# Now every commit auto-deploys
git commit -m "Your changes"
```

**What it does:**
- ✅ Automatically deploys on every commit
- ✅ Only if `FLY_AUTO_DEPLOY=true` is set

---

## ✅ VERIFY DEPLOYMENT

```bash
# Check status
fly status --app mike-agent-project

# View logs
fly logs --app mike-agent-project

# View app
fly open --app mike-agent-project
```

---

## 🎯 RECOMMENDED WORKFLOW

1. **Make changes**
2. **Test locally**
3. **Deploy:**
   ```bash
   ./deploy_to_fly.sh
   ```
4. **Verify:**
   ```bash
   fly logs --app mike-agent-project | tail -50
   ```

---

**That's it! Your changes are now live on Fly.io!**
