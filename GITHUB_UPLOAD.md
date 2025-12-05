# Uploading Mike Agent to GitHub

## ⚠️ IMPORTANT: Before Uploading

### 1. Check for Sensitive Information
The following files contain API keys and should NOT be committed:
- `config.py` - Contains your Alpaca API keys
- `*.csv` - Trade history files may contain sensitive data
- `*.log` - Log files

**These are already in `.gitignore` and will NOT be uploaded.**

### 2. Verify .gitignore
Make sure `.gitignore` includes:
- `config.py` (use `config.py.example` instead)
- `*.log` and `logs/`
- `*.csv` (except example files)
- `venv/` and other Python cache files

## 📤 Upload Steps

### Step 1: Check Current Status
```bash
git status
```

### Step 2: Add Files to Git
```bash
# Add all files (respecting .gitignore)
git add .

# Or add specific files
git add *.py *.md *.txt *.sh
git add core/
git add config.py.example
git add requirements.txt
```

### Step 3: Commit Changes
```bash
git commit -m "Initial commit: Mike Agent v3 - RL Trading Agent with comprehensive risk management"
```

### Step 4: Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., `mike-agent-v3`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Copy the repository URL

### Step 5: Add Remote and Push
```bash
# Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Or if you prefer SSH:
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🔐 Security Checklist

Before pushing, verify:
- [ ] `config.py` is in `.gitignore` (contains API keys)
- [ ] `*.csv` trade files are excluded
- [ ] `*.log` files are excluded
- [ ] No API keys in any committed files
- [ ] `config.py.example` exists with placeholder keys

## 📝 Recommended Files to Include

✅ **Include:**
- All `.py` files (agent code)
- `README.md` and documentation
- `requirements.txt`
- `config.py.example` (template, no real keys)
- `.gitignore`
- Documentation files (`.md`)

❌ **Exclude (already in .gitignore):**
- `config.py` (real API keys)
- `*.log` files
- `*.csv` trade data
- `venv/` directory
- `__pycache__/` directories

## 🚀 Quick Upload Command

```bash
# 1. Check what will be committed
git status

# 2. Add files
git add .

# 3. Commit
git commit -m "Mike Agent v3: RL trading agent with 5-tier TP, volatility regimes, and comprehensive risk management"

# 4. Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/mike-agent-v3.git

# 5. Push
git push -u origin main
```

## 📋 After Uploading

1. **Update README.md** with:
   - Project description
   - Setup instructions
   - How to configure API keys (point to `config.py.example`)
   - Usage examples

2. **Add Repository Topics:**
   - `trading-bot`
   - `reinforcement-learning`
   - `options-trading`
   - `alpaca-api`
   - `python`

3. **Add License** (if desired):
   - MIT, Apache 2.0, or your preferred license

## 🔄 Future Updates

To push future changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

## ⚠️ If You Accidentally Committed Sensitive Data

If you accidentally committed `config.py` or other sensitive files:

1. **Remove from Git history:**
```bash
git rm --cached config.py
git commit -m "Remove sensitive config file"
git push
```

2. **Rotate your API keys** in Alpaca dashboard (old keys are now exposed)

3. **Update .gitignore** to prevent future commits

---

**Ready to upload?** Follow the steps above, and your project will be on GitHub! 🚀

