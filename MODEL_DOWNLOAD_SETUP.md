# 🚀 Model Auto-Download Setup (Fully Automatic)

## ✅ Status: Fully Automatic - No Manual Intervention Needed

The model download is now **100% automatic**. Once you set the `MODEL_URL` secret, the agent will:
- ✅ Auto-detect missing model
- ✅ Auto-download from URL
- ✅ Auto-verify download
- ✅ Auto-start trading

**No manual steps required after initial setup.**

---

## 📥 Quick Setup (One-Time)

### Step 1: Upload Your Model

Choose one option:

#### Option A: GitHub Releases (Easiest)
1. Go to your GitHub repo → Releases → Create new release
2. Upload `mike_momentum_model_v3_lstm.zip`
3. Copy the download URL (e.g., `https://github.com/user/repo/releases/download/v1.0/model.zip`)

#### Option B: Public S3 Bucket
```bash
aws s3 cp models/mike_momentum_model_v3_lstm.zip \
  s3://your-public-bucket/models/mike_momentum_model_v3_lstm.zip \
  --acl public-read
```
URL: `https://your-public-bucket.s3.amazonaws.com/models/mike_momentum_model_v3_lstm.zip`

#### Option C: Any Public URL
Upload to any web server and get the direct download URL.

---

### Step 2: Set Fly Secret (One Command)

```bash
fly secrets set MODEL_URL=https://your-url.com/models/mike_momentum_model_v3_lstm.zip
```

---

### Step 3: Deploy (That's It!)

```bash
fly deploy
```

**The agent will automatically:**
1. Check for model on startup
2. Download if missing
3. Cache locally
4. Start trading

**No manual intervention needed!**

---

## 🔄 How It Works (Automatic)

1. **Startup**: Agent checks `models/mike_momentum_model_v3_lstm.zip`
2. **If Missing**: Downloads from `MODEL_URL` automatically
3. **If Exists**: Uses cached model (no download)
4. **Verification**: Confirms model file exists
5. **Trading**: Agent starts with model loaded

**All automatic - runs on every container start.**

---

## 📊 Supported URL Types

- ✅ `https://...` - HTTP/HTTPS (most common)
- ✅ `http://...` - HTTP
- ✅ `s3://bucket/path` - AWS S3 (requires AWS credentials)

---

## 🛠️ Troubleshooting

### Model Not Downloading?

1. Check URL is accessible:
   ```bash
   curl -I https://your-url.com/models/mike_momentum_model_v3_lstm.zip
   ```

2. Check Fly secret is set:
   ```bash
   fly secrets list
   ```

3. Check logs:
   ```bash
   fly logs | grep -i model
   ```

---

## ✅ Verification

After deployment, check logs:

```bash
fly logs | grep -E "(Model|download|ready)"
```

You should see:
```
✅ Model ready at models/mike_momentum_model_v3_lstm.zip (auto-downloaded, no manual intervention)
```

