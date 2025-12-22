#!/bin/bash
# Cloud startup script for Fly.io
# Runs both the trading agent and Streamlit dashboard

set -e  # Exit on error

# Get port from Fly.io environment variable (required for HTTP service)
PORT=${PORT:-8080}
echo "🚀 Starting Cloud Deployment on PORT ${PORT}"

# Set timezone to EST/EDT (America/New_York) - CRITICAL for consistent timestamps
export TZ=America/New_York
echo "🕐 Timezone set to: $TZ (EST/EDT)"

# Set Python path to include user-installed packages
export PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH
export PATH=/root/.local/bin:$PATH

# Ensure PORT is exported for child processes
export PORT

# Get Git version info (if available)
GIT_VERSION=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "📦 Version Info: Git ${GIT_VERSION}"

# Verify stable-baselines3 is accessible (graceful - don't exit if version.txt is missing)
echo "🔍 Verifying dependencies..."
if python -c "import stable_baselines3; print('✅ stable-baselines3:', getattr(stable_baselines3, '__version__', 'installed'))" 2>/dev/null; then
    echo "✅ stable-baselines3 verified"
elif python -c "from stable_baselines3 import PPO; print('✅ stable-baselines3 importable (version check skipped)')" 2>/dev/null; then
    echo "✅ stable-baselines3 importable (version.txt missing but package works)"
else
    echo "⚠️  WARNING: stable-baselines3 import failed. Checking installation..."
    python -c "import sys; print('Python path:', sys.path)" 2>/dev/null || true
    pip list | grep stable || echo "Not in pip list"
    echo "⚠️  Continuing anyway - agent may fail at runtime if stable-baselines3 is truly missing"
    # Don't exit - let the agent try to start and fail gracefully if needed
fi

# Check for required environment variables
if [ -z "$ALPACA_KEY" ] || [ -z "$ALPACA_SECRET" ]; then
    echo "⚠️  WARNING: ALPACA_KEY or ALPACA_SECRET not set. Agent may fail to start."
    echo "   Set secrets with: fly secrets set ALPACA_KEY=... ALPACA_SECRET=..."
fi

# Determine mode from environment variable
MODE=${MODE:-paper}
echo "🧪 Starting Agent in ${MODE^^} mode..."

# Model download logic (FULLY AUTOMATIC - no manual intervention needed)
# Use the trained historical model (5M timesteps, 23.9 years of data)
MODEL_PATH="models/mike_23feature_model_final.zip"
MODEL_DIR="models"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model exists locally
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Model not found locally at $MODEL_PATH"
    
    # Try to download from MODEL_URL if set (AUTOMATIC)
    if [ -n "$MODEL_URL" ]; then
        echo "📥 Auto-downloading model from $MODEL_URL (no manual intervention needed)..."
        
        # Check if it's an S3 URL
        if [[ "$MODEL_URL" == s3://* ]]; then
            # S3 download (automatic - tries multiple methods)
            if command -v aws &> /dev/null; then
                aws s3 cp "$MODEL_URL" "$MODEL_PATH" 2>&1 && echo "✅ Model auto-downloaded from S3" || {
                    echo "⚠️  AWS CLI failed, trying Python boto3..."
                    python -c "
import boto3
import sys
import os
try:
    s3 = boto3.client('s3')
    url = sys.argv[1]
    path = sys.argv[2]
    bucket = url.split('/')[2]
    key = '/'.join(url.split('/')[3:])
    s3.download_file(bucket, key, path)
    print('✅ Model auto-downloaded from S3 via boto3')
except Exception as e:
    print(f'❌ S3 download failed: {e}')
    sys.exit(1)
" "$MODEL_URL" "$MODEL_PATH" 2>&1 || echo "⚠️  S3 download failed - check AWS credentials in Fly secrets"
                }
            elif python -c "import boto3" 2>/dev/null; then
                # Use Python boto3 automatically
                python -c "
import boto3
import sys
import os
try:
    s3 = boto3.client('s3')
    url = sys.argv[1]
    path = sys.argv[2]
    bucket = url.split('/')[2]
    key = '/'.join(url.split('/')[3:])
    s3.download_file(bucket, key, path)
    print('✅ Model auto-downloaded from S3')
except Exception as e:
    print(f'❌ S3 download failed: {e}')
    sys.exit(1)
" "$MODEL_URL" "$MODEL_PATH" 2>&1 || echo "⚠️  S3 download failed - check AWS credentials"
            else
                echo "⚠️  S3 URL provided but neither aws CLI nor boto3 available"
            fi
        # Check if it's an HTTP/HTTPS URL (MOST COMMON - FULLY AUTOMATIC)
        elif [[ "$MODEL_URL" == http://* ]] || [[ "$MODEL_URL" == https://* ]]; then
            # HTTP download - try Python urllib first (most reliable), then curl
            echo "📥 Downloading model from URL (automatic, no manual intervention)..."
            
            # Method 1: Python urllib (always available, most reliable)
            if python -c "import urllib.request" 2>/dev/null; then
                echo "📥 Using Python to download (automatic)..."
                if python -c "
import urllib.request
import sys
import os
try:
    url = sys.argv[1]
    path = sys.argv[2]
    print(f'Downloading from: {url}')
    # Create request with User-Agent (GitHub requires this)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    # urlretrieve handles redirects automatically, but we'll use urlopen for better control
    with urllib.request.urlopen(req) as response:
        with open(path, 'wb') as f:
            f.write(response.read())
    if os.path.exists(path) and os.path.getsize(path) > 0:
        size = os.path.getsize(path)
        print(f'✅ Model auto-downloaded from URL ({size:,} bytes)')
    else:
        print('❌ Download failed - file is empty or missing')
        sys.exit(1)
except Exception as e:
    print(f'❌ Python download failed: {e}')
    sys.exit(1)
" "$MODEL_URL" "$MODEL_PATH" 2>&1; then
                    echo "✅ Model download successful"
                else
                    echo "⚠️  Python download failed, trying curl..."
                    # Fallback to curl
                    if command -v curl &> /dev/null; then
                        if curl -L -f -o "$MODEL_PATH" "$MODEL_URL" 2>&1; then
                            echo "✅ Model auto-downloaded from URL via curl"
                            ls -lh "$MODEL_PATH" 2>/dev/null || true
                        else
                            echo "⚠️  curl download also failed - check URL accessibility"
                            echo "   URL: $MODEL_URL"
                        fi
                    else
                        echo "⚠️  curl not available - download failed"
                    fi
                fi
            # Method 2: curl fallback (if Python fails)
            elif command -v curl &> /dev/null; then
                echo "📥 Using curl to download (automatic)..."
                if curl -L -f -o "$MODEL_PATH" "$MODEL_URL" 2>&1; then
                    echo "✅ Model auto-downloaded from URL via curl"
                    ls -lh "$MODEL_PATH" 2>/dev/null || true
                else
                    echo "⚠️  curl download failed - check URL"
                fi
            else
                echo "⚠️  No download tools available (Python urllib/curl)"
            fi
        else
            echo "⚠️  MODEL_URL format not recognized (use s3://, http://, or https://)"
        fi
    else
        echo "ℹ️  MODEL_URL not set. Model will be loaded from local path if available."
        echo "   To enable auto-download, set: fly secrets set MODEL_URL=<url>"
    fi
    
    # Final automatic check
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  Model still not found. Agent will attempt to start but may fail to load model."
        echo "   Set MODEL_URL secret to enable automatic download: fly secrets set MODEL_URL=<url>"
    else
        echo "✅ Model ready at $MODEL_PATH (auto-downloaded, no manual intervention)"
        ls -lh "$MODEL_PATH" 2>/dev/null || true
    fi
else
    echo "✅ Model found locally at $MODEL_PATH (no download needed)"
fi

# Start Streamlit dashboard first (it's the HTTP service Fly.io expects)
# Use PORT from environment (set by Fly.io) or default to 8080
STREAMLIT_PORT=${PORT:-8080}
export STREAMLIT_SERVER_PORT=${STREAMLIT_PORT}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

echo "📊 Starting Streamlit dashboard on port ${STREAMLIT_PORT} (address: 0.0.0.0)..."
streamlit run dashboard_app.py \
    --server.port=${STREAMLIT_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.runOnSave=false \
    --server.fileWatcherType=none &
DASHBOARD_PID=$!

# Wait a moment for dashboard to start
sleep 3

# Check if dashboard started successfully
if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "❌ Dashboard failed to start. Check logs."
    exit 1
fi

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"

# Start the trading agent in the background
echo "🤖 Starting trading agent..."
# Use unbuffered Python output and redirect to both file and stdout
python -u mike_agent_live_safe.py 2>&1 | tee /tmp/agent.log &
AGENT_PID=$!

# Wait a moment for agent to initialize
sleep 5

# Check if agent started successfully
if ! kill -0 $AGENT_PID 2>/dev/null; then
    echo "⚠️  Agent may have failed to start, checking logs..."
    cat /tmp/agent.log 2>/dev/null || echo "No agent logs found"
    echo "⚠️  Continuing with dashboard..."
else
    echo "✅ Agent started (PID: $AGENT_PID)"
    # Show first few lines of agent output
    echo "📋 Agent startup output:"
    head -20 /tmp/agent.log 2>/dev/null || echo "No agent output yet"
    # Also tail the log file in background to show ongoing output
    tail -f /tmp/agent.log &
    TAIL_PID=$!
fi

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down..."
    kill $AGENT_PID 2>/dev/null || true
    kill $DASHBOARD_PID 2>/dev/null || true
    wait
    exit 0
}

trap cleanup SIGTERM SIGINT

# Keep script running - wait for dashboard (primary process)
# If dashboard dies, the container should restart
wait $DASHBOARD_PID
EXIT_CODE=$?

echo "Dashboard exited with code $EXIT_CODE"
exit $EXIT_CODE
