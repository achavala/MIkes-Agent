#!/bin/bash
#
# Prepare Training Package for Desktop Mac
#
# Packages all necessary files for running training on a separate Mac
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_NAME="mike-agent-training-package"
PACKAGE_DIR="${PACKAGE_NAME}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${PACKAGE_NAME}_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "=" | tr '=' '═'
echo "📦 PREPARING TRAINING PACKAGE FOR DESKTOP MAC"
echo "=" | tr '=' '═'
echo ""

# Create package directory
echo "📁 Creating package directory..."
mkdir -p "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/data/historical/enriched"
mkdir -p "$PACKAGE_DIR/models"

# Copy data files
echo "📊 Copying data files..."
if [ -d "data/historical/enriched" ]; then
    cp data/historical/enriched/*.pkl "$PACKAGE_DIR/data/historical/enriched/" 2>/dev/null || true
    echo "   ✅ Data files copied"
else
    echo "   ⚠️  Warning: Data directory not found"
fi

# Copy training scripts
echo "📝 Copying training scripts..."
TRAINING_FILES=(
    "historical_training_system.py"
    "train_historical_model.py"
    "quant_features_collector.py"
    "institutional_features.py"
    "greeks_calculator.py"
    "latency_monitor.py"
    "prevent_sleep.sh"
    "start_training.sh"
    "TRAINING_TIME_ESTIMATE.py"
    "COMPLETE_TRAINING_PLAN.md"
    "PARALLEL_TRAINING_SETUP.md"
)

for file in "${TRAINING_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$PACKAGE_DIR/"
        echo "   ✅ $file"
    else
        echo "   ⚠️  $file not found (optional)"
    fi
done

# Copy requirements
echo "📋 Copying requirements..."
if [ -f "requirements.txt" ]; then
    cp requirements.txt "$PACKAGE_DIR/"
    echo "   ✅ requirements.txt"
else
    echo "   ⚠️  requirements.txt not found - creating minimal one..."
    cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
stable-baselines3[extra]>=2.0.0
gymnasium>=0.28.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
yfinance>=0.2.0
scikit-learn>=1.3.0
EOF
fi

# Create setup script for desktop Mac
echo "🔧 Creating setup script..."
cat > "$PACKAGE_DIR/setup_desktop_mac.sh" << 'EOF'
#!/bin/bash
#
# Setup Script for Desktop Mac
# Run this after extracting the package
#

set -e

echo "=" | tr '=' '═'
echo "🚀 SETTING UP MIKE AGENT TRAINING ON DESKTOP MAC"
echo "=" | tr '=' '═'
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take 5-10 minutes..."
pip install -r requirements.txt

# Install RL dependencies
echo "🤖 Installing RL dependencies..."
pip install "stable-baselines3[extra]" gymnasium --quiet

# Make scripts executable
echo "🔨 Making scripts executable..."
chmod +x prevent_sleep.sh
chmod +x start_training.sh 2>/dev/null || true

# Verify data files
echo "✅ Verifying data files..."
if [ -f "data/historical/enriched/SPY_enriched_2002-01-01_latest.pkl" ]; then
    echo "   ✅ SPY data found"
else
    echo "   ⚠️  Warning: SPY data not found"
fi

if [ -f "data/historical/enriched/QQQ_enriched_2002-01-01_latest.pkl" ]; then
    echo "   ✅ QQQ data found"
else
    echo "   ⚠️  Warning: QQQ data not found"
fi

if [ -f "data/historical/enriched/SPX_enriched_2002-01-01_latest.pkl" ]; then
    echo "   ✅ SPX data found"
else
    echo "   ⚠️  Warning: SPX data not found"
fi

echo ""
echo "=" | tr '=' '═'
echo "✅ SETUP COMPLETE!"
echo "=" | tr '=' '═'
echo ""
echo "Next steps:"
echo ""
echo "1. Prevent Mac from sleeping:"
echo "   ./prevent_sleep.sh start"
echo ""
echo "2. Start training:"
echo "   ./start_training.sh"
echo ""
echo "   OR manually:"
echo "   source venv/bin/activate"
echo "   nohup python train_historical_model.py \\"
echo "       --symbols SPY,QQQ,SPX \\"
echo "       --start-date 2002-01-01 \\"
echo "       --timesteps 5000000 \\"
echo "       --use-greeks \\"
echo "       --regime-balanced \\"
echo "       --save-freq 100000 \\"
echo "       > training.log 2>&1 &"
echo ""
echo "3. Monitor progress:"
echo "   tail -f training.log"
echo ""
echo "4. After training completes:"
echo "   ./prevent_sleep.sh stop"
echo ""
EOF

chmod +x "$PACKAGE_DIR/setup_desktop_mac.sh"

# Create README for desktop Mac
echo "📄 Creating README..."
cat > "$PACKAGE_DIR/README_DESKTOP.md" << 'EOF'
# Mike Agent Training Package

This package contains everything needed to run 7-day training on your desktop Mac.

## Quick Start

1. **Extract this package:**
   ```bash
   tar -xzf mike-agent-training-package_*.tar.gz
   cd mike-agent-training-package_*/
   ```

2. **Run setup:**
   ```bash
   ./setup_desktop_mac.sh
   ```

3. **Start training:**
   ```bash
   ./prevent_sleep.sh start
   ./start_training.sh
   ```

4. **Monitor:**
   ```bash
   tail -f training.log
   ```

## What's Included

- ✅ Historical data files (SPY, QQQ, SPX - 23.9 years)
- ✅ Training scripts
- ✅ Setup script (automated)
- ✅ Sleep prevention script
- ✅ All dependencies

## Training Time

- **Estimated:** ~7 days (CPU) or ~1.75 days (GPU)
- **Timesteps:** 5,000,000
- **Checkpoints:** Every 100,000 steps

## Output

Trained models will be saved in:
```
models/
├── mike_rl_model_step_100000.zip
├── mike_rl_model_step_200000.zip
└── ...
```

Final model: `models/mike_rl_model_step_5000000.zip`

## After Training

Copy the final model back to your M2 Mac laptop for live trading.

See `PARALLEL_TRAINING_SETUP.md` for full details.
EOF

# Create archive
echo "📦 Creating archive..."
tar -czf "$OUTPUT_FILE" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".DS_Store" \
    "$PACKAGE_DIR"

# Calculate size
SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "=" | tr '=' '═'
echo "✅ PACKAGE CREATED SUCCESSFULLY!"
echo "=" | tr '=' '═'
echo ""
echo "📦 Package: $OUTPUT_FILE"
echo "📊 Size: $SIZE"
echo ""
echo "📍 Next Steps:"
echo ""
echo "1. Transfer to Desktop Mac:"
echo "   • USB Drive: cp $OUTPUT_FILE /Volumes/USB_DRIVE/"
echo "   • Cloud: Upload to iCloud/Dropbox/Google Drive"
echo "   • Network: Use file sharing"
echo ""
echo "2. On Desktop Mac:"
echo "   tar -xzf $OUTPUT_FILE"
echo "   cd $PACKAGE_DIR"
echo "   ./setup_desktop_mac.sh"
echo "   ./prevent_sleep.sh start"
echo "   ./start_training.sh"
echo ""
echo "3. Continue daily work on M2 Mac laptop (no changes needed)"
echo ""
echo "📄 Full guide: See PARALLEL_TRAINING_SETUP.md"
echo ""

# Cleanup temp directory
rm -rf "$PACKAGE_DIR"

echo "✅ Done!"

