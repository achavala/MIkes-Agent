# Quick Fix Instructions

## The Problem
PyTorch MPS deadlock on Apple Silicon. Some packages reinstall MPS/GPU PyTorch after CPU-only installation.

## The Solution (Run These in Order)

### Step 1: Quick Fix (30 seconds)
```bash
cd /Users/chavala/Mike-agent-project
source venv/bin/activate

# Uninstall any torch
pip uninstall torch torchvision torchaudio -y

# Install CPU-only torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install basic packages
pip install scikit-learn numpy
```

### Step 2: Test (Must finish in < 2 seconds)
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
python -c "from mike_ai_agent import MikeAIAgent; print('DONE')"
```

If you still see mutex lock → run this:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=1
python -c "from mike_ai_agent import MikeAIAgent; print('✓ 100% fixed')"
```

### Step 3: Full Fix (if quick fix doesn't work)
```bash
./fix_pytorch_deadlock.sh
```

### Step 4: Start Dashboard
```bash
./run.sh
```

## Or Use the Quick Fix Script
```bash
./quick_fix.sh
```

## Why This Happens
These packages silently reinstall MPS/GPU PyTorch:
- `langchain`
- `langchain-community`
- `langchain-openai`
- `chromadb`
- `sentence-transformers`
- `transformers`

**Solution**: Install them with `--no-deps` after CPU-only PyTorch is installed.

## Verification
After fix, this should finish instantly (< 2 seconds):
```bash
python -c "from mike_ai_agent import MikeAIAgent; print('✓ Mike AI Agent finally works!')"
```

If it hangs → PyTorch MPS is still being used → run fix again.

