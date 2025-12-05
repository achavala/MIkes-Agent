# Fix PyTorch MPS Deadlock

## Quick Fix

Run this script directly in your terminal (not through Cursor):

```bash
cd /Users/chavala/Mike-agent-project
./fix_pytorch_deadlock.sh
```

## Manual Steps (if script doesn't work)

```bash
cd /Users/chavala/Mike-agent-project
source venv/bin/activate

# 1. Upgrade pip
pip install --upgrade pip

# 2. Uninstall any torch
pip uninstall torch torchvision torchaudio -y

# 3. Install CPU-only PyTorch FIRST
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other packages with --no-deps
pip install scikit-learn numpy pandas streamlit plotly \
            langchain langchain-openai langchain-community \
            chromadb sentence-transformers openai tiktoken \
            faiss-cpu transformers accelerate --no-deps

# 5. Test
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=1

python -c "import numpy, sklearn; print('✓ Basics OK')"
python -c "from mike_ai_agent import MikeAIAgent; print('✓ MikeAIAgent OK')"
```

## What This Fix Does

1. **Uninstalls any MPS/GPU version of PyTorch**
2. **Installs CPU-only PyTorch** from the official CPU index
3. **Installs other packages with --no-deps** to prevent them from downgrading torch
4. **Sets environment variables** to prevent MPS usage

The key is that we need to ensure:
- PyTorch is CPU-only
- Environment variables are set
- run.sh has the proper exports

