#!/usr/bin/env python3
"""
Test imports with PyTorch MPS deadlock prevention
Run this directly: python test_imports.py
"""
import os

# Set environment variables BEFORE any imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'

print("Testing imports...")

try:
    import numpy
    print("✓ numpy OK")
except Exception as e:
    print(f"✗ numpy failed: {e}")

try:
    import sklearn
    print("✓ sklearn OK")
except Exception as e:
    print(f"✗ sklearn failed: {e}")

try:
    import torch
    print(f"✓ torch OK (device: {torch.device('cpu')})")
    # Verify it's CPU-only
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("⚠️  WARNING: MPS is available - this may cause deadlocks!")
    else:
        print("✓ MPS not available (good - CPU-only)")
except Exception as e:
    print(f"✗ torch failed: {e}")

try:
    from mike_ai_agent import MikeAIAgent
    print("✓ MikeAIAgent OK")
except Exception as e:
    print(f"✗ MikeAIAgent failed: {e}")

print("\n✅ All imports complete!")
