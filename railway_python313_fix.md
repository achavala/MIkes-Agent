# Railway Python 3.13 Compatibility Fix

## Issue
Pandas compilation failing on Python 3.13 because:
- pandas==2.2.2 doesn't have pre-built wheels for Python 3.13
- Railway tries to compile from source, which fails

## Solution
Updated all dependencies to versions with Python 3.13 pre-built wheels:

### Key Changes:
- `pandas==2.2.2` → `pandas>=2.2.3` (has Python 3.13 wheels)
- `numpy==2.1.1` → `numpy>=2.1.3` (has Python 3.13 wheels)
- `yfinance==0.2.40` → `yfinance>=0.2.47` (updated)
- Added `scipy>=1.14.0` and `scikit-learn>=1.5.0` explicitly

### Why This Works:
- Python 3.13 requires packages with pre-built wheels
- pandas 2.2.3+ has wheels for Python 3.13
- numpy 2.1.3+ has wheels for Python 3.13
- No compilation needed = faster builds

## Alternative: Use Python 3.12
If issues persist, you can configure Railway to use Python 3.12:
- In Railway settings, set Python version to 3.12
- This allows older package versions

