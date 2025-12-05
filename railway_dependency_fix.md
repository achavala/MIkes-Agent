# Railway Dependency Conflict Fix

## Issue
Dependency conflict between:
- `gymnasium==1.0.0` (specified)
- `stable-baselines3==2.3.2` requires `gymnasium<0.30 and >=0.28.1`

These versions are incompatible.

## Solution
Updated to compatible versions:
- `stable-baselines3==2.3.2` → `stable-baselines3[extra]>=2.7.0`
- `gymnasium==1.0.0` → `gymnasium>=1.0.0`

### Why This Works:
- stable-baselines3 2.7.0+ supports gymnasium 1.0.0+
- Both are compatible with Python 3.13
- All dependencies now resolve correctly

## Alternative Solution
If you need to keep stable-baselines3 2.3.2:
- Use `gymnasium>=0.28.1,<0.30` instead
- But this is not recommended as gymnasium 1.0.0+ has better Python 3.13 support

