#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM/CNN BACKBONE VALIDATION
Validates if RL agent has been upgraded from MLP to LSTM/CNN/Attention
"""

import sys
import os
sys.path.insert(0, '.')

print("=" * 100)
print("LSTM/CNN BACKBONE VALIDATION - MODEL ARCHITECTURE CHECK")
print("=" * 100)
print()

validation_results = {}

# ==================== CHECK CURRENT MODEL ARCHITECTURE ====================
print("1️⃣  Current RL Model Architecture")
print("-" * 100)

try:
    # Check training script
    with open('train_historical_model.py', 'r') as f:
        train_content = f.read()
    
    # Check if using MLP
    using_mlp = '"MlpPolicy"' in train_content or "'MlpPolicy'" in train_content or 'MlpPolicy' in train_content
    
    # Check if using LSTM
    using_lstm_policy = '"LstmPolicy"' in train_content or "'LstmPolicy'" in train_content or 'LstmPolicy' in train_content
    
    # Check if using CNN
    using_cnn_policy = '"CnnPolicy"' in train_content or "'CnnPolicy'" in train_content or 'CnnPolicy' in train_content
    
    # Check if custom policy with LSTM
    has_lstm_import = 'from stable_baselines3.common.policies import' in train_content or 'LSTM' in train_content or 'Lstm' in train_content
    
    # Check if attention mechanism
    has_attention = 'Attention' in train_content or 'attention' in train_content.lower()
    
    # Check if temporal convolution
    has_tcn = 'TemporalConv' in train_content or 'temporal.*conv' in train_content.lower() or 'TCN' in train_content
    
    if using_lstm_policy or (has_lstm_import and 'LSTM' in train_content):
        print("   ✅ LSTM POLICY: Found")
        validation_results['LSTM_Policy'] = 'FOUND'
    elif using_mlp:
        print("   ❌ MLP POLICY: Still using basic MLP")
        print("      - No temporal memory")
        print("      - Cannot detect patterns across bars")
        print("      - Cannot track volatility regimes")
        print("      - Cannot detect transitions")
        validation_results['LSTM_Policy'] = 'MISSING - Using MLP'
    else:
        print("   ⚠️  UNKNOWN: Policy type unclear")
        validation_results['LSTM_Policy'] = 'UNKNOWN'
    
    if using_cnn_policy or has_tcn:
        print("   ✅ CNN/TCN: Found")
        validation_results['CNN_TCN'] = 'FOUND'
    else:
        print("   ❌ CNN/TCN: Not found")
        validation_results['CNN_TCN'] = 'MISSING'
    
    if has_attention:
        print("   ✅ ATTENTION: Found")
        validation_results['Attention'] = 'FOUND'
    else:
        print("   ❌ ATTENTION: Not found")
        validation_results['Attention'] = 'MISSING'
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    validation_results['LSTM_Policy'] = 'ERROR'
print()

# ==================== CHECK IF SEPARATE LSTM EXISTS ====================
print("2️⃣  Separate LSTM Model (Non-RL)")
print("-" * 100)
try:
    if os.path.exists('mike_ai_agent.py'):
        with open('mike_ai_agent.py', 'r') as f:
            ai_agent_content = f.read()
        
        if 'LSTM' in ai_agent_content and 'tensorflow' in ai_agent_content.lower():
            print("   ✅ SEPARATE LSTM: Found in mike_ai_agent.py")
            print("      ⚠️  BUT: This is NOT integrated into RL model")
            print("      - Separate AI agent with LSTM for direction prediction")
            print("      - RL model still uses MLP")
            validation_results['Separate_LSTM'] = 'FOUND_BUT_NOT_INTEGRATED'
        else:
            print("   ❌ SEPARATE LSTM: Not found")
            validation_results['Separate_LSTM'] = 'MISSING'
    else:
        print("   ❌ mike_ai_agent.py: File not found")
        validation_results['Separate_LSTM'] = 'FILE_NOT_FOUND'
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    validation_results['Separate_LSTM'] = 'ERROR'
print()

# ==================== CHECK FOR TEMPORAL PATTERN DETECTION ====================
print("3️⃣  Temporal Pattern Detection Capabilities")
print("-" * 100)
try:
    # Check if observation includes temporal dimension
    if '(window_size' in train_content or '(20,' in train_content:
        print("   ✅ TEMPORAL WINDOW: Found (20 timesteps)")
        print("      ⚠️  BUT: MLP processes as flat features, no sequence memory")
        validation_results['Temporal_Window'] = 'EXISTS_BUT_NO_MEMORY'
    else:
        print("   ❌ TEMPORAL WINDOW: Not found")
        validation_results['Temporal_Window'] = 'MISSING'
    
    # Check if volatility regime tracking exists in model
    if 'volatility.*regime' in train_content.lower() or 'regime' in train_content.lower():
        print("   ✅ VOLATILITY REGIME: Feature exists")
        print("      ⚠️  BUT: MLP cannot track regime transitions")
        validation_results['Regime_Tracking'] = 'FEATURE_EXISTS_NO_TRANSITION_TRACKING'
    else:
        print("   ❌ VOLATILITY REGIME: Not found")
        validation_results['Regime_Tracking'] = 'MISSING'
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    validation_results['Temporal_Window'] = 'ERROR'
print()

# ==================== CHECK FOR CUSTOM POLICY IMPLEMENTATION ====================
print("4️⃣  Custom Policy with LSTM/CNN Backbone")
print("-" * 100)
try:
    # Look for custom policy files
    custom_policy_files = [
        'institutional_rl_model.py',
        'custom_policy.py',
        'lstm_policy.py',
        'temporal_policy.py'
    ]
    
    found_custom = False
    for file in custom_policy_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
                if 'LSTM' in content or 'Lstm' in content or 'temporal' in content.lower():
                    print(f"   ✅ CUSTOM POLICY: Found in {file}")
                    found_custom = True
                    validation_results['Custom_Policy'] = f'FOUND_IN_{file}'
                    break
    
    if not found_custom:
        print("   ❌ CUSTOM POLICY: Not found")
        print("      - No custom LSTM/CNN policy implementation")
        validation_results['Custom_Policy'] = 'MISSING'
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    validation_results['Custom_Policy'] = 'ERROR'
print()

# ==================== SUMMARY ====================
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)
print()

missing_count = sum(1 for v in validation_results.values() if 'MISSING' in str(v) or 'USING_MLP' in str(v))
found_count = sum(1 for v in validation_results.values() if 'FOUND' in str(v) and 'NOT_INTEGRATED' not in str(v))

for feature, status in validation_results.items():
    if "FOUND" in str(status) and "NOT_INTEGRATED" not in str(status):
        icon = "✅"
    elif "MISSING" in str(status) or "USING_MLP" in str(status):
        icon = "❌"
    elif "NOT_INTEGRATED" in str(status) or "NO_MEMORY" in str(status) or "NO_TRANSITION" in str(status):
        icon = "⚠️ "
    else:
        icon = "❓"
    print(f"{icon} {feature:30s}: {status}")

print()
print(f"✅ Implemented: {found_count}/4")
print(f"❌ Missing: {missing_count}/4")
print()

# Final verdict
if missing_count >= 3:
    print("🎯 STATUS: ❌ NOT COMPLETE")
    print()
    print("   Current State:")
    print("   - ❌ Using basic MLP policy (no memory)")
    print("   - ❌ Cannot detect patterns across bars")
    print("   - ❌ Cannot track volatility regime transitions")
    print("   - ❌ No temporal convolution or attention")
    print()
    print("   Required Upgrade:")
    print("   - ✅ LSTM encoding (for temporal memory)")
    print("   - ✅ Temporal Convolution Networks (for pattern detection)")
    print("   - ✅ Attention mechanisms (for regime transitions)")
    print()
    print("   Action Required: Upgrade MLP → LSTM/CNN/Attention backbone")
elif missing_count >= 2:
    print("⚠️  STATUS: PARTIAL - Some components missing")
else:
    print("✅ STATUS: COMPLETE - LSTM/CNN backbone implemented")

print()

