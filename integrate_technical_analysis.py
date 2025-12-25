#!/usr/bin/env python3
"""
Integration Guide: How to integrate Technical Analysis Engine into mike_agent_live_safe.py
"""

# This file shows how to integrate the technical analysis engine
# into the main trading agent

"""
STEP 1: Import the engine
==========================
Add to mike_agent_live_safe.py imports:

from technical_analysis_engine import TechnicalAnalysisEngine
"""

"""
STEP 2: Initialize the engine
==============================
Add to run_safe_live_trading() function, after model loading:

# Initialize Technical Analysis Engine
ta_engine = TechnicalAnalysisEngine(lookback_bars=50)
risk_mgr.log("✅ Technical Analysis Engine initialized", "INFO")
"""

"""
STEP 3: Use TA in symbol analysis loop
========================================
In the main trading loop, before RL inference:

# Get historical data for technical analysis
symbol_hist = get_market_data(symbol, period="2d", interval="1m", api=api, risk_mgr=risk_mgr)
if len(symbol_hist) >= 20:
    # Run technical analysis
    ta_result = ta_engine.analyze_symbol(
        data=symbol_hist,
        symbol=symbol,
        current_price=current_price
    )
    
    # Boost confidence if pattern detected
    if ta_result['best_pattern']:
        pattern = ta_result['best_pattern']
        confidence_boost = ta_result['confidence_boost']
        
        # Log pattern detection
        risk_mgr.log(
            f"🎯 TA Pattern: {pattern['pattern_type']} ({pattern['direction']}) "
            f"| Confidence: {pattern['confidence']:.2f} | "
            f"Boost: +{confidence_boost:.2f}",
            "INFO"
        )
        
        # Use target-based strike if pattern detected
        if ta_result['strike_suggestion']:
            strike = ta_result['strike_suggestion']
            risk_mgr.log(
                f"🎯 TA Strike: ${strike:.2f} (based on {pattern['pattern_type']})",
                "INFO"
            )
"""

"""
STEP 4: Boost RL confidence with TA
====================================
After RL inference, boost confidence if TA pattern detected:

# In symbol_actions calculation:
if ta_result and ta_result['best_pattern']:
    # Boost confidence based on TA pattern
    base_confidence = action_strength  # From RL model
    ta_boost = ta_result['confidence_boost']
    boosted_confidence = min(0.95, base_confidence + ta_boost)
    
    # Update action strength
    action_strength = boosted_confidence
    
    risk_mgr.log(
        f"🚀 Confidence Boost: {base_confidence:.3f} → {boosted_confidence:.3f} "
        f"(+{ta_boost:.3f} from TA pattern)",
        "INFO"
    )
"""

"""
STEP 5: Use TA-based strike selection
======================================
When selecting strike, use TA suggestion if available:

# In strike selection (for CALL and PUT):
if ta_result and ta_result['strike_suggestion']:
    # Use TA-based strike
    strike = ta_result['strike_suggestion']
    risk_mgr.log(
        f"🎯 Using TA-based strike: ${strike:.2f} "
        f"(pattern: {ta_result['best_pattern']['pattern_type']})",
        "INFO"
    )
else:
    # Fallback to fixed offset
    strike = find_atm_strike(symbol_price, option_type='call' or 'put')
"""

"""
STEP 6: Entry confirmation
============================
Before executing trade, check for confirmation:

if ta_result and ta_result['best_pattern']:
    pattern = ta_result['best_pattern']
    
    # Check if pattern requires confirmation
    if pattern['pattern_type'] == 'trendline_break':
        if not pattern.get('confirmed', False):
            risk_mgr.log(
                f"⏳ Waiting for confirmation: {pattern['reason']}",
                "INFO"
            )
            continue  # Skip this iteration, wait for confirmation
    
    # Check invalidation levels
    if 'invalidation_level' in pattern:
        if current_price > pattern['invalidation_level']:  # For PUTS
            risk_mgr.log(
                f"❌ Pattern invalidated: Price ${current_price:.2f} > "
                f"${pattern['invalidation_level']:.2f}",
                "INFO"
            )
            continue  # Skip trade
"""

"""
STEP 7: Log TA analysis
========================
Add detailed logging:

if ta_result['patterns']:
    risk_mgr.log(
        f"📊 TA Analysis for {symbol}:",
        "INFO"
    )
    for pattern in ta_result['patterns']:
        risk_mgr.log(
            f"  • {pattern['pattern_type']}: {pattern['reason']} "
            f"(confidence: {pattern['confidence']:.2f})",
            "INFO"
        )
    
    if ta_result['targets']:
        risk_mgr.log(
            f"  🎯 Targets: ${ta_result['targets']['target1']:.2f} / "
            f"${ta_result['targets']['target2']:.2f}",
            "INFO"
        )
"""





