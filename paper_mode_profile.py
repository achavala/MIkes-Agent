"""
PAPER MODE CONFIGURATION
Settings for paper trading mode (transition from behavioral to live)
"""
from typing import Dict, Any


# Paper mode profile - full risk constraints, no behavioral relaxations
PAPER_MODE_PROFILE = {
    "risk_manager": {
        "gamma_limit_multiplier": 1.0,  # Full constraints (no relaxation)
        "delta_limit_multiplier": 1.0,  # Full constraints (no relaxation)
        "vix_kill_switch": True,  # Enable VIX kill switch for paper trading
        "daily_loss_limit_multiplier": 1.0,  # Full constraints
    },
    "ensemble": {
        "min_agent_agreement": 2,  # Require 2+ agents to agree (standard)
        "confidence_threshold": 0.3,  # Standard threshold
    },
    "signal_floor": {
        "enabled": False,  # Disabled in paper mode (rely on natural signals)
        "rl_confidence_min": 0.52,
        "ensemble_confidence_min": 0.50,
        "min_size_multiplier": 1.0,  # Full size
    },
    "action_nudge": {
        "enabled": True,  # Keep action nudge temporarily (will be removed after stability)
        "rl_action_raw_threshold": 0.15,  # Keep same threshold
        "force_probe_trade": False,  # No probe trades in paper mode
        "probe_trade_size": 0.1,
    },
    "execution": {
        "apply_iv_crush": True,  # Full execution penalties
        "apply_theta_penalty": True,  # Full execution penalties
        "slippage_multiplier": 1.0,  # Full slippage impact
    },
    "trading": {
        "min_trades_per_day": 0,  # No minimum (organic signals only)
        "max_trades_per_day": 20,  # Cap to prevent overtrading
    },
    "verdict": {
        "use_fallback_scoring": False,  # No fallback scoring in paper mode
    }
}


def get_paper_mode_profile() -> Dict[str, Any]:
    """Get paper mode profile"""
    return PAPER_MODE_PROFILE.copy()


def apply_paper_mode_overrides(config: Dict[str, Any], profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Apply paper mode overrides to config
    
    Args:
        config: Original configuration
        profile: Paper mode profile (defaults to PAPER_MODE_PROFILE)
        
    Returns:
        Modified configuration with paper mode overrides
    """
    if profile is None:
        profile = PAPER_MODE_PROFILE
    
    # Deep copy to avoid modifying original
    result = config.copy()
    
    # Apply risk manager overrides
    if "risk_manager" in profile:
        if "risk_manager" not in result:
            result["risk_manager"] = {}
        result["risk_manager"].update(profile["risk_manager"])
    
    # Apply ensemble overrides
    if "ensemble" in profile:
        if "ensemble" not in result:
            result["ensemble"] = {}
        result["ensemble"].update(profile["ensemble"])
    
    # Apply execution overrides
    if "execution" in profile:
        if "execution" not in result:
            result["execution"] = {}
        result["execution"].update(profile["execution"])
    
    # Apply trading overrides
    if "trading" in profile:
        if "trading" not in result:
            result["trading"] = {}
        result["trading"].update(profile["trading"])
    
    return result





