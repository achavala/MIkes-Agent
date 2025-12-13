"""
BEHAVIORAL BACKTEST RELAXATION PROFILE
Config for behavioral backtests to test decision logic, not capital survival
"""
from typing import Dict, Any


# Behavioral backtest profile - loosens constraints for decision testing
BEHAVIORAL_PROFILE = {
    "risk_manager": {
        "gamma_limit_multiplier": 1.5,  # 50% more lenient
        "delta_limit_multiplier": 1.5,  # 50% more lenient
        "vix_kill_switch": False,  # Disable for behavioral (keep for live)
        "daily_loss_limit_multiplier": 2.0,  # More lenient
    },
    "ensemble": {
        "min_agent_agreement": 1,  # Allow single agent to propose (behavioral testing only)
        "confidence_threshold": 0.25,  # Lower threshold (from 0.3)
    },
    "signal_floor": {
        "enabled": True,  # Behavioral signal floor - allow weak-but-consistent signals
        "rl_confidence_min": 0.52,  # Minimum RL confidence to allow trade
        "ensemble_confidence_min": 0.50,  # Minimum ensemble confidence to allow trade
        "min_size_multiplier": 0.5,  # Use smaller size for exploratory trades
    },
    "action_nudge": {
        "enabled": True,  # Behavioral action nudge - allow weak directional intent
        "rl_action_raw_threshold": 0.15,  # Minimum abs(rl_action_raw) to propose trade
        "force_probe_trade": False,  # DISABLED: No forced probe trades (moved to organic signal expression)
        "probe_trade_size": 0.1,  # Probe trade size multiplier (10% of normal)
    },
    "execution": {
        "apply_iv_crush": False,  # Still log, but don't penalize
        "apply_theta_penalty": False,  # Still log, but don't penalize
        "slippage_multiplier": 0.5,  # Reduce slippage impact
    },
    "trading": {
        "min_trades_per_day": 1,  # Expect at least 1 trade per active day
        "max_trades_per_day": 20,  # Cap to prevent overtrading
    }
}


def get_behavioral_profile() -> Dict[str, Any]:
    """Get behavioral backtest profile"""
    return BEHAVIORAL_PROFILE.copy()


def apply_behavioral_overrides(config: Dict[str, Any], profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Apply behavioral profile overrides to config
    
    Args:
        config: Original configuration
        profile: Behavioral profile (defaults to BEHAVIORAL_PROFILE)
        
    Returns:
        Modified configuration with behavioral overrides
    """
    if profile is None:
        profile = BEHAVIORAL_PROFILE
    
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

