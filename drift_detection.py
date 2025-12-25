"""
DRIFT DETECTION MODULE
Monitors ensemble and RL model for drift
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque


class DriftDetector:
    """
    Detects model drift in ensemble and RL signals
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.rl_history = deque(maxlen=window_size)
        self.ensemble_history = deque(maxlen=window_size)
        self.regime_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
    
    def record_rl_signal(self, action: int, confidence: float):
        """Record RL signal"""
        self.rl_history.append({
            'action': action,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def record_ensemble_signal(self, action: int, confidence: float, regime: str):
        """Record ensemble signal"""
        self.ensemble_history.append({
            'action': action,
            'confidence': confidence,
            'regime': regime,
            'timestamp': datetime.now()
        })
        self.regime_history.append(regime)
        self.confidence_history.append(confidence)
    
    def check_rl_drift(self) -> Dict[str, any]:
        """Check for RL model drift"""
        if len(self.rl_history) < 20:
            return {'drift_detected': False, 'reason': 'Insufficient history'}
        
        recent = list(self.rl_history)[-10:]
        earlier = list(self.rl_history)[-20:-10]
        
        # Check confidence degradation
        recent_conf = np.mean([s['confidence'] for s in recent])
        earlier_conf = np.mean([s['confidence'] for s in earlier])
        
        if recent_conf < earlier_conf * 0.7:  # 30% drop
            return {
                'drift_detected': True,
                'type': 'confidence_degradation',
                'recent_avg': recent_conf,
                'earlier_avg': earlier_conf,
                'drop_pct': (earlier_conf - recent_conf) / earlier_conf * 100
            }
        
        # Check action distribution change
        recent_actions = [s['action'] for s in recent]
        earlier_actions = [s['action'] for s in earlier]
        
        recent_dist = {0: recent_actions.count(0), 1: recent_actions.count(1), 2: recent_actions.count(2)}
        earlier_dist = {0: earlier_actions.count(0), 1: earlier_actions.count(1), 2: earlier_actions.count(2)}
        
        # Calculate distribution shift
        total_recent = sum(recent_dist.values())
        total_earlier = sum(earlier_dist.values())
        
        if total_recent > 0 and total_earlier > 0:
            shift = sum(abs(recent_dist[k] / total_recent - earlier_dist[k] / total_earlier) for k in [0, 1, 2])
            if shift > 0.4:  # 40% distribution shift
                return {
                    'drift_detected': True,
                    'type': 'action_distribution_shift',
                    'shift': shift,
                    'recent_dist': recent_dist,
                    'earlier_dist': earlier_dist
                }
        
        return {'drift_detected': False, 'reason': 'No drift detected'}
    
    def check_ensemble_drift(self) -> Dict[str, any]:
        """Check for ensemble drift"""
        if len(self.ensemble_history) < 20:
            return {'drift_detected': False, 'reason': 'Insufficient history'}
        
        recent = list(self.ensemble_history)[-10:]
        earlier = list(self.ensemble_history)[-20:-10]
        
        # Check confidence degradation
        recent_conf = np.mean([s['confidence'] for s in recent])
        earlier_conf = np.mean([s['confidence'] for s in earlier])
        
        if recent_conf < earlier_conf * 0.7:
            return {
                'drift_detected': True,
                'type': 'confidence_degradation',
                'recent_avg': recent_conf,
                'earlier_avg': earlier_conf
            }
        
        # Check regime stability
        recent_regimes = list(self.regime_history)[-10:]
        regime_changes = len(set(recent_regimes))
        
        if regime_changes > 3:  # More than 3 different regimes in 10 signals
            return {
                'drift_detected': True,
                'type': 'regime_instability',
                'regime_changes': regime_changes,
                'regimes': recent_regimes
            }
        
        return {'drift_detected': False, 'reason': 'No drift detected'}
    
    def check_regime_drift(self) -> Dict[str, any]:
        """Check for regime drift"""
        if len(self.regime_history) < 20:
            return {'drift_detected': False, 'reason': 'Insufficient history'}
        
        recent_regimes = list(self.regime_history)[-10:]
        earlier_regimes = list(self.regime_history)[-20:-10]
        
        # Check for rapid regime changes
        recent_changes = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i-1])
        earlier_changes = sum(1 for i in range(1, len(earlier_regimes)) if earlier_regimes[i] != earlier_regimes[i-1])
        
        if recent_changes > earlier_changes * 2:  # 2x more changes
            return {
                'drift_detected': True,
                'type': 'rapid_regime_changes',
                'recent_changes': recent_changes,
                'earlier_changes': earlier_changes,
                'recent_regimes': recent_regimes
            }
        
        return {'drift_detected': False, 'reason': 'No drift detected'}
    
    def get_drift_report(self) -> Dict[str, any]:
        """Get comprehensive drift report"""
        rl_drift = self.check_rl_drift()
        ensemble_drift = self.check_ensemble_drift()
        regime_drift = self.check_regime_drift()
        
        any_drift = rl_drift.get('drift_detected') or ensemble_drift.get('drift_detected') or regime_drift.get('drift_detected')
        
        return {
            'any_drift': any_drift,
            'rl_drift': rl_drift,
            'ensemble_drift': ensemble_drift,
            'regime_drift': regime_drift,
            'timestamp': datetime.now()
        }


# Global instance
_drift_detector: Optional[DriftDetector] = None


def initialize_drift_detector(window_size: int = 50) -> DriftDetector:
    """Initialize global drift detector"""
    global _drift_detector
    _drift_detector = DriftDetector(window_size=window_size)
    return _drift_detector


def get_drift_detector() -> Optional[DriftDetector]:
    """Get global drift detector instance"""
    return _drift_detector





