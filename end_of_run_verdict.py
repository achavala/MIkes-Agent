"""
END-OF-RUN VERDICT SYSTEM
Produces single summary at day 30 with recommendation
"""
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np


class EndOfRunVerdict:
    """
    End-of-run verdict system
    
    Produces one single summary at day 30 with recommendation:
    - ❌ Reject
    - ⚠️ Revise
    - ✅ Proceed to limited live
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def generate_verdict(
        self,
        backtest_start_date: str,
        backtest_end_date: str
    ) -> Dict:
        """
        Generate end-of-run verdict
        
        Args:
            backtest_start_date: Start date (YYYY-MM-DD)
            backtest_end_date: End date (YYYY-MM-DD)
            
        Returns:
            Verdict with recommendation
        """
        from institutional_logging import get_logger
        logger = get_logger()
        
        if not logger:
            return {'error': 'Logger not initialized'}
        
        # Get all logs
        decisions = logger.get_logs("decisions", start_date=backtest_start_date, end_date=backtest_end_date)
        risk_checks = logger.get_logs("risk", start_date=backtest_start_date, end_date=backtest_end_date)
        executions = logger.get_logs("execution", start_date=backtest_start_date, end_date=backtest_end_date)
        positions = logger.get_logs("positions", start_date=backtest_start_date, end_date=backtest_end_date)
        learning = logger.get_logs("learning", start_date=backtest_start_date, end_date=backtest_end_date)
        
        # Generate scorecards
        behavior_scorecard = self._generate_behavior_scorecard(decisions, positions)
        risk_scorecard = self._generate_risk_scorecard(risk_checks)
        execution_scorecard = self._generate_execution_scorecard(executions)
        learning_scorecard = self._generate_learning_scorecard(learning)
        
        # Calculate overall scores
        behavior_score = behavior_scorecard.get('overall_score', 0)
        risk_score = risk_scorecard.get('overall_score', 0)
        execution_score = execution_scorecard.get('overall_score', 0)
        learning_score = learning_scorecard.get('overall_score', 0)
        
        # Determine recommendation
        recommendation = self._determine_recommendation(
            behavior_score, risk_score, execution_score, learning_score,
            risk_scorecard.get('violations', 0)
        )
        
        verdict = {
            "backtest_period": {
                "start": backtest_start_date,
                "end": backtest_end_date,
                "days": 30
            },
            "scorecards": {
                "behavior": behavior_scorecard,
                "risk": risk_scorecard,
                "execution": execution_scorecard,
                "learning": learning_scorecard
            },
            "overall_scores": {
                "behavior": behavior_score,
                "risk": risk_score,
                "execution": execution_score,
                "learning": learning_score,
                "average": (behavior_score + risk_score + execution_score + learning_score) / 4
            },
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save verdict
        self._save_verdict(verdict)
        
        return verdict
    
    def _generate_behavior_scorecard(
        self,
        decisions: List[Dict],
        positions: List[Dict]
    ) -> Dict:
        """Generate behavior scorecard"""
        if not decisions:
            return {'overall_score': 0, 'error': 'No decisions'}
        
        df_decisions = pd.DataFrame(decisions)
        df_positions = pd.DataFrame(positions) if positions else pd.DataFrame()
        
        scores = {}
        
        # Consistency across regimes
        if 'regime' in df_decisions.columns:
            regimes = df_decisions['regime'].unique()
            regime_consistency = len(regimes) > 0
            scores['regime_consistency'] = 1.0 if regime_consistency else 0.0
        
        # HOLD vs BUY balance
        if 'action_final' in df_decisions.columns:
            hold_count = len(df_decisions[df_decisions['action_final'] == 'HOLD'])
            buy_count = len(df_decisions[df_decisions['action_final'].str.contains('BUY', na=False)])
            total = len(df_decisions)
            hold_rate = (hold_count / total) if total > 0 else 0
            # Ideal: 40-70% HOLD rate
            if 0.4 <= hold_rate <= 0.7:
                scores['hold_balance'] = 1.0
            elif 0.3 <= hold_rate < 0.4 or 0.7 < hold_rate <= 0.8:
                scores['hold_balance'] = 0.7
            else:
                scores['hold_balance'] = 0.3
        
        # Ensemble influence
        if 'rl_action' in df_decisions.columns and 'ensemble_action' in df_decisions.columns:
            overrides = df_decisions[df_decisions['rl_action'] != df_decisions['ensemble_action']]
            override_rate = len(overrides) / len(df_decisions) if len(df_decisions) > 0 else 0
            # Ideal: 20-50% override rate (ensemble has influence but not dominant)
            if 0.2 <= override_rate <= 0.5:
                scores['ensemble_influence'] = 1.0
            elif 0.1 <= override_rate < 0.2 or 0.5 < override_rate <= 0.7:
                scores['ensemble_influence'] = 0.7
            else:
                scores['ensemble_influence'] = 0.3
        
        # Position lifecycle quality
        if len(df_positions) > 0 and 'final_pnl' in df_positions.columns:
            winning = len(df_positions[df_positions['final_pnl'] > 0])
            win_rate = (winning / len(df_positions)) if len(df_positions) > 0 else 0
            # Ideal: 60-80% win rate
            if 0.6 <= win_rate <= 0.8:
                scores['position_quality'] = 1.0
            elif 0.5 <= win_rate < 0.6 or 0.8 < win_rate <= 0.9:
                scores['position_quality'] = 0.7
            else:
                scores['position_quality'] = 0.3
        
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'metrics': {
                'total_decisions': len(df_decisions),
                'hold_rate': (hold_count / total * 100) if total > 0 else 0,
                'ensemble_override_rate': (len(overrides) / len(df_decisions) * 100) if len(df_decisions) > 0 else 0,
                'win_rate': (winning / len(df_positions) * 100) if len(df_positions) > 0 else 0
            }
        }
    
    def _generate_risk_scorecard(self, risk_checks: List[Dict]) -> Dict:
        """Generate risk scorecard"""
        if not risk_checks:
            return {'overall_score': 0, 'error': 'No risk checks'}
        
        df = pd.DataFrame(risk_checks)
        
        # Count violations
        blocked = df[df.get('risk_action') == 'BLOCK']
        violations = len(blocked)
        
        # Check for gamma breaches
        gamma_breaches = blocked[blocked.get('risk_reason', '').str.contains('GAMMA', na=False)]
        gamma_breach_count = len(gamma_breaches)
        
        # Score: 0 violations = 1.0, 1-5 = 0.7, >5 = 0.3
        if violations == 0:
            overall_score = 1.0
        elif violations <= 5:
            overall_score = 0.7
        else:
            overall_score = 0.3
        
        return {
            'overall_score': overall_score,
            'violations': violations,
            'gamma_breaches': gamma_breach_count,
            'total_checks': len(df),
            'block_rate_pct': (violations / len(df) * 100) if len(df) > 0 else 0,
            'passed': violations == 0
        }
    
    def _generate_execution_scorecard(self, executions: List[Dict]) -> Dict:
        """Generate execution scorecard"""
        if not executions:
            return {'overall_score': 0, 'error': 'No executions'}
        
        df = pd.DataFrame(executions)
        
        scores = {}
        
        # Slippage realism
        if 'slippage_pct' in df.columns:
            avg_slippage = df['slippage_pct'].mean()
            # Ideal: 0.3-0.8%
            if 0.3 <= avg_slippage <= 0.8:
                scores['slippage_realism'] = 1.0
            elif 0.2 <= avg_slippage < 0.3 or 0.8 < avg_slippage <= 1.2:
                scores['slippage_realism'] = 0.7
            else:
                scores['slippage_realism'] = 0.3
        
        # Execution cost components present
        has_gamma = 'gamma_impact' in df.columns and df['gamma_impact'].notna().any()
        has_iv_crush = 'iv_crush_impact' in df.columns and df['iv_crush_impact'].notna().any()
        has_theta = 'theta_impact' in df.columns and df['theta_impact'].notna().any()
        
        component_score = sum([has_gamma, has_iv_crush, has_theta]) / 3.0
        scores['execution_components'] = component_score
        
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'metrics': {
                'total_fills': len(df),
                'avg_slippage_pct': df['slippage_pct'].mean() if 'slippage_pct' in df.columns else 0,
                'has_gamma_impact': has_gamma,
                'has_iv_crush_impact': has_iv_crush,
                'has_theta_impact': has_theta
            }
        }
    
    def _generate_learning_scorecard(self, learning: List[Dict]) -> Dict:
        """Generate learning scorecard"""
        if not learning:
            return {'overall_score': 0.5, 'note': 'No learning events'}
        
        df = pd.DataFrame(learning)
        
        scores = {}
        
        # Retraining frequency
        retrained = df[df.get('retrained') == True]
        retrain_count = len(retrained)
        # Ideal: 3-7 retrains in 30 days
        if 3 <= retrain_count <= 7:
            scores['retrain_frequency'] = 1.0
        elif 1 <= retrain_count < 3 or 7 < retrain_count <= 10:
            scores['retrain_frequency'] = 0.7
        else:
            scores['retrain_frequency'] = 0.3
        
        # Model improvements
        if 'sharpe_candidate' in df.columns and 'sharpe_prod' in df.columns:
            improvements = []
            for _, row in retrained.iterrows():
                if pd.notna(row.get('sharpe_candidate')) and pd.notna(row.get('sharpe_prod')):
                    improvement = row['sharpe_candidate'] - row['sharpe_prod']
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                # Positive improvement = good
                if avg_improvement > 0.1:
                    scores['model_improvement'] = 1.0
                elif avg_improvement > 0:
                    scores['model_improvement'] = 0.7
                else:
                    scores['model_improvement'] = 0.3
            else:
                scores['model_improvement'] = 0.5
        
        # Stability (low variance in improvements)
        if len(improvements) > 1:
            improvement_std = np.std(improvements)
            # Low std = stable
            if improvement_std < 0.2:
                scores['stability'] = 1.0
            elif improvement_std < 0.5:
                scores['stability'] = 0.7
            else:
                scores['stability'] = 0.3
        else:
            scores['stability'] = 0.5
        
        overall_score = np.mean(list(scores.values())) if scores else 0.5
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'metrics': {
                'retrain_count': retrain_count,
                'avg_improvement': np.mean(improvements) if improvements else 0,
                'improvement_std': np.std(improvements) if len(improvements) > 1 else 0
            }
        }
    
    def _determine_recommendation(
        self,
        behavior_score: float,
        risk_score: float,
        execution_score: float,
        learning_score: float,
        risk_violations: int
    ) -> Dict:
        """Determine final recommendation"""
        avg_score = (behavior_score + risk_score + execution_score + learning_score) / 4
        
        # Critical: Zero tolerance for risk violations
        if risk_violations > 0:
            return {
                'decision': 'REJECT',
                'reason': f'Risk violations detected: {risk_violations}',
                'next_steps': [
                    'Review risk log for violation details',
                    'Fix risk management logic',
                    'Re-run backtest after fixes'
                ]
            }
        
        # High scores across the board
        if avg_score >= 0.8 and behavior_score >= 0.7 and execution_score >= 0.7:
            return {
                'decision': 'PROCEED_TO_LIMITED_LIVE',
                'reason': 'All scorecards passed with high scores',
                'next_steps': [
                    'Begin with paper trading',
                    'Monitor closely for first week',
                    'Gradually increase position sizes'
                ]
            }
        
        # Medium scores or mixed results
        if avg_score >= 0.6:
            return {
                'decision': 'REVISE',
                'reason': f'Mixed results: avg_score={avg_score:.2f}',
                'next_steps': [
                    'Review low-scoring areas',
                    'Make targeted improvements',
                    'Re-run backtest on improvements'
                ],
                'areas_to_improve': self._identify_weak_areas(behavior_score, risk_score, execution_score, learning_score)
            }
        
        # Low scores
        return {
            'decision': 'REJECT',
            'reason': f'Low overall score: {avg_score:.2f}',
            'next_steps': [
                'Major revision required',
                'Review all scorecards',
                'Address fundamental issues before re-testing'
            ]
        }
    
    def _identify_weak_areas(
        self,
        behavior_score: float,
        risk_score: float,
        execution_score: float,
        learning_score: float
    ) -> List[str]:
        """Identify areas that need improvement"""
        areas = []
        threshold = 0.7
        
        if behavior_score < threshold:
            areas.append('Behavior (decision quality, regime consistency)')
        if risk_score < threshold:
            areas.append('Risk (violations, constraint adherence)')
        if execution_score < threshold:
            areas.append('Execution (slippage realism, cost modeling)')
        if learning_score < threshold:
            areas.append('Learning (retraining effectiveness, stability)')
        
        return areas
    
    def _save_verdict(self, verdict: Dict):
        """Save verdict to file"""
        verdict_file = self.log_dir / "end_of_run_verdict.json"
        try:
            with open(verdict_file, 'w') as f:
                json.dump(verdict, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save verdict: {e}")


# Global instance
_verdict_system: Optional[EndOfRunVerdict] = None


def initialize_verdict_system(log_dir: str = "logs") -> EndOfRunVerdict:
    """Initialize global verdict system"""
    global _verdict_system
    _verdict_system = EndOfRunVerdict(log_dir=log_dir)
    return _verdict_system


def get_verdict_system() -> Optional[EndOfRunVerdict]:
    """Get global verdict system instance"""
    return _verdict_system

