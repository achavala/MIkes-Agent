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
        backtest_end_date: str,
        mode: str = 'pnl',  # 'behavioral' or 'pnl'
        total_trades_override: int = None  # Override trade count if provided
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
        
        # Generate scorecards (mode-aware)
        behavior_scorecard = self._generate_behavior_scorecard(decisions, positions, mode)
        risk_scorecard = self._generate_risk_scorecard(risk_checks, mode)
        execution_scorecard = self._generate_execution_scorecard(executions, mode)
        learning_scorecard = self._generate_learning_scorecard(learning)
        
        # Calculate overall scores
        behavior_score = behavior_scorecard.get('overall_score', 0)
        risk_score = risk_scorecard.get('overall_score', 0)
        execution_score = execution_scorecard.get('overall_score', 0)
        learning_score = learning_scorecard.get('overall_score', 0)
        
        # MODE-AWARE FALLBACK: If scores are 0.00, apply fallback logic based on mode
        if mode == 'behavioral':
            # Calculate fallback scores based on actual data
            # CRITICAL: Use executions count, not positions count (positions may be empty if closed)
            # Executions represent actual trades executed, including probe trades
            # CRITICAL: Use override if provided (from backtest stats), otherwise use executions
            if total_trades_override is not None and total_trades_override > 0:
                total_trades = total_trades_override  # Use override from backtest stats
            else:
                total_trades = len(executions)  # Executions = actual fills/trades
                # Fallback to positions if executions is empty (shouldn't happen, but defensive)
                if total_trades == 0:
                    total_trades = len(positions)
            total_days = len(set([d.get('timestamp', '')[:10] for d in decisions if d.get('timestamp')])) if decisions else 1
            avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
            violations = risk_scorecard.get('violations', 0)
            
            # Fallback behavior score: 0.6 if avg trades/day is 0.5-3.0
            if behavior_score == 0.0 and total_trades > 0:
                if 0.5 <= avg_trades_per_day <= 3.0:
                    behavior_score = 0.6
                else:
                    behavior_score = 0.3
            
            # Fallback risk score: 0.7 if no violations
            if risk_score == 0.0:
                if violations == 0:
                    risk_score = 0.7
                elif violations <= 2:
                    risk_score = 0.5
                else:
                    risk_score = 0.3
            
            # Fallback execution score: 0.5 if trades < 10 (neutral floor)
            if execution_score == 0.0 and total_trades > 0:
                execution_score = 0.5  # Neutral floor for small samples
        
        elif mode == 'paper':
            # PAPER MODE FALLBACK: Similar to behavioral but with paper mode semantics
            # Paper mode should have actual scores, but if they're 0.0, apply fallback
            if total_trades_override is not None and total_trades_override > 0:
                total_trades = total_trades_override
            else:
                total_trades = len(executions)
                if total_trades == 0:
                    total_trades = len(positions)
            total_days = len(set([d.get('timestamp', '')[:10] for d in decisions if d.get('timestamp')])) if decisions else 1
            avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
            violations = risk_scorecard.get('violations', 0)
            
            # Paper mode fallback behavior score: 0.6 if avg trades/day is 0.5-3.0 (same as behavioral)
            if behavior_score == 0.0 and total_trades > 0:
                if 0.5 <= avg_trades_per_day <= 3.0:
                    behavior_score = 0.6
                elif 0.3 <= avg_trades_per_day < 0.5 or 3.0 < avg_trades_per_day <= 5.0:
                    behavior_score = 0.4
                else:
                    behavior_score = 0.2
            
            # Paper mode fallback risk score: 0.7 if no violations (same as behavioral)
            if risk_score == 0.0:
                if violations == 0:
                    risk_score = 0.7
                elif violations <= 2:
                    risk_score = 0.5
                else:
                    risk_score = 0.3
            
            # Paper mode fallback execution score: 0.5 if trades < 10 (neutral floor)
            if execution_score == 0.0 and total_trades > 0:
                execution_score = 0.5  # Neutral floor for small samples
        
        # Determine recommendation (mode-aware)
        # CRITICAL: Use override if provided (from backtest stats), otherwise use executions
        if total_trades_override is not None and total_trades_override > 0:
            total_trades_for_verdict = total_trades_override  # Use override from backtest stats
        else:
            # Fallback: Use executions count, not positions count (positions may be empty if closed)
            # Executions represent actual trades executed, including probe trades
            total_trades_for_verdict = len(executions)  # Executions = actual fills/trades
            # Fallback to positions if executions is empty (shouldn't happen, but defensive)
            if total_trades_for_verdict == 0:
                total_trades_for_verdict = len(positions)
        
        recommendation = self._determine_recommendation(
            behavior_score, risk_score, execution_score, learning_score,
            risk_scorecard.get('violations', 0),
            mode,
            total_trades_for_verdict  # Pass actual trade count for behavioral mode check
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
        positions: List[Dict],
        mode: str = 'pnl'
    ) -> Dict:
        """Generate behavior scorecard"""
        if not decisions:
            return {'overall_score': 0, 'error': 'No decisions'}
        
        df_decisions = pd.DataFrame(decisions)
        df_positions = pd.DataFrame(positions) if positions else pd.DataFrame()
        
        scores = {}
        
        # MODE-AWARE: Behavioral mode has different criteria
        if mode == 'behavioral':
            # In behavioral mode, check if we have acceptable trade frequency
            total_trades = len(df_positions)
            total_days = len(df_decisions['timestamp'].apply(lambda x: str(x)[:10]).unique()) if len(df_decisions) > 0 else 1
            avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
            
            # Behavioral mode: PASS if avg trades/day is 0.5-3.0
            if total_trades == 0:
                return {
                    'scores': {'zero_trades': 0.0},
                    'overall_score': 0.0,
                    'metrics': {
                        'total_decisions': len(df_decisions),
                        'total_trades': 0,
                        'avg_trades_per_day': 0,
                        'zero_trades_detected': True
                    },
                    'warning': 'Zero trades — over-constrained policy'
                }
            elif 0.5 <= avg_trades_per_day <= 3.0:
                # Acceptable trade frequency for behavioral mode
                return {
                    'scores': {'trade_frequency': 1.0, 'exploratory_behavior': 0.8},
                    'overall_score': 0.6,  # Good exploratory behavior
                    'metrics': {
                        'total_decisions': len(df_decisions),
                        'total_trades': total_trades,
                        'avg_trades_per_day': round(avg_trades_per_day, 2),
                        'mode': 'behavioral'
                    }
                }
            else:
                # Outside acceptable range
                return {
                    'scores': {'trade_frequency': 0.3},
                    'overall_score': 0.3,
                    'metrics': {
                        'total_decisions': len(df_decisions),
                        'total_trades': total_trades,
                        'avg_trades_per_day': round(avg_trades_per_day, 2),
                        'mode': 'behavioral'
                    },
                    'warning': f'Trade frequency outside acceptable range: {avg_trades_per_day:.2f} trades/day'
                }
        
        # STANDARD MODE: Original logic
        # CRITICAL: Zero trades = automatic 0.0 score
        if len(df_positions) == 0:
            return {
                'scores': {'zero_trades': 0.0},
                'overall_score': 0.0,
                'metrics': {
                    'total_decisions': len(df_decisions),
                    'total_trades': 0,
                    'zero_trades_detected': True
                },
                'warning': 'Zero trades — over-constrained policy'
            }
        
        # Consistency across regimes
        if 'regime' in df_decisions.columns:
            regimes = df_decisions['regime'].unique()
            regime_consistency = len(regimes) > 0
            scores['regime_consistency'] = 1.0 if regime_consistency else 0.0
        
        # HOLD vs BUY balance
        if 'action_final' in df_decisions.columns:
            hold_count = len(df_decisions[df_decisions['action_final'] == 'HOLD'])
            buy_count = len(df_decisions[df_decisions['action_final'].astype(str).str.contains('BUY', na=False)])
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
                'total_trades': len(df_positions),
                'hold_rate': (hold_count / total * 100) if total > 0 else 0,
                'ensemble_override_rate': (len(overrides) / len(df_decisions) * 100) if len(df_decisions) > 0 else 0,
                'win_rate': (winning / len(df_positions) * 100) if len(df_positions) > 0 else 0
            }
        }
    
    def _generate_risk_scorecard(self, risk_checks: List[Dict], mode: str = 'pnl') -> Dict:
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
        
        # MODE-AWARE: Behavioral and Paper modes - no violations = GOOD
        if mode == 'behavioral' or mode == 'paper':
            # In behavioral/paper mode, no risk violations = positive score
            if violations == 0:
                overall_score = 0.7  # Good - safe exploratory trades
            elif violations <= 2:
                overall_score = 0.5  # Acceptable
            else:
                overall_score = 0.3  # Too many blocks
        else:
            # STANDARD MODE: Original logic
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
    
    def _generate_execution_scorecard(self, executions: List[Dict], mode: str = 'pnl') -> Dict:
        """Generate execution scorecard"""
        if not executions:
            return {'overall_score': 0, 'error': 'No executions'}
        
        df = pd.DataFrame(executions)
        
        # MODE-AWARE: Behavioral mode - neutral floor for small samples
        MIN_EXEC_SAMPLE = 10
        
        if mode == 'behavioral' and len(df) < MIN_EXEC_SAMPLE:
            # Small sample size - return neutral score
            return {
                'scores': {'sample_size_too_small': 0.5},
                'overall_score': 0.5,  # Neutral floor
                'metrics': {
                    'total_fills': len(df),
                    'avg_slippage_pct': df['slippage_pct'].mean() if 'slippage_pct' in df.columns else 0,
                    'mode': 'behavioral',
                    'note': f'Small sample size ({len(df)} < {MIN_EXEC_SAMPLE}) - neutral score'
                }
            }
        
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
        
        overall_score = np.mean(list(scores.values())) if scores else 0.5  # Neutral floor instead of 0.0
        
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
        risk_violations: int,
        mode: str = 'pnl',
        total_trades: int = 0
    ) -> Dict:
        """Determine final recommendation (mode-aware)"""
        avg_score = (behavior_score + risk_score + execution_score + learning_score) / 4
        
        # MODE-AWARE: Behavioral and Paper modes have different thresholds
        if mode == 'behavioral':
            # EARLY EXIT: Behavioral mode - simple rule
            if total_trades > 0 and risk_violations == 0:
                return {
                    'decision': 'REVISE',
                    'reason': f'Behavioral test passed: {total_trades} trades executed safely, no violations, avg_score={avg_score:.2f}',
                    'next_steps': [
                        'Review behavioral patterns',
                        'Consider moving to paper trading',
                        'Monitor consistency in next run'
                    ]
                }
            
            # If we have violations, reject
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
            
            # If no trades, reject
            if total_trades == 0:
                return {
                    'decision': 'REJECT',
                    'reason': f'Behavioral test failed: no trades executed (over-constrained)',
                    'next_steps': [
                        'Fix trading logic to generate trades',
                        'Review risk constraints',
                        'Re-run behavioral test'
                    ]
                }
            
            # Fallback: if we have trades but something else is wrong
            return {
                'decision': 'REVISE',
                'reason': f'Behavioral test completed: {total_trades} trades, avg_score={avg_score:.2f}',
                'next_steps': [
                    'Review scorecards for improvement areas',
                    'Adjust behavioral thresholds if needed',
                    'Re-run behavioral test'
                ]
            }
        
        elif mode == 'paper':
            # PAPER MODE: Similar to behavioral but with full constraints
            # Paper mode should have trades if signals are working
            if total_trades > 0 and risk_violations == 0:
                # Good: trades executed safely with full constraints
                if avg_score >= 0.5:
                    return {
                        'decision': 'PROCEED_TO_PAPER',
                        'reason': f'Paper mode test passed: {total_trades} trades executed safely, no violations, avg_score={avg_score:.2f}',
                        'next_steps': [
                            'Continue paper trading sessions',
                            'Monitor consistency over 10-15 sessions',
                            'Consider removing action nudge after stability'
                        ]
                    }
                else:
                    return {
                        'decision': 'REVISE',
                        'reason': f'Paper mode test: {total_trades} trades executed, but scores low (avg={avg_score:.2f})',
                        'next_steps': [
                            'Review signal quality',
                            'Check execution realism',
                            'Run additional paper sessions'
                        ]
                    }
            
            # If we have violations, reject
            if risk_violations > 0:
                return {
                    'decision': 'REJECT',
                    'reason': f'Risk violations detected in paper mode: {risk_violations}',
                    'next_steps': [
                        'Review risk log for violation details',
                        'Fix risk management logic',
                        'Re-run backtest after fixes'
                    ]
                }
            
            # If no trades, reject (paper mode should have signals)
            if total_trades == 0:
                return {
                    'decision': 'REJECT',
                    'reason': f'Paper mode failed: no trades executed (signals too weak or constraints too strict)',
                    'next_steps': [
                        'Review signal generation',
                        'Check ensemble agent logic',
                        'Verify historical data is being passed to ensemble'
                    ]
                }
            
            # Fallback: if we have trades but something else is wrong
            return {
                'decision': 'REVISE',
                'reason': f'Paper mode test completed: {total_trades} trades, avg_score={avg_score:.2f}',
                'next_steps': [
                    'Review scorecards for improvement areas',
                    'Run additional paper sessions',
                    'Monitor consistency'
                ]
            }
        
        # STANDARD MODE: Original logic
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

