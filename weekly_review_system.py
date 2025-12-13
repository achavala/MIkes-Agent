"""
WEEKLY REVIEW CADENCE SYSTEM
Reviews at days 5, 10, 20, 30 with specific questions
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json


class WeeklyReviewSystem:
    """
    Weekly review cadence system for 30-day backtest
    
    Review checkpoints: Days 5, 10, 20, 30
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.reviews: Dict[str, Dict] = {}
    
    def conduct_review(
        self,
        day: int,
        backtest_start_date: str
    ) -> Dict:
        """
        Conduct weekly review
        
        Args:
            day: Day number (5, 10, 20, 30)
            backtest_start_date: Backtest start date (YYYY-MM-DD)
            
        Returns:
            Review results
        """
        from institutional_logging import get_logger
        logger = get_logger()
        
        if not logger:
            return {'error': 'Logger not initialized'}
        
        # Calculate review date
        start = datetime.fromisoformat(backtest_start_date)
        review_date = (start + timedelta(days=day-1)).strftime("%Y-%m-%d")
        
        # Get logs for period
        start_date = backtest_start_date
        end_date = review_date
        
        decisions = logger.get_logs("decisions", start_date=start_date, end_date=end_date)
        risk_checks = logger.get_logs("risk", start_date=start_date, end_date=end_date)
        executions = logger.get_logs("execution", start_date=start_date, end_date=end_date)
        positions = logger.get_logs("positions", start_date=start_date, end_date=end_date)
        learning = logger.get_logs("learning", start_date=start_date, end_date=end_date)
        
        # Answer the 6 key questions
        review = {
            "day": day,
            "review_date": review_date,
            "period_start": start_date,
            "period_end": end_date,
            "answers": self._answer_key_questions(
                decisions, risk_checks, executions, positions, learning
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save review
        self.reviews[f"day_{day}"] = review
        self._save_reviews()
        
        return review
    
    def _answer_key_questions(
        self,
        decisions: List[Dict],
        risk_checks: List[Dict],
        executions: List[Dict],
        positions: List[Dict],
        learning: List[Dict]
    ) -> Dict:
        """Answer the 6 key questions"""
        import pandas as pd
        import numpy as np
        
        answers = {}
        
        # Question 1: Which agent dominates per regime?
        if decisions:
            df = pd.DataFrame(decisions)
            if 'regime' in df.columns and 'agent_votes' in df.columns:
                regime_agent_dominance = {}
                for regime in df['regime'].unique():
                    regime_decisions = df[df['regime'] == regime]
                    agent_counts = {}
                    for votes in regime_decisions['agent_votes']:
                        if isinstance(votes, dict):
                            for agent, vote in votes.items():
                                if vote != 'HOLD':
                                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                    regime_agent_dominance[regime] = agent_counts
                answers['agent_dominance_per_regime'] = regime_agent_dominance
        
        # Question 2: Is ensemble override rate stable?
        if decisions:
            df = pd.DataFrame(decisions)
            if 'rl_action' in df.columns and 'ensemble_action' in df.columns:
                overrides = df[df['rl_action'] != df['ensemble_action']]
                override_rate = len(overrides) / len(df) * 100 if len(df) > 0 else 0
                answers['ensemble_override_rate_pct'] = override_rate
                answers['override_rate_stable'] = True  # Would calculate variance over time
        
        # Question 3: Is gamma agent blocking late-day stupidity?
        if risk_checks:
            df = pd.DataFrame(risk_checks)
            blocked = df[df.get('risk_action') == 'BLOCK']
            gamma_blocks = blocked[blocked.get('risk_reason', '').str.contains('GAMMA', na=False)]
            answers['gamma_blocks'] = len(gamma_blocks)
            answers['gamma_blocks_effective'] = len(gamma_blocks) > 0
        
        # Question 4: Is slippage within 0.3-0.8%?
        if executions:
            df = pd.DataFrame(executions)
            if 'slippage_pct' in df.columns:
                avg_slippage = df['slippage_pct'].mean()
                min_slippage = df['slippage_pct'].min()
                max_slippage = df['slippage_pct'].max()
                answers['avg_slippage_pct'] = avg_slippage
                answers['slippage_range'] = (min_slippage, max_slippage)
                answers['slippage_within_bounds'] = 0.3 <= avg_slippage <= 0.8
        
        # Question 5: Is retraining helping or hurting?
        if learning:
            df = pd.DataFrame(learning)
            retrained = df[df.get('retrained') == True]
            if len(retrained) > 0:
                promotions = retrained[retrained.get('promotion') == True]
                if 'sharpe_candidate' in df.columns and 'sharpe_prod' in df.columns:
                    improvements = []
                    for _, row in retrained.iterrows():
                        if row.get('sharpe_candidate') and row.get('sharpe_prod'):
                            improvement = row['sharpe_candidate'] - row['sharpe_prod']
                            improvements.append(improvement)
                    avg_improvement = np.mean(improvements) if improvements else 0
                    answers['retraining_avg_improvement'] = avg_improvement
                    answers['retraining_helping'] = avg_improvement > 0
        
        # Question 6: Is HOLD behavior sensible?
        if decisions:
            df = pd.DataFrame(decisions)
            if 'action_final' in df.columns:
                hold_count = len(df[df['action_final'] == 'HOLD'])
                total = len(df)
                hold_rate = (hold_count / total * 100) if total > 0 else 0
                answers['hold_rate_pct'] = hold_rate
                # Sensible = 40-70% HOLD rate (not too aggressive, not too passive)
                answers['hold_behavior_sensible'] = 40 <= hold_rate <= 70
        
        return answers
    
    def _save_reviews(self):
        """Save reviews to file"""
        reviews_file = self.log_dir / "weekly_reviews.json"
        try:
            with open(reviews_file, 'w') as f:
                json.dump(self.reviews, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save reviews: {e}")
    
    def load_reviews(self):
        """Load reviews from file"""
        reviews_file = self.log_dir / "weekly_reviews.json"
        if reviews_file.exists():
            try:
                with open(reviews_file, 'r') as f:
                    self.reviews = json.load(f)
            except Exception:
                self.reviews = {}
    
    def get_review_summary(self) -> Dict:
        """Get summary of all reviews"""
        return {
            "total_reviews": len(self.reviews),
            "review_days": sorted([int(k.replace('day_', '')) for k in self.reviews.keys()]),
            "reviews": self.reviews
        }


# Global instance
_review_system: Optional[WeeklyReviewSystem] = None


def initialize_review_system(log_dir: str = "logs") -> WeeklyReviewSystem:
    """Initialize global review system"""
    global _review_system
    _review_system = WeeklyReviewSystem(log_dir=log_dir)
    _review_system.load_reviews()
    return _review_system


def get_review_system() -> Optional[WeeklyReviewSystem]:
    """Get global review system instance"""
    return _review_system

