"""
INSTITUTIONAL-GRADE LOGGING SYSTEM
Structured logs for 30-day backtest validation
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from collections import defaultdict


class InstitutionalLogger:
    """
    Institutional-grade logging system for backtest validation
    
    Logs:
    - Decision Log (every bar)
    - Risk Log (every trade + risk check)
    - Execution Log (every fill)
    - Position Lifecycle Log
    - Online Learning Log (daily)
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "decisions").mkdir(exist_ok=True)
        (self.log_dir / "risk").mkdir(exist_ok=True)
        (self.log_dir / "execution").mkdir(exist_ok=True)
        (self.log_dir / "positions").mkdir(exist_ok=True)
        (self.log_dir / "learning").mkdir(exist_ok=True)
        (self.log_dir / "feedback").mkdir(exist_ok=True)
        
        # In-memory buffers for current day
        self.decision_buffer: List[Dict] = []
        self.risk_buffer: List[Dict] = []
        self.execution_buffer: List[Dict] = []
        self.position_buffer: Dict[str, Dict] = {}
        
        self.current_date: Optional[str] = None
    
    def _get_log_file(self, category: str, date: str) -> Path:
        """Get log file path for category and date"""
        return self.log_dir / category / f"{date}.jsonl"
    
    def _append_log(self, category: str, log_entry: Dict):
        """Append log entry to file"""
        date = log_entry.get('timestamp', datetime.now().isoformat())[:10]  # Extract date
        
        log_file = self._get_log_file(category, date)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Update daily index (for quick metadata lookup)
        self._update_daily_index(category, date)
    
    def _update_daily_index(self, category: str, date: str):
        """Update daily index metadata (counts per day)"""
        index_file = self.log_dir / f"{category}_index.json"
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {}
            
            if date not in index:
                index[date] = {'count': 0, 'last_updated': datetime.now().isoformat()}
            
            index[date]['count'] += 1
            index[date]['last_updated'] = datetime.now().isoformat()
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass  # Don't fail on index updates
    
    def log_decision(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
        regime: str,
        time_to_expiry_min: int,
        action_final: str,
        confidence_final: float,
        rl_action: str,
        rl_confidence: float,
        ensemble_action: str,
        ensemble_confidence: float,
        agent_votes: Dict[str, str],
        action_scores: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log decision (every bar, even if HOLD)
        
        Args:
            timestamp: Decision timestamp
            symbol: Trading symbol
            price: Current price
            regime: Market regime
            time_to_expiry_min: Minutes to expiration
            action_final: Final action (HOLD/BUY_CALL/BUY_PUT)
            confidence_final: Final confidence
            rl_action: RL agent action
            rl_confidence: RL confidence
            ensemble_action: Ensemble action
            ensemble_confidence: Ensemble confidence
            agent_votes: Individual agent votes
            action_scores: Action scores breakdown
        """
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "price": price,
            "regime": regime,
            "time_to_expiry_min": time_to_expiry_min,
            "action_final": action_final,
            "confidence_final": confidence_final,
            "rl_action": rl_action,
            "rl_confidence": rl_confidence,
            "ensemble_action": ensemble_action,
            "ensemble_confidence": ensemble_confidence,
            "agent_votes": agent_votes,
            "action_scores": action_scores or {},
            "metadata": metadata or {}
        }
        
        self._append_log("decisions", log_entry)
        self.decision_buffer.append(log_entry)
    
    def log_risk_check(
        self,
        timestamp: datetime,
        symbol: str,
        portfolio_delta: float,
        portfolio_gamma: float,
        portfolio_theta: float,
        portfolio_vega: float,
        gamma_limit: float,
        delta_limit: float,
        risk_action: str,  # ALLOW or BLOCK
        risk_reason: Optional[str] = None,
        proposed_trade: Optional[Dict] = None
    ):
        """
        Log risk check (every trade + every risk check)
        
        Args:
            timestamp: Check timestamp
            symbol: Trading symbol
            portfolio_delta: Current portfolio delta
            portfolio_gamma: Current portfolio gamma
            portfolio_theta: Current portfolio theta
            portfolio_vega: Current portfolio vega
            gamma_limit: Gamma limit
            delta_limit: Delta limit
            risk_action: ALLOW or BLOCK
            risk_reason: Reason if blocked
            proposed_trade: Proposed trade details
        """
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "portfolio_delta": portfolio_delta,
            "portfolio_gamma": portfolio_gamma,
            "portfolio_theta": portfolio_theta,
            "portfolio_vega": portfolio_vega,
            "gamma_limit": gamma_limit,
            "delta_limit": delta_limit,
            "risk_action": risk_action,
            "risk_reason": risk_reason,
            "proposed_trade": proposed_trade
        }
        
        self._append_log("risk", log_entry)
        self.risk_buffer.append(log_entry)
    
    def log_execution(
        self,
        timestamp: datetime,
        symbol: str,
        order_type: str,
        mid_price: float,
        fill_price: float,
        spread: float,
        slippage_pct: float,
        qty: int,
        gamma_impact: Optional[float] = None,
        iv_crush_impact: Optional[float] = None,
        theta_impact: Optional[float] = None,
        liquidity_factor: Optional[float] = None,
        execution_details: Optional[Dict] = None
    ):
        """
        Log execution (every fill)
        
        Args:
            timestamp: Execution timestamp
            symbol: Trading symbol
            order_type: BUY_CALL/BUY_PUT/SELL
            mid_price: Midpoint price
            fill_price: Actual fill price
            spread: Bid-ask spread
            slippage_pct: Slippage percentage
            qty: Quantity filled
            gamma_impact: Gamma squeeze impact
            iv_crush_impact: IV collapse impact
            theta_impact: Theta explosion impact
            liquidity_factor: Liquidity factor
            execution_details: Additional execution details
        """
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "order_type": order_type,
            "mid_price": mid_price,
            "fill_price": fill_price,
            "spread": spread,
            "slippage_pct": slippage_pct,
            "qty": qty,
            "gamma_impact": gamma_impact,
            "iv_crush_impact": iv_crush_impact,
            "theta_impact": theta_impact,
            "liquidity_factor": liquidity_factor,
            "execution_details": execution_details or {}
        }
        
        self._append_log("execution", log_entry)
        self.execution_buffer.append(log_entry)
    
    def log_position_entry(
        self,
        trade_id: str,
        timestamp: datetime,
        symbol: str,
        action: str,
        qty: int,
        entry_price: float,
        strike: float,
        premium: float,
        is_probe_trade: bool = False
    ):
        """
        Log position entry
        
        Args:
            trade_id: Unique trade identifier
            timestamp: Entry timestamp
            symbol: Trading symbol
            action: Action type (BUY_CALL/BUY_PUT)
            qty: Quantity
            entry_price: Entry price
            strike: Strike price
            premium: Option premium
            is_probe_trade: Whether this is a probe trade (behavioral testing)
        """
        if trade_id not in self.position_buffer:
            self.position_buffer[trade_id] = {
                "trade_id": trade_id,
                "symbol": symbol,
                "entry_time": timestamp.isoformat(),
                "entry_action": action,
                "entry_qty": qty,
                "entry_price": entry_price,
                "strike": strike,
                "entry_premium": premium,
                "is_probe_trade": is_probe_trade,
                "exit_time": None,
                "exit_reason": None,
                "exit_price": None,
                "exit_premium": None,
                "hold_minutes": None,
                "max_unrealized": None,
                "final_pnl": None,
                "max_unrealized_pct": None,
                "final_pnl_pct": None
            }
    
    def log_position_exit(
        self,
        trade_id: str,
        timestamp: datetime,
        exit_reason: str,
        exit_price: float,
        exit_premium: float,
        final_pnl: float,
        max_unrealized: Optional[float] = None
    ):
        """Log position exit"""
        if trade_id in self.position_buffer:
            position = self.position_buffer[trade_id]
            position["exit_time"] = timestamp.isoformat()
            position["exit_reason"] = exit_reason
            position["exit_price"] = exit_price
            position["exit_premium"] = exit_premium
            position["final_pnl"] = final_pnl
            position["max_unrealized"] = max_unrealized
            
            # Calculate hold time
            entry_time = datetime.fromisoformat(position["entry_time"])
            exit_time = timestamp
            hold_minutes = (exit_time - entry_time).total_seconds() / 60
            position["hold_minutes"] = hold_minutes
            
            # Calculate percentages
            entry_premium = position["entry_premium"]
            if entry_premium > 0:
                position["final_pnl_pct"] = (final_pnl / (entry_premium * position["entry_qty"] * 100)) * 100
                if max_unrealized:
                    position["max_unrealized_pct"] = (max_unrealized / (entry_premium * position["entry_qty"] * 100)) * 100
            
            # Write to file
            self._append_log("positions", position)
            
            # Remove from buffer
            del self.position_buffer[trade_id]
    
    def log_learning_event(
        self,
        date: str,
        regime: str,
        retrained: bool,
        model_candidate: Optional[str] = None,
        production_model: Optional[str] = None,
        sharpe_candidate: Optional[float] = None,
        sharpe_prod: Optional[float] = None,
        promotion: bool = False,
        retrain_reason: Optional[str] = None
    ):
        """
        Log online learning event (daily)
        
        Args:
            date: Date (YYYY-MM-DD)
            regime: Market regime
            retrained: Whether model was retrained
            model_candidate: New model version ID
            production_model: Current production model ID
            sharpe_candidate: Candidate Sharpe ratio
            sharpe_prod: Production Sharpe ratio
            promotion: Whether candidate was promoted
            retrain_reason: Reason for retraining
        """
        log_entry = {
            "date": date,
            "regime": regime,
            "retrained": retrained,
            "model_candidate": model_candidate,
            "production_model": production_model,
            "sharpe_candidate": sharpe_candidate,
            "sharpe_prod": sharpe_prod,
            "promotion": promotion,
            "retrain_reason": retrain_reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self._append_log("learning", log_entry)
    
    def log_feedback(
        self,
        date: str,
        reviewer: str,
        comment: str,
        severity: str = "MEDIUM",  # LOW, MEDIUM, HIGH
        category: str = "BEHAVIOR",  # BEHAVIOR, RISK, EXECUTION, LEARNING
        trade_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Log human feedback/review
        
        Args:
            date: Date (YYYY-MM-DD)
            reviewer: Reviewer name
            comment: Feedback comment
            severity: LOW, MEDIUM, HIGH
            category: BEHAVIOR, RISK, EXECUTION, LEARNING
            trade_id: Associated trade ID (if applicable)
            timestamp: Specific timestamp (if applicable)
        """
        log_entry = {
            "date": date,
            "reviewer": reviewer,
            "comment": comment,
            "severity": severity,
            "category": category,
            "trade_id": trade_id,
            "timestamp": timestamp.isoformat() if timestamp else datetime.now().isoformat()
        }
        
        self._append_log("feedback", log_entry)
    
    def flush_buffers(self):
        """Flush all buffers (call at end of day)"""
        # Buffers are already written, just clear them
        self.decision_buffer.clear()
        self.risk_buffer.clear()
        self.execution_buffer.clear()
    
    def get_logs(
        self,
        category: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get logs with filters
        
        Args:
            category: decisions/risk/execution/positions/learning/feedback
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Filter by symbol
            filters: Additional filters
            
        Returns:
            List of log entries
        """
        logs = []
        
        # Determine date range
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            # Get all available dates
            category_dir = self.log_dir / category
            if not category_dir.exists():
                return []
            dates = [f.stem for f in category_dir.glob("*.jsonl")]
        
        # Load logs
        for date in dates:
            log_file = self._get_log_file(category, date)
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Apply filters
                            if symbol and entry.get('symbol') != symbol:
                                continue
                            
                            if filters:
                                for key, value in filters.items():
                                    if entry.get(key) != value:
                                        break
                                else:
                                    continue
                            
                            logs.append(entry)
                        except json.JSONDecodeError:
                            continue
        
        return logs
    
    def generate_daily_summary(self, date: str) -> Dict:
        """Generate daily summary from logs"""
        decisions = self.get_logs("decisions", start_date=date, end_date=date)
        risk_checks = self.get_logs("risk", start_date=date, end_date=date)
        executions = self.get_logs("execution", start_date=date, end_date=date)
        positions = self.get_logs("positions", start_date=date, end_date=date)
        learning = self.get_logs("learning", start_date=date, end_date=date)
        
        # Calculate statistics
        total_decisions = len(decisions)
        hold_count = sum(1 for d in decisions if d.get('action_final') == 'HOLD')
        buy_count = sum(1 for d in decisions if 'BUY' in d.get('action_final', ''))
        
        risk_blocks = sum(1 for r in risk_checks if r.get('risk_action') == 'BLOCK')
        risk_allows = sum(1 for r in risk_checks if r.get('risk_action') == 'ALLOW')
        
        total_executions = len(executions)
        avg_slippage = np.mean([e.get('slippage_pct', 0) for e in executions]) if executions else 0.0
        
        total_positions = len(positions)
        winning_trades = sum(1 for p in positions if p.get('final_pnl', 0) > 0)
        win_rate = (winning_trades / total_positions * 100) if total_positions > 0 else 0.0
        
        # Ensemble override rate
        ensemble_overrides = sum(1 for d in decisions 
                                if d.get('rl_action') != d.get('ensemble_action') 
                                and d.get('action_final') == d.get('ensemble_action'))
        override_rate = (ensemble_overrides / total_decisions * 100) if total_decisions > 0 else 0.0
        
        # Regime distribution
        regimes = defaultdict(int)
        for d in decisions:
            regimes[d.get('regime', 'UNKNOWN')] += 1
        
        summary = {
            "date": date,
            "decisions": {
                "total": total_decisions,
                "hold": hold_count,
                "buy": buy_count,
                "ensemble_override_rate_pct": override_rate
            },
            "risk": {
                "total_checks": len(risk_checks),
                "blocks": risk_blocks,
                "allows": risk_allows,
                "block_rate_pct": (risk_blocks / len(risk_checks) * 100) if risk_checks else 0.0
            },
            "execution": {
                "total_fills": total_executions,
                "avg_slippage_pct": avg_slippage
            },
            "positions": {
                "total": total_positions,
                "winning": winning_trades,
                "win_rate_pct": win_rate
            },
            "learning": learning[-1] if learning else None,
            "regime_distribution": dict(regimes)
        }
        
        return summary


# Global instance
_institutional_logger: Optional[InstitutionalLogger] = None


def initialize_logger(log_dir: str = "logs") -> InstitutionalLogger:
    """Initialize global institutional logger"""
    global _institutional_logger
    _institutional_logger = InstitutionalLogger(log_dir=log_dir)
    return _institutional_logger


def get_logger() -> Optional[InstitutionalLogger]:
    """Get global institutional logger instance"""
    return _institutional_logger

