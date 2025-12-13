"""
TRADE BLOCK REASON AGGREGATOR
Instruments WHY trades were blocked for diagnostics
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path


class TradeBlockAggregator:
    """
    Aggregates trade block reasons to diagnose over-constrained policies
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.daily_blocks: Dict[str, Dict[str, int]] = {}  # date -> {reason: count}
        self.total_blocks: Dict[str, int] = defaultdict(int)
    
    def log_block(
        self,
        date: str,
        reason: str,
        symbol: Optional[str] = None
    ):
        """
        Log a trade block
        
        Args:
            date: Date (YYYY-MM-DD)
            reason: Block reason (e.g., "GAMMA_LIMIT", "MACRO_RISK_OFF", "ENSEMBLE_DISAGREEMENT")
            symbol: Trading symbol (optional)
        """
        if date not in self.daily_blocks:
            self.daily_blocks[date] = defaultdict(int)
        
        self.daily_blocks[date][reason] += 1
        self.total_blocks[reason] += 1
    
    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """
        Get daily block summary
        
        Returns:
            {
                "total_blocks": int,
                "by_reason": {reason: count},
                "percentages": {reason: percentage}
            }
        """
        if date not in self.daily_blocks:
            return {
                "total_blocks": 0,
                "by_reason": {},
                "percentages": {}
            }
        
        blocks = self.daily_blocks[date]
        total = sum(blocks.values())
        
        percentages = {
            reason: (count / total * 100) if total > 0 else 0
            for reason, count in blocks.items()
        }
        
        return {
            "total_blocks": total,
            "by_reason": dict(blocks),
            "percentages": percentages
        }
    
    def get_period_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get period summary (e.g., 30-day backtest)
        
        Returns:
            {
                "total_blocks": int,
                "by_reason": {reason: count},
                "percentages": {reason: percentage},
                "daily_average": float
            }
        """
        period_blocks = defaultdict(int)
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        days_count = 0
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            if date_str in self.daily_blocks:
                for reason, count in self.daily_blocks[date_str].items():
                    period_blocks[reason] += count
                days_count += 1
            current = current.replace(day=current.day + 1) if current.day < 28 else current.replace(month=current.month + 1, day=1)
            if current > end:
                break
        
        total = sum(period_blocks.values())
        percentages = {
            reason: (count / total * 100) if total > 0 else 0
            for reason, count in period_blocks.items()
        }
        
        daily_avg = total / days_count if days_count > 0 else 0
        
        return {
            "total_blocks": total,
            "by_reason": dict(period_blocks),
            "percentages": percentages,
            "daily_average": daily_avg,
            "days_analyzed": days_count
        }
    
    def print_daily_summary(self, date: str):
        """Print formatted daily summary"""
        summary = self.get_daily_summary(date)
        
        if summary["total_blocks"] == 0:
            print(f"   📊 Trade Blocks Summary ({date}): No blocks recorded")
            return
        
        print(f"\n   📊 Trade Blocks Summary ({date}):")
        print(f"      Total blocks: {summary['total_blocks']}")
        print(f"      By reason:")
        
        # Sort by count (descending)
        sorted_reasons = sorted(
            summary["by_reason"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons:
            pct = summary["percentages"].get(reason, 0)
            print(f"        - {reason}: {count} ({pct:.1f}%)")
    
    def print_period_summary(self, start_date: str, end_date: str):
        """Print formatted period summary"""
        summary = self.get_period_summary(start_date, end_date)
        
        if summary["total_blocks"] == 0:
            print(f"\n📊 Trade Blocks Summary ({start_date} to {end_date}): No blocks recorded")
            return
        
        print(f"\n📊 Trade Blocks Summary ({start_date} to {end_date}):")
        print(f"   Total blocks: {summary['total_blocks']}")
        print(f"   Daily average: {summary['daily_average']:.1f}")
        print(f"   Days analyzed: {summary['days_analyzed']}")
        print(f"   By reason:")
        
        # Sort by count (descending)
        sorted_reasons = sorted(
            summary["by_reason"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for reason, count in sorted_reasons:
            pct = summary["percentages"].get(reason, 0)
            print(f"     - {reason}: {count} ({pct:.1f}%)")
    
    def save_summary(self, start_date: str, end_date: str, filepath: Optional[str] = None):
        """Save period summary to file"""
        summary = self.get_period_summary(start_date, end_date)
        
        if filepath is None:
            filepath = self.log_dir / "trade_blocks_summary.json"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


# Global instance
_block_aggregator: Optional[TradeBlockAggregator] = None


def initialize_block_aggregator(log_dir: str = "logs") -> TradeBlockAggregator:
    """Initialize global block aggregator"""
    global _block_aggregator
    _block_aggregator = TradeBlockAggregator(log_dir=log_dir)
    return _block_aggregator


def get_block_aggregator() -> Optional[TradeBlockAggregator]:
    """Get global block aggregator instance"""
    return _block_aggregator

