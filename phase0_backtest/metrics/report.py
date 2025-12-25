"""
Phase 0 Metrics & Reporting

Generates detailed analysis of Phase 0 backtest results.
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class Phase0Report:
    """
    Phase 0 Backtest Report Generator
    """
    
    def __init__(self, results: dict):
        """
        Initialize report generator
        
        Args:
            results: Backtest results dictionary
        """
        self.results = results
    
    def generate_summary(self) -> str:
        """Generate summary report"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 0 BACKTEST - SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic stats
        report.append(f"Period: {self.results['start_date']} to {self.results['end_date']}")
        report.append(f"Trading Days: {self.results['trading_days']}")
        report.append(f"Total Trades: {self.results['total_trades']}")
        report.append(f"Total Rejections: {self.results['total_rejections']}")
        report.append("")
        
        # Daily summaries
        if self.results['daily_summaries']:
            report.append("DAILY BREAKDOWN:")
            report.append("-" * 80)
            for daily in self.results['daily_summaries']:
                report.append(f"  {daily['date']}: "
                             f"Trades={daily['trades_taken']}, "
                             f"P&L=${daily['total_pnl']:.2f}, "
                             f"Halted={daily['trading_halted']}")
            report.append("")
        
        # Trade analysis
        if self.results['trade_log']:
            report.append("TRADE ANALYSIS:")
            report.append("-" * 80)
            trades_df = pd.DataFrame(self.results['trade_log'])
            
            # Group by symbol
            if 'symbol' in trades_df.columns:
                symbol_counts = trades_df['symbol'].value_counts()
                report.append("Trades by Symbol:")
                for symbol, count in symbol_counts.items():
                    report.append(f"  {symbol}: {count}")
                report.append("")
            
            # Group by action
            if 'action' in trades_df.columns:
                action_counts = trades_df['action'].value_counts()
                report.append("Trades by Action:")
                for action, count in action_counts.items():
                    report.append(f"  {action}: {count}")
                report.append("")
        
        # Rejection analysis
        if self.results['rejection_log']:
            report.append("REJECTION ANALYSIS:")
            report.append("-" * 80)
            rejections_df = pd.DataFrame(self.results['rejection_log'])
            
            if 'reason' in rejections_df.columns:
                reason_counts = rejections_df['reason'].value_counts()
                report.append("Rejections by Reason:")
                for reason, count in reason_counts.items():
                    report.append(f"  {reason}: {count}")
                report.append("")
        
        return "\n".join(report)
    
    def generate_detailed_analysis(self) -> str:
        """Generate detailed trade-by-trade analysis"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 0 BACKTEST - DETAILED TRADE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Trade-by-trade breakdown
        if self.results['trade_log']:
            report.append("TRADE-BY-TRADE BREAKDOWN:")
            report.append("-" * 80)
            
            for i, trade in enumerate(self.results['trade_log'], 1):
                report.append(f"\nTrade #{i}:")
                report.append(f"  Date: {trade.get('date', 'N/A')}")
                report.append(f"  Time: {trade.get('time', 'N/A')}")
                report.append(f"  Symbol: {trade.get('symbol', 'N/A')}")
                report.append(f"  Option: {trade.get('option_symbol', 'N/A')}")
                report.append(f"  Action: {trade.get('action', 'N/A')}")
                report.append(f"  Strike: ${trade.get('strike', 0):.2f}")
                report.append(f"  Entry Price: ${trade.get('entry_price', 0):.2f}")
                report.append(f"  Entry Premium: ${trade.get('entry_premium', 0):.2f}")
                report.append(f"  Quantity: {trade.get('qty', 0)}")
                report.append(f"  Confidence: {trade.get('confidence', 0):.3f}")
                report.append(f"  VIX: {trade.get('vix', 0):.1f}")
                report.append(f"  Expected Move: ${trade.get('expected_move', 0):.2f}")
                report.append(f"  Breakeven Move: ${trade.get('breakeven_move', 0):.2f}")
                report.append("")
        
        # Rejection-by-rejection breakdown
        if self.results['rejection_log']:
            report.append("\n" + "=" * 80)
            report.append("REJECTION-BY-REJECTION BREAKDOWN:")
            report.append("-" * 80)
            
            for i, rejection in enumerate(self.results['rejection_log'], 1):
                report.append(f"\nRejection #{i}:")
                report.append(f"  Date: {rejection.get('date', 'N/A')}")
                report.append(f"  Time: {rejection.get('time', 'N/A')}")
                report.append(f"  Symbol: {rejection.get('symbol', 'N/A')}")
                report.append(f"  RL Action: {rejection.get('rl_action', 'N/A')}")
                report.append(f"  Confidence: {rejection.get('confidence', 0):.3f}")
                report.append(f"  Reason: {rejection.get('reason', 'N/A')}")
                if 'expected_move' in rejection and 'breakeven_move' in rejection:
                    report.append(f"  Expected Move: ${rejection.get('expected_move', 0):.2f}")
                    report.append(f"  Breakeven Move: ${rejection.get('breakeven_move', 0):.2f}")
                report.append("")
        
        return "\n".join(report)
    
    def save_report(self, output_file: str):
        """Save complete report to file"""
        with open(output_file, 'w') as f:
            f.write(self.generate_summary())
            f.write("\n\n")
            f.write(self.generate_detailed_analysis())
        
        print(f"✅ Report saved to: {output_file}")


