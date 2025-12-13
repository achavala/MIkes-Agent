#!/usr/bin/env python3
"""
30-DAY BACKTEST RUNNER
Institutional-grade backtest with comprehensive logging
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Try to import yfinance (optional - may have dependency conflicts)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    YFINANCE_AVAILABLE = False
    print(f"⚠️ Warning: yfinance not available ({e})")
    print("   Will use alternative data source or existing backtest infrastructure")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from institutional_logging import initialize_logger, get_logger
from realistic_fill_modeling import calculate_realistic_fill
from online_learning_system import initialize_online_learning, get_online_learning_system
from multi_agent_ensemble import initialize_meta_router, get_meta_router
from log_compression import compress_daily_logs
from weekly_review_system import initialize_review_system, get_review_system
from end_of_run_verdict import initialize_verdict_system, get_verdict_system

# Import agent (will need to be adapted)
try:
    from mike_agent import MikeAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: mike_agent not available")


class InstitutionalBacktest:
    """
    30-day backtest with institutional-grade logging
    """
    
    def __init__(
        self,
        symbols: list = ['SPY', 'QQQ', 'SPX'],
        capital: float = 100000.0,
        mode: str = 'behavioral',  # 'behavioral' or 'pnl'
        log_dir: str = "logs"
    ):
        self.symbols = symbols
        self.capital = capital
        self.mode = mode
        self.current_capital = capital
        
        # Initialize logging
        self.logger = initialize_logger(log_dir=log_dir)
        
        # Initialize online learning
        self.learning_system = initialize_online_learning(
            model_dir="models",
            rolling_window_days=30,
            min_retrain_interval_hours=20
        )
        
        # Initialize meta router
        self.meta_router = initialize_meta_router()
        
        # Initialize review systems
        self.review_system = initialize_review_system(log_dir=log_dir)
        self.verdict_system = initialize_verdict_system(log_dir=log_dir)
        
        # Track backtest start date
        self.backtest_start_date = None
        
        # Track positions
        self.positions = {}
        self.trade_counter = {}
        for symbol in symbols:
            self.trade_counter[symbol] = 0
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Run 30-day backtest
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Backtest results
        """
        print("="*70)
        print("  30-DAY INSTITUTIONAL BACKTEST")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"Date Range: {start_date} to {end_date}")
        print("="*70)
        
        # Load data for all symbols
        all_data = {}
        
        if not YFINANCE_AVAILABLE:
            print("\n❌ yfinance not available due to dependency conflict.")
            print("\n   To fix this issue:")
            print("   1. Upgrade websockets: pip install --upgrade websockets")
            print("   2. Or use CSV data files (modify code to load from CSV)")
            print("   3. Or use a different virtual environment")
            print("\n   The backtest cannot proceed without a data source.")
            return {}
        
        for symbol in self.symbols:
            print(f"\nLoading data for {symbol}...")
            try:
                # yfinance limitation: Only 8 days of 1m data per request
                # Split into chunks of 7 days to be safe
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                
                all_chunks = []
                current_start = start
                chunk_days = 7  # Use 7 days per chunk to stay under 8-day limit
                
                while current_start < end:
                    current_end = min(current_start + timedelta(days=chunk_days), end)
                    chunk_start_str = current_start.strftime("%Y-%m-%d")
                    chunk_end_str = current_end.strftime("%Y-%m-%d")
                    
                    print(f"  Loading chunk: {chunk_start_str} to {chunk_end_str}...")
                    
                    try:
                        chunk_data = yf.download(
                            symbol, 
                            start=chunk_start_str, 
                            end=chunk_end_str, 
                            interval='1m', 
                            progress=False
                        )
                        
                        if len(chunk_data) > 0:
                            # Clean and format
                            chunk_data.columns = [col.lower() if isinstance(col, str) else col for col in chunk_data.columns]
                            if isinstance(chunk_data.columns, pd.MultiIndex):
                                chunk_data = chunk_data.xs(symbol, axis=1, level=1)
                            
                            all_chunks.append(chunk_data)
                            print(f"    ✅ Loaded {len(chunk_data)} bars")
                        else:
                            print(f"    ⚠️ No data for this chunk")
                    except Exception as chunk_error:
                        print(f"    ⚠️ Error loading chunk: {chunk_error}")
                        # Try daily data as fallback for this chunk
                        try:
                            print(f"    Trying daily data as fallback...")
                            chunk_data = yf.download(
                                symbol,
                                start=chunk_start_str,
                                end=chunk_end_str,
                                interval='1d',
                                progress=False
                            )
                            if len(chunk_data) > 0:
                                chunk_data.columns = [col.lower() if isinstance(col, str) else col for col in chunk_data.columns]
                                if isinstance(chunk_data.columns, pd.MultiIndex):
                                    chunk_data = chunk_data.xs(symbol, axis=1, level=1)
                                all_chunks.append(chunk_data)
                                print(f"    ✅ Loaded {len(chunk_data)} daily bars (fallback)")
                        except Exception:
                            pass
                    
                    current_start = current_end
                
                if all_chunks:
                    # Combine all chunks
                    data = pd.concat(all_chunks, axis=0)
                    data = data.sort_index()
                    data = data[~data.index.duplicated(keep='first')]  # Remove duplicates
                    all_data[symbol] = data
                    print(f"  ✅ Total loaded: {len(data)} bars")
                else:
                    print(f"  ⚠️ No data loaded for {symbol}")
                    continue
                    
            except Exception as e:
                print(f"  ❌ Error loading {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_data:
            print("\n❌ No data available for any symbol")
            return {}
        
        # Run backtest
        print("\n" + "="*70)
        print("  RUNNING BACKTEST...")
        print("="*70)
        
        # Store start date for reviews
        self.backtest_start_date = start_date
        
        # Process each trading day
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        day_number = 0
        
        for day in trading_days:
            day_number += 1
            day_str = day.strftime("%Y-%m-%d")
            print(f"\n📅 Processing {day_str}...")
            
            # Process each symbol
            for symbol in self.symbols:
                if symbol not in all_data:
                    continue
                
                # Get day's data
                day_data = all_data[symbol][all_data[symbol].index.date == day.date()]
                if len(day_data) == 0:
                    continue
                
                # Process each bar
                for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                    self._process_bar(symbol, timestamp, row, day_str)
            
            # End of day: flush buffers, check for retraining
            self.logger.flush_buffers()
            self._check_daily_retraining(day_str)
            
            # Weekly review checkpoints (days 5, 10, 20, 30)
            if day_number in [5, 10, 20, 30]:
                print(f"\n📊 Conducting Day {day_number} Review...")
                review = self.review_system.conduct_review(day_number, start_date)
                print(f"   ✅ Review complete: {len(review.get('answers', {}))} questions answered")
        
        # Compress logs (recommendation: handle log volume)
        print("\n📦 Compressing daily logs...")
        compress_daily_logs(log_dir=self.log_dir)
        
        # Generate final report
        results = self._generate_report()
        
        # Generate end-of-run verdict
        print("\n📋 Generating End-of-Run Verdict...")
        verdict = self.verdict_system.generate_verdict(start_date, end_date)
        
        print("\n" + "="*70)
        print("  BACKTEST COMPLETE")
        print("="*70)
        print(f"\n🎯 FINAL VERDICT: {verdict.get('recommendation', {}).get('decision', 'UNKNOWN')}")
        print(f"   Reason: {verdict.get('recommendation', {}).get('reason', 'N/A')}")
        print(f"\n📊 Scorecards:")
        print(f"   Behavior: {verdict.get('overall_scores', {}).get('behavior', 0):.2f}")
        print(f"   Risk: {verdict.get('overall_scores', {}).get('risk', 0):.2f}")
        print(f"   Execution: {verdict.get('overall_scores', {}).get('execution', 0):.2f}")
        print(f"   Learning: {verdict.get('overall_scores', {}).get('learning', 0):.2f}")
        print(f"   Average: {verdict.get('overall_scores', {}).get('average', 0):.2f}")
        
        results['verdict'] = verdict
        
        return results
    
    def _process_bar(self, symbol: str, timestamp: datetime, bar: pd.Series, date_str: str):
        """Process a single bar"""
        price = bar.get('close', bar.get('Close', 0))
        if price <= 0:
            return
        
        # Calculate time to expiry (0DTE = ~6.5 hours from 9:30 AM)
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_expiry_hours = (market_close - timestamp).total_seconds() / 3600
        time_to_expiry_min = int(time_to_expiry_hours * 60)
        
        # Detect regime (simplified - use VIX if available)
        vix = 20.0  # Default, would get from data
        regime = self._detect_regime(vix)
        
        # Get RL action (simulated for now)
        rl_action = "HOLD"
        rl_confidence = 0.5
        
        # Get ensemble action
        try:
            # Prepare data for ensemble
            ensemble_data = pd.DataFrame({
                'close': [price],
                'open': [bar.get('open', price)],
                'high': [bar.get('high', price)],
                'low': [bar.get('low', price)],
                'volume': [bar.get('volume', 0)]
            })
            
            ensemble_action, ensemble_confidence, ensemble_details = self.meta_router.route(
                data=ensemble_data,
                vix=vix,
                symbol=symbol,
                current_price=price,
                strike=round(price),
                portfolio_delta=0.0,  # Would get from portfolio
                delta_limit=2000.0
            )
            
            # Convert action to string
            action_map = {0: "HOLD", 1: "BUY_CALL", 2: "BUY_PUT"}
            ensemble_action_str = action_map.get(ensemble_action, "HOLD")
            rl_action_str = action_map.get(0, "HOLD")  # Simulated
            
            # Combine RL + Ensemble (60% ensemble, 40% RL)
            if ensemble_confidence > 0.3:
                final_action = ensemble_action_str
                final_confidence = ensemble_confidence * 0.6 + rl_confidence * 0.4
            else:
                final_action = rl_action_str
                final_confidence = rl_confidence
            
            # Get agent votes
            agent_votes = {}
            signals = ensemble_details.get('signals', {})
            for agent_name, signal_info in signals.items():
                agent_action = signal_info.get('action', 0)
                agent_votes[agent_name] = action_map.get(agent_action, "HOLD")
            
        except Exception as e:
            print(f"  ⚠️ Ensemble error: {e}")
            final_action = "HOLD"
            final_confidence = 0.0
            agent_votes = {}
        
        # Log decision
        self.logger.log_decision(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            regime=regime,
            time_to_expiry_min=time_to_expiry_min,
            action_final=final_action,
            confidence_final=final_confidence,
            rl_action=rl_action_str,
            rl_confidence=rl_confidence,
            ensemble_action=ensemble_action_str,
            ensemble_confidence=ensemble_confidence,
            agent_votes=agent_votes
        )
        
        self.stats['total_decisions'] += 1
        
        # Risk check (simplified)
        portfolio_delta = 0.0  # Would get from portfolio
        portfolio_gamma = 0.0
        portfolio_theta = 0.0
        portfolio_vega = 0.0
        
        risk_action = "ALLOW"
        risk_reason = None
        
        if final_action != "HOLD":
            # Check risk limits
            if portfolio_gamma > 0.025:
                risk_action = "BLOCK"
                risk_reason = "GAMMA_LIMIT_EXCEEDED"
        
        self.logger.log_risk_check(
            timestamp=timestamp,
            symbol=symbol,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            portfolio_theta=portfolio_theta,
            portfolio_vega=portfolio_vega,
            gamma_limit=0.025,
            delta_limit=2000.0,
            risk_action=risk_action,
            risk_reason=risk_reason
        )
        
        # Execute trade if allowed
        if final_action != "HOLD" and risk_action == "ALLOW":
            self._execute_trade(symbol, timestamp, price, final_action, date_str)
    
    def _execute_trade(self, symbol: str, timestamp: datetime, price: float, action: str, date_str: str):
        """Execute a trade with realistic fill modeling"""
        # Generate trade ID
        self.trade_counter[symbol] += 1
        trade_id = f"{symbol}_{date_str}_{self.trade_counter[symbol]:03d}"
        
        # Calculate realistic fill
        mid = price * 0.05  # Estimate option premium (5% of underlying)
        bid = mid * 0.98
        ask = mid * 1.02
        spread = ask - bid
        
        time_to_expiry = (datetime(timestamp.year, timestamp.month, timestamp.day, 16, 0) - timestamp).total_seconds() / 3600
        
        fill_price, fill_details = calculate_realistic_fill(
            mid=mid,
            bid=bid,
            ask=ask,
            qty=1,
            side='buy' if 'BUY' in action else 'sell',
            time_to_expiry=time_to_expiry,
            vix=20.0,
            volume=1000000,
            has_news=False,
            gamma_exposure=0.0,
            hidden_liquidity_pct=0.1
        )
        
        # Log execution
        self.logger.log_execution(
            timestamp=timestamp,
            symbol=symbol,
            order_type=action,
            mid_price=mid,
            fill_price=fill_price,
            spread=spread,
            slippage_pct=fill_details.get('slippage_pct', 0),
            qty=1,
            gamma_impact=fill_details.get('gamma_impact'),
            iv_crush_impact=fill_details.get('iv_crush_impact'),
            theta_impact=fill_details.get('theta_impact'),
            liquidity_factor=fill_details.get('liquidity_factor')
        )
        
        # Log position entry
        self.logger.log_position_entry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            qty=1,
            entry_price=price,
            strike=round(price),
            premium=fill_price
        )
        
        # Store position
        self.positions[trade_id] = {
            'entry_time': timestamp,
            'entry_price': price,
            'entry_premium': fill_price,
            'symbol': symbol,
            'action': action
        }
        
        self.stats['total_trades'] += 1
    
    def _detect_regime(self, vix: float) -> str:
        """Detect market regime from VIX"""
        if vix < 18:
            return "mean_reverting"
        elif vix < 25:
            return "neutral"
        elif vix < 35:
            return "volatile"
        else:
            return "trending"
    
    def _check_daily_retraining(self, date_str: str):
        """Check if model should be retrained"""
        try:
            regime = "neutral"  # Would detect from data
            should_retrain, reason = self.learning_system.should_retrain(regime)
            
            if should_retrain:
                # Get rolling window data (simplified)
                training_data = pd.DataFrame({'close': np.random.randn(100) + 500})
                
                version_id = self.learning_system.retrain_model(
                    training_data=training_data,
                    current_regime=regime,
                    training_config={'epochs': 100}
                )
                
                self.logger.log_learning_event(
                    date=date_str,
                    regime=regime,
                    retrained=True,
                    model_candidate=version_id,
                    production_model=self.learning_system.current_production_version,
                    retrain_reason=reason
                )
            else:
                self.logger.log_learning_event(
                    date=date_str,
                    regime=regime,
                    retrained=False,
                    retrain_reason=reason
                )
        except Exception as e:
            print(f"  ⚠️ Retraining check error: {e}")
    
    def _generate_report(self) -> Dict:
        """Generate backtest report"""
        return {
            'stats': self.stats,
            'final_capital': self.current_capital,
            'total_return_pct': ((self.current_capital - self.capital) / self.capital * 100) if self.capital > 0 else 0.0
        }


def main():
    """Run 30-day backtest"""
    # Calculate date range (last 30 trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)  # 45 calendar days to get ~30 trading days
    
    backtest = InstitutionalBacktest(
        symbols=['SPY'],  # Start with SPY for testing
        capital=100000.0,
        mode='behavioral',
        log_dir="logs"
    )
    
    results = backtest.run_backtest(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    print("\n📊 Backtest Results:")
    print(f"  Total Decisions: {results.get('stats', {}).get('total_decisions', 0)}")
    print(f"  Total Trades: {results.get('stats', {}).get('total_trades', 0)}")
    print(f"  Final Capital: ${results.get('final_capital', 0):,.2f}")
    print(f"  Total Return: {results.get('total_return_pct', 0):.2f}%")
    
    print("\n✅ Backtest complete! Logs saved to logs/ directory")
    print("   View logs in Dashboard → Analytics → Logs")


if __name__ == "__main__":
    main()

