#!/usr/bin/env python3
"""
Mike Agent v3 - Regime-by-Regime Backtesting
Full backtest on Mike's 20-day dataset (Nov 3 - Dec 1, 2025)
Shows performance breakdown by volatility regime
"""
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import agent logic (handle if not available)
try:
    import sys
    sys.path.insert(0, '.')
    from mike_agent_live_safe import VOL_REGIMES, estimate_premium, find_atm_strike
except ImportError:
    # Fallback if import fails
    VOL_REGIMES = {
        "calm": {"risk": 0.10, "max_pct": 0.30, "sl": -0.15, "tp1": 0.30, "tp2": 0.60, "tp3": 1.20, "trail": 0.50},
        "normal": {"risk": 0.07, "max_pct": 0.25, "sl": -0.20, "tp1": 0.40, "tp2": 0.80, "tp3": 1.50, "trail": 0.60},
        "storm": {"risk": 0.05, "max_pct": 0.20, "sl": -0.28, "tp1": 0.60, "tp2": 1.20, "tp3": 2.50, "trail": 0.90},
        "crash": {"risk": 0.03, "max_pct": 0.15, "sl": -0.35, "tp1": 1.00, "tp2": 2.00, "tp3": 4.00, "trail": 1.50}
    }
    
    def estimate_premium(price, strike, option_type):
        from scipy.stats import norm
        T, r, sigma = 1/365, 0.04, 0.20
        if T <= 0:
            return max(0.01, abs(price - strike))
        d1 = (np.log(price / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            return strike * np.exp(-r * T) * norm.cdf(-d2) - price * norm.cdf(-d1)
    
    def find_atm_strike(price):
        return round(price)

# ==================== REGIME CLASSIFICATION ====================
def get_regime(vix: float) -> str:
    """Determine volatility regime based on VIX"""
    if vix < 18:
        return "calm"
    elif vix < 25:
        return "normal"
    elif vix < 35:
        return "storm"
    else:
        return "crash"

# ==================== BACKTEST SIMULATION ====================
def simulate_trade(regime: str, entry_price: float, exit_price: float, direction: str = 'call') -> float:
    """Simulate a single trade based on regime"""
    regime_params = VOL_REGIMES.get(regime, VOL_REGIMES["normal"])
    
    strike = find_atm_strike(entry_price)
    entry_premium = estimate_premium(entry_price, strike, direction)
    
    # Calculate PnL based on price movement
    if direction == 'call':
        price_move = (exit_price - entry_price) / entry_price
    else:
        price_move = (entry_price - exit_price) / entry_price
    
    # Simplified: premium change ≈ price move * 10 (leverage approximation)
    premium_change = price_move * 10
    exit_premium = entry_premium * (1 + premium_change)
    
    pnl_pct = (exit_premium - entry_premium) / entry_premium
    
    # Apply take-profit tiers
    if pnl_pct >= regime_params['tp3']:
        return regime_params['tp3']  # Full exit at TP3
    elif pnl_pct >= regime_params['tp2']:
        return regime_params['tp2'] * 0.7 + regime_params['tp3'] * 0.3  # 70% at TP2, 30% at TP3
    elif pnl_pct >= regime_params['tp1']:
        return regime_params['tp1'] * 0.5 + regime_params['tp2'] * 0.3 + regime_params['tp3'] * 0.2  # 50/30/20
    elif pnl_pct <= regime_params['sl']:
        return regime_params['sl']  # Stop-loss hit
    else:
        return pnl_pct  # Hold

def backtest_regimes(start_date: str = "2025-11-03", end_date: str = "2025-12-02"):
    """Backtest agent performance by volatility regime"""
    print("=" * 70)
    print("MIKE AGENT v3 – REGIME-BY-REGIME BACKTEST")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Downloading real SPY and VIX data...")
    
    # Download real data
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, interval="5m")
        vix = yf.download("^VIX", start=start_date, end=end_date, interval="5m")
        
        # Fix MultiIndex if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # Align data
        spy = spy.dropna()
        vix = vix.dropna()
        
        # Resample VIX to match SPY timestamps (forward fill)
        vix_resampled = vix['Close'].reindex(spy.index, method='ffill')
        
        df = spy.copy()
        df['VIX'] = vix_resampled
        df = df.dropna()
        
        print(f"✓ Downloaded {len(df)} bars")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Classify regimes
    df['regime'] = df['VIX'].apply(get_regime)
    df['date'] = df.index.date
    
    # Group by date and regime
    results = []
    capital = 1000.0
    peak_capital = 1000.0
    
    regime_stats = {
        "calm": {"days": 0, "trades": 0, "wins": 0, "total_return": 0.0, "daily_returns": []},
        "normal": {"days": 0, "trades": 0, "wins": 0, "total_return": 0.0, "daily_returns": []},
        "storm": {"days": 0, "trades": 0, "wins": 0, "total_return": 0.0, "daily_returns": []},
        "crash": {"days": 0, "trades": 0, "wins": 0, "total_return": 0.0, "daily_returns": []}
    }
    
    for date, group in df.groupby('date'):
        if len(group) < 10:  # Skip days with insufficient data
            continue
        
        regime = group['regime'].mode()[0] if len(group['regime'].mode()) > 0 else "normal"
        vix_avg = group['VIX'].mean()
        
        regime_stats[regime]["days"] += 1
        
        # Simulate trades for this day
        # Use actual price movements from the data
        day_start_price = group['Close'].iloc[0]
        day_end_price = group['Close'].iloc[-1]
        day_high = group['High'].max()
        day_low = group['Low'].min()
        
        # Determine direction based on gap (simplified)
        gap = (day_start_price - df[df.index.date == date - pd.Timedelta(days=1)]['Close'].iloc[-1] if len(df[df.index.date == date - pd.Timedelta(days=1)]) > 0 else day_start_price) / day_start_price
        
        direction = 'call' if gap < -0.005 else 'put'  # Gap down → call, gap up → put
        
        # Simulate trade
        regime_params = VOL_REGIMES.get(regime, VOL_REGIMES["normal"])
        risk_pct = regime_params['risk']
        
        # Calculate position size
        entry_premium = estimate_premium(day_start_price, find_atm_strike(day_start_price), direction)
        risk_dollar = capital * risk_pct
        contracts = max(1, int(risk_dollar / (entry_premium * 100)))
        
        # Calculate PnL
        if direction == 'call':
            best_exit = day_high
        else:
            best_exit = day_low
        
        pnl_pct = simulate_trade(regime, day_start_price, best_exit, direction)
        
        # Apply stop-loss if needed
        if pnl_pct <= regime_params['sl']:
            pnl_pct = regime_params['sl']
        
        # Calculate daily return
        position_value = contracts * entry_premium * 100
        pnl_dollar = position_value * pnl_pct
        daily_return_pct = pnl_dollar / capital
        
        # Update capital
        capital += pnl_dollar
        peak_capital = max(peak_capital, capital)
        drawdown = (capital - peak_capital) / peak_capital
        
        # Track regime stats
        regime_stats[regime]["trades"] += 1
        if pnl_pct > 0:
            regime_stats[regime]["wins"] += 1
        regime_stats[regime]["total_return"] += daily_return_pct
        regime_stats[regime]["daily_returns"].append(daily_return_pct)
        
        results.append({
            'date': date,
            'regime': regime,
            'vix': vix_avg,
            'daily_return': daily_return_pct,
            'capital': capital,
            'drawdown': drawdown,
            'pnl_pct': pnl_pct
        })
    
    # Print results
    print("\n" + "=" * 70)
    print("REGIME-BY-REGIME PERFORMANCE")
    print("=" * 70)
    print(f"{'Regime':<10} {'Days':<6} {'Win Rate':<10} {'Avg Return/Day':<15} {'Total Return':<15} {'Max DD':<10} {'Sharpe':<8}")
    print("-" * 70)
    
    for regime_name in ["calm", "normal", "storm", "crash"]:
        stats = regime_stats[regime_name]
        if stats["days"] == 0:
            print(f"{regime_name.upper():<10} {'0':<6} {'—':<10} {'—':<15} {'—':<15} {'—':<10} {'—':<8}")
            continue
        
        win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        avg_daily_return = (stats["total_return"] / stats["days"] * 100) if stats["days"] > 0 else 0
        total_return = stats["total_return"] * 100
        
        # Calculate Sharpe ratio
        if len(stats["daily_returns"]) > 1:
            returns_array = np.array(stats["daily_returns"])
            sharpe = (np.mean(returns_array) / (np.std(returns_array) + 1e-6)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Find max drawdown for this regime
        regime_results = [r for r in results if r['regime'] == regime_name]
        max_dd = min([r['drawdown'] for r in regime_results]) if regime_results else 0.0
        
        print(f"{regime_name.upper():<10} {stats['days']:<6} {win_rate:.0f}%{'':<6} {avg_daily_return:+.1f}%{'':<8} {total_return:+.1f}%{'':<8} {max_dd:.1%}{'':<5} {sharpe:.1f}")
    
    print("=" * 70)
    print(f"\nFINAL RESULTS:")
    print(f"  Starting Capital: $1,000.00")
    print(f"  Final Capital: ${capital:,.2f}")
    print(f"  Total Return: +{((capital/1000)-1)*100:.1f}%")
    print(f"  Max Drawdown: {min([r['drawdown'] for r in results]):.1%}")
    
    overall_wins = sum([s["wins"] for s in regime_stats.values()])
    overall_trades = sum([s["trades"] for s in regime_stats.values()])
    overall_win_rate = (overall_wins / overall_trades * 100) if overall_trades > 0 else 0
    
    print(f"  Overall Win Rate: {overall_win_rate:.0f}%")
    print(f"  Total Days: {len(results)}")
    print("=" * 70)
    
    return results, regime_stats

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest Mike Agent by volatility regime")
    parser.add_argument('--start', type=str, default='2025-11-03', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-02', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    results, stats = backtest_regimes(args.start, args.end)

