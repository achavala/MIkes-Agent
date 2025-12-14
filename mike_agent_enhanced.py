#!/usr/bin/env python3
"""
MikeAgent Enhanced - Complete High-Frequency Options Scalping Bot
Based on 20-day dataset (Nov 3-Dec 1, 2025)
Features: Gap detection, re-entries, avg-down, trims, SLs, VIX/IV filters, gamma proxies, EOD curl detection
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd
import time
import sys
from typing import Optional, Dict, Any, List
import yfinance as yf
try:
    import config
except ImportError:
    # Create a mock config from environment variables
    class Config:
        ALPACA_KEY = os.environ.get('ALPACA_KEY', '')
        ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
        ALPACA_BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        SYMBOLS = ['SPY', 'QQQ']
        START_CAPITAL = 1000.0
        RISK_PCT = 0.07
        VIX_THRESHOLD = 28.0
        IV_THRESHOLD = 30.0
    config = Config()
    import os  # Ensure os is imported for the fallback

try:
    from alpaca_trade_api.rest import REST as AlpacaREST
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class Signal:
    """Trading signal"""
    def __init__(self, symbol, action, size, strike, strategy, confidence, metadata=None):
        self.symbol = symbol
        self.action = action
        self.size = int(size)
        self.strike = strike
        self.strategy = strategy
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"Signal({self.action} {self.size}x {self.strike} @ {self.symbol})"


class Action:
    """Trading actions"""
    BUY = 'buy'
    SELL = 'sell'


class MikeAgent:
    """
    Enhanced MikeAgent - Complete Gap-Scalp-ReEntry Strategy
    
    Features:
    - Gap detection with VIX/IV filters
    - Re-entry and avg-down logic
    - Trim exits at +30%/+60%
    - Stop loss on rejection or -20%
    - Gamma proxy calculations
    - EOD curl detection
    - Monte Carlo backtesting
    """
    
    def __init__(self, mode='backtest', symbols=None, capital=None, risk_pct=None,
                 vix_threshold=None, iv_threshold=None):
        self.mode = mode
        self.symbols = symbols or config.SYMBOLS
        self.capital = capital or config.START_CAPITAL
        self.current_capital = self.capital
        self.risk_pct = risk_pct or config.RISK_PCT
        self.vix_threshold = vix_threshold or config.VIX_THRESHOLD
        self.iv_threshold = iv_threshold or config.IV_THRESHOLD
        
        # Position tracking per symbol
        self.entry_premium: Dict[str, float] = {}
        self.avg_premium: Dict[str, float] = {}
        self.position_size: Dict[str, int] = {}
        self.direction: Dict[str, Optional[str]] = {}
        self.strike: Dict[str, Optional[float]] = {}
        self.pt_level: Dict[str, float] = {}
        self.sl_level: Dict[str, float] = {}
        self.has_avg_down: Dict[str, bool] = {}
        self.entry_price: Dict[str, float] = {}
        self.yesterday_close: Dict[str, float] = {}
        self.entry_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.logs: List[str] = []
        self.trades: List[Dict] = []
        self.pnl_data: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # VIX/IV tracking
        self.current_vix: float = 0.0
        self.current_iv: Dict[str, float] = {}
        
        # Alpaca API
        if mode == 'paper' and ALPACA_AVAILABLE:
            if config.ALPACA_KEY == 'YOUR_PAPER_KEY':
                raise ValueError("Please set ALPACA_KEY and ALPACA_SECRET in config.py")
            self.api = AlpacaREST(config.ALPACA_KEY, config.ALPACA_SECRET, 
                                 base_url=config.ALPACA_BASE_URL)
        else:
            self.api = None
        
        # Initialize tracking
        for symbol in self.symbols:
            self._init_symbol(symbol)
    
    def _init_symbol(self, symbol: str):
        """Initialize tracking for a symbol"""
        self.entry_premium[symbol] = 0.0
        self.avg_premium[symbol] = 0.0
        self.position_size[symbol] = 0
        self.direction[symbol] = None
        self.strike[symbol] = None
        self.pt_level[symbol] = 0.0
        self.sl_level[symbol] = 0.0
        self.has_avg_down[symbol] = False
        self.entry_price[symbol] = 0.0
        self.yesterday_close[symbol] = 0.0
        self.entry_time[symbol] = None
        self.current_iv[symbol] = 0.20  # Default IV
        self.daily_pnl[symbol] = 0.0
    
    def _get_vix(self) -> float:
        """Get current VIX level"""
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='1d')
            if len(hist) > 0:
                self.current_vix = hist['Close'].iloc[-1]
                return self.current_vix
        except Exception as e:
            self.log(f"Error fetching VIX: {e}")
        return self.current_vix if self.current_vix > 0 else 20.0  # Default
    
    def _get_iv(self, symbol: str) -> float:
        """Get implied volatility for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            # Try to get IV from options chain
            expirations = ticker.options
            if expirations:
                chain = ticker.option_chain(expirations[0])
                # Use ATM option IV
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                calls = chain.calls
                calls['distance'] = abs(calls['strike'] - current_price)
                atm_call = calls.loc[calls['distance'].idxmin()]
                if 'impliedVolatility' in atm_call and not pd.isna(atm_call['impliedVolatility']):
                    self.current_iv[symbol] = atm_call['impliedVolatility']
                    return self.current_iv[symbol]
        except Exception as e:
            self.log(f"Error fetching IV for {symbol}: {e}")
        return self.current_iv.get(symbol, 0.20)  # Default
    
    def _calculate_gamma_proxy(self, S: float, K: float, direction: str, iv: float) -> float:
        """
        Calculate gamma proxy for position sizing
        Higher gamma = more leverage = smaller size needed
        """
        T = config.T
        r = config.R
        sigma = iv
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Gamma proxy: inverse relationship with size
        # High gamma (>0.1) = reduce size by 20%
        if gamma > 0.1:
            return 0.8
        elif gamma > 0.05:
            return 0.9
        return 1.0
    
    def _detect_eod_curl(self, bar: Dict[str, Any], symbol: str) -> bool:
        """
        Detect end-of-day curl pattern
        EOD curl: Strong move in last hour, potential reversal
        """
        # Simplified: check if we're near market close (3:30-4:00 PM ET)
        now = datetime.now()
        if self.mode == 'backtest':
            # In backtest, check if it's late in the day
            return False  # Simplified for backtest
        
        # For live/paper: check time
        if 15 <= now.hour < 16:  # 3-4 PM
            # Check for strong move
            move_pct = abs(bar['close'] - bar['open']) / bar['open']
            if move_pct > 0.01:  # >1% move
                return True
        return False
    
    def _get_yesterday_close(self, symbol: str) -> float:
        """Get yesterday's closing price"""
        if self.yesterday_close[symbol] > 0:
            return self.yesterday_close[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            if len(hist) >= 2:
                self.yesterday_close[symbol] = hist['Close'].iloc[-2]
                return self.yesterday_close[symbol]
        except Exception as e:
            self.log(f"Error fetching yesterday close for {symbol}: {e}")
        return 0.0
    
    def log(self, message: str):
        """Log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.logs.append(log_msg)
        print(log_msg)
    
    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Main strategy logic - called on each bar
        
        Args:
            symbol: Trading symbol
            bar: Dictionary with 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        
        Returns:
            Signal if action needed, None otherwise
        """
        current_price = bar['close']
        
        # Update VIX and IV
        vix = self._get_vix()
        iv = self._get_iv(symbol)
        
        # Entry Logic: Gap fill detection with filters
        if self.position_size[symbol] == 0:
            # Check VIX filter for puts
            if self._detect_gap(symbol, bar):
                gap_direction = 'put' if bar['open'] > bar['close'] * 1.005 else 'call'
                
                # VIX filter: Only trade puts if VIX > threshold
                if gap_direction == 'put' and vix < self.vix_threshold:
                    return None
                
                # IV filter: Check if IV is reasonable
                if iv < self.iv_threshold * 0.5:  # Too low IV
                    return None
                
                # EOD curl detection: Avoid entries near close
                if self._detect_eod_curl(bar, symbol):
                    self.log(f"{symbol}: EOD curl detected, skipping entry")
                    return None
                
                strike = self._find_strike_near_gap(current_price, gap_direction)
                if strike:
                    premium = self._estimate_premium(current_price, strike, gap_direction, iv)
                    if premium > 0:
                        # Calculate size with gamma proxy
                        gamma_proxy = self._calculate_gamma_proxy(current_price, strike, gap_direction, iv)
                        size = self._calculate_size(premium, current_price)
                        initial_size = max(1, int(size * config.INITIAL_SIZE_PCT * gamma_proxy))
                        
                        # Store position info
                        self.entry_premium[symbol] = premium
                        self.avg_premium[symbol] = premium
                        self.position_size[symbol] = initial_size
                        self.direction[symbol] = gap_direction
                        self.strike[symbol] = strike
                        self.entry_price[symbol] = current_price
                        self.entry_time[symbol] = bar.get('timestamp', datetime.now())
                        
                        # Set profit target and stop loss
                        if gap_direction == 'call':
                            self.pt_level[symbol] = current_price * (1 + config.PT_PCT)
                        else:
                            self.pt_level[symbol] = current_price * (1 - config.PT_PCT)
                        
                        self.sl_level[symbol] = premium * (1 - config.SL_PCT)
                        self.has_avg_down[symbol] = False
                        
                        self.log(f"{symbol}: ENTRY {gap_direction.upper()} strike={strike:.2f} "
                                f"size={initial_size} premium=${premium:.2f} VIX={vix:.1f} IV={iv:.1%}")
                        
                        return Signal(
                            symbol=symbol,
                            action=Action.BUY,
                            size=initial_size,
                            strike=strike,
                            strategy="MikeAgent",
                            confidence=0.75,
                            metadata={
                                'reason': 'entry',
                                'premium': premium,
                                'vix': vix,
                                'iv': iv,
                                'gamma_proxy': gamma_proxy
                            }
                        )
        
        # Position Management
        if self.position_size[symbol] > 0:
            current_premium = self._estimate_premium(
                current_price, self.strike[symbol], self.direction[symbol], iv
            )
            
            # Theta check: Exit if holding too long (0DTE)
            if self.entry_time[symbol]:
                hold_time = (bar.get('timestamp', datetime.now()) - self.entry_time[symbol]).total_seconds() / 3600
                if hold_time > 6:  # >6 hours for 0DTE
                    self.log(f"{symbol}: Theta flag - holding >6 hours, exiting")
                    exit_size = self.position_size[symbol]
                    self.position_size[symbol] = 0
                    return Signal(
                        symbol=symbol,
                        action=Action.SELL,
                        size=exit_size,
                        strike=self.strike[symbol],
                        strategy="MikeAgent",
                        confidence=1.0,
                        metadata={'reason': 'theta_flag', 'hold_time': hold_time}
                    )
            
            # Avg-Down Logic: -10% to -30%
            if not self.has_avg_down[symbol]:
                pnl_pct = (current_premium - self.avg_premium[symbol]) / self.avg_premium[symbol]
                if config.AVG_DOWN_MIN <= pnl_pct <= config.AVG_DOWN_MAX:
                    # Add 50% more (1.5x total)
                    add_size = max(1, int(self.position_size[symbol] * 0.5))
                    total_cost = (self.avg_premium[symbol] * self.position_size[symbol] * 100) + \
                                (current_premium * add_size * 100)
                    self.avg_premium[symbol] = total_cost / ((self.position_size[symbol] + add_size) * 100)
                    self.position_size[symbol] += add_size
                    self.has_avg_down[symbol] = True
                    
                    self.log(f"{symbol}: AVG-DOWN add {add_size} contracts at ${current_premium:.2f}")
                    
                    return Signal(
                        symbol=symbol,
                        action=Action.BUY,
                        size=add_size,
                        strike=self.strike[symbol],
                        strategy="MikeAgent",
                        confidence=0.80,
                        metadata={'reason': 'avg_down', 'new_premium': current_premium}
                    )
            
            # Exit Logic
            pnl_pct = (current_premium - self.avg_premium[symbol]) / self.avg_premium[symbol]
            
            # Trim 70% at +60%
            if pnl_pct >= 0.60:
                trim_size = int(self.position_size[symbol] * config.TRIM_60_PCT)
                self.position_size[symbol] -= trim_size
                self.log(f"{symbol}: TRIM 70% at +{pnl_pct*100:.1f}%")
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=trim_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=0.90,
                    metadata={'reason': 'trim_60', 'pnl_pct': pnl_pct * 100}
                )
            
            # Trim 50% at +30%
            elif pnl_pct >= 0.30:
                trim_size = int(self.position_size[symbol] * config.TRIM_30_PCT)
                self.position_size[symbol] -= trim_size
                self.log(f"{symbol}: TRIM 50% at +{pnl_pct*100:.1f}%")
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=trim_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=0.85,
                    metadata={'reason': 'trim_30', 'pnl_pct': pnl_pct * 100}
                )
            
            # Stop Loss: -20% or rejection
            elif pnl_pct <= -config.SL_PCT or self._is_rejected(bar, symbol):
                exit_size = self.position_size[symbol]
                self.position_size[symbol] = 0
                reason = 'stop_loss' if pnl_pct <= -config.SL_PCT else 'rejection'
                self.log(f"{symbol}: {reason.upper()} at {pnl_pct*100:.1f}%")
                return Signal(
                    symbol=symbol,
                    action=Action.SELL,
                    size=exit_size,
                    strike=self.strike[symbol],
                    strategy="MikeAgent",
                    confidence=1.0,
                    metadata={'reason': reason, 'pnl_pct': pnl_pct * 100}
                )
        
        return None
    
    def _detect_gap(self, symbol: str, bar: Dict[str, Any]) -> bool:
        """Detect if there's a significant gap (>0.5%)"""
        yesterday_close = self._get_yesterday_close(symbol)
        if yesterday_close == 0:
            return False
        
        gap_pct = abs(bar['open'] - yesterday_close) / yesterday_close
        return gap_pct > config.GAP_THRESHOLD
    
    def _find_strike_near_gap(self, price: float, direction: str) -> Optional[float]:
        """Find strike price near gap fill level"""
        if direction == 'call':
            return round(price + 1, 2)  # Slightly OTM call
        else:
            return round(price - 1, 2)  # Slightly OTM put
    
    def _estimate_premium(self, S: float, K: float, direction: str, iv: Optional[float] = None) -> float:
        """Estimate option premium using Black-Scholes"""
        T = config.T
        r = config.R
        sigma = iv if iv else config.DEFAULT_SIGMA
        
        if T <= 0:
            # Intrinsic value only
            if direction == 'call':
                return max(0.01, S - K)
            else:
                return max(0.01, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if direction == 'call':
            premium = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            premium = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0.01, premium)
    
    def _calculate_size(self, premium: float, price: float) -> int:
        """Calculate position size based on risk percentage"""
        if premium <= 0:
            return 0
        
        risk_dollar = self.current_capital * self.risk_pct
        contracts = int(risk_dollar / (premium * 100))
        
        # "Size for $0" on lottos (very cheap options)
        if premium < config.LOTTO_THRESHOLD:
            contracts = max(contracts, 1)
        
        return max(1, contracts)
    
    def _is_rejected(self, bar: Dict[str, Any], symbol: str) -> bool:
        """Check if price was rejected from profit target"""
        if self.pt_level[symbol] == 0:
            return False
        
        if self.direction[symbol] == 'call':
            return bar['high'] > self.pt_level[symbol] and bar['close'] < self.pt_level[symbol]
        else:  # put
            return bar['low'] < self.pt_level[symbol] and bar['close'] > self.pt_level[symbol]
    
    def backtest(self, csv_file: Optional[str] = None, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, monte_carlo: bool = False) -> Dict:
        """Run backtest on historical data"""
        print(f"\n{'='*60}")
        print("BACKTEST MODE - Enhanced MikeAgent")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"Monte Carlo: {monte_carlo}")
        
        if monte_carlo:
            return self._monte_carlo_backtest(csv_file, start_date, end_date)
        
        total_pnl = 0.0
        self.current_capital = self.capital
        
        for symbol in self.symbols:
            self._init_symbol(symbol)
            
            if csv_file:
                data = self._load_csv_data(csv_file, symbol)
            else:
                data = self._load_yahoo_data(symbol, start_date or config.BACKTEST_START_DATE,
                                            end_date or config.BACKTEST_END_DATE)
            
            if data is None or len(data) == 0:
                print(f"No data for {symbol}")
                continue
            
            if len(data) > 1:
                self.yesterday_close[symbol] = data.iloc[0]['close']
            
            symbol_pnl = 0.0
            
            for idx, (timestamp, row) in enumerate(data.iterrows()):
                bar = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0),
                    'timestamp': timestamp
                }
                
                if idx > 0:
                    self.yesterday_close[symbol] = data.iloc[idx - 1]['close']
                
                signal = self.on_bar(symbol, bar)
                
                if signal:
                    trade_pnl = self._simulate_trade(signal, bar, symbol)
                    symbol_pnl += trade_pnl
                    total_pnl += trade_pnl
                    self.current_capital += trade_pnl
                    
                    self.pnl_data.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'pnl': self.current_capital - self.capital,
                        'capital': self.current_capital
                    })
        
        return self._generate_backtest_report(total_pnl)
    
    def _monte_carlo_backtest(self, csv_file: Optional[str], start_date: Optional[str],
                              end_date: Optional[str]) -> Dict:
        """Monte Carlo simulation for backtesting"""
        results = []
        
        for sim in range(config.MONTE_CARLO_SIMULATIONS):
            # Reset for each simulation
            self.current_capital = self.capital
            self.trades = []
            self.pnl_data = []
            
            for symbol in self.symbols:
                self._init_symbol(symbol)
            
            # Run backtest with slight randomization
            # (In production, add noise to prices/volatility)
            # Create a temporary agent to avoid recursion
            temp_agent = MikeAgent(
                mode='backtest',
                symbols=self.symbols,
                capital=self.capital,
                risk_pct=self.risk_pct
            )
            result = temp_agent.backtest(csv_file, start_date, end_date, monte_carlo=False)
            results.append(result)
        
        # Aggregate results
        returns = [r['total_return'] for r in results]
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'win_rate': np.mean([r['win_rate'] for r in results]),
            'simulations': config.MONTE_CARLO_SIMULATIONS
        }
    
    def _load_csv_data(self, csv_file: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load backtest data from CSV"""
        try:
            df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            self.log(f"Error loading CSV: {e}")
            return None
    
    def _load_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if len(df) == 0:
                return None
            df.columns = df.columns.str.lower()
            df.index.name = 'date'
            return df
        except Exception as e:
            self.log(f"Error loading Yahoo data for {symbol}: {e}")
            return None
    
    def _simulate_trade(self, signal: Signal, bar: Dict[str, Any], symbol: str) -> float:
        """Simulate trade execution for backtesting"""
        if signal.action == Action.BUY:
            return 0.0
        
        # Calculate PnL
        current_premium = self._estimate_premium(
            bar['close'], signal.strike, self.direction[symbol]
        )
        
        if signal.metadata.get('reason') in ['trim_30', 'trim_60']:
            pnl = (current_premium - self.avg_premium[symbol]) * signal.size * 100
        else:
            pnl = (current_premium - self.avg_premium[symbol]) * signal.size * 100
        
        self.trades.append({
            'symbol': symbol,
            'timestamp': bar.get('timestamp', datetime.now()),
            'action': signal.action,
            'reason': signal.metadata.get('reason', 'exit'),
            'strike': signal.strike,
            'size': signal.size,
            'price': bar['close'],
            'premium': current_premium,
            'pnl': pnl
        })
        
        return pnl
    
    def _generate_backtest_report(self, total_pnl: float) -> Dict:
        """Generate backtest report"""
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return_pct = (total_pnl / self.capital) * 100
        
        report = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'final_capital': self.current_capital,
            'total_return': return_pct
        }
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        for key, value in report.items():
            if isinstance(value, float):
                if 'return' in key or 'rate' in key:
                    print(f"{key}: {value:.2f}%")
                elif 'pnl' in key or 'capital' in key or 'win' in key or 'loss' in key:
                    print(f"{key}: ${value:,.2f}")
                else:
                    print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print(f"{'='*60}\n")
        
        return report
    
    def run_paper_trade(self):
        """Run paper trading loop"""
        if not self.api:
            raise ValueError("Alpaca API not configured")
        
        self.log("Starting paper trading...")
        
        try:
            while True:
                for symbol in self.symbols:
                    bar = self._get_latest_bar(symbol)
                    if bar:
                        signal = self.on_bar(symbol, bar)
                        if signal:
                            self._execute_signal(signal, bar)
                time.sleep(60)
        except KeyboardInterrupt:
            self.log("Paper trading stopped")
    
    def _get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest bar from Alpaca"""
        if not self.api:
            return None
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=1).df
            if len(bars) == 0:
                return None
            latest = bars.iloc[-1]
            return {
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'volume': int(latest['volume']),
                'timestamp': latest.name
            }
        except Exception as e:
            self.log(f"Error fetching bar for {symbol}: {e}")
            return None
    
    def _execute_signal(self, signal: Signal, bar: Dict[str, Any]):
        """Execute signal in paper trading"""
        self.log(f"EXECUTING: {signal}")
        # TODO: Implement actual Alpaca order placement
        # self.api.submit_order(...)

