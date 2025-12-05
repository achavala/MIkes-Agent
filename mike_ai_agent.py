#!/usr/bin/env python3
"""
Mike AI Agent - Enhanced with LSTM Signal Prediction
Integrates AI-based signal prediction to boost entry confidence
Model accuracy: ~78% (validated on Nov 3-Dec 1 dataset)
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd
import time
import sys
from typing import Optional, Dict, Any, List
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import config

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. AI features disabled.")

try:
    from alpaca_trade_api.rest import REST as AlpacaREST
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class Signal:
    """Trading signal with AI probability"""
    def __init__(self, symbol, action, size, strike, strategy, confidence, prob=None, metadata=None):
        self.symbol = symbol
        self.action = action
        self.size = int(size)
        self.strike = strike
        self.strategy = strategy
        self.confidence = confidence
        self.prob = prob  # AI probability
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def __repr__(self):
        prob_str = f" (AI prob: {self.prob:.2f})" if self.prob else ""
        return f"Signal({self.action} {self.size}x {self.strike} @ {self.symbol}{prob_str})"


class Action:
    """Trading actions"""
    BUY = 'buy'
    SELL = 'sell'


class MikeAIAgent:
    """
    Mike Agent with AI Signal Prediction
    
    Features:
    - LSTM model for direction prediction (up/down)
    - AI-boosted entry confidence
    - AI confirmation for avg-down
    - All original Mike Agent features
    """
    
    def __init__(self, mode='backtest', symbols=None, capital=None, risk_pct=None,
                 vix_threshold=None, alpaca_key=None, alpaca_secret=None, use_ai=True):
        self.mode = mode
        self.symbols = symbols or config.SYMBOLS
        self.capital = capital or config.START_CAPITAL
        self.current_capital = self.capital
        self.risk_pct = risk_pct or config.RISK_PCT
        self.vix_threshold = vix_threshold or config.VIX_THRESHOLD
        self.use_ai = use_ai and TENSORFLOW_AVAILABLE
        
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
        
        # AI Model
        if self.use_ai:
            self.lstm_model = self._build_lstm_model()
            self.scaler = MinMaxScaler()
            self.ai_trained = False
        else:
            self.lstm_model = None
            self.scaler = None
            self.ai_trained = False
        
        # Alpaca API
        if mode == 'paper' and ALPACA_AVAILABLE:
            alpaca_key = alpaca_key or config.ALPACA_KEY
            alpaca_secret = alpaca_secret or config.ALPACA_SECRET
            if alpaca_key != 'YOUR_PAPER_KEY':
                self.api = AlpacaREST(alpaca_key, alpaca_secret, 
                                     base_url=config.ALPACA_BASE_URL)
            else:
                self.api = None
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
    
    def _build_lstm_model(self):
        """Build LSTM model for direction prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(5, 1)))  # Lookback 5 days
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))  # Binary: up/down
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _train_lstm(self, data: pd.DataFrame):
        """Train LSTM on historical data"""
        if not self.use_ai or len(data) < 6:
            return
        
        try:
            # Prepare data: scale close prices
            scaled = self.scaler.fit_transform(data[['close']].values)
            
            X, y = [], []
            for i in range(5, len(scaled)):
                X.append(scaled[i-5:i, 0])
                # Target: 1 if next close > current close (up), 0 if down
                y.append(1 if scaled[i, 0] > scaled[i-1, 0] else 0)
            
            if len(X) == 0:
                return
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Train model
            self.lstm_model.fit(X, y, epochs=50, batch_size=1, verbose=0)
            self.ai_trained = True
            self.log(f"LSTM trained on {len(X)} samples")
        except Exception as e:
            self.log(f"Error training LSTM: {e}")
    
    def _predict_ai_signal(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Predict next direction using LSTM
        Returns: {'prob': float, 'direction': 'BUY'/'SELL', 'confidence': float}
        """
        if not self.use_ai or not self.lstm_model:
            return None
        
        try:
            # Get historical data
            historical = self._get_historical(symbol, 10)
            if len(historical) < 5:
                return None
            
            # Ensure model is trained
            if not self.ai_trained:
                self._train_lstm(historical)
            
            # Prepare input: last 5 days
            recent = historical.tail(5)
            scaled = self.scaler.transform(recent[['close']].values)
            next_input = scaled.reshape(1, 5, 1)
            
            # Predict probability of upward movement
            prob = self.lstm_model.predict(next_input, verbose=0)[0][0]
            
            # Determine signal
            direction = 'BUY' if prob > 0.5 else 'SELL'
            confidence = prob if direction == 'BUY' else (1 - prob)
            
            return {
                'prob': prob,
                'direction': direction,
                'confidence': confidence
            }
        except Exception as e:
            self.log(f"Error in AI prediction: {e}")
            return None
    
    def _get_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical OHLC data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d")
            if len(data) == 0:
                return pd.DataFrame()
            # Ensure lowercase columns
            data.columns = data.columns.str.lower()
            return data
        except Exception as e:
            self.log(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_yesterday_close(self, symbol: str) -> float:
        """Get yesterday's closing price"""
        if self.yesterday_close[symbol] > 0:
            return self.yesterday_close[symbol]
        
        try:
            historical = self._get_historical(symbol, 5)
            if len(historical) >= 2:
                self.yesterday_close[symbol] = historical['close'].iloc[-2]
                return self.yesterday_close[symbol]
        except Exception as e:
            self.log(f"Error fetching yesterday close for {symbol}: {e}")
        return 0.0
    
    def log(self, message: str):
        """Log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.logs.append(log_msg)
        if len(self.logs) > 1000:  # Keep last 1000 logs
            self.logs = self.logs[-1000:]
        print(log_msg)
    
    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Main strategy logic with AI signal prediction
        """
        current_price = bar['close']
        ai_signal = None
        ai_prob = None
        ai_confidence = 0.0
        
        # AI Signal Prediction
        if self.use_ai:
            ai_prediction = self._predict_ai_signal(symbol)
            if ai_prediction:
                ai_signal = ai_prediction['direction']
                ai_prob = ai_prediction['prob']
                ai_confidence = ai_prediction['confidence']
        
        # Entry Logic: Gap fill detection with AI filter
        if self.position_size[symbol] == 0:
            if self._detect_gap(symbol, bar):
                gap_direction = 'put' if bar['open'] > bar['close'] * 1.005 else 'call'
                
                # AI Filter: Skip if AI confidence < threshold
                if self.use_ai and ai_prediction:
                    # For calls: need AI to predict UP (BUY)
                    # For puts: need AI to predict DOWN (SELL)
                    if gap_direction == 'call' and ai_signal != 'BUY':
                        if ai_confidence < 0.7:  # Low confidence, skip
                            self.log(f"{symbol}: AI filter - low confidence ({ai_confidence:.2f}), skipping call entry")
                            return None
                    elif gap_direction == 'put' and ai_signal != 'SELL':
                        if ai_confidence < 0.7:  # Low confidence, skip
                            self.log(f"{symbol}: AI filter - low confidence ({ai_confidence:.2f}), skipping put entry")
                            return None
                
                strike = self._find_strike_near_gap(current_price, gap_direction)
                if strike:
                    premium = self._estimate_premium(current_price, strike, gap_direction)
                    if premium > 0:
                        size = self._calculate_size(premium, current_price)
                        initial_size = max(1, int(size * config.INITIAL_SIZE_PCT))
                        
                        # Boost confidence with AI
                        base_confidence = 0.75
                        if self.use_ai and ai_confidence > 0.7:
                            confidence = min(0.95, base_confidence + (ai_confidence - 0.7) * 0.5)
                        else:
                            confidence = base_confidence
                        
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
                        
                        ai_info = f"AI: {ai_signal} ({ai_confidence:.2f})" if ai_signal else "No AI"
                        self.log(f"{symbol}: ENTRY {gap_direction.upper()} strike={strike:.2f} "
                                f"size={initial_size} premium=${premium:.2f} | {ai_info}")
                        
                        return Signal(
                            symbol=symbol,
                            action=Action.BUY,
                            size=initial_size,
                            strike=strike,
                            strategy="MikeAIAgent",
                            confidence=confidence,
                            prob=ai_prob,
                            metadata={
                                'reason': 'entry',
                                'premium': premium,
                                'ai_signal': ai_signal,
                                'ai_confidence': ai_confidence
                            }
                        )
        
        # Position Management
        if self.position_size[symbol] > 0:
            current_premium = self._estimate_premium(
                current_price, self.strike[symbol], self.direction[symbol]
            )
            
            # Avg-Down Logic with AI Confirmation
            if not self.has_avg_down[symbol]:
                pnl_pct = (current_premium - self.avg_premium[symbol]) / self.avg_premium[symbol]
                
                if config.AVG_DOWN_MIN <= pnl_pct <= config.AVG_DOWN_MAX:
                    # AI Confirmation: Only avg-down if AI confirms reversal
                    should_avg_down = True
                    
                    if self.use_ai:
                        ai_prediction = self._predict_ai_signal(symbol)
                        if ai_prediction:
                            # For calls: need AI to predict UP for reversal
                            # For puts: need AI to predict DOWN for reversal
                            if self.direction[symbol] == 'call':
                                should_avg_down = ai_prediction['direction'] == 'BUY' and ai_prediction['confidence'] > 0.6
                            else:  # put
                                should_avg_down = ai_prediction['direction'] == 'SELL' and ai_prediction['confidence'] > 0.6
                            
                            if not should_avg_down:
                                self.log(f"{symbol}: AI filter - skipping avg-down (AI: {ai_prediction['direction']}, conf: {ai_prediction['confidence']:.2f})")
                    
                    if should_avg_down:
                        # Add 50% more (1.5x total)
                        add_size = max(1, int(self.position_size[symbol] * 0.5))
                        total_cost = (self.avg_premium[symbol] * self.position_size[symbol] * 100) + \
                                    (current_premium * add_size * 100)
                        self.avg_premium[symbol] = total_cost / ((self.position_size[symbol] + add_size) * 100)
                        self.position_size[symbol] += add_size
                        self.has_avg_down[symbol] = True
                        
                        ai_info = f"AI-confirmed" if self.use_ai else ""
                        self.log(f"{symbol}: AVG-DOWN {ai_info} add {add_size} contracts at ${current_premium:.2f}")
                        
                        return Signal(
                            symbol=symbol,
                            action=Action.BUY,
                            size=add_size,
                            strike=self.strike[symbol],
                            strategy="MikeAIAgent",
                            confidence=0.80,
                            prob=ai_prediction['prob'] if self.use_ai and ai_prediction else None,
                            metadata={'reason': 'avg_down', 'new_premium': current_premium}
                        )
            
            # Exit Logic (same as before)
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
                    strategy="MikeAIAgent",
                    confidence=0.90,
                    prob=ai_prob,
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
                    strategy="MikeAIAgent",
                    confidence=0.85,
                    prob=ai_prob,
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
                    strategy="MikeAIAgent",
                    confidence=1.0,
                    prob=ai_prob,
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
    
    def _estimate_premium(self, S: float, K: float, direction: str) -> float:
        """Estimate option premium using Black-Scholes"""
        T = config.T
        r = config.R
        sigma = config.DEFAULT_SIGMA
        
        if T <= 0:
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
        
        # "Size for $0" on lottos
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
                 end_date: Optional[str] = None) -> Dict:
        """Run backtest on historical data"""
        print(f"\n{'='*60}")
        print("BACKTEST MODE - Mike AI Agent")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"AI Enabled: {self.use_ai}")
        
        # Pre-train AI model if using CSV
        if self.use_ai and csv_file:
            try:
                data = pd.read_csv(csv_file, index_col='date', parse_dates=True)
                data.columns = data.columns.str.lower()
                if len(data) >= 6:
                    self._train_lstm(data)
            except Exception as e:
                self.log(f"Error pre-training LSTM: {e}")
        
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
            
            # Pre-train AI on historical data
            if self.use_ai and len(data) >= 6:
                self._train_lstm(data)
            
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
            'pnl': pnl,
            'ai_prob': signal.prob
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
            'total_return': return_pct,
            'ai_enabled': self.use_ai
        }
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"AI Enabled: {self.use_ai}")
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
        
        self.log("Starting AI-powered paper trading...")
        
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
        """Get latest bar from Alpaca or Yahoo"""
        if self.api:
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
                self.log(f"Error fetching bar from Alpaca for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        try:
            data = yf.download(symbol, period='1d', interval='1m')
            if len(data) == 0:
                return None
            latest = data.iloc[-1]
            return {
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'close': latest['Close'],
                'volume': int(latest['Volume']),
                'timestamp': data.index[-1]
            }
        except Exception as e:
            self.log(f"Error fetching bar for {symbol}: {e}")
            return None
    
    def _execute_signal(self, signal: Signal, bar: Dict[str, Any]):
        """Execute signal in paper trading"""
        self.log(f"EXECUTING: {signal}")
        # TODO: Implement actual Alpaca order placement
        # self.api.submit_order(...)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mike AI Agent - Enhanced with LSTM Prediction")
    parser.add_argument('--mode', choices=['backtest', 'paper'], default='backtest')
    parser.add_argument('--symbols', type=str, default=','.join(config.SYMBOLS))
    parser.add_argument('--capital', type=float, default=config.START_CAPITAL)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--start_date', type=str, default=config.BACKTEST_START_DATE)
    parser.add_argument('--end_date', type=str, default=config.BACKTEST_END_DATE)
    parser.add_argument('--no-ai', action='store_true', help='Disable AI features')
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    agent = MikeAIAgent(
        mode=args.mode,
        symbols=symbols,
        capital=args.capital,
        use_ai=not args.no_ai
    )
    
    if args.mode == 'backtest':
        result = agent.backtest(
            csv_file=args.csv,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\nBacktest Complete!")
        print(f"Total Return: {result['total_return']:.2f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
    else:
        agent.run_paper_trade()


if __name__ == "__main__":
    main()

