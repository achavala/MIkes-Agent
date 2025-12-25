"""
Phase 0 Risk Book - Daily Risk State & Kill Switches

Tracks daily risk state and enforces hard daily loss limits.
Once a daily limit is hit, trading is halted for that day.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, date
import pytz


class RiskBook:
    """
    Phase 0 Risk Book
    
    Purpose: Track daily risk state and enforce kill switches.
    Philosophy: Once halted, no recovery attempts.
    """
    
    def __init__(self, daily_loss_limit: float = -250.0, max_trades_per_day: int = 5):
        """
        Initialize risk book
        
        Args:
            daily_loss_limit: Hard dollar loss limit per day (default: -$250)
            max_trades_per_day: Maximum trades per day (default: 5)
        """
        self.daily_loss_limit = daily_loss_limit
        self.max_trades_per_day = max_trades_per_day
        
        # Daily state
        self.current_date: Optional[date] = None
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.max_intraday_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.trades_taken: int = 0
        self.trading_halted: bool = False
        
        # Position tracking
        self.open_positions: Dict[str, dict] = {}
        
        # Daily history
        self.daily_history: list = []
        
        self.est = pytz.timezone('US/Eastern')
    
    def reset_day(self, trading_date: date, starting_equity: float = 10000.0):
        """
        Reset risk book for new trading day
        
        Args:
            trading_date: Date for this trading day
            starting_equity: Starting equity for the day
        """
        # Save previous day if exists
        if self.current_date is not None:
            self.daily_history.append({
                'date': self.current_date,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'total_pnl': self.realized_pnl + self.unrealized_pnl,
                'max_drawdown': self.max_intraday_drawdown,
                'trades_taken': self.trades_taken,
                'trading_halted': self.trading_halted
            })
        
        # Reset for new day
        self.current_date = trading_date
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_intraday_drawdown = 0.0
        self.peak_equity = starting_equity
        self.trades_taken = 0
        self.trading_halted = False
        self.open_positions = {}
    
    def can_open_new_trade(self) -> Tuple[bool, str]:
        """
        Check if new trade can be opened
        
        Returns:
            (can_open, reason_if_blocked)
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, "Trading halted (daily loss limit exceeded)"
        
        # Check max trades per day
        if self.trades_taken >= self.max_trades_per_day:
            return False, f"Max trades per day ({self.max_trades_per_day}) reached"
        
        # Check daily loss limit
        total_pnl = self.realized_pnl + self.unrealized_pnl
        if total_pnl <= self.daily_loss_limit:
            self.trading_halted = True
            return False, f"Daily loss limit (${self.daily_loss_limit:.2f}) exceeded: ${total_pnl:.2f}"
        
        return True, "OK"
    
    def record_trade(self, symbol: str, entry_price: float, qty: int, premium: float, action: int):
        """
        Record a new trade
        
        Args:
            symbol: Option symbol
            entry_price: Entry price (underlying price)
            qty: Number of contracts
            premium: Premium paid per contract
            action: Action taken (1=BUY_CALL, 2=BUY_PUT)
        """
        self.trades_taken += 1
        
        # Record position
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'entry_premium': premium,
            'qty': qty,
            'action': action,
            'entry_time': datetime.now(self.est),
            'cost': qty * premium * 100  # Total cost
        }
        
        # Update realized PnL (negative for entry cost)
        self.realized_pnl -= qty * premium * 100
    
    def close_position(self, symbol: str, exit_price: float, exit_premium: float) -> float:
        """
        Close a position and update PnL
        
        Args:
            symbol: Option symbol
            exit_price: Exit price (underlying price)
            exit_premium: Premium received per contract
        
        Returns:
            Realized PnL for this trade
        """
        if symbol not in self.open_positions:
            return 0.0
        
        pos = self.open_positions[symbol]
        entry_premium = pos['entry_premium']
        qty = pos['qty']
        
        # Calculate realized PnL
        trade_pnl = (exit_premium - entry_premium) * qty * 100
        self.realized_pnl += trade_pnl
        
        # Remove position
        del self.open_positions[symbol]
        
        return trade_pnl
    
    def update_unrealized_pnl(self, symbol: str, current_premium: float):
        """
        Update unrealized PnL for a position
        
        Args:
            symbol: Option symbol
            current_premium: Current premium per contract
        """
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        entry_premium = pos['entry_premium']
        qty = pos['qty']
        
        # Calculate unrealized PnL
        unrealized = (current_premium - entry_premium) * qty * 100
        self.unrealized_pnl = sum(
            (current_premium - self.open_positions[s]['entry_premium']) * self.open_positions[s]['qty'] * 100
            for s in self.open_positions.keys()
        )
        
        # Update peak equity and drawdown
        current_equity = self.peak_equity + self.realized_pnl + self.unrealized_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (current_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown < self.max_intraday_drawdown:
            self.max_intraday_drawdown = drawdown
    
    def daily_loss_exceeded(self) -> bool:
        """Check if daily loss limit exceeded"""
        total_pnl = self.realized_pnl + self.unrealized_pnl
        if total_pnl <= self.daily_loss_limit:
            self.trading_halted = True
            return True
        return False
    
    def finalize_day(self) -> dict:
        """
        Finalize current day and return summary
        
        Returns:
            Daily summary dictionary
        """
        # Close any remaining positions at end of day (assume worthless for 0DTE)
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            # Assume 0DTE options expire worthless if not closed
            self.close_position(symbol, 0.0, 0.01)  # $0.01 minimum value
        
        total_pnl = self.realized_pnl + self.unrealized_pnl
        
        summary = {
            'date': self.current_date,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': total_pnl,
            'max_drawdown': self.max_intraday_drawdown,
            'trades_taken': self.trades_taken,
            'trading_halted': self.trading_halted,
            'final_equity': self.peak_equity + total_pnl
        }
        
        return summary
    
    def get_current_state(self) -> dict:
        """Get current risk state"""
        return {
            'date': self.current_date,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'max_drawdown': self.max_intraday_drawdown,
            'trades_taken': self.trades_taken,
            'trading_halted': self.trading_halted,
            'open_positions': len(self.open_positions)
        }

