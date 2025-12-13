from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
import os
from datetime import datetime
import pytz
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import alpaca_trade_api as tradeapi
    import config
    from trade_database import TradeDatabase
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Dependencies not found")

app = FastAPI(title="Mike Agent API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---
def get_alpaca_api():
    if not ALPACA_AVAILABLE:
        return None
    return tradeapi.REST(
        config.ALPACA_KEY,
        config.ALPACA_SECRET,
        config.ALPACA_BASE_URL,
        api_version='v2'
    )

# --- Endpoints ---

@app.get("/api/portfolio")
async def get_portfolio():
    if not ALPACA_AVAILABLE:
        return {"equity": 0, "daily_pnl_pct": 0, "daily_pnl_usd": 0, "buying_power": 0, "currency": "USD"}
    
    try:
        api = get_alpaca_api()
        if not api:
            return {"equity": 0, "daily_pnl_pct": 0, "daily_pnl_usd": 0, "buying_power": 0, "currency": "USD"}
        
        account = api.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity) if hasattr(account, 'last_equity') and account.last_equity else equity
        
        daily_pnl_pct = ((equity - last_equity) / last_equity) * 100 if last_equity > 0 else 0.0
        daily_pnl_usd = equity - last_equity
        
        return {
            "equity": equity,
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "daily_pnl_usd": round(daily_pnl_usd, 2),
            "buying_power": float(account.buying_power),
            "currency": account.currency if hasattr(account, 'currency') else "USD"
        }
    except Exception as e:
        print(f"Error fetching portfolio: {e}")
        # Return fallback data instead of raising exception
        return {"equity": 0, "daily_pnl_pct": 0, "daily_pnl_usd": 0, "buying_power": 0, "currency": "USD"}

@app.get("/api/positions")
async def get_positions():
    if not ALPACA_AVAILABLE:
        return []
    
    try:
        api = get_alpaca_api()
        if not api:
            return []
        
        positions = api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price)
            }
            for p in positions
        ]
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []  # Return empty list instead of raising exception

@app.get("/api/agent/status")
async def get_agent_status():
    running = False
    # Check process
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'mike_agent_live_safe.py' in cmdline:
                    running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
        
    return {
        "status": "running" if running else "stopped",
        "last_heartbeat": datetime.now().isoformat(),
        "mode": "live"  # Or dynamic based on config
    }

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = 50):
    try:
        db = TradeDatabase()
        trades = db.get_all_trades()[-limit:]
        # Reverse to show newest first
        return trades[::-1]
    except Exception as e:
        # Fallback if DB fails
        return []

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    # In a real scenario, this would read shared state or DB
    # For now, returning configured limits
    return {
        "daily_loss_limit_pct": -15.0,
        "max_drawdown_pct": 30.0,
        "max_position_size_pct": 25.0,
        "vix_kill_switch": 28.0,
        "current_vix": 20.5  # Placeholder or fetch real
    }

# --- Backtesting Endpoints ---

class BacktestRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    symbols: List[str]
    initial_capital: float = 10000.0
    model_path: Optional[str] = None

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run backtest on historical data"""
    try:
        # Import backtesting modules
        MikeAgent = None
        try:
            from mike_agent import MikeAgent
        except ImportError:
            try:
                from mike_agent_enhanced import MikeAgent
            except ImportError:
                pass
        
        if MikeAgent is None:
            raise HTTPException(status_code=500, detail="MikeAgent module not found. Please ensure mike_agent.py or mike_agent_enhanced.py exists.")
        
        # Create agent instance - check what parameters it accepts
        try:
            # Try with mode parameter
            agent = MikeAgent(
                mode='backtest',
                symbols=request.symbols,
                capital=request.initial_capital
            )
        except TypeError:
            # Fallback: try without mode parameter
            agent = MikeAgent(
                symbols=request.symbols,
                capital=request.initial_capital
            )
        
        # Run backtest
        results = agent.backtest(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Format results - handle different return types
        if isinstance(results, dict):
            total_pnl = results.get('total_pnl', 0)
            return_pct = results.get('return_pct', (total_pnl / request.initial_capital * 100) if request.initial_capital > 0 else 0)
            final_capital = results.get('final_capital', request.initial_capital + total_pnl)
            
            return {
                "success": True,
                "total_pnl": round(float(total_pnl), 2),
                "return_pct": round(float(return_pct), 2),
                "total_trades": results.get('total_trades', 0),
                "win_rate": round(float(results.get('win_rate', 0)), 2),
                "max_drawdown": round(float(results.get('max_drawdown', 0)), 2),
                "final_capital": round(float(final_capital), 2),
                "trades": results.get('trades', [])
            }
        else:
            # If results is return_pct (float) - calculate total_pnl from it
            return_pct = float(results) if results is not None else 0.0
            total_pnl = (return_pct / 100.0) * request.initial_capital
            final_capital = request.initial_capital + total_pnl
            
            # Try to get additional info from agent if available
            total_trades = 0
            if hasattr(agent, 'trades') and agent.trades:
                total_trades = len(agent.trades)
            
            return {
                "success": True,
                "total_pnl": round(total_pnl, 2),
                "return_pct": round(return_pct, 2),
                "total_trades": total_trades,
                "win_rate": 0.0,  # Would need to calculate from trades
                "max_drawdown": 0.0,  # Would need to track during backtest
                "final_capital": round(final_capital, 2),
                "trades": []
            }
            
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.get("/api/backtest/available-dates")
async def get_available_dates():
    """Get available date range for backtesting"""
    from datetime import datetime, timedelta
    
    # Get today's date
    today = datetime.now().date()
    
    # Calculate earliest available date (e.g., 2 years back)
    earliest = today - timedelta(days=730)
    
    return {
        "earliest_date": earliest.strftime("%Y-%m-%d"),
        "latest_date": today.strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

