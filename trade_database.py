#!/usr/bin/env python3
"""
Persistent Trade Database - SQLite
Stores ALL trade history permanently - never loses data
"""
import sqlite3
import os
from datetime import datetime, date
from typing import List, Dict, Optional
import pandas as pd

# Database file path (persistent, not in git)
# Can be overridden via environment variable for shared access (Railway deployment)
DB_PATH = os.getenv('TRADES_DATABASE_PATH', "trades_database.db")

class TradeDatabase:
    """Persistent SQLite database for all trade history"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table with comprehensive fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                underlying TEXT,
                expiration_date TEXT,
                strike_price REAL,
                option_type TEXT,
                is_0dte INTEGER DEFAULT 0,
                action TEXT NOT NULL,
                qty INTEGER NOT NULL,
                entry_premium REAL,
                exit_premium REAL,
                entry_price REAL,
                exit_price REAL,
                fill_price REAL,
                pnl REAL,
                pnl_pct REAL,
                regime TEXT,
                vix REAL,
                reason TEXT,
                order_id TEXT,
                source TEXT DEFAULT 'alpaca',
                submitted_at TEXT,
                filled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol, action, qty)
            )
        """)
        
        # Create index for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_0dte ON trades(is_0dte)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expiration ON trades(expiration_date)
        """)
        
        conn.commit()
        conn.close()
    
    def _parse_option_symbol(self, symbol: str) -> Dict[str, any]:
        """Parse Alpaca option symbol to extract components"""
        # Format: SPY241203C00450000
        # SPY = underlying
        # 241203 = YYMMDD expiration
        # C = Call (P = Put)
        # 00450000 = Strike * 1000
        
        result = {
            'underlying': None,
            'expiration_date': None,
            'strike_price': None,
            'option_type': None,
            'is_0dte': False
        }
        
        try:
            if len(symbol) < 15:
                return result
            
            # Find underlying (SPY, QQQ, SPX, etc.)
            for i in range(len(symbol)):
                if symbol[i].isdigit():
                    result['underlying'] = symbol[:i]
                    date_str = symbol[i:i+6]  # YYMMDD
                    break
            
            if not result['underlying'] or not date_str:
                return result
            
            # Parse expiration date
            try:
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                exp_date = date(year, month, day)
                result['expiration_date'] = exp_date.isoformat()
                
                # Check if 0DTE (expires today)
                today = date.today()
                result['is_0dte'] = (exp_date == today)
            except:
                pass
            
            # Parse option type and strike
            if len(symbol) >= 15:
                option_type_char = symbol[9]  # C or P
                result['option_type'] = 'call' if option_type_char == 'C' else 'put'
                
                strike_str = symbol[10:]  # Last 8 digits
                result['strike_price'] = float(strike_str) / 1000.0
            
        except Exception as e:
            # If parsing fails, return partial result
            pass
        
        return result
    
    def save_trade(self, trade_data: Dict[str, any]) -> int:
        """Save a trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Parse option symbol to extract details
        symbol = trade_data.get('symbol', '')
        parsed = self._parse_option_symbol(symbol)
        
        # Prepare trade record
        trade_record = {
            'timestamp': trade_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'symbol': symbol,
            'underlying': parsed.get('underlying') or trade_data.get('underlying'),
            'expiration_date': parsed.get('expiration_date') or trade_data.get('expiration_date'),
            'strike_price': parsed.get('strike_price') or trade_data.get('strike_price'),
            'option_type': parsed.get('option_type') or trade_data.get('option_type'),
            'is_0dte': 1 if parsed.get('is_0dte') else 0,
            'action': trade_data.get('action', trade_data.get('side', 'BUY')),
            'qty': trade_data.get('qty', trade_data.get('quantity', 0)),
            'entry_premium': trade_data.get('entry_premium', trade_data.get('premium', 0)),
            'exit_premium': trade_data.get('exit_premium', 0),
            'entry_price': trade_data.get('entry_price', trade_data.get('price', 0)),
            'exit_price': trade_data.get('exit_price', 0),
            'fill_price': trade_data.get('fill_price', trade_data.get('filled_avg_price', 0)),
            'pnl': trade_data.get('pnl', 0),
            'pnl_pct': trade_data.get('pnl_pct', 0),
            'regime': trade_data.get('regime', trade_data.get('vol_regime')),
            'vix': trade_data.get('vix', 0),
            'reason': trade_data.get('reason', trade_data.get('metadata', '')),
            'order_id': trade_data.get('order_id', trade_data.get('id', '')),
            'source': trade_data.get('source', 'alpaca'),
            'submitted_at': trade_data.get('submitted_at', ''),
            'filled_at': trade_data.get('filled_at', '')
        }
        
        try:
            # Check if submitted_at and filled_at columns exist, add if not
            cursor.execute("PRAGMA table_info(trades)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'submitted_at' not in columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN submitted_at TEXT")
            if 'filled_at' not in columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN filled_at TEXT")
            
            # Use INSERT OR IGNORE to prevent duplicates and never delete trades
            # The UNIQUE constraint on (timestamp, symbol, action, qty) prevents exact duplicates
            cursor.execute("""
                INSERT OR IGNORE INTO trades (
                    timestamp, symbol, underlying, expiration_date, strike_price, option_type,
                    is_0dte, action, qty, entry_premium, exit_premium, entry_price, exit_price,
                    fill_price, pnl, pnl_pct, regime, vix, reason, order_id, source, submitted_at, filled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_record['timestamp'],
                trade_record['symbol'],
                trade_record['underlying'],
                trade_record['expiration_date'],
                trade_record['strike_price'],
                trade_record['option_type'],
                trade_record['is_0dte'],
                trade_record['action'],
                trade_record['qty'],
                trade_record['entry_premium'],
                trade_record['exit_premium'],
                trade_record['entry_price'],
                trade_record['exit_price'],
                trade_record['fill_price'],
                trade_record['pnl'],
                trade_record['pnl_pct'],
                trade_record['regime'],
                trade_record['vix'],
                trade_record['reason'],
                trade_record['order_id'],
                trade_record['source'],
                trade_record['submitted_at'],
                trade_record['filled_at']
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            return trade_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_all_trades(self, filter_0dte: bool = False) -> List[Dict]:
        """Get all trades from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades ORDER BY timestamp DESC"
        if filter_0dte:
            query = "SELECT * FROM trades WHERE is_0dte = 1 ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df.to_dict('records')
    
    def get_trades_by_date(self, start_date: str, end_date: str, filter_0dte: bool = False) -> List[Dict]:
        """Get trades within date range"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM trades 
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params = [start_date, end_date]
        
        if filter_0dte:
            query += " AND is_0dte = 1"
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df.to_dict('records')
    
    def get_0dte_trades_only(self) -> List[Dict]:
        """Get only 0DTE trades"""
        return self.get_all_trades(filter_0dte=True)
    
    def get_trade_statistics(self, filter_0dte: bool = False) -> Dict:
        """Get trade statistics"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades"
        if filter_0dte:
            query += " WHERE is_0dte = 1"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        total_trades = len(df[df['action'] == 'BUY'])
        winning_trades = len(df[(df['action'] == 'SELL') & (df['pnl'] > 0)])
        losing_trades = len(df[(df['action'] == 'SELL') & (df['pnl'] < 0)])
        
        total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0.0
        
        wins = df[(df['action'] == 'SELL') & (df['pnl'] > 0)]['pnl']
        losses = df[(df['action'] == 'SELL') & (df['pnl'] < 0)]['pnl']
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0.0,
            'total_pnl': float(total_pnl),
            'avg_win': float(wins.mean()) if len(wins) > 0 else 0.0,
            'avg_loss': float(losses.mean()) if len(losses) > 0 else 0.0
        }
    
    def get_daily_pnl_summary(self, filter_0dte: bool = False) -> pd.DataFrame:
        """
        Get daily P&L summary - groups trades by date and calculates daily totals
        Returns DataFrame with columns: date, num_trades, total_pnl, winning_trades, losing_trades
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(DISTINCT CASE WHEN action = 'BUY' THEN symbol || timestamp END) as num_trades,
                SUM(CASE WHEN action = 'SELL' THEN pnl ELSE 0 END) as total_pnl,
                SUM(CASE WHEN action = 'SELL' AND pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN action = 'SELL' AND pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(CASE WHEN action = 'SELL' AND pnl > 0 THEN pnl ELSE 0 END) as total_wins,
                SUM(CASE WHEN action = 'SELL' AND pnl < 0 THEN pnl ELSE 0 END) as total_losses
            FROM trades
        """
        
        if filter_0dte:
            query += " WHERE is_0dte = 1"
        
        query += " GROUP BY DATE(timestamp) ORDER BY date DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Calculate win rate per day
            df['win_rate'] = (df['winning_trades'] / (df['winning_trades'] + df['losing_trades']) * 100).fillna(0.0)
            # Round to 2 decimals
            df['total_pnl'] = df['total_pnl'].round(2)
            df['total_wins'] = df['total_wins'].round(2)
            df['total_losses'] = df['total_losses'].round(2)
            df['win_rate'] = df['win_rate'].round(1)
        
        return df
    
    def get_trades_by_date_range(self, start_date: str = None, end_date: str = None, filter_0dte: bool = False) -> pd.DataFrame:
        """
        Get all trades with full details, optionally filtered by date range
        Returns DataFrame with all trade columns
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if filter_0dte:
            query += " AND is_0dte = 1"
        
        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        return df
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create backup of database"""
        if backup_path is None:
            backup_path = f"trades_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return backup_path
    
    def export_to_csv(self, output_path: str = None, filter_0dte: bool = False) -> str:
        """Export trades to CSV"""
        if output_path is None:
            output_path = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades = self.get_all_trades(filter_0dte=filter_0dte)
        if trades:
            df = pd.DataFrame(trades)
            df.to_csv(output_path, index=False)
            return output_path
        return None


