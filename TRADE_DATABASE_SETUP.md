# Trade Database & 0DTE Filtering Setup

## ✅ Completed Features

### 1. Persistent SQLite Database (`trade_database.py`)
- **Location**: `trades_database.db` (local file, never lost)
- **Features**:
  - Stores ALL trade history permanently
  - Automatic 0DTE detection from option symbols
  - Parses option symbols (SPY241203C00450000 format)
  - Tracks entry/exit premiums, PnL, regime, VIX
  - Indexed for fast queries
  - Backup and export functionality

### 2. Database Integration (`mike_agent_live_safe.py`)
- **Automatic Trade Saving**:
  - Saves every BUY order to database
  - Saves every SELL/exit to database
  - Includes all trade metadata (regime, VIX, PnL, etc.)
- **No Data Loss**: Trades are saved immediately, even if code changes

### 3. Dashboard 0DTE Filtering (`app.py`)
- **Current Positions**: Shows ONLY 0DTE positions
- **Trading History**: Shows ONLY 0DTE trades
- **Statistics**: Calculated from 0DTE trades only
- **Automatic Filtering**: All Alpaca data filtered to 0DTE before display

## How It Works

### Option Symbol Parsing
Alpaca option symbols: `SPY241203C00450000`
- `SPY` = Underlying
- `241203` = YYMMDD expiration date
- `C` = Call (P = Put)
- `00450000` = Strike price * 1000

The database automatically:
1. Parses the expiration date from symbol
2. Compares with today's date
3. Marks `is_0dte = 1` if expiration == today
4. Filters all queries to show only 0DTE when requested

### Database Schema
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    symbol TEXT,
    underlying TEXT,
    expiration_date TEXT,
    strike_price REAL,
    option_type TEXT,
    is_0dte INTEGER,  -- 1 if 0DTE, 0 otherwise
    action TEXT,      -- BUY or SELL
    qty INTEGER,
    entry_premium REAL,
    exit_premium REAL,
    pnl REAL,
    pnl_pct REAL,
    regime TEXT,
    vix REAL,
    reason TEXT,
    ...
)
```

## Usage

### View All Trades
```python
from trade_database import TradeDatabase

db = TradeDatabase()
all_trades = db.get_all_trades()
```

### View Only 0DTE Trades
```python
odte_trades = db.get_0dte_trades_only()
```

### Get Statistics
```python
# All trades
stats = db.get_trade_statistics(filter_0dte=False)

# 0DTE only
odte_stats = db.get_trade_statistics(filter_0dte=True)
```

### Backup Database
```python
backup_path = db.backup_database()
```

### Export to CSV
```python
csv_path = db.export_to_csv(filter_0dte=True)
```

## Testing

Run the test suite:
```bash
source venv/bin/activate
python test_trade_database.py
```

Tests validate:
- ✅ Database creation
- ✅ Trade saving
- ✅ 0DTE filtering
- ✅ Statistics calculation
- ✅ Dashboard filtering function

## Dashboard Changes

The dashboard (`app.py`) now:
1. **Filters positions** to show only 0DTE
2. **Filters trade history** to show only 0DTE
3. **Calculates statistics** from 0DTE trades only
4. **Uses database** as primary source (with Alpaca fallback)

## Database Location

- **File**: `trades_database.db` (in project root)
- **Backup**: Can create backups with `backup_database()`
- **Export**: Can export to CSV with `export_to_csv()`

## Important Notes

1. **Never Delete Database**: The database file contains all your trade history. Keep it safe!
2. **Automatic Saving**: Trades are saved automatically by the agent - no manual action needed
3. **0DTE Detection**: Works automatically from option symbols - no configuration needed
4. **Code Changes Safe**: Database persists even if you modify code or restart the agent

## Next Steps

1. ✅ Database created and tested
2. ✅ Agent integration complete
3. ✅ Dashboard filtering complete
4. ✅ Test suite validated

**Your trade history is now permanently stored and protected!**


