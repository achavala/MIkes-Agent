# 🎨 Mike Agent Professional Dashboard

Modern, resilient GUI dashboard that works across reboots and system changes.

## ✨ Features

- **🌐 Multi-Page Navigation**: Dashboard, Analytics, Trades, Agents, Risk Management, Settings
- **💾 Persistent State**: Settings and state saved to disk, survives reboots
- **🎨 Modern Dark Theme**: Professional look inspired by institutional trading platforms
- **🔄 Auto-Refresh**: Configurable auto-refresh with customizable intervals
- **🛡️ Error Recovery**: Graceful handling of missing data and API failures
- **📊 Real-Time Data**: Live portfolio, positions, and trade data from Alpaca
- **📈 Interactive Charts**: Plotly charts for performance visualization

## 🚀 Quick Start

### Option 1: Use Startup Script (Recommended)

```bash
./start_dashboard_pro.sh
```

### Option 2: Run Directly

```bash
streamlit run dashboard_app.py
```

The dashboard will open at: **http://localhost:8501**

## 📋 Pages

### 1. Dashboard
- Portfolio overview with total balance
- Today's P&L and trade count
- Active positions table
- Portfolio value chart
- P&L distribution chart
- Agent status indicator

### 2. Analytics
- Performance metrics (win rate, avg win/loss)
- Trade performance over time
- Symbol distribution charts
- Detailed statistics table

### 3. Trades
- Complete trade history
- Filters by symbol, date range, status
- Trade details view
- Export capabilities

### 4. Agents
- Agent status (running/stopped)
- Agent control (start/stop/restart)
- Agent configuration
- Model information

### 5. Risk Management
- Current risk limits
- Safeguards status
- Risk metrics over time
- Real-time risk monitoring

### 6. Settings
- Theme selection (Dark/Light)
- Auto-refresh configuration
- Timezone settings
- API configuration
- Trading preferences

## 🔧 Configuration

### Persistent Settings

Settings are saved to `dashboard_data/settings.json`:
- Theme preference
- Auto-refresh enabled/interval
- Default symbols
- Timezone

### State Management

Application state is saved to `dashboard_data/state.json`:
- Last refresh time
- Selected page
- View mode

## 🛡️ Resilience Features

1. **Graceful Degradation**: Works even if Alpaca API is unavailable
2. **Error Recovery**: Handles missing files and data gracefully
3. **Process Detection**: Detects agent status even without psutil
4. **File-Based Persistence**: Settings survive reboots
5. **Auto-Recovery**: Automatically recovers from errors

## 📦 Dependencies

All dependencies are in `requirements.txt`:
- streamlit
- plotly
- pandas
- numpy
- alpaca-trade-api (optional)
- psutil (optional, has fallback)

## 🎨 Customization

### Theme

Edit the CSS in `inject_custom_css()` function to customize:
- Colors
- Fonts
- Layout
- Animations

### Data Sources

Modify data fetching functions to:
- Add new data sources
- Change update intervals
- Add caching
- Implement real-time streaming

## 🔍 Troubleshooting

### Dashboard won't start

1. Check if port 8501 is available:
   ```bash
   lsof -i :8501
   ```

2. Kill existing Streamlit processes:
   ```bash
   pkill -f streamlit
   ```

3. Check dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data not showing

1. Check Alpaca API keys in `config.py`
2. Verify trade database exists
3. Check logs in `logs/` directory

### Settings not persisting

1. Check `dashboard_data/` directory exists
2. Verify write permissions
3. Check disk space

## 📝 Notes

- Dashboard works independently of the trading agent
- All data is read-only (no trading actions from dashboard)
- Settings are user-specific (stored locally)
- Auto-refresh can be disabled in Settings

## 🚀 Future Enhancements

- [ ] Real-time WebSocket updates
- [ ] Mobile-responsive design
- [ ] Export reports (PDF/CSV)
- [ ] Custom alerts and notifications
- [ ] Multi-account support
- [ ] Advanced charting tools
- [ ] Strategy backtesting interface

---

**Built for Mike Agent v3 - Professional Trading System**





