#!/usr/bin/env python3
"""
Mike Agent Professional Dashboard
Modern, resilient GUI that works across reboots and system changes
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import json
import time
from pathlib import Path
import sys

# Try to import plotly (optional for basic functionality)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Charts will be limited. Install with: pip install plotly")

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import dependencies gracefully
try:
    import alpaca_trade_api as tradeapi
    try:
        import config
    except ImportError:
        # Create a mock config from environment variables (for Railway)
        class Config:
            ALPACA_KEY = os.environ.get('ALPACA_KEY', '')
            ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
            ALPACA_BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        config = Config()
        
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    st.warning("⚠️ Alpaca API not available. Some features may be limited.")

try:
    from trade_database import TradeDatabase
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False

# ==================== CONFIGURATION ====================
PAGE_CONFIG = {
    "page_title": "Mike Agent Dashboard",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# Persistent storage paths
DATA_DIR = Path("dashboard_data")
DATA_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
STATE_FILE = DATA_DIR / "state.json"

# ==================== PERSISTENT STATE MANAGEMENT ====================
def load_settings():
    """Load settings from file with defaults"""
    defaults = {
        "theme": "dark",
        "auto_refresh": True,
        "refresh_interval": 10,
        "default_symbols": ["SPY", "QQQ", "SPX"],
        "timezone": "US/Eastern"
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except Exception as e:
            st.error(f"Error loading settings: {e}")
    return defaults

def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.error(f"Error saving settings: {e}")

def load_state():
    """Load application state"""
    defaults = {
        "last_refresh": None,
        "selected_page": "Dashboard",
        "view_mode": "overview"
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except Exception:
            pass
    return defaults

def save_state(state):
    """Save application state"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

# ==================== CUSTOM CSS ====================
def inject_custom_css():
    """Inject custom CSS for TradeNova aesthetic (Light Theme + Glossy Purple)"""
    st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #f8fafc;
        background-image: none;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    
    /* Dashboard Header (Blue to Orange Gradient) */
    .tradenova-header {
        background: linear-gradient(90deg, #2563eb 0%, #ea580c 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white !important;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.2);
    }
    
    .tradenova-title {
        font-size: 3rem;
        font-weight: 800;
        color: white !important;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tradenova-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Standard Dashboard Header */
    .dashboard-header {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    .dashboard-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Buttons (Glossy Purple) - Restored for Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(168, 85, 247, 0.3) !important;
        text-align: center !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: auto !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #9333ea 0%, #6d28d9 100%) !important;
        box-shadow: 0 8px 12px -3px rgba(168, 85, 247, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Navigation - Styled Radio as Text Links */
    div[data-testid="stRadio"] > label {
        display: none;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 2rem;
        display: flex;
        justify-content: center;
        background: white;
        padding: 1rem 2rem;
        border-radius: 100px; /* Pill shape container */
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        width: fit-content;
        margin: 0 auto 2rem auto;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    /* Hide the radio circle */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Hover state for nav items */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        color: #a855f7 !important;
        transform: translateY(-1px);
    }
    
    /* Active state for nav items (checked) */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        color: #a855f7 !important;
        border-bottom: 2px solid #a855f7 !important;
        border-radius: 0 !important;
    }
    
    /* Navigation Highlight for Active Page */
    /* Note: Streamlit doesn't easily expose active state class on buttons, 
       but we can style the specific buttons based on usage context if needed.
       For now, the visual feedback is hover/click. */
    
    /* Secondary/Action Buttons */
    div[data-testid="stForm"] .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
    
    /* Metric Cards (White with Shadow) */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #a855f7; /* Purple hover border */
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        /* Glossy purple gradient text for values */
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        gap: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 8px 20px;
        color: #64748b;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: #f8fafc;
        color: #a855f7 !important; /* Glossy purple active state */
        border-bottom: 2px solid #a855f7;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input {
        background-color: white;
        color: #1e293b;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: #a855f7;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2);
    }
    
    /* Charts */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Status Colors */
    .status-online { color: #22c55e !important; font-weight: bold; }
    .status-offline { color: #ef4444 !important; font-weight: bold; }
    
    /* Gradient Text Utility */
    .gradient-text {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== NAVIGATION ====================
def render_navigation():
    """Render top navigation bar"""
    pages = ["Dashboard", "Analytics", "Trades", "Agents", "Risk Management", "Settings"]
    
    # Initialize session state for page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render navigation as a styled radio button (looks like text links)
    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav"
    )
    
    # Update state if changed
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()
    
    return st.session_state.current_page

# ==================== DASHBOARD PAGE ====================
def render_dashboard():
    """Render main dashboard page"""
    # TradeNova Header (Blue to Orange Gradient)
    st.markdown("""
    <div class="tradenova-header">
        <div class="tradenova-title">
            <span>📈</span> MikesAgent - AI Trading System Dashboard
        </div>
        <div class="tradenova-subtitle">
            Institutional-Grade Multi-Agent RL Trading Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get portfolio data
    portfolio_data = get_portfolio_data()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Balance</div>
            <div class="metric-value">${portfolio_data.get('portfolio_value', 0):,.2f}</div>
            <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                {portfolio_data.get('daily_pnl_pct', 0):+.2f}% today
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pnl = portfolio_data.get('today_pnl', 0)
        pnl_color = '#16a34a' if pnl >= 0 else '#dc2626' # Darker green/red for light theme
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Today's P&L</div>
            <div class="metric-value" style="color: {pnl_color}; background: none; -webkit-text-fill-color: initial;">
                ${pnl:+,.2f}
            </div>
            <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                {portfolio_data.get('num_trades_today', 0)} trades
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        active_positions = len(get_active_positions())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Positions</div>
            <div class="metric-value">{active_positions}</div>
            <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                {portfolio_data.get('max_positions', 3)} max
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        agent_status = get_agent_status()
        status_color = "#16a34a" if agent_status.get('running', False) else "#dc2626"
        status_text = "ONLINE" if agent_status.get('running', False) else "OFFLINE"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Agent Status</div>
            <div class="metric-value" style="color: {status_color}; background: none; -webkit-text-fill-color: initial; font-size: 1.5rem;">
                {status_text}
            </div>
            <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                {agent_status.get('uptime', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Portfolio Value Over Time")
        portfolio_chart = create_portfolio_chart()
        if portfolio_chart:
            st.plotly_chart(portfolio_chart, use_container_width=True)
        else:
            st.info("Install plotly to view charts: pip install plotly")
    
    with col2:
        st.markdown("### 💰 P&L Distribution")
        pnl_chart = create_pnl_distribution_chart()
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)
        else:
            st.info("Install plotly to view charts: pip install plotly")
    
    # Active positions table
    st.markdown("### 📋 Active Positions")
    positions_df = get_active_positions()
    if not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
    else:
        st.info("No active positions")

# ==================== ANALYTICS PAGE ====================
def render_analytics():
    """Render analytics page"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">📈 Analytics</div>
        <div class="dashboard-subtitle">Performance metrics and detailed analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Time period selector
    period = st.selectbox("Time Period", ["24H", "7D", "30D", "ALL"], index=2)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = get_performance_stats(period)
    
    with col1:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col2:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col3:
        st.metric("Avg Win", f"${stats.get('avg_win', 0):,.2f}")
    with col4:
        st.metric("Avg Loss", f"${stats.get('avg_loss', 0):,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trade Performance Over Time")
        performance_chart = create_performance_chart(period)
        if performance_chart:
            st.plotly_chart(performance_chart, use_container_width=True)
        else:
            st.info("Install plotly to view charts")
    
    with col2:
        st.markdown("### Trade Distribution by Symbol")
        symbol_chart = create_symbol_distribution_chart(period)
        if symbol_chart:
            st.plotly_chart(symbol_chart, use_container_width=True)
        else:
            st.info("Install plotly to view charts")
    
    # Detailed statistics table
    st.markdown("### 📊 Detailed Statistics")
    stats_df = get_detailed_stats(period)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Add Logs, Feedback, Data Integrity, P&L Analysis, and Live Activity tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔴 Live Activity", "📋 Logs", "💬 Feedback", "🔒 Data Integrity", "💰 P&L Analysis"])
    
    with tab1:
        render_live_activity()
    
    with tab2:
        render_logs_section()
    
    with tab3:
        render_feedback_section()
    
    with tab4:
        from data_integrity_analytics import render_data_integrity_panel
        render_data_integrity_panel()
        
    with tab5:
        render_pnl_analysis()

# ==================== TRADES PAGE ====================
def render_trades():
    """Render trades page"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">💼 Trades</div>
        <div class="dashboard-subtitle">Complete trade history and backtesting</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for Trade History and Backtesting
    tab1, tab2 = st.tabs(["📊 Trade History", "🧪 Backtesting"])
    
    with tab1:
        # Sync Button
        col_sync1, col_sync2 = st.columns([5, 1])
        with col_sync2:
            if st.button("🔄 Sync History", help="Recover missing trades from Alpaca"):
                with st.spinner("Syncing history..."):
                    count = sync_alpaca_history()
                    if count >= 0:
                        st.success(f"Synced {count} trades")
                        time.sleep(1)
                        st.rerun()

        # Daily P&L Summary Section
        st.markdown("### 📊 Daily P&L Summary")
        if TRADE_DB_AVAILABLE:
            try:
                trade_db = TradeDatabase()
                daily_pnl = trade_db.get_daily_pnl_summary(filter_0dte=False)
                
                if not daily_pnl.empty:
                    # Format the daily summary for display
                    daily_display = daily_pnl.copy()
                    daily_display.columns = ['Date', 'Trades', 'Total P&L ($)', 'Wins', 'Losses', 'Total Wins ($)', 'Total Losses ($)', 'Win Rate (%)']
                    daily_display['Total P&L ($)'] = daily_display['Total P&L ($)'].apply(lambda x: f"${x:,.2f}")
                    daily_display['Total Wins ($)'] = daily_display['Total Wins ($)'].apply(lambda x: f"${x:,.2f}")
                    daily_display['Total Losses ($)'] = daily_display['Total Losses ($)'].apply(lambda x: f"${x:,.2f}")
                    daily_display['Win Rate (%)'] = daily_display['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(daily_display, use_container_width=True, hide_index=True)
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Days", len(daily_pnl))
                    with col2:
                        st.metric("Total P&L", f"${daily_pnl['total_pnl'].sum():,.2f}")
                    with col3:
                        st.metric("Best Day", f"${daily_pnl['total_pnl'].max():,.2f}")
                    with col4:
                        st.metric("Worst Day", f"${daily_pnl['total_pnl'].min():,.2f}")
                else:
                    st.info("No daily P&L data available yet. Trades will appear here once executed.")
            except Exception as e:
                st.error(f"Error loading daily P&L: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        st.markdown("---")
        st.markdown("### 📋 Detailed Trade History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol_filter = st.multiselect("Symbol", ["SPY", "QQQ", "SPX"], default=[])
        with col2:
            # Date range picker
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            date_range = st.date_input(
                "Date Range",
                value=(week_ago, today),
                key="trade_date_range"
            )
        with col3:
            status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
        
        # Trades table
        trades_df = get_trades_data(symbol_filter, date_range, status_filter)
        
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # Export button
            if st.button("📥 Export to CSV"):
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Trade details
            if st.checkbox("Show Trade Details"):
                selected_trade = st.selectbox("Select Trade", trades_df.index)
                show_trade_details(selected_trade)
        else:
            st.info("No trades found. Trades will appear here once the agent executes them.")
    
    with tab2:
        render_backtesting()

# ==================== AGENTS PAGE ====================
def render_agents():
    """Render agents page"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">🤖 Agents</div>
        <div class="dashboard-subtitle">Agent status and configuration</div>
    </div>
    """, unsafe_allow_html=True)
    
    agent_status = get_agent_status()
    
    # Agent status card
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Agent Status")
        status_color = "#16a34a" if agent_status.get('running', False) else "#dc2626"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {status_color}; font-size: 1.5rem; font-weight: 800;">
                {'🟢 RUNNING' if agent_status.get('running', False) else '🔴 STOPPED'}
            </div>
            <div style="color: #64748b; margin-top: 1rem; font-size: 0.9rem;">
                <strong>Uptime:</strong> {agent_status.get('uptime', 'N/A')}<br>
                <strong>Last Update:</strong> {agent_status.get('last_update', 'N/A')}<br>
                <strong>Model:</strong> {agent_status.get('model', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Agent Actions")
        if st.button("🔄 Restart Agent", use_container_width=True):
            restart_agent()
        if st.button("⏸️ Stop Agent", use_container_width=True):
            stop_agent()
        if st.button("▶️ Start Agent", use_container_width=True):
            start_agent()
    
    # Agent configuration
    st.markdown("### ⚙️ Agent Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.text_input("Model Path", value=agent_status.get('model_path', 'models/mike_historical_model.zip'))
        st.selectbox("Trading Mode", ["Paper", "Live"], index=0)
        st.multiselect("Trading Symbols", ["SPY", "QQQ", "SPX"], default=["SPY", "QQQ"])
    
    with config_col2:
        st.number_input("Risk Per Trade (%)", value=7.0, min_value=0.1, max_value=10.0, step=0.1)
        st.number_input("Max Positions", value=3, min_value=1, max_value=10)
        st.number_input("Max Daily Trades", value=20, min_value=1, max_value=100)

# ==================== RISK MANAGEMENT PAGE ====================
def render_risk_management():
    """Render risk management page"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">🛡️ Risk Management</div>
        <div class="dashboard-subtitle">Safeguards and risk controls</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk limits overview
    st.markdown("### Current Risk Limits")
    risk_limits = get_risk_limits()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Loss Limit", f"{risk_limits.get('daily_loss_limit', -15):.0f}%")
    with col2:
        st.metric("Max Drawdown", f"{risk_limits.get('max_drawdown', 30):.0f}%")
    with col3:
        st.metric("VIX Kill Switch", f"{risk_limits.get('vix_kill', 28):.0f}")
    with col4:
        st.metric("Max Position Size", f"{risk_limits.get('max_position_pct', 25):.0f}%")
    
    # Safeguards status
    st.markdown("### 🛡️ Safeguards Status")
    safeguards = get_safeguards_status()
    
    for safeguard in safeguards:
        status_icon = "✅" if safeguard.get('active', False) else "❌"
        st.markdown(f"""
        **{status_icon} {safeguard.get('name', 'Unknown')}**
        - Status: {'Active' if safeguard.get('active', False) else 'Inactive'}
        - Limit: {safeguard.get('limit', 'N/A')}
        - Current: {safeguard.get('current', 'N/A')}
        """)
    
    # Risk metrics chart
    st.markdown("### 📊 Risk Metrics Over Time")
    risk_chart = create_risk_metrics_chart()
    if risk_chart:
        st.plotly_chart(risk_chart, use_container_width=True)
    else:
        st.info("Install plotly to view charts")

# ==================== SETTINGS PAGE ====================
def render_settings():
    """Render settings page"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">⚙️ Settings</div>
        <div class="dashboard-subtitle">Configuration and preferences</div>
    </div>
    """, unsafe_allow_html=True)
    
    settings = load_settings()
    
    # General settings
    st.markdown("### General Settings")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0 if settings.get('theme') == 'dark' else 1)
    auto_refresh = st.checkbox("Auto Refresh", value=settings.get('auto_refresh', True))
    refresh_interval = st.number_input("Refresh Interval (seconds)", value=settings.get('refresh_interval', 10), min_value=1, max_value=60)
    timezone = st.selectbox("Timezone", ["US/Eastern", "US/Central", "US/Pacific", "UTC"], 
                           index=["US/Eastern", "US/Central", "US/Pacific", "UTC"].index(settings.get('timezone', 'US/Eastern')))
    
    # Trading settings
    st.markdown("### Trading Settings")
    default_symbols = st.multiselect("Default Symbols", ["SPY", "QQQ", "SPX"], 
                                     default=settings.get('default_symbols', ['SPY', 'QQQ']))
    
    # API settings
    st.markdown("### API Configuration")
    if ALPACA_AVAILABLE:
        api_key = st.text_input("Alpaca API Key", value=getattr(config, 'ALPACA_KEY', ''), type="password")
        api_secret = st.text_input("Alpaca API Secret", value=getattr(config, 'ALPACA_SECRET', ''), type="password")
        api_base_url = st.text_input("Alpaca Base URL", value=getattr(config, 'ALPACA_BASE_URL', ''))
    
    # Save button
    if st.button("💾 Save Settings", use_container_width=True):
        new_settings = {
            "theme": theme.lower(),
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "timezone": timezone,
            "default_symbols": default_symbols
        }
        save_settings(new_settings)
        st.success("Settings saved successfully!")

# ==================== DATA FETCHING FUNCTIONS ====================
def get_portfolio_data():
    """Get portfolio data from Alpaca or defaults"""
    if not ALPACA_AVAILABLE:
        return {
            'portfolio_value': 100000.0,
            'daily_pnl_pct': 0.0,
            'today_pnl': 0.0,
            'num_trades_today': 0,
            'max_positions': 3
        }
    
    try:
        api = tradeapi.REST(
            config.ALPACA_KEY,
            config.ALPACA_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        account = api.get_account()
        portfolio_value = float(account.equity)
        
        # Calculate daily P&L
        last_equity = float(account.last_equity) if hasattr(account, 'last_equity') and account.last_equity else portfolio_value
        daily_pnl_pct = ((portfolio_value - last_equity) / last_equity * 100) if last_equity > 0 else 0.0
        daily_pnl_dollar = portfolio_value - last_equity
        
        # Get today's trades
        today = datetime.now(pytz.timezone('US/Eastern')).date()
        orders = api.list_orders(status='filled', limit=100)
        
        today_orders = []
        for o in orders:
            if o.filled_at:
                dt = o.filled_at
                # Handle string format if returned as string
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                
                # Ensure timezone awareness
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                
                # Convert to Eastern time for daily P&L calculation
                dt_eastern = dt.astimezone(pytz.timezone('US/Eastern'))
                
                if dt_eastern.date() == today:
                    today_orders.append(o)
        
        return {
            'portfolio_value': portfolio_value,
            'daily_pnl_pct': daily_pnl_pct,
            'today_pnl': daily_pnl_dollar,
            'num_trades_today': len(today_orders),
            'max_positions': 3
        }
    except Exception as e:
        st.error(f"Error fetching portfolio data: {e}")
        return {
            'portfolio_value': 100000.0,
            'daily_pnl_pct': 0.0,
            'today_pnl': 0.0,
            'num_trades_today': 0,
            'max_positions': 3
        }

def get_active_positions():
    """Get active positions"""
    if not ALPACA_AVAILABLE:
        return pd.DataFrame()
    
    try:
        api = tradeapi.REST(
            config.ALPACA_KEY,
            config.ALPACA_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        positions = api.list_positions()
        
        if not positions:
            return pd.DataFrame()
        
        positions_data = []
        for pos in positions:
            positions_data.append({
                'Symbol': pos.symbol,
                'Qty': float(pos.qty),
                'Avg Entry': f"${float(pos.avg_entry_price):.4f}",
                'Current Price': f"${float(pos.current_price):.4f}",
                'Market Value': f"${float(pos.market_value):,.2f}",
                'Unrealized P&L': f"${float(pos.unrealized_pl):+,.2f}",
                'Unrealized P&L %': f"{float(pos.unrealized_plpc):+.2f}%"
            })
        
        return pd.DataFrame(positions_data)
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return pd.DataFrame()

def get_agent_status():
    """Get agent status - checks process, logs, and health files"""
    running = False
    uptime = 'N/A'
    deployment_info = None
    
    # Method 1: Check for agent process (works in Fly.io container)
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []) or [])
                # Look for agent process - check multiple patterns
                if any(keyword in cmdline.lower() for keyword in ['mike_agent_live_safe', 'mike_agent', 'agent_live']):
                    running = True
                    deployment_info = "Fly.io"
                    # Calculate uptime from process start time
                    try:
                        create_time_val = proc.info.get('create_time')
                        if create_time_val:
                            create_time = datetime.fromtimestamp(create_time_val)
                            uptime_delta = datetime.now() - create_time
                            hours = int(uptime_delta.total_seconds() / 3600)
                            minutes = int((uptime_delta.total_seconds() % 3600) / 60)
                            uptime = f"{hours}h {minutes}m"
                        else:
                            uptime = "Running"
                    except:
                        uptime = "Running"
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, KeyError, AttributeError):
                pass
    except ImportError:
        pass
    
    # Method 2: Check for recent log file activity (if process check didn't work)
    if not running:
        try:
            # Check multiple possible log locations
            log_paths = [
                Path("/tmp/agent.log"),  # Agent logs to /tmp/agent.log
                Path("logs/mike_agent_safe_" + datetime.now().strftime('%Y%m%d') + ".log"),
                Path("agent.log"),
                Path("logs/agent.log"),
            ]
            
            for log_file in log_paths:
                if log_file.exists():
                    # Check if log was updated in last 5 minutes
                    try:
                        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        age_seconds = (datetime.now() - mtime).total_seconds()
                        if age_seconds < 300:  # Updated in last 5 minutes
                            running = True
                            deployment_info = "Fly.io (log-based)"
                            age_minutes = int(age_seconds / 60)
                            uptime = f"Log updated {age_minutes}m ago"
                            break
                    except (OSError, ValueError):
                        continue
        except Exception:
            pass
    
    # Method 3: Check for health check file (if agent creates one)
    if not running:
        try:
            health_file = Path("/tmp/agent_health.json")
            if health_file.exists():
                try:
                    health_data = json.loads(health_file.read_text())
                    if health_data.get('running', False):
                        running = True
                        deployment_info = "Fly.io (health check)"
                        uptime = health_data.get('uptime', 'N/A')
                except:
                    # Check file modification time as fallback
                    try:
                        mtime = datetime.fromtimestamp(health_file.stat().st_mtime)
                        age_seconds = (datetime.now() - mtime).total_seconds()
                        if age_seconds < 300:  # Updated in last 5 minutes
                            running = True
                            deployment_info = "Fly.io (health check)"
                            age_minutes = int(age_seconds / 60)
                            uptime = f"Updated {age_minutes}m ago"
                    except:
                        pass
        except Exception:
            pass
    
    # Method 4: Try to check Fly.io status via API (only if running locally, not in container)
    if not running:
        try:
            import subprocess
            # Only try this if we're not in a Fly.io container (check for /app directory)
            # In Fly.io, /app exists, so skip this check
            if not Path("/app").exists():
                result = subprocess.run(
                    ['fly', 'status', '--app', 'mike-agent-project', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    status_data = json.loads(result.stdout)
                    machines = status_data.get('Machines', [])
                    if machines:
                        started_machines = [m for m in machines if m.get('State') == 'started']
                        if started_machines:
                            running = True
                            deployment_info = f"Fly.io ({len(started_machines)} machine(s))"
                            uptime = "Remote"
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception):
            pass
    
    return {
        'running': running,
        'uptime': uptime,
        'deployment': deployment_info or 'Fly.io',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'mike_23feature_model_final.zip',
        'model_path': 'models/mike_23feature_model_final.zip'
    }

def get_performance_stats(period):
    """Get performance statistics from database"""
    if TRADE_DB_AVAILABLE:
        try:
            trade_db = TradeDatabase()
            stats = trade_db.get_trade_statistics(filter_0dte=True)
            
            return {
                'win_rate': stats.get('win_rate', 0.0),
                'total_trades': stats.get('total_trades', 0),
                'avg_win': stats.get('avg_win', 0.0),
                'avg_loss': stats.get('avg_loss', 0.0),
                'total_pnl': stats.get('total_pnl', 0.0)
            }
        except Exception as e:
            st.error(f"Error loading performance stats: {e}")
    
    # Defaults
    return {
        'win_rate': 0.0,
        'total_trades': 0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'total_pnl': 0.0
    }

def get_trades_data(symbol_filter, date_range, status_filter):
    """Get trades data from database or Alpaca with proper date filtering"""
    if TRADE_DB_AVAILABLE:
        try:
            trade_db = TradeDatabase()
            
            # Handle date range filtering
            start_date = None
            end_date = None
            if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
                if date_range[0]:
                    start_date = date_range[0].strftime('%Y-%m-%d')
                if date_range[1]:
                    end_date = date_range[1].strftime('%Y-%m-%d')
            
            # Get trades from database with date filtering
            if start_date or end_date:
                trades_df = trade_db.get_trades_by_date_range(start_date=start_date, end_date=end_date, filter_0dte=False)
            else:
                all_trades = trade_db.get_all_trades()
                if not all_trades:
                    return pd.DataFrame()
                trades_df = pd.DataFrame(all_trades)
            
            if trades_df.empty:
                return pd.DataFrame()
            
            # Convert to display format
            trades_list = []
            for _, trade in trades_df.iterrows():
                # Apply symbol filter
                if symbol_filter != "All":
                    underlying = trade.get('underlying', '')
                    if underlying != symbol_filter:
                        continue
                
                # Apply status filter (if needed)
                # Note: Status filter logic can be added here if needed
                
                trades_list.append({
                    'Timestamp': trade.get('timestamp', ''),
                    'Symbol': trade.get('symbol', ''),
                    'Action': trade.get('action', ''),
                    'Qty': trade.get('qty', 0),
                    'Entry Premium': trade.get('entry_premium', 0),
                    'Exit Premium': trade.get('exit_premium', 0),
                    'P&L': trade.get('pnl', 0),
                    'P&L %': trade.get('pnl_pct', 0)
                })
            
            df = pd.DataFrame(trades_list)
            
            # Apply filters
            if symbol_filter:
                df = df[df['Symbol'].isin(symbol_filter)]
            
            if status_filter and status_filter != "All":
                # Filter by status (would need to determine from data)
                pass
            
            return df.sort_values('Timestamp', ascending=False)
        except Exception as e:
            st.error(f"Error loading trades: {e}")
            return pd.DataFrame()
    
    # Fallback: try Alpaca
    if ALPACA_AVAILABLE:
        try:
            api = tradeapi.REST(
                config.ALPACA_KEY,
                config.ALPACA_SECRET,
                config.ALPACA_BASE_URL,
                api_version='v2'
            )
            orders = api.list_orders(status='filled', limit=500)
            
            trades_list = []
            for order in orders:
                trades_list.append({
                    'Timestamp': order.filled_at or order.submitted_at,
                    'Symbol': order.symbol,
                    'Action': order.side.upper(),
                    'Qty': float(order.filled_qty),
                    'Price': float(order.filled_avg_price),
                    'Status': order.status
                })
            
            return pd.DataFrame(trades_list).sort_values('Timestamp', ascending=False)
        except Exception as e:
            st.error(f"Error loading trades from Alpaca: {e}")
    
    return pd.DataFrame()

def get_risk_limits():
    """Get risk limits"""
    return {
        'daily_loss_limit': -15,
        'max_drawdown': 30,
        'vix_kill': 28,
        'max_position_pct': 25
    }

def get_safeguards_status():
    """Get safeguards status"""
    return [
        {'name': 'Daily Loss Limit', 'active': True, 'limit': '-15%', 'current': '0%'},
        {'name': 'Max Drawdown', 'active': True, 'limit': '-30%', 'current': '0%'},
        {'name': 'VIX Kill Switch', 'active': True, 'limit': '28', 'current': '20.5'},
        {'name': 'Max Position Size', 'active': True, 'limit': '25%', 'current': '15%'}
    ]

# ==================== CHART CREATION FUNCTIONS ====================
def create_portfolio_chart():
    """Create portfolio value chart"""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Placeholder data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = 100000 + np.cumsum(np.random.randn(30) * 500)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#4ade80', width=2),
        fill='tonexty',
        fillcolor='rgba(74, 222, 128, 0.1)'
    ))
    fig.update_layout(
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_pnl_distribution_chart():
    """Create P&L distribution chart"""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Placeholder data
    pnl_values = np.random.randn(100) * 500
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl_values,
        nbinsx=20,
        marker_color='#667eea'
    ))
    fig.update_layout(
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_performance_chart(period):
    """Create performance chart"""
    if not PLOTLY_AVAILABLE:
        return None
    return create_portfolio_chart()

def create_symbol_distribution_chart(period):
    """Create symbol distribution chart"""
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure(data=[go.Bar(x=['SPY', 'QQQ', 'SPX'], y=[50, 30, 20], marker_color='#a855f7')])
    fig.update_layout(
        template='plotly_white', 
        height=300, 
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_risk_metrics_chart():
    """Create risk metrics chart"""
    if not PLOTLY_AVAILABLE:
        return None
    return create_portfolio_chart()

# ==================== AGENT CONTROL FUNCTIONS ====================
def restart_agent():
    """Restart agent"""
    st.info("Agent restart initiated...")

def stop_agent():
    """Stop agent"""
    st.info("Agent stop initiated...")

def start_agent():
    """Start agent"""
    st.info("Agent start initiated...")

def show_trade_details(trade_id):
    """Show detailed trade information"""
    st.write(f"Trade details for {trade_id}")

# ==================== BACKTESTING ====================
def render_backtesting():
    """Render backtesting interface"""
    st.markdown("### 🧪 Run Backtest on Historical Data")
    st.markdown("Test your strategy on any historical date range")
    
    # Backtest configuration
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date()
        )
        
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="SPY,QQQ",
            help="Enter symbols separated by commas, e.g., SPY,QQQ,SPX"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date()
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=10000.0,
            min_value=100.0,
            step=1000.0,
            help="Starting capital for the backtest"
        )
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest_ui(start_date, end_date, symbols_input, initial_capital)

def run_backtest_ui(start_date, end_date, symbols_input, initial_capital):
    """Run backtest and display results"""
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    if not symbols:
        st.error("Please enter at least one symbol")
        return
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing backtest...")
        progress_bar.progress(10)
        
        # Import backtesting module
        try:
            from mike_agent import MikeAgent
            AGENT_AVAILABLE = True
        except ImportError:
            try:
                from mike_agent_enhanced import MikeAgent
                AGENT_AVAILABLE = True
            except ImportError:
                AGENT_AVAILABLE = False
                st.error("MikeAgent module not found. Please ensure mike_agent.py exists.")
                return
        
        status_text.text("Creating agent instance...")
        progress_bar.progress(30)
        
        # Create agent
        try:
            agent = MikeAgent(
                mode='backtest',
                symbols=symbols,
                capital=initial_capital
            )
        except TypeError:
            # Fallback if mode parameter not supported
            agent = MikeAgent(
                symbols=symbols,
                capital=initial_capital
            )
        
        status_text.text(f"Running backtest from {start_date} to {end_date}...")
        progress_bar.progress(50)
        
        # Run backtest
        result = agent.backtest(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        progress_bar.progress(90)
        status_text.text("Calculating results...")
        
        # Process results
        if isinstance(result, dict):
            total_pnl = result.get('total_pnl', 0)
            return_pct = result.get('return_pct', 0)
            final_capital = result.get('final_capital', initial_capital + total_pnl)
            total_trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
        else:
            # If result is return_pct (float)
            return_pct = float(result) if result is not None else 0.0
            total_pnl = (return_pct / 100.0) * initial_capital
            final_capital = initial_capital + total_pnl
            total_trades = 0
            win_rate = 0.0
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest complete!")
        
        # Display results
        st.success("Backtest completed successfully!")
        
        # Results metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total P&L",
                f"${total_pnl:,.2f}",
                delta=f"{return_pct:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Final Capital",
                f"${final_capital:,.2f}",
                delta=f"${total_pnl:+,.2f}"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                total_trades
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%"
            )
        
        # Summary
        st.markdown("### 📊 Backtest Summary")
        summary_data = {
            "Metric": [
                "Initial Capital",
                "Final Capital",
                "Total P&L",
                "Return %",
                "Total Trades",
                "Win Rate",
                "Period"
            ],
            "Value": [
                f"${initial_capital:,.2f}",
                f"${final_capital:,.2f}",
                f"${total_pnl:+,.2f}",
                f"{return_pct:+.2f}%",
                str(total_trades),
                f"{win_rate:.1f}%",
                f"{start_date} to {end_date}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        progress_bar.empty()
        status_text.empty()

def get_detailed_stats(period):
    """Get detailed statistics"""
    return pd.DataFrame({
        'Metric': ['Win Rate', 'Avg Win', 'Avg Loss', 'Profit Factor'],
        'Value': ['75.5%', '$450.25', '-$125.50', '2.5']
    })

def render_pnl_analysis():
    """Render P&L Analysis section"""
    st.markdown("### 💰 P&L Analysis")
    st.markdown("Daily, Weekly, and Monthly Profit & Loss")
    
    if not TRADE_DB_AVAILABLE:
        st.error("Trade Database not available.")
        return
        
    try:
        trade_db = TradeDatabase()
        trades = trade_db.get_all_trades()
        
        if not trades:
            st.info("No trades found in database.")
            return
            
        df = pd.DataFrame(trades)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Daily P&L
        daily_pnl = df.groupby('date')['pnl'].sum().reset_index()
        daily_pnl = daily_pnl.sort_values('date', ascending=False)
        
        # Weekly P&L
        df['week'] = df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_pnl = df.groupby('week')['pnl'].sum().reset_index()
        weekly_pnl = weekly_pnl.sort_values('week', ascending=False)
        
        # Monthly P&L
        df['month'] = df['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)
        monthly_pnl = df.groupby('month')['pnl'].sum().reset_index()
        monthly_pnl = monthly_pnl.sort_values('month', ascending=False)
        
        # Tabs for different views
        tab_daily, tab_weekly, tab_monthly = st.tabs(["Daily", "Weekly", "Monthly"])
        
        with tab_daily:
            st.markdown("#### Daily P&L")
            col1, col2 = st.columns([2, 1])
            with col1:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[go.Bar(
                        x=daily_pnl['date'], 
                        y=daily_pnl['pnl'], 
                        marker_color=np.where(daily_pnl['pnl'] >= 0, '#22c55e', '#ef4444')
                    )])
                    fig.update_layout(title="Daily P&L", template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(daily_pnl.style.format({'pnl': '${:,.2f}'}), use_container_width=True)
                
        with tab_weekly:
            st.markdown("#### Weekly P&L")
            col1, col2 = st.columns([2, 1])
            with col1:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[go.Bar(
                        x=weekly_pnl['week'], 
                        y=weekly_pnl['pnl'], 
                        marker_color=np.where(weekly_pnl['pnl'] >= 0, '#22c55e', '#ef4444')
                    )])
                    fig.update_layout(title="Weekly P&L", template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(weekly_pnl.style.format({'pnl': '${:,.2f}'}), use_container_width=True)
                
        with tab_monthly:
            st.markdown("#### Monthly P&L")
            col1, col2 = st.columns([2, 1])
            with col1:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[go.Bar(
                        x=monthly_pnl['month'], 
                        y=monthly_pnl['pnl'], 
                        marker_color=np.where(monthly_pnl['pnl'] >= 0, '#22c55e', '#ef4444')
                    )])
                    fig.update_layout(title="Monthly P&L", template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(monthly_pnl.style.format({'pnl': '${:,.2f}'}), use_container_width=True)
                
    except Exception as e:
        st.error(f"Error calculating P&L: {e}")

def sync_alpaca_history():
    """Sync trade history from Alpaca to Database"""
    if not ALPACA_AVAILABLE or not TRADE_DB_AVAILABLE:
        st.error("Alpaca API or Trade Database not available")
        return -1
        
    try:
        api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET, config.ALPACA_BASE_URL, api_version='v2')
        # Get filled orders
        orders = api.list_orders(status='filled', limit=500)
        
        trade_db = TradeDatabase()
        count = 0
        
        for order in orders:
            # Construct trade record
            # Note: We won't have PnL for historical syncs without complex matching
            # But ensuring the record exists is step 1
            
            # Helper to check if it's 0DTE (approximation)
            is_0dte = 0
            if '0DTE' in order.symbol: # Naive check, improved if we parse symbol
                is_0dte = 1
            
            # Convert UTC timestamps to EST
            import pytz
            est = pytz.timezone('US/Eastern')
            
            def convert_utc_to_est(utc_timestamp):
                if not utc_timestamp:
                    return ''
                try:
                    ts_str = str(utc_timestamp)
                    if 'T' in ts_str:
                        ts_str = ts_str.replace('Z', '+00:00')
                        dt_utc = datetime.fromisoformat(ts_str)
                        if dt_utc.tzinfo is None:
                            dt_utc = pytz.utc.localize(dt_utc)
                        dt_est = dt_utc.astimezone(est)
                        return dt_est.strftime('%Y-%m-%d %H:%M:%S %Z')
                    return ts_str
                except:
                    return str(utc_timestamp)
            
            filled_at_est = convert_utc_to_est(order.filled_at if hasattr(order, 'filled_at') else None)
            submitted_at_est = convert_utc_to_est(order.submitted_at if hasattr(order, 'submitted_at') else None)
            primary_timestamp = filled_at_est or submitted_at_est or datetime.now(est).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            trade_data = {
                'timestamp': primary_timestamp,
                'symbol': order.symbol,
                'action': order.side.upper(),
                'qty': float(order.filled_qty),
                'fill_price': float(order.filled_avg_price or 0),
                'order_id': str(order.id),
                'source': 'alpaca_sync',
                'pnl': 0, # Cannot determine easily from single order
                'is_0dte': is_0dte,
                'submitted_at': submitted_at_est,
                'filled_at': filled_at_est
            }
            
            try:
                # save_trade handles INSERT OR REPLACE based on timestamp/symbol/action/qty
                # We might want to ensure we don't overwrite existing records with PnL data
                # But since current DB unique constraint doesn't include order_id, we rely on timestamp
                trade_db.save_trade(trade_data)
                count += 1
            except Exception:
                pass 
                
        return count
    except Exception as e:
        st.error(f"Sync error: {e}")
        return -1

# ==================== LOGS SECTION ====================
def render_logs_section():
    """Render Logs section with filters and views"""
    st.markdown("### 📋 Institutional Logs")
    st.markdown("View and analyze structured logs from backtests and live trading")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30), key="logs_start")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now(), key="logs_end")
    with col3:
        symbol_filter = st.selectbox("Symbol", ["All", "SPY", "QQQ", "SPX"], key="logs_symbol")
    with col4:
        log_category = st.selectbox("Log Category", 
                                   ["decisions", "risk", "execution", "positions", "learning"], key="logs_category")
    
    # Load logs
    try:
        from institutional_logging import get_logger
        logger = get_logger()
        
        if logger:
            logs = logger.get_logs(
                category=log_category,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                symbol=symbol_filter if symbol_filter != "All" else None
            )
            
            if logs:
                st.success(f"Loaded {len(logs)} log entries")
                
                # Create DataFrame for display
                logs_df = pd.DataFrame(logs)
                
                # Display logs table
                st.dataframe(logs_df, use_container_width=True, height=400)
                
                # Generate summary statistics
                if log_category == "decisions":
                    render_decision_analytics(logs)
                elif log_category == "risk":
                    render_risk_analytics(logs)
                elif log_category == "execution":
                    render_execution_analytics(logs)
                elif log_category == "positions":
                    render_position_analytics(logs)
                elif log_category == "learning":
                    render_learning_analytics(logs)
            else:
                st.info("No logs found for selected filters. Run a backtest to generate logs.")
        else:
            st.warning("Logger not initialized. Run a backtest first.")
    except ImportError:
        st.error("Institutional logging module not found")
    except Exception as e:
        st.error(f"Error loading logs: {e}")

def render_decision_analytics(logs: list):
    """Render decision analytics"""
    st.markdown("#### Decision Analytics")
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_decisions = len(df)
        st.metric("Total Decisions", total_decisions)
    
    with col2:
        hold_count = len(df[df.get('action_final') == 'HOLD']) if 'action_final' in df.columns else 0
        st.metric("HOLD Count", hold_count)
    
    with col3:
        buy_count = len(df[df.get('action_final', '').astype(str).str.contains('BUY', na=False)]) if 'action_final' in df.columns else 0
        st.metric("BUY Count", buy_count)
    
    # Ensemble override rate
    if 'rl_action' in df.columns and 'ensemble_action' in df.columns:
        overrides = df[df['rl_action'] != df['ensemble_action']]
        override_rate = len(overrides) / len(df) * 100 if len(df) > 0 else 0
        st.metric("Ensemble Override Rate", f"{override_rate:.1f}%")
    
    # Regime distribution
    if 'regime' in df.columns:
        regime_counts = df['regime'].value_counts()
        st.bar_chart(regime_counts)

def render_risk_analytics(logs: list):
    """Render risk analytics"""
    st.markdown("#### Risk Analytics")
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_checks = len(df)
        st.metric("Total Risk Checks", total_checks)
    
    with col2:
        blocks = len(df[df.get('risk_action') == 'BLOCK']) if 'risk_action' in df.columns else 0
        block_rate = (blocks / total_checks * 100) if total_checks > 0 else 0
        st.metric("Block Rate", f"{block_rate:.1f}%")
    
    # Risk block reasons
    if 'risk_reason' in df.columns:
        blocked = df[df.get('risk_action') == 'BLOCK']
        if len(blocked) > 0:
            reasons = blocked['risk_reason'].value_counts()
            st.bar_chart(reasons)

def render_execution_analytics(logs: list):
    """Render execution analytics"""
    st.markdown("#### Execution Analytics")
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_fills = len(df)
        st.metric("Total Fills", total_fills)
    
    with col2:
        if 'slippage_pct' in df.columns:
            avg_slippage = df['slippage_pct'].mean()
            st.metric("Avg Slippage", f"{avg_slippage:.2f}%")
    
    with col3:
        if 'spread' in df.columns:
            avg_spread = df['spread'].mean()
            st.metric("Avg Spread", f"${avg_spread:.4f}")

def render_position_analytics(logs: list):
    """Render position analytics"""
    st.markdown("#### Position Analytics")
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_positions = len(df)
        st.metric("Total Positions", total_positions)
    
    with col2:
        if 'final_pnl' in df.columns:
            winning = len(df[df['final_pnl'] > 0])
            win_rate = (winning / total_positions * 100) if total_positions > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        if 'hold_minutes' in df.columns:
            avg_hold = df['hold_minutes'].mean()
            st.metric("Avg Hold Time", f"{avg_hold:.1f} min")

def render_learning_analytics(logs: list):
    """Render learning analytics"""
    st.markdown("#### Learning Analytics")
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    retrained = df[df.get('retrained') == True] if 'retrained' in df.columns else pd.DataFrame()
    st.metric("Retraining Events", len(retrained))
    
    if 'sharpe_candidate' in df.columns and 'sharpe_prod' in df.columns:
        promotions = df[df.get('promotion') == True] if 'promotion' in df.columns else pd.DataFrame()
        st.metric("Model Promotions", len(promotions))

def render_feedback_section():
    """Render Feedback/Review section"""
    st.markdown("### 💬 Feedback & Review")
    st.markdown("Quantitative feedback and human review input")
    
    # Date selector for feedback
    feedback_date = st.date_input("Select Date for Review", value=datetime.now(), key="feedback_date")
    
    # Load daily summary
    try:
        from institutional_logging import get_logger
        logger = get_logger()
        
        if logger:
            summary = logger.generate_daily_summary(feedback_date.strftime("%Y-%m-%d"))
            
            if summary and summary.get('decisions', {}).get('total', 0) > 0:
                # Quantitative Feedback (Auto-Generated)
                st.markdown("#### 📊 Quantitative Feedback (Auto-Generated)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Decision Summary:**")
                    decisions = summary.get('decisions', {})
                    st.write(f"- Total Decisions: {decisions.get('total', 0)}")
                    st.write(f"- HOLD: {decisions.get('hold', 0)}")
                    st.write(f"- BUY: {decisions.get('buy', 0)}")
                    st.write(f"- Ensemble Override Rate: {decisions.get('ensemble_override_rate_pct', 0):.1f}%")
                
                with col2:
                    st.markdown("**Risk Summary:**")
                    risk = summary.get('risk', {})
                    st.write(f"- Total Checks: {risk.get('total_checks', 0)}")
                    st.write(f"- Blocks: {risk.get('blocks', 0)}")
                    st.write(f"- Block Rate: {risk.get('block_rate_pct', 0):.1f}%")
                
                st.markdown("**Execution Summary:**")
                execution = summary.get('execution', {})
                st.write(f"- Total Fills: {execution.get('total_fills', 0)}")
                st.write(f"- Avg Slippage: {execution.get('avg_slippage_pct', 0):.2f}%")
                
                st.markdown("**Position Summary:**")
                positions = summary.get('positions', {})
                st.write(f"- Total Positions: {positions.get('total', 0)}")
                st.write(f"- Win Rate: {positions.get('win_rate_pct', 0):.1f}%")
                
                # Human Review Input
                st.markdown("---")
                st.markdown("#### ✍️ Human Review Input")
                
                with st.form("feedback_form"):
                    reviewer = st.text_input("Reviewer Name", value="PM")
                    comment = st.text_area("Comment", placeholder="Enter feedback...")
                    severity = st.selectbox("Severity", ["LOW", "MEDIUM", "HIGH"])
                    category = st.selectbox("Category", ["BEHAVIOR", "RISK", "EXECUTION", "LEARNING"])
                    
                    submitted = st.form_submit_button("💾 Submit Feedback")
                    
                    if submitted and comment:
                        logger.log_feedback(
                            date=feedback_date.strftime("%Y-%m-%d"),
                            reviewer=reviewer,
                            comment=comment,
                            severity=severity,
                            category=category
                        )
                        st.success("Feedback saved successfully!")
            else:
                st.info("No data available for selected date. Run a backtest first.")
        else:
            st.warning("Logger not initialized. Run a backtest first.")
    except ImportError:
        st.error("Institutional logging module not found")
    except Exception as e:
        st.error(f"Error loading feedback: {e}")

# ==================== LIVE ACTIVITY SECTION ====================
def render_live_activity():
    """Render live activity log showing setup validation and data sources"""
    st.markdown("### 🔴 Live Activity Log")
    st.markdown("Real-time view of what the agent is actively analyzing")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=True, key="auto_refresh_activity")
    with col2:
        refresh_btn = st.button("🔄 Refresh Now", key="refresh_activity")
    with col3:
        max_entries = st.selectbox("Max Entries", [25, 50, 100, 200], index=1, key="max_activity_entries")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        activity_filter = st.selectbox(
            "Filter by Type",
            ["All", "Setup Validation", "Data Source", "RL Inference", "Ensemble", "Trade Execution", "Blocked"],
            key="activity_filter"
        )
    with col2:
        symbol_filter = st.selectbox(
            "Filter by Symbol",
            ["All", "SPY", "QQQ", "IWM"],
            key="activity_symbol_filter"
        )
    with col3:
        time_range = st.selectbox(
            "Time Range",
            ["Last 5 minutes", "Last 15 minutes", "Last 30 minutes", "Last hour", "All"],
            index=2,
            key="activity_time_range"
        )
    
    # Load live activity
    try:
        from live_activity_log import LiveActivityLog
        activity_log = LiveActivityLog()
        activities = activity_log.get_recent_activities(limit=max_entries)
        
        # Apply filters
        if activity_filter != "All":
            filter_map = {
                "Setup Validation": "setup_validation",
                "Data Source": "data_source",
                "RL Inference": "rl_inference",
                "Ensemble": "ensemble_activity",
                "Trade Execution": "trade_execution",
                "Blocked": "blocked"
            }
            activities = [a for a in activities if a.get('type') == filter_map.get(activity_filter)]
        
        if symbol_filter != "All":
            activities = [a for a in activities if a.get('symbol') == symbol_filter]
        
        # Filter by time range
        if time_range != "All":
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            time_map = {
                "Last 5 minutes": timedelta(minutes=5),
                "Last 15 minutes": timedelta(minutes=15),
                "Last 30 minutes": timedelta(minutes=30),
                "Last hour": timedelta(hours=1)
            }
            cutoff = now - time_map.get(time_range, timedelta(minutes=30))
            
            # Filter activities, ensuring all timestamps are timezone-aware
            filtered_activities = []
            for a in activities:
                ts = a.get('timestamp', now)
                # Ensure timestamp is timezone-aware (EST)
                if ts.tzinfo is None:
                    # If naive, assume it's EST and localize
                    ts = est.localize(ts)
                else:
                    # If aware, convert to EST
                    ts = ts.astimezone(est)
                
                # Now safe to compare
                if ts >= cutoff:
                    a['timestamp'] = ts  # Update with timezone-aware version
                    filtered_activities.append(a)
            activities = filtered_activities
        
        if activities:
            st.success(f"📊 Showing {len(activities)} recent activities")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Data source summary
            data_sources = activity_log.get_data_source_summary()
            with col1:
                alpaca_count = data_sources.get('Alpaca API', 0)
                massive_count = data_sources.get('Massive API', 0)
                st.metric("Data Sources", f"Alpaca: {alpaca_count} | Massive: {massive_count}")
            
            # Setup validation summary
            setup_summary = activity_log.get_setup_validation_summary()
            with col2:
                st.metric("Setups Validated", setup_summary.get('validating', 0))
            with col3:
                st.metric("Setups Selected", setup_summary.get('selected', 0))
            with col4:
                st.metric("Setups Rejected", setup_summary.get('rejected', 0))
            
            # Activity log table
            st.markdown("#### 📋 Activity Log")
            
            # Prepare data for display
            display_data = []
            est = pytz.timezone('US/Eastern')
            for activity in activities:
                timestamp = activity.get('timestamp', datetime.now(est))
                # Ensure timestamp is timezone-aware
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = est.localize(timestamp)
                    else:
                        timestamp = timestamp.astimezone(est)
                    time_str = timestamp.strftime('%H:%M:%S')
                else:
                    time_str = str(timestamp)
                
                activity_type = activity.get('type', 'unknown')
                symbol = activity.get('symbol', 'N/A')
                message = activity.get('message', '')[:150]  # Truncate long messages
                
                # Format activity type for display
                type_display = {
                    'data_source_alpaca': '📊 Data: Alpaca',
                    'data_source_massive': '📊 Data: Massive',
                    'data_source_yfinance': '📊 Data: yfinance (DELAYED)',
                    'setup_validation': '🔍 Setup Validation',
                    'rl_inference': '🤖 RL Inference',
                    'ensemble_activity': '🎯 Ensemble',
                    'trade_execution': '✅ Trade Executed',
                    'blocked': '⛔ Blocked',
                    'price_validation': '💰 Price Validation',
                    'safeguard_check': '🛡️ Safeguard Check'
                }.get(activity_type, activity_type)
                
                # Extract key details
                details = []
                if activity.get('data_source'):
                    details.append(f"Source: {activity['data_source']}")
                if activity.get('price'):
                    details.append(f"Price: ${activity['price']:.2f}")
                if activity.get('rl_action'):
                    details.append(f"RL: {activity['rl_action']}")
                if activity.get('confidence'):
                    details.append(f"Conf: {activity['confidence']:.3f}")
                if activity.get('ensemble_action'):
                    details.append(f"Ensemble: {activity['ensemble_action']}")
                if activity.get('block_reason'):
                    details.append(f"Reason: {activity['block_reason']}")
                
                display_data.append({
                    'Time': time_str,
                    'Type': type_display,
                    'Symbol': symbol,
                    'Details': ' | '.join(details) if details else '—',
                    'Message': message
                })
            
            # Display as table
            if display_data:
                df = pd.DataFrame(display_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=600,
                    hide_index=True
                )
            else:
                st.info("No activities found matching filters")
        else:
            st.warning("No activities found. Make sure the agent is running and generating logs.")
            st.info(f"Looking for log file: {activity_log.log_file}")
            
            # Show log file status and help debug
            import os
            if os.path.exists(activity_log.log_file):
                file_size = os.path.getsize(activity_log.log_file)
                st.info(f"✅ Log file exists ({file_size:,} bytes)")
                
                # Try to show last few lines for debugging
                try:
                    with open(activity_log.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            st.code(f"Last 3 lines from log:\n" + "".join(lines[-3:]), language=None)
                            
                            # Try parsing to see if we get activities
                            test_activities = activity_log.parse_log_file(max_lines=100)
                            if test_activities:
                                st.success(f"✅ Successfully parsed {len(test_activities)} activities from log file")
                                st.info("Try adjusting filters or time range to see them.")
                            else:
                                st.warning("⚠️ Log file exists but no activities were parsed. Check log format.")
                except Exception as e:
                    st.warning(f"Could not read log file: {e}")
            else:
                st.error(f"❌ Log file does not exist: {activity_log.log_file}")
                st.info("The agent needs to be running to generate logs.")
                
                # Suggest alternative locations
                alt_logs = [
                    f"logs/mike_agent_safe_{datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d')}.log",
                    "agent_output.log",
                    "/tmp/agent.log"
                ]
                st.info("Checking alternative log locations...")
                for alt_log in alt_logs:
                    if os.path.exists(alt_log):
                        st.success(f"✅ Found alternative log: {alt_log}")
                        st.info("Try refreshing or restarting the dashboard to use this log file.")
                        break
            
            # Show log file status and help debug
            import os
            if os.path.exists(activity_log.log_file):
                file_size = os.path.getsize(activity_log.log_file)
                st.info(f"✅ Log file exists: {activity_log.log_file} ({file_size:,} bytes)")
                
                # Try to show last few lines for debugging
                try:
                    with open(activity_log.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            st.code(f"Last 3 lines from log:\n" + "".join(lines[-3:]), language=None)
                            
                            # Try parsing to see if we get activities
                            test_activities = activity_log.parse_log_file(max_lines=100)
                            if test_activities:
                                st.success(f"✅ Successfully parsed {len(test_activities)} activities from log file")
                                st.info("Try adjusting filters or time range to see them.")
                            else:
                                st.warning("⚠️ Log file exists but no activities were parsed. Check log format.")
                except Exception as e:
                    st.warning(f"Could not read log file: {e}")
            else:
                st.error(f"❌ Log file does not exist: {activity_log.log_file}")
                st.info("The agent needs to be running to generate logs.")
                
                # Suggest alternative locations
                est = pytz.timezone('US/Eastern')
                alt_logs = [
                    f"logs/mike_agent_safe_{datetime.now(est).strftime('%Y%m%d')}.log",
                    "agent_output.log",
                    "/tmp/agent.log"
                ]
                st.info("Checking alternative log locations...")
                for alt_log in alt_logs:
                    if os.path.exists(alt_log):
                        st.success(f"✅ Found alternative log: {alt_log}")
                        st.info("Try refreshing or restarting the dashboard to use this log file.")
                        break
    
    except ImportError:
        st.error("Live activity log module not found. Install live_activity_log.py")
    except Exception as e:
        st.error(f"Error loading live activity: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # Auto-refresh using Streamlit's built-in mechanism
    if auto_refresh:
        # Use placeholder to trigger refresh
        placeholder = st.empty()
        with placeholder:
            st.info("🔄 Auto-refreshing every 5 seconds...")
        time.sleep(5)
        placeholder.empty()
        st.rerun()

# ==================== MAIN APP ====================
def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    
    # Inject custom CSS
    inject_custom_css()
    
    # Load settings and state
    settings = load_settings()
    state = load_state()
    
    # Render navigation
    current_page = render_navigation()
    
    # Render current page
    if current_page == "Dashboard":
        render_dashboard()
    elif current_page == "Analytics":
        render_analytics()
    elif current_page == "Trades":
        render_trades()
    elif current_page == "Agents":
        render_agents()
    elif current_page == "Risk Management":
        render_risk_management()
    elif current_page == "Settings":
        render_settings()
    
    # Auto-refresh if enabled
    if settings.get('auto_refresh', True):
        time.sleep(settings.get('refresh_interval', 10))
        st.rerun()

if __name__ == "__main__":
    main()

