import streamlit as st
import datetime
import pytz
import time
import os

st.set_page_config(page_title="Mike Agent v3", layout="wide")
st.title("Mike Agent Dashboard")

# === LEFT SIDEBAR — YOUR ORIGINAL BACKTEST GUI (unchanged) ===
with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["backtest", "live"])
    
    symbols = st.text_input("Symbols (comma-separated)", "SPY,QQQ")
    capital = st.number_input("Initial Capital ($)", value=1000.0)
    risk_pct = st.slider("Risk Per Trade (%)", 0.01, 5.0, 0.07, 0.01)

    st.markdown("### Backtest Options")
    use_csv = st.checkbox("Use CSV File")
    start_date = st.date_input("Start Date", datetime.date(2025, 11, 3))
    end_date = st.date_input("End Date", datetime.date(2025, 12, 1))
    monte_carlo = st.checkbox("Monte Carlo Simulation")

    if st.button("Run Backtest", type="primary"):
        st.success("Backtest would run here (code not included in this minimal version)")

# === RIGHT SIDE — LIVE STATUS + REAL-TIME LOG ===
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## Live Agent Brain Activity")
    status_box = st.empty()
    log_box = st.empty()

def update_live_status():
    est = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(est)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now.weekday() >= 5:
        next_open = market_open + datetime.timedelta(days=(7-now.weekday()))
    elif now > market_close:
        next_open = market_open + datetime.timedelta(days=1)
    elif now < market_open:
        next_open = market_open
    else:
        next_open = market_close

    is_trading = market_open <= now <= market_close and now.weekday() < 5
    status = "MARKET OPEN — HUNTING" if is_trading else "WAITING FOR MARKET"
    color = "#00ff00" if is_trading else "#ffaa00"
    countdown = next_open - now
    h, rem = divmod(int(countdown.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m}m" if h or m else f"{s}s"

    status_html = f"""
    <div style="padding:20px; border-radius:12px; background:#111; text-align:center; border:3px solid {color}">
        <h2 style="margin:0; color:{color}">Live Status: {status}</h2>
        <h1 style="margin:10px 0; color:#00ff88">{time_str}</h1>
        <p style="color:#aaa">Next: {next_open.strftime('%A %I:%M %p EST')}</p>
    </div>
    """
    status_box.markdown(status_html, unsafe_allow_html=True)

    # Real log from your running agent
    try:
        if os.path.exists("mike.log"):
            with open("mike.log", "r") as f:
                lines = f.readlines()[-20:]
            log_text = "".join(lines)
        else:
            log_text = "Agent running in background...\nmike.log will appear here when active"
    except:
        log_text = "Reading log..."

    log_box.text_area("Real-Time Activity Log", log_text, height=380)

# Auto-refresh every 8 seconds
update_live_status()
time.sleep(8)
st.rerun()
