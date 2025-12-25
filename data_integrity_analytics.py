"""
DATA INTEGRITY ANALYTICS PANEL
Analytics panel for monitoring data provider usage
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def render_data_integrity_panel():
    """Render Data Integrity panel in Analytics"""
    st.markdown("### 🔒 Data Integrity & Provider Usage")
    st.markdown("Monitor data provider usage and ensure institutional-grade data sources")
    
    try:
        from data_provider_router import get_data_router
        router = get_data_router()
        
        if router:
            # Get provider statistics
            stats = router.get_provider_stats()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Fetches", stats.get('total_fetches', 0))
            
            with col2:
                massive_pct = stats.get('percentages', {}).get('massive', 0)
                st.metric("Massive Usage", f"{massive_pct:.1f}%")
            
            with col3:
                alpaca_pct = stats.get('percentages', {}).get('alpaca', 0)
                st.metric("Alpaca Usage", f"{alpaca_pct:.1f}%")
            
            with col4:
                yfinance_count = stats.get('yfinance_usage', 0)
                yfinance_flag = stats.get('yfinance_red_flag', False)
                color = "🔴" if yfinance_flag else "🟢"
                st.metric(f"{color} yfinance Usage", yfinance_count)
            
            # Provider usage breakdown
            st.markdown("#### Provider Usage Breakdown")
            usage_df = pd.DataFrame([
                {
                    "Provider": provider,
                    "Count": stats.get('by_provider', {}).get(provider, 0),
                    "Percentage": stats.get('percentages', {}).get(provider, 0)
                }
                for provider in ['massive', 'alpaca', 'polygon', 'yfinance']
            ])
            st.dataframe(usage_df, use_container_width=True, hide_index=True)
            
            # Red flag warning
            if stats.get('yfinance_red_flag', False):
                st.error("⚠️ **RED FLAG**: yfinance usage detected in institutional mode!")
                st.warning("This indicates data provider failures. Review logs for details.")
            
            # Provider logs
            st.markdown("#### Recent Provider Logs")
            logs = router.get_provider_logs()
            
            if logs:
                # Show last 100 logs
                recent_logs = logs[-100:]
                logs_df = pd.DataFrame(recent_logs)
                st.dataframe(logs_df, use_container_width=True, height=400)
            else:
                st.info("No provider logs available yet. Run a backtest to generate logs.")
        else:
            st.warning("Data router not initialized. Run a backtest first.")
    except ImportError:
        st.error("Data provider router module not found")
    except Exception as e:
        st.error(f"Error loading data integrity panel: {e}")





