#!/bin/bash
# Quick dashboard restart script

echo "Stopping existing Streamlit processes..."
pkill -f "streamlit run"

sleep 2

echo "Starting dashboard on port 8501..."
cd /Users/chavala/Mike-agent-project
source venv/bin/activate
streamlit run app.py --server.port=8501 > dashboard.log 2>&1 &

sleep 3
echo ""
echo "✅ Dashboard restarted!"
echo "📊 Access at: http://localhost:8501"
echo "📋 Check logs: tail -f dashboard.log"
echo ""

