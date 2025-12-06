
#!/bin/bash
# Quick agent restart script

echo "Stopping old agent..."
pkill -f mike_agent_live_safe.py
sleep 2

echo "Starting agent with fixed code..."
cd /Users/chavala/Mike-agent-project
source venv/bin/activate
nohup python mike_agent_live_safe.py > agent_output.log 2>&1 &

sleep 3
echo ""
echo "✅ Agent restarted!"
echo "📊 Check logs with: tail -f agent_output.log"
echo ""
tail -20 agent_output.log

