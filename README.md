# Mike Agent v3 - RL Trading Bot

Autonomous reinforcement learning agent for SPY 0DTE options trading with 12 institutional-grade safeguards.

## Quick Start

```bash
pip install -r requirements.txt
python mike_rl_agent.py --train
python mike_agent_live_safe.py
```

## Deployed Live

🚀 https://mike-agent-123.railway.app (mobile-friendly)

## Safeguards

- Daily loss limit: -15%
- Position size: 25% equity max
- VIX kill-switch: >28
- Hard stops: -30% / -20%
- 12 total protections

Built December 2025 on Apple Silicon.
