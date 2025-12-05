"""
Strategy manager for coordinating agents
"""
from typing import List, Dict
from core.agents import MikeAgent
from core.data.options_feed import OptionsFeed
from core.utils.risk_manager import RiskManager
from core.utils.regime_engine import RegimeEngine
from core.utils.microstructure import Microstructure


class StrategyManager:
    """Manages all trading agents"""
    
    def __init__(self, account_value: float = 10000.0):
        self.options_feed = OptionsFeed()
        self.microstructure = Microstructure()
        self.regime_engine = RegimeEngine()
        self.risk_manager = RiskManager(account_value)
        
        # Initialize agents
        self.agents = []
        self.agents.append(
            MikeAgent(
                self.options_feed,
                self.microstructure,
                self.regime_engine,
                self.risk_manager
            )
        )
    
    def get_agent(self, name: str):
        """Get agent by name"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def update_account_value(self, new_value: float):
        """Update account value across all managers"""
        self.risk_manager.update_account_value(new_value)

