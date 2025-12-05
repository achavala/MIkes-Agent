"""
Risk management utilities
"""
from typing import Optional


class RiskManager:
    """Manages position sizing and risk"""
    
    def __init__(self, account_value: float = 10000.0):
        self.account_value = account_value
        
    def calculate_size(self, symbol: str, premium: float, underlying_price: float, 
                      risk_pct: float = 0.07) -> int:
        """
        Calculate position size based on risk percentage.
        
        Args:
            symbol: Trading symbol
            premium: Option premium per contract
            underlying_price: Current underlying price
            risk_pct: Risk percentage (default 7% for Mike strategy)
            
        Returns:
            Number of contracts
        """
        if premium <= 0:
            return 0
        
        risk_amount = self.account_value * risk_pct
        contracts = int(risk_amount / (premium * 100))  # Each contract = 100 shares
        
        # Mike style: "size for $0" on lottos (very cheap options)
        if premium < 0.10:  # Lotto threshold
            contracts = max(contracts, 1)  # At least 1 contract
        
        return max(1, contracts)  # Minimum 1 contract
    
    def update_account_value(self, new_value: float):
        """Update account value after trades"""
        self.account_value = new_value

