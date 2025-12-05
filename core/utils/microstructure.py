"""
Market microstructure analysis
"""
from typing import Optional, Dict
from core.strategy.base import Bar


class Microstructure:
    """Analyzes order flow and microstructure"""
    
    def __init__(self):
        pass
    
    def analyze_bar(self, bar: Bar) -> Dict:
        """Analyze bar for microstructure signals"""
        return {
            'volume': bar.volume,
            'spread': bar.high - bar.low,
            'body': abs(bar.close - bar.open),
            'upper_wick': bar.high - max(bar.open, bar.close),
            'lower_wick': min(bar.open, bar.close) - bar.low
        }

