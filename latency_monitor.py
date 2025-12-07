"""
⏱️ LATENCY MONITORING MODULE

Tracks execution latency for order placement and fills.
Addresses the critical gap identified in architect review:
"1ms delay = 5-10% fill slippage on gamma ramps"

Provides:
- Order placement timing
- Fill latency tracking
- Execution performance metrics
- Latency alerts

Author: Mike Agent Institutional Upgrade
Date: December 6, 2025
"""

import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import deque
import statistics


class LatencyMonitor:
    """
    Monitor and track execution latency for trading orders
    """
    
    def __init__(self, alert_threshold_ms: float = 500.0, max_history: int = 100):
        """
        Initialize latency monitor
        
        Args:
            alert_threshold_ms: Alert if latency exceeds this (default 500ms)
            max_history: Maximum number of latency records to keep
        """
        self.alert_threshold_ms = alert_threshold_ms
        self.latency_history = deque(maxlen=max_history)
        self.order_timings = {}  # Track active orders by order_id
    
    def start_order_timing(self, order_id: str, symbol: str) -> None:
        """
        Start timing an order placement
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
        """
        self.order_timings[order_id] = {
            'symbol': symbol,
            'start_time': time.perf_counter(),
            'start_wall_time': datetime.now()
        }
    
    def record_order_placed(self, order_id: str) -> Optional[float]:
        """
        Record when order was placed (submitted to broker)
        
        Args:
            order_id: Order identifier
            
        Returns:
            Placement latency in milliseconds, or None if order not found
        """
        if order_id not in self.order_timings:
            return None
        
        placement_time = time.perf_counter()
        placement_latency_ms = (placement_time - self.order_timings[order_id]['start_time']) * 1000
        
        self.order_timings[order_id]['placement_time'] = placement_time
        self.order_timings[order_id]['placement_latency_ms'] = placement_latency_ms
        
        return placement_latency_ms
    
    def record_order_filled(
        self,
        order_id: str,
        fill_price: Optional[float] = None,
        fill_qty: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """
        Record when order was filled
        
        Args:
            order_id: Order identifier
            fill_price: Fill price (optional)
            fill_qty: Fill quantity (optional)
            
        Returns:
            Dictionary with latency metrics, or None if order not found
        """
        if order_id not in self.order_timings:
            return None
        
        fill_time = time.perf_counter()
        timing_data = self.order_timings[order_id]
        
        # Calculate latencies
        total_latency_ms = (fill_time - timing_data['start_time']) * 1000
        
        placement_latency_ms = timing_data.get('placement_latency_ms', 0.0)
        fill_latency_ms = total_latency_ms - placement_latency_ms
        
        # Record metrics
        metrics = {
            'total_latency_ms': total_latency_ms,
            'placement_latency_ms': placement_latency_ms,
            'fill_latency_ms': fill_latency_ms,
            'symbol': timing_data['symbol'],
            'fill_price': fill_price,
            'fill_qty': fill_qty,
            'timestamp': timing_data['start_wall_time']
        }
        
        # Add to history
        self.latency_history.append(metrics)
        
        # Clean up
        del self.order_timings[order_id]
        
        return metrics
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get statistical summary of latency metrics
        
        Returns:
            Dictionary with mean, median, min, max, stddev for:
            - total_latency_ms
            - placement_latency_ms
            - fill_latency_ms
        """
        if not self.latency_history:
            return {
                'count': 0,
                'total_latency_mean_ms': 0.0,
                'total_latency_median_ms': 0.0,
                'total_latency_min_ms': 0.0,
                'total_latency_max_ms': 0.0,
                'total_latency_stddev_ms': 0.0,
                'placement_latency_mean_ms': 0.0,
                'fill_latency_mean_ms': 0.0
            }
        
        total_latencies = [m['total_latency_ms'] for m in self.latency_history]
        placement_latencies = [m.get('placement_latency_ms', 0.0) for m in self.latency_history]
        fill_latencies = [m.get('fill_latency_ms', 0.0) for m in self.latency_history]
        
        return {
            'count': len(self.latency_history),
            'total_latency_mean_ms': statistics.mean(total_latencies),
            'total_latency_median_ms': statistics.median(total_latencies),
            'total_latency_min_ms': min(total_latencies),
            'total_latency_max_ms': max(total_latencies),
            'total_latency_stddev_ms': statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0.0,
            'placement_latency_mean_ms': statistics.mean(placement_latencies),
            'fill_latency_mean_ms': statistics.mean(fill_latencies)
        }
    
    def check_latency_alert(self, latency_ms: float) -> Tuple[bool, str]:
        """
        Check if latency exceeds alert threshold
        
        Args:
            latency_ms: Latency in milliseconds
            
        Returns:
            Tuple of (should_alert, message)
        """
        if latency_ms > self.alert_threshold_ms:
            return True, f"⚠️ High latency: {latency_ms:.2f}ms > {self.alert_threshold_ms:.0f}ms threshold"
        return False, ""
    
    def get_recent_latencies(self, n: int = 10) -> List[Dict[str, float]]:
        """
        Get most recent latency records
        
        Args:
            n: Number of recent records to return
            
        Returns:
            List of recent latency metrics (most recent first)
        """
        return list(self.latency_history)[-n:]
    
    def format_latency_report(self) -> str:
        """
        Format a human-readable latency report
        
        Returns:
            Formatted string with latency statistics
        """
        stats = self.get_latency_stats()
        
        if stats['count'] == 0:
            return "⏱️ Latency Monitor: No orders tracked yet"
        
        report = [
            "⏱️ LATENCY STATISTICS",
            f"   Orders Tracked: {stats['count']}",
            f"   Total Latency (mean): {stats['total_latency_mean_ms']:.2f}ms",
            f"   Total Latency (median): {stats['total_latency_median_ms']:.2f}ms",
            f"   Total Latency (min/max): {stats['total_latency_min_ms']:.2f}ms / {stats['total_latency_max_ms']:.2f}ms",
            f"   Placement Latency (mean): {stats['placement_latency_mean_ms']:.2f}ms",
            f"   Fill Latency (mean): {stats['fill_latency_mean_ms']:.2f}ms"
        ]
        
        return "\n".join(report)


# Global instance for easy access
_global_latency_monitor: Optional[LatencyMonitor] = None


def get_latency_monitor(alert_threshold_ms: float = 500.0) -> LatencyMonitor:
    """
    Get or create global latency monitor instance
    
    Args:
        alert_threshold_ms: Alert threshold (only used if creating new instance)
        
    Returns:
        LatencyMonitor instance
    """
    global _global_latency_monitor
    if _global_latency_monitor is None:
        _global_latency_monitor = LatencyMonitor(alert_threshold_ms=alert_threshold_ms)
    return _global_latency_monitor


def reset_latency_monitor() -> None:
    """Reset global latency monitor"""
    global _global_latency_monitor
    _global_latency_monitor = None

