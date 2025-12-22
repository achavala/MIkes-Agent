#!/usr/bin/env python3
"""
Live Activity Log Parser
Extracts real-time activity from agent logs for dashboard display
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import pytz

class LiveActivityLog:
    """Parse and display live agent activity from log files"""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize with log file path"""
        self.est = pytz.timezone('US/Eastern')
        now_est = datetime.now(self.est)
        
        if log_file is None:
            # Try to find today's log file (priority order)
            daily_log = f"logs/mike_agent_safe_{now_est.strftime('%Y%m%d')}.log"
            
            # Check multiple possible log locations in priority order
            log_candidates = [
                daily_log,  # Today's daily log (highest priority)
                "agent_output.log",  # Fallback log
                "/tmp/agent.log",  # Fly.io container log
                f"logs/agent_{now_est.strftime('%Y%m%d')}.log",  # Alternative daily log
            ]
            
            # Use first existing log file
            for candidate in log_candidates:
                if os.path.exists(candidate):
                    self.log_file = candidate
                    break
            else:
                # If none found, use daily log (will show empty if doesn't exist)
                self.log_file = daily_log
        else:
            self.log_file = log_file
        
        self.today = now_est.date()
    
    def parse_log_file(self, max_lines: int = 500) -> List[Dict]:
        """Parse log file and extract activity events"""
        if not os.path.exists(self.log_file):
            return []
        
        activities = []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Read last N lines (most recent activity)
                lines = deque(f, maxlen=max_lines)
                
                for line in lines:
                    activity = self.parse_activity_line(line)
                    if activity:
                        activities.append(activity)
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
        return activities[:100]  # Return last 100 activities
    
    def parse_activity_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line and extract activity information"""
        if not line.strip():
            return None
        
        # Extract timestamp
        timestamp = self.extract_timestamp(line)
        
        # Extract log level
        level_match = re.search(r'\[(INFO|WARNING|ERROR|TRADE|DEBUG|CRITICAL)\]', line)
        level = level_match.group(1) if level_match else 'INFO'
        
        # Extract message
        message_match = re.search(r'\]\s+(.+)', line)
        message = message_match.group(1) if message_match else line.strip()
        
        # Determine activity type
        activity_type = self.classify_activity(message, level)
        
        if not activity_type:
            return None
        
        # Extract details based on activity type
        details = self.extract_activity_details(message, activity_type)
        
        return {
            'timestamp': timestamp,
            'level': level,
            'type': activity_type,
            'message': message,
            **details
        }
    
    def extract_timestamp(self, line: str) -> datetime:
        """Extract timestamp from log line - ALWAYS returns timezone-aware datetime (EST)"""
        # Try multiple timestamp formats
        patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',  # 2025-12-22 10:30:00
            r'\[(\d{2}:\d{2}:\d{2})\]',  # [10:30:00]
            r'(\d{2}:\d{2}:\d{2})',  # 10:30:00
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    time_str = match.group(1)
                    if len(time_str) > 8:  # Full datetime
                        # Parse as naive, then make timezone-aware (EST)
                        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                        # If already timezone-aware, convert to EST; otherwise localize to EST
                        if dt.tzinfo is None:
                            dt = self.est.localize(dt)
                        else:
                            dt = dt.astimezone(self.est)
                        return dt
                    else:  # Time only
                        today = datetime.now(self.est).replace(hour=0, minute=0, second=0, microsecond=0)
                        time_parts = time_str.split(':')
                        dt = today.replace(hour=int(time_parts[0]), minute=int(time_parts[1]), second=int(time_parts[2]))
                        # Ensure timezone-aware
                        if dt.tzinfo is None:
                            dt = self.est.localize(dt)
                        return dt
                except Exception as e:
                    # Log error but continue
                    pass
        
        # Default to now if no timestamp found - ALWAYS timezone-aware (EST)
        return datetime.now(self.est)
    
    def classify_activity(self, message: str, level: str) -> Optional[str]:
        """Classify activity type from message"""
        message_lower = message.lower()
        
        # Data source activity
        if any(x in message_lower for x in ['alpaca api', 'massive api', 'yfinance', 'data source', 'data:', 'price:']):
            if 'alpaca api' in message_lower:
                return 'data_source_alpaca'
            elif 'massive api' in message_lower:
                return 'data_source_massive'
            elif 'yfinance' in message_lower:
                return 'data_source_yfinance'
            else:
                return 'data_source'
        
        # Setup validation
        if any(x in message_lower for x in ['setup', 'validating', 'symbol selection', 'eligible symbols']):
            return 'setup_validation'
        
        # RL inference
        if any(x in message_lower for x in ['rl action', 'rl inference', 'rl probs', 'action_strength', 'rl model']):
            return 'rl_inference'
        
        # Ensemble activity
        if any(x in message_lower for x in ['ensemble', 'meta-router', 'trend agent', 'reversal agent', 'volatility agent']):
            return 'ensemble_activity'
        
        # Trade execution
        if any(x in message_lower for x in ['executed', 'order submitted', 'trade', 'buy', 'sell', 'position']):
            if 'executed' in message_lower or 'order submitted' in message_lower:
                return 'trade_execution'
            else:
                return 'trade_activity'
        
        # Blocking/rejection
        if any(x in message_lower for x in ['blocked', 'reject', 'skip', 'confidence too low', 'safeguard']):
            return 'blocked'
        
        # Price validation
        if any(x in message_lower for x in ['price validation', 'price:', 'current_price', 'data is', 'stale']):
            return 'price_validation'
        
        # Safeguard check
        if any(x in message_lower for x in ['safeguard', 'risk check', 'cooldown', 'max position']):
            return 'safeguard_check'
        
        return None
    
    def extract_activity_details(self, message: str, activity_type: str) -> Dict:
        """Extract specific details based on activity type"""
        details = {}
        message_lower = message.lower()
        
        # Extract symbol
        symbol_match = re.search(r'\b(SPY|QQQ|IWM|SPX)\b', message, re.IGNORECASE)
        if symbol_match:
            details['symbol'] = symbol_match.group(1).upper()
        
        # Extract data source
        if 'data_source' in activity_type:
            if 'alpaca api' in message_lower:
                details['data_source'] = 'Alpaca API'
            elif 'massive api' in message_lower:
                details['data_source'] = 'Massive API'
            elif 'yfinance' in message_lower:
                details['data_source'] = 'yfinance (DELAYED)'
            else:
                details['data_source'] = 'Unknown'
            
            # Extract price if available
            price_match = re.search(r'\$(\d+\.?\d*)', message)
            if price_match:
                details['price'] = float(price_match.group(1))
            
            # Extract data age
            age_match = re.search(r'(\d+\.?\d*)\s*min', message)
            if age_match:
                details['data_age_minutes'] = float(age_match.group(1))
        
        # Extract setup validation details
        if activity_type == 'setup_validation':
            if 'eligible' in message_lower:
                details['status'] = 'validating'
            elif 'selected' in message_lower:
                details['status'] = 'selected'
            elif 'blocked' in message_lower or 'reject' in message_lower:
                details['status'] = 'rejected'
        
        # Extract RL inference details
        if activity_type == 'rl_inference':
            # Extract action
            action_match = re.search(r'action[=:]?\s*(\d+)', message, re.IGNORECASE)
            if action_match:
                action_map = {0: 'HOLD', 1: 'BUY CALL', 2: 'BUY PUT', 3: 'TRIM 50%', 4: 'TRIM 70%', 5: 'FULL EXIT'}
                action_num = int(action_match.group(1))
                details['rl_action'] = action_map.get(action_num, f'Action {action_num}')
            
            # Extract confidence
            conf_match = re.search(r'(?:strength|confidence)[=:]?\s*(\d+\.?\d*)', message, re.IGNORECASE)
            if conf_match:
                details['confidence'] = float(conf_match.group(1))
        
        # Extract ensemble details
        if activity_type == 'ensemble_activity':
            # Extract ensemble action
            action_match = re.search(r'action[=:]?\s*(\d+)', message, re.IGNORECASE)
            if action_match:
                action_map = {0: 'HOLD', 1: 'BUY CALL', 2: 'BUY PUT'}
                action_num = int(action_match.group(1))
                details['ensemble_action'] = action_map.get(action_num, f'Action {action_num}')
            
            # Extract confidence
            conf_match = re.search(r'confidence[=:]?\s*(\d+\.?\d*)', message, re.IGNORECASE)
            if conf_match:
                details['ensemble_confidence'] = float(conf_match.group(1))
            
            # Extract regime
            regime_match = re.search(r'regime[=:]?\s*(\w+)', message, re.IGNORECASE)
            if regime_match:
                details['regime'] = regime_match.group(1)
        
        # Extract blocking reason
        if activity_type == 'blocked':
            if 'confidence too low' in message_lower:
                details['block_reason'] = 'Confidence below threshold'
            elif 'safeguard' in message_lower:
                details['block_reason'] = 'Safeguard triggered'
            elif 'cooldown' in message_lower:
                details['block_reason'] = 'Cooldown active'
            elif 'max position' in message_lower:
                details['block_reason'] = 'Max positions reached'
            else:
                details['block_reason'] = 'Blocked'
        
        # Extract trade details
        if 'trade_execution' in activity_type:
            # Extract action
            if 'buy call' in message_lower:
                details['trade_action'] = 'BUY CALL'
            elif 'buy put' in message_lower:
                details['trade_action'] = 'BUY PUT'
            elif 'sell' in message_lower:
                details['trade_action'] = 'SELL'
            
            # Extract quantity
            qty_match = re.search(r'(\d+)\s*x\s*', message, re.IGNORECASE)
            if qty_match:
                details['quantity'] = int(qty_match.group(1))
        
        return details
    
    def get_recent_activities(self, limit: int = 50) -> List[Dict]:
        """Get most recent activities"""
        activities = self.parse_log_file(max_lines=1000)
        return activities[:limit]
    
    def get_activities_by_type(self, activity_type: str, limit: int = 20) -> List[Dict]:
        """Get activities filtered by type"""
        activities = self.parse_log_file(max_lines=1000)
        filtered = [a for a in activities if a.get('type') == activity_type]
        return filtered[:limit]
    
    def get_data_source_summary(self) -> Dict:
        """Get summary of data sources used"""
        activities = self.parse_log_file(max_lines=1000)
        data_sources = {}
        
        for activity in activities:
            if 'data_source' in activity.get('type', ''):
                source = activity.get('data_source', 'Unknown')
                data_sources[source] = data_sources.get(source, 0) + 1
        
        return data_sources
    
    def get_setup_validation_summary(self) -> Dict:
        """Get summary of setup validations"""
        activities = self.parse_log_file(max_lines=1000)
        summary = {
            'validating': 0,
            'selected': 0,
            'rejected': 0,
            'by_symbol': {}
        }
        
        for activity in activities:
            if activity.get('type') == 'setup_validation':
                status = activity.get('status', 'unknown')
                if status in summary:
                    summary[status] += 1
                
                symbol = activity.get('symbol')
                if symbol:
                    if symbol not in summary['by_symbol']:
                        summary['by_symbol'][symbol] = {'validating': 0, 'selected': 0, 'rejected': 0}
                    if status in summary['by_symbol'][symbol]:
                        summary['by_symbol'][symbol][status] += 1
        
        return summary

