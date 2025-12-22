#!/usr/bin/env python3
"""
Setup Analytics Dashboard
Shows current status of setup validation, picking, and rejection
"""
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import json

class SetupAnalytics:
    def __init__(self, log_file: Optional[str] = None):
        # Try to find the most recent log file
        if log_file is None:
            daily_log = f"logs/mike_agent_safe_{datetime.now().strftime('%Y%m%d')}.log"
            if os.path.exists(daily_log):
                self.log_file = daily_log
            elif os.path.exists("agent_output.log"):
                self.log_file = "agent_output.log"
            else:
                self.log_file = daily_log  # Will show empty if doesn't exist
        else:
            self.log_file = log_file
        
        # Today's date for filtering
        self.today = datetime.now().date()
        self.setup_status = {
            'validating': [],
            'picked': [],
            'rejected': [],
            'executed': []
        }
        self.recent_logs = deque(maxlen=1000)  # Keep last 1000 log entries
        self.stats = {
            'total_setups_validated': 0,
            'total_picked': 0,
            'total_rejected': 0,
            'total_executed': 0,
            'rejection_reasons': defaultdict(int),
            'symbol_activity': defaultdict(int),
            'confidence_distribution': []
        }
        
    def parse_option_expiration(self, option_symbol: str) -> Optional[datetime]:
        """Parse expiration date from option symbol (e.g., SPY251210C00688000 -> 2025-12-10)"""
        # Format: SPY251210C00688000 = SPY + YYMMDD + C/P + Strike
        match = re.search(r'(\w{3})(\d{6})([CP])(\d+)', option_symbol)
        if match:
            underlying = match.group(1)
            date_str = match.group(2)  # YYMMDD
            option_type = match.group(3)
            
            try:
                # Parse YYMMDD to datetime
                year = 2000 + int(date_str[:2])  # YY -> 20YY
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                return datetime(year, month, day).date()
            except:
                return None
        return None
    
    def is_option_expired(self, option_symbol: str) -> bool:
        """Check if option is expired"""
        exp_date = self.parse_option_expiration(option_symbol)
        if exp_date:
            return exp_date < self.today
        return False
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line and extract setup information"""
        if not line.strip():
            return None
        
        # Try to extract full date-time if available (some logs have dates)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        log_date = None
        if date_match:
            try:
                log_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
            except:
                pass
        
        # Extract timestamp
        timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # If no date in log, assume today if timestamp is recent, or check file date
        if log_date is None:
            # For agent_output.log, we can't determine date from line, so we'll filter later
            log_date = self.today  # Assume today, will filter by file date
        
        # Extract log level
        level_match = re.search(r'\[(INFO|WARNING|ERROR|TRADE|DEBUG|CRITICAL)\]', line)
        level = level_match.group(1) if level_match else 'INFO'
        
        # Extract message
        message_match = re.search(r'\]\s+(.+)', line)
        message = message_match.group(1) if message_match else line
        
        return {
            'timestamp': timestamp,
            'date': log_date,
            'level': level,
            'message': message,
            'raw': line
        }
    
    def analyze_setup_validation(self, log_entry: Dict):
        """Analyze log entry for setup validation patterns"""
        msg = log_entry['message']
        
        # Filter: Only process entries from today
        log_date = log_entry.get('date')
        if log_date and log_date != self.today:
            return  # Skip old entries
        
        # Data validation
        if 'CRITICAL: Data is from' in msg or 'data validation failed' in msg:
            self.stats['total_rejected'] += 1
            self.stats['rejection_reasons']['Stale Data'] += 1
            self.setup_status['rejected'].append({
                'time': log_entry['timestamp'],
                'reason': 'Stale Data',
                'details': msg
            })
        
        # Price validation
        if 'PRICE MISMATCH' in msg or 'Price Validation' in msg:
            if 'match' in msg.lower():
                # Price validated successfully
                pass
            else:
                self.stats['rejection_reasons']['Price Mismatch'] += 1
        
        # RL Inference - multiple patterns
        if 'RL Inference:' in msg or 'RL Action=' in msg or 'RL Debug:' in msg or '🧠' in msg:
            # Try to extract symbol (SPY, QQQ, IWM, SPX)
            symbol_match = re.search(r'\b(SPY|QQQ|IWM|SPX)\b', msg)
            symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
            
            # Extract action and strength - multiple patterns
            action_match = re.search(r'action[=:]?\s*(\d+)', msg)
            strength_match = re.search(r'(?:Strength|strength)[=:]?\s*([\d.]+)', msg)
            
            # Also check for Raw= pattern
            if not action_match:
                raw_match = re.search(r'Raw=([\d.]+)', msg)
                if raw_match:
                    raw_val = float(raw_match.group(1))
                    # Infer action from raw value (heuristic)
                    if raw_val > 0.5:
                        action = 1  # BUY CALL
                    else:
                        action = 0  # HOLD
            
            action = int(action_match.group(1)) if action_match else None
            strength = float(strength_match.group(1)) if strength_match else None
            
            if strength:
                self.stats['confidence_distribution'].append(strength)
            
            self.setup_status['validating'].append({
                'time': log_entry['timestamp'],
                'symbol': symbol,
                'action': action,
                'strength': strength,
                'message': msg
            })
            self.stats['total_setups_validated'] += 1
            self.stats['symbol_activity'][symbol] += 1
        
        # TA Pattern Detection
        if 'TA Pattern:' in msg:
            pattern_match = re.search(r'(\w+)\s+TA Pattern:\s*(\w+)', msg)
            if pattern_match:
                symbol = pattern_match.group(1)
                pattern = pattern_match.group(2)
                self.stats['symbol_activity'][symbol] += 1
        
        # Confidence threshold rejection
        if 'BLOCKED:' in msg and 'Confidence too low' in msg:
            symbol_match = re.search(r'symbol\s+(\w+)', msg)
            strength_match = re.search(r'strength=([\d.]+)', msg)
            
            symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
            strength = float(strength_match.group(1)) if strength_match else None
            
            self.stats['total_rejected'] += 1
            self.stats['rejection_reasons']['Low Confidence'] += 1
            self.setup_status['rejected'].append({
                'time': log_entry['timestamp'],
                'symbol': symbol,
                'reason': 'Low Confidence',
                'strength': strength,
                'details': msg
            })
        
        # No eligible symbols
        if 'BLOCKED: No eligible symbols' in msg:
            self.stats['total_rejected'] += 1
            self.stats['rejection_reasons']['No Eligible Symbols'] += 1
            self.setup_status['rejected'].append({
                'time': log_entry['timestamp'],
                'reason': 'No Eligible Symbols',
                'details': msg
            })
        
        # Symbol selection
        if 'SYMBOL SELECTION:' in msg:
            symbol_match = re.search(r'(\w+)\s+selected', msg)
            strength_match = re.search(r'strength=([\d.]+)', msg)
            
            symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
            strength = float(strength_match.group(1)) if strength_match else None
            
            self.setup_status['picked'].append({
                'time': log_entry['timestamp'],
                'symbol': symbol,
                'strength': strength,
                'details': msg
            })
            self.stats['total_picked'] += 1
        
        # Trade execution
        if 'EXECUTED:' in msg or '✓ EXECUTED:' in msg:
            symbol_match = re.search(r'BUY\s+(\d+)x\s+(\w+)', msg)
            if symbol_match:
                qty = int(symbol_match.group(1))
                symbol = symbol_match.group(2)
                
                # Check if option is expired
                is_expired = self.is_option_expired(symbol)
                exp_date = self.parse_option_expiration(symbol)
                
                # Extract underlying for activity tracking
                underlying_match = re.search(r'^(\w{3})', symbol)
                underlying = underlying_match.group(1) if underlying_match else symbol[:3]
                
                self.setup_status['executed'].append({
                    'time': log_entry['timestamp'],
                    'symbol': symbol,
                    'underlying': underlying,
                    'qty': qty,
                    'expiration': exp_date.strftime('%Y-%m-%d') if exp_date else None,
                    'is_expired': is_expired,
                    'details': msg
                })
                self.stats['total_executed'] += 1
                self.stats['symbol_activity'][underlying] += 1
    
    def load_recent_logs(self, lines: int = 500):
        """Load recent log entries from file, filtering to today's entries"""
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                
                # If log file has date in name (mike_agent_safe_YYYYMMDD.log), only read that day
                # Otherwise, read last N lines and filter by date in parse_log_line
                if 'mike_agent_safe_' in self.log_file and self.log_file.endswith('.log'):
                    # Extract date from filename
                    date_match = re.search(r'mike_agent_safe_(\d{8})\.log', self.log_file)
                    if date_match:
                        file_date_str = date_match.group(1)
                        try:
                            file_date = datetime.strptime(file_date_str, '%Y%m%d').date()
                            if file_date != self.today:
                                # File is from a different day, return empty
                                return []
                        except:
                            pass
                
                return all_lines[-lines:]  # Last N lines
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def analyze_logs(self, lines: int = 500):
        """Analyze recent log entries"""
        log_lines = self.load_recent_logs(lines)
        
        # Clear old status (keep only recent)
        self.setup_status['validating'] = []
        self.setup_status['picked'] = []
        self.setup_status['rejected'] = []
        
        for line in log_lines:
            log_entry = self.parse_log_line(line)
            if log_entry:
                self.recent_logs.append(log_entry)
                self.analyze_setup_validation(log_entry)
        
        # Keep only last 50 entries in each category
        for key in self.setup_status:
            if len(self.setup_status[key]) > 50:
                self.setup_status[key] = self.setup_status[key][-50:]
    
    def get_current_status(self) -> Dict:
        """Get current status summary"""
        # Get most recent validation
        current_validating = self.setup_status['validating'][-1] if self.setup_status['validating'] else None
        current_picked = self.setup_status['picked'][-1] if self.setup_status['picked'] else None
        current_rejected = self.setup_status['rejected'][-1] if self.setup_status['rejected'] else None
        current_executed = self.setup_status['executed'][-1] if self.setup_status['executed'] else None
        
        # Calculate average confidence
        avg_confidence = sum(self.stats['confidence_distribution']) / len(self.stats['confidence_distribution']) if self.stats['confidence_distribution'] else 0.0
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_validating': current_validating,
            'current_picked': current_picked,
            'current_rejected': current_rejected,
            'current_executed': current_executed,
            'stats': {
                'total_validated': self.stats['total_setups_validated'],
                'total_picked': self.stats['total_picked'],
                'total_rejected': self.stats['total_rejected'],
                'total_executed': self.stats['total_executed'],
                'avg_confidence': round(avg_confidence, 3),
                'rejection_reasons': dict(self.stats['rejection_reasons']),
                'symbol_activity': dict(self.stats['symbol_activity'])
            },
            'recent_validating': self.setup_status['validating'][-10:],  # Last 10
            'recent_picked': self.setup_status['picked'][-10:],
            'recent_rejected': self.setup_status['rejected'][-10:],
            'recent_executed': self.setup_status['executed'][-10:]
        }
    
    def print_dashboard(self):
        """Print formatted dashboard"""
        self.analyze_logs()
        status = self.get_current_status()
        
        print("\n" + "="*80)
        print("🎯 SETUP ANALYTICS DASHBOARD")
        print("="*80)
        print(f"📅 Last Updated: {status['timestamp']}")
        print(f"📊 Log File: {self.log_file}")
        print(f"📆 Filtering: Today ({self.today}) only")
        
        # Check if we're reading from agent_output.log (multi-day file)
        if 'agent_output.log' in self.log_file:
            print(f"⚠️  WARNING: Reading from agent_output.log (multi-day file)")
            print(f"   Showing all entries - some may be from previous days")
            print(f"   Use daily log file for today-only: logs/mike_agent_safe_{self.today.strftime('%Y%m%d')}.log")
        print()
        
        # Current Status
        print("🔍 CURRENT STATUS")
        print("-" * 80)
        
        if status['current_validating']:
            v = status['current_validating']
            strength = v.get('strength') or 0.0
            print(f"  ⏳ Validating: {v.get('symbol', 'UNKNOWN')} | "
                  f"Action={v.get('action', 'N/A')} | Strength={strength:.3f} | "
                  f"Time: {v.get('time', 'N/A')}")
        else:
            print("  ⏳ Validating: None")
        
        if status['current_picked']:
            p = status['current_picked']
            strength = p.get('strength') or 0.0
            print(f"  ✅ Picked: {p.get('symbol', 'UNKNOWN')} | "
                  f"Strength={strength:.3f} | "
                  f"Time: {p.get('time', 'N/A')}")
        else:
            print("  ✅ Picked: None")
        
        if status['current_rejected']:
            r = status['current_rejected']
            print(f"  ❌ Rejected: {r.get('symbol', 'UNKNOWN')} | "
                  f"Reason: {r.get('reason', 'Unknown')} | "
                  f"Time: {r.get('time', 'N/A')}")
        else:
            print("  ❌ Rejected: None")
        
        if status['current_executed']:
            e = status['current_executed']
            symbol = e.get('symbol', 'UNKNOWN')
            exp_date = e.get('expiration')
            is_expired = e.get('is_expired', False)
            expired_warning = " ⚠️ EXPIRED" if is_expired else ""
            exp_info = f" (Exp: {exp_date})" if exp_date else ""
            print(f"  🚀 Executed: {symbol}{exp_info}{expired_warning} | "
                  f"Qty: {e.get('qty', 0)}x | "
                  f"Time: {e.get('time', 'N/A')}")
        else:
            print("  🚀 Executed: None")
        
        print()
        
        # Statistics
        print("📈 STATISTICS")
        print("-" * 80)
        stats = status['stats']
        print(f"  Total Validated: {stats['total_validated']}")
        print(f"  Total Picked: {stats['total_picked']}")
        print(f"  Total Rejected: {stats['total_rejected']}")
        print(f"  Total Executed: {stats['total_executed']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
        print()
        
        # Rejection Reasons
        if stats['rejection_reasons']:
            print("❌ REJECTION REASONS")
            print("-" * 80)
            for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
            print()
        
        # Symbol Activity
        if stats['symbol_activity']:
            print("📊 SYMBOL ACTIVITY")
            print("-" * 80)
            for symbol, count in sorted(stats['symbol_activity'].items(), key=lambda x: -x[1]):
                print(f"  {symbol}: {count} events")
            print()
        
        # Recent Activity
        print("🕐 RECENT ACTIVITY (Last 10)")
        print("-" * 80)
        
        print("\n  Validating:")
        for v in status['recent_validating'][-5:]:
            strength = v.get('strength') or 0.0
            print(f"    [{v.get('time', 'N/A')}] {v.get('symbol', 'UNKNOWN')} | "
                  f"Action={v.get('action', 'N/A')} | Strength={strength:.3f}")
        
        print("\n  Picked:")
        for p in status['recent_picked'][-5:]:
            strength = p.get('strength') or 0.0
            print(f"    [{p.get('time', 'N/A')}] {p.get('symbol', 'UNKNOWN')} | "
                  f"Strength={strength:.3f}")
        
        print("\n  Rejected:")
        for r in status['recent_rejected'][-5:]:
            print(f"    [{r.get('time', 'N/A')}] {r.get('symbol', 'UNKNOWN')} | "
                  f"Reason: {r.get('reason', 'Unknown')}")
        
        print("\n  Executed:")
        for e in status['recent_executed'][-5:]:
            symbol = e.get('symbol', 'UNKNOWN')
            exp_date = e.get('expiration')
            is_expired = e.get('is_expired', False)
            expired_warning = " ⚠️ EXPIRED" if is_expired else ""
            exp_info = f" (Exp: {exp_date})" if exp_date else ""
            print(f"    [{e.get('time', 'N/A')}] {symbol}{exp_info}{expired_warning} | "
                  f"Qty: {e.get('qty', 0)}x")
        
        print("\n" + "="*80)
    
    def export_json(self, filename: str = None):
        """Export current status to JSON"""
        self.analyze_logs()
        status = self.get_current_status()
        
        if filename is None:
            filename = f"setup_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        print(f"✅ Exported analytics to {filename}")
        return filename

def main():
    import sys
    
    analytics = SetupAnalytics()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--export' or sys.argv[1] == '-e':
            analytics.export_json()
            return
        elif sys.argv[1] == '--file' or sys.argv[1] == '-f':
            if len(sys.argv) > 2:
                analytics.log_file = sys.argv[2]
            else:
                print("Error: --file requires a log file path")
                return
    
    # Print dashboard
    analytics.print_dashboard()
    
    # Auto-refresh option
    if len(sys.argv) > 1 and (sys.argv[1] == '--watch' or sys.argv[1] == '-w'):
        import time
        print("\n🔄 Watching for updates (Ctrl+C to stop)...")
        try:
            while True:
                time.sleep(5)  # Refresh every 5 seconds
                os.system('clear' if os.name != 'nt' else 'cls')
                analytics.print_dashboard()
        except KeyboardInterrupt:
            print("\n👋 Stopped watching")

if __name__ == '__main__':
    main()

