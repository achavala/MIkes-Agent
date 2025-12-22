#!/usr/bin/env python3
"""
Detailed Analysis of Today's Market Setups
Analyzes what setups were validated, picked, rejected, and why
"""
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json

class SetupAnalyzer:
    def __init__(self, log_file=None):
        self.today = datetime.now().date()
        self.log_file = log_file or f"logs/mike_agent_safe_{self.today.strftime('%Y%m%d')}.log"
        
        # Try agent_output.log if daily log doesn't exist
        if not os.path.exists(self.log_file) and os.path.exists("agent_output.log"):
            self.log_file = "agent_output.log"
        
        self.setups = []
        self.rejections = []
        self.executions = []
        self.stats = defaultdict(int)
        
    def parse_log_line(self, line):
        """Parse log line and extract relevant information"""
        if not line.strip():
            return None
        
        # Extract timestamp
        timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # Extract date if present
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        log_date = None
        if date_match:
            try:
                log_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
            except:
                pass
        
        # Extract level
        level_match = re.search(r'\[(INFO|WARNING|ERROR|TRADE|DEBUG|CRITICAL)\]', line)
        level = level_match.group(1) if level_match else 'INFO'
        
        # Extract message
        message_match = re.search(r'\]\s+(.+)', line)
        message = message_match.group(1) if message_match else line
        
        return {
            'timestamp': timestamp,
            'date': log_date or self.today,
            'level': level,
            'message': message,
            'raw': line
        }
    
    def extract_setup_info(self, log_entry):
        """Extract setup validation information from log entry"""
        msg = log_entry['message']
        result = {
            'time': log_entry['timestamp'],
            'symbol': None,
            'action': None,
            'strength': None,
            'source': None,
            'reason': None,
            'status': None
        }
        
        # Extract symbol (SPY, QQQ, IWM, SPX)
        symbol_match = re.search(r'\b(SPY|QQQ|IWM|SPX)\b', msg)
        if symbol_match:
            result['symbol'] = symbol_match.group(1)
        
        # RL Inference patterns - multiple formats
        if 'RL Inference' in msg or 'RL Action=' in msg or '🧠' in msg or 'RL Debug:' in msg:
            result['status'] = 'validating'
            result['source'] = 'RL'
            
            # Extract action - multiple patterns
            action_match = re.search(r'action[=:]?\s*(\d+)', msg, re.IGNORECASE)
            if not action_match:
                # Try "Action=1 (BUY CALL)" pattern
                action_match = re.search(r'Action=(\d+)', msg)
            if not action_match:
                # Try "→ Action=1" pattern
                action_match = re.search(r'→\s*Action=(\d+)', msg)
            
            if action_match:
                result['action'] = int(action_match.group(1))
            
            # Extract strength/confidence - multiple patterns
            strength_match = re.search(r'(?:Strength|strength|confidence)[=:]?\s*([\d.]+)', msg, re.IGNORECASE)
            if not strength_match:
                # Try Raw= pattern (this is the raw RL output before calibration)
                raw_match = re.search(r'Raw=([\d.]+)', msg)
                if raw_match:
                    raw_val = float(raw_match.group(1))
                    # Convert raw to estimated strength (heuristic)
                    if raw_val >= 0.5:
                        result['strength'] = 0.50 + (raw_val - 0.5) * 0.3  # Map 0.5-1.0 to 0.5-0.65
                    else:
                        result['strength'] = raw_val * 0.5  # Map 0.0-0.5 to 0.0-0.25
                    result['raw_value'] = raw_val
            
            return result
        
        # TA Pattern detection
        if 'TA Pattern:' in msg:
            result['status'] = 'validating'
            result['source'] = 'TA'
            
            pattern_match = re.search(r'Pattern:\s*(\w+)', msg)
            if pattern_match:
                result['reason'] = f"TA Pattern: {pattern_match.group(1)}"
            
            return result
        
        # Symbol selection (picked)
        if 'SYMBOL SELECTION:' in msg or 'selected for' in msg.lower():
            result['status'] = 'picked'
            
            strength_match = re.search(r'strength=([\d.]+)', msg)
            if strength_match:
                result['strength'] = float(strength_match.group(1))
            
            return result
        
        # Rejections
        if 'BLOCKED:' in msg:
            result['status'] = 'rejected'
            
            # Extract rejection reason
            if 'Confidence too low' in msg:
                result['reason'] = 'Low Confidence'
                strength_match = re.search(r'strength=([\d.]+)', msg)
                if strength_match:
                    result['strength'] = float(strength_match.group(1))
            elif 'No eligible symbols' in msg:
                result['reason'] = 'No Eligible Symbols'
            elif 'Stale Data' in msg or 'data validation failed' in msg.lower():
                result['reason'] = 'Stale Data'
            elif 'cooldown' in msg.lower():
                result['reason'] = 'Cooldown Active'
            elif 'Max positions' in msg or 'concurrent' in msg.lower():
                result['reason'] = 'Max Positions Reached'
            elif 'Safeguard active' in msg:
                result['reason'] = 'Safeguard Active'
                # Extract specific safeguard reason
                if 'After 14:30' in msg or 'theta' in msg.lower():
                    result['reason'] = 'Time Filter (After 14:30 EST)'
                elif 'VIX' in msg:
                    result['reason'] = 'VIX Kill Switch'
                elif 'Max concurrent' in msg or 'Max positions' in msg:
                    result['reason'] = 'Max Positions Reached'
                elif 'Daily loss' in msg.lower():
                    result['reason'] = 'Daily Loss Limit'
                elif 'Drawdown' in msg:
                    result['reason'] = 'Max Drawdown'
            else:
                result['reason'] = 'Other Block'
            
            return result
        
        # Executions
        if 'EXECUTED:' in msg or '✓ EXECUTED:' in msg:
            result['status'] = 'executed'
            
            symbol_match = re.search(r'BUY\s+(\d+)x\s+(\w+)', msg)
            if symbol_match:
                result['qty'] = int(symbol_match.group(1))
                result['option_symbol'] = symbol_match.group(2)
            
            return result
        
        return None
    
    def analyze_logs(self):
        """Analyze log file for setup validations"""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        print(f"📊 Analyzing log file: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        # Filter to today's entries (if possible)
        today_lines = []
        current_hour = datetime.now().hour
        
        for line in lines:
            # Check if line is from today - multiple patterns
            is_today = False
            
            # Pattern 1: Date in line
            if self.today.strftime('%Y-%m-%d') in line or 'Dec 19' in line or '2025-12-19' in line:
                is_today = True
            
            # Pattern 2: Recent timestamps (14:xx, 15:xx, 16:xx for today)
            timestamp_match = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\]', line)
            if timestamp_match:
                hour = int(timestamp_match.group(1))
                # If it's afternoon/evening, look for 14-16 hour entries
                if current_hour >= 14:
                    if hour >= 14 and hour <= 16:
                        is_today = True
                # If it's morning, look for 8-12 hour entries
                elif current_hour < 14:
                    if hour >= 8 and hour <= 12:
                        is_today = True
            
            if is_today:
                today_lines.append(line)
        
        # If no date filtering possible, use recent lines
        if not today_lines:
            today_lines = lines[-2000:]  # Last 2000 lines
            print(f"⚠️  No date filtering possible, using last {len(today_lines)} lines")
        
        print(f"📝 Processing {len(today_lines)} log entries...")
        
        for line in today_lines:
            log_entry = self.parse_log_line(line)
            if not log_entry:
                continue
            
            setup_info = self.extract_setup_info(log_entry)
            if setup_info and setup_info['status']:
                if setup_info['status'] == 'validating':
                    self.setups.append(setup_info)
                    self.stats['total_validated'] += 1
                elif setup_info['status'] == 'picked':
                    self.setups.append(setup_info)
                    self.stats['total_picked'] += 1
                elif setup_info['status'] == 'rejected':
                    self.rejections.append(setup_info)
                    self.stats['total_rejected'] += 1
                    if setup_info['reason']:
                        self.stats[f"rejected_{setup_info['reason'].lower().replace(' ', '_')}"] += 1
                elif setup_info['status'] == 'executed':
                    self.executions.append(setup_info)
                    self.stats['total_executed'] += 1
    
    def generate_report(self):
        """Generate detailed analysis report"""
        print("\n" + "="*80)
        print("📊 DETAILED MARKET ANALYSIS - DECEMBER 19, 2025")
        print("="*80)
        
        # Overall Statistics
        print("\n📈 OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total Setups Validated: {self.stats['total_validated']}")
        print(f"Total Setups Picked: {self.stats['total_picked']}")
        print(f"Total Setups Rejected: {self.stats['total_rejected']}")
        print(f"Total Trades Executed: {self.stats['total_executed']}")
        
        # Rejection Reasons Breakdown
        if self.rejections:
            print("\n❌ REJECTION REASONS BREAKDOWN")
            print("-" * 80)
            rejection_reasons = defaultdict(int)
            for rejection in self.rejections:
                if rejection['reason']:
                    rejection_reasons[rejection['reason']] += 1
            
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
        
        # Detailed Rejections
        if self.rejections:
            print("\n🔍 DETAILED REJECTION ANALYSIS")
            print("-" * 80)
            for i, rejection in enumerate(self.rejections[-20:], 1):  # Last 20 rejections
                symbol = rejection.get('symbol', 'UNKNOWN')
                reason = rejection.get('reason', 'Unknown')
                strength = rejection.get('strength')
                time = rejection.get('time', 'N/A')
                
                strength_str = f" | Strength: {strength:.3f}" if strength else ""
                print(f"{i}. [{time}] {symbol} | Reason: {reason}{strength_str}")
        
        # Setup Validations
        if self.setups:
            print("\n🔍 SETUP VALIDATIONS")
            print("-" * 80)
            
            # Group by symbol
            by_symbol = defaultdict(list)
            for setup in self.setups:
                symbol = setup.get('symbol', 'UNKNOWN')
                by_symbol[symbol].append(setup)
            
            for symbol, setups in sorted(by_symbol.items()):
                print(f"\n  {symbol}:")
                for setup in setups[-10:]:  # Last 10 per symbol
                    action = setup.get('action', 'N/A')
                    strength = setup.get('strength')
                    source = setup.get('source', 'Unknown')
                    time = setup.get('time', 'N/A')
                    
                    action_names = {0: 'HOLD', 1: 'BUY CALL', 2: 'BUY PUT', 3: 'TRIM', 4: 'EXIT'}
                    action_name = action_names.get(action, f'Action {action}')
                    
                    strength_str = f" | Strength: {strength:.3f}" if strength else ""
                    print(f"    [{time}] {action_name} ({source}){strength_str}")
        
        # Executions
        if self.executions:
            print("\n✅ EXECUTED TRADES")
            print("-" * 80)
            for i, execution in enumerate(self.executions[-10:], 1):  # Last 10
                symbol = execution.get('symbol', 'UNKNOWN')
                option = execution.get('option_symbol', 'N/A')
                qty = execution.get('qty', 0)
                time = execution.get('time', 'N/A')
                print(f"{i}. [{time}] {symbol} | {qty}x {option}")
        
        # Confidence Analysis
        strengths = [s.get('strength') for s in self.setups if s.get('strength')]
        if strengths:
            print("\n📊 CONFIDENCE ANALYSIS")
            print("-" * 80)
            print(f"  Average Confidence: {sum(strengths) / len(strengths):.3f}")
            print(f"  Min Confidence: {min(strengths):.3f}")
            print(f"  Max Confidence: {max(strengths):.3f}")
            print(f"  Below Threshold (< 0.52): {sum(1 for s in strengths if s < 0.52)}")
            print(f"  Above Threshold (>= 0.52): {sum(1 for s in strengths if s >= 0.52)}")
        
        # Summary
        print("\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        
        if self.stats['total_rejected'] > 0:
            print(f"\n⚠️  {self.stats['total_rejected']} setups were rejected today.")
            print("   Main reasons:")
            top_reasons = sorted(
                [(k.replace('rejected_', ''), v) for k, v in self.stats.items() if k.startswith('rejected_')],
                key=lambda x: -x[1]
            )[:3]
            for reason, count in top_reasons:
                print(f"   - {reason.replace('_', ' ').title()}: {count}")
        
        if self.stats['total_executed'] > 0:
            print(f"\n✅ {self.stats['total_executed']} trades were executed today.")
        
        if self.stats['total_validated'] > 0 and self.stats['total_executed'] == 0:
            print("\n🤔 Analysis:")
            print("   Many setups were validated but none executed.")
            print("   This suggests confidence thresholds or safeguards are working.")
            print("   Check rejection reasons above for details.")
        
        print("\n" + "="*80)

def main():
    analyzer = SetupAnalyzer()
    analyzer.analyze_logs()
    analyzer.generate_report()

if __name__ == '__main__':
    main()

