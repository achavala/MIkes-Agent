#!/usr/bin/env python3
"""
Real-Time Agent Activity Monitor
Shows what the agent is checking, selecting, and executing
"""

import subprocess
import re
import sys
from datetime import datetime

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

def extract_timestamp(line):
    """Extract timestamp from log line"""
    match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
    return match.group(1) if match else datetime.now().strftime('%H:%M:%S')

def parse_rl_signal(line):
    """Parse RL inference signal"""
    # Pattern: "🔍 SPY RL Action=1, Strength=0.725"
    patterns = [
        r'🔍\s*(\w+)\s+RL.*action[=:](\d+).*strength[=:]?([\d.]+)',
        r'RL Action.*(\w+).*action[=:](\d+).*strength[=:]?([\d.]+)',
        r'(\w+).*RL.*Action=(\d+).*Strength=([\d.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            symbol = match.group(1)
            action = int(match.group(2))
            strength = float(match.group(3))
            
            action_names = {
                0: "HOLD",
                1: "BUY CALL",
                2: "BUY PUT",
                3: "TRIM 50%",
                4: "TRIM 70%",
                5: "EXIT"
            }
            action_name = action_names.get(action, f"ACTION {action}")
            strength_pct = strength * 100
            
            if strength >= 0.65:
                return f"{Colors.GREEN}[RL]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} → {Colors.GREEN}{action_name}{Colors.NC} ({Colors.GREEN}{strength_pct:.1f}%{Colors.NC})"
            else:
                return f"{Colors.YELLOW}[RL]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} → {Colors.YELLOW}{action_name}{Colors.NC} ({Colors.YELLOW}{strength_pct:.1f}%{Colors.NC}) {Colors.RED}[BLOCKED: Low Confidence]{Colors.NC}"
    return None

def parse_symbol_selection(line):
    """Parse symbol selection"""
    match = re.search(r'(?:Symbol selected|SYMBOL SELECTION).*?(\w+).*?(BUY CALL|BUY PUT|CALL|PUT)', line, re.IGNORECASE)
    if match:
        symbol = match.group(1)
        action = match.group(2)
        return f"{Colors.MAGENTA}[SELECT]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} selected for {Colors.GREEN}{action}{Colors.NC}"
    return None

def parse_trade_execution(line):
    """Parse trade execution"""
    if re.search(r'(TRADE_OPENED|NEW ENTRY|EXECUTED.*BUY)', line, re.IGNORECASE):
        symbol_match = re.search(r'(\w+\d+[CP]\d+)', line)
        qty_match = re.search(r'qty[=:](\d+)', line, re.IGNORECASE)
        premium_match = re.search(r'premium[=:]\$?([\d.]+)', line, re.IGNORECASE)
        strike_match = re.search(r'strike[=:]\$?([\d.]+)', line, re.IGNORECASE)
        
        symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
        qty = qty_match.group(1) if qty_match else "?"
        premium = premium_match.group(1) if premium_match else "?"
        strike = strike_match.group(1) if strike_match else "?"
        
        output = [
            f"{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.NC}",
            f"{Colors.GREEN}✅ TRADE EXECUTED{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC}",
            f"   {Colors.WHITE}Qty:{Colors.NC} {qty} contracts",
            f"   {Colors.WHITE}Premium:{Colors.NC} ${premium}",
            f"   {Colors.WHITE}Strike:{Colors.NC} ${strike}",
            f"{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.NC}"
        ]
        return "\n".join(output)
    return None

def parse_blocked(line):
    """Parse blocked trade"""
    if re.search(r'(BLOCKED|⛔)', line):
        symbol_match = re.search(r'(\w+).*BLOCKED', line, re.IGNORECASE)
        reason_match = re.search(r'BLOCKED[:\s]+(.*?)(?:\s+\||$)', line, re.IGNORECASE)
        
        symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
        reason = reason_match.group(1).strip() if reason_match else "Unknown reason"
        
        return f"{Colors.RED}⛔ BLOCKED{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC}: {Colors.YELLOW}{reason}{Colors.NC}"
    return None

def parse_ensemble(line):
    """Parse ensemble signal"""
    match = re.search(r'Ensemble.*?(\w+).*?confidence[=:]?([\d.]+)', line, re.IGNORECASE)
    if match:
        symbol = match.group(1)
        confidence = float(match.group(2))
        conf_pct = confidence * 100
        return f"{Colors.BLUE}[ENSEMBLE]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} confidence: {Colors.BLUE}{conf_pct:.1f}%{Colors.NC}"
    return None

def parse_exit(line):
    """Parse position exit"""
    if re.search(r'(STOP-LOSS|TAKE-PROFIT|TRAILING|Position closed)', line, re.IGNORECASE):
        symbol_match = re.search(r'(\w+\d+[CP]\d+)', line)
        symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
        
        if re.search(r'STOP-LOSS', line, re.IGNORECASE):
            return f"{Colors.RED}[EXIT]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} {line.strip()}"
        else:
            return f"{Colors.GREEN}[EXIT]{Colors.NC} {Colors.CYAN}{symbol}{Colors.NC} {line.strip()}"
    return None

def parse_safeguard(line):
    """Parse safeguard trigger"""
    if re.search(r'SAFEGUARD.*TRIGGERED', line, re.IGNORECASE):
        return f"{Colors.RED}🚨 SAFEGUARD{Colors.NC} {line.strip()}"
    return None

def format_line(line):
    """Format a log line with appropriate colors and parsing"""
    timestamp = extract_timestamp(line)
    
    # Try different parsers in order
    parsed = (
        parse_rl_signal(line) or
        parse_symbol_selection(line) or
        parse_trade_execution(line) or
        parse_blocked(line) or
        parse_ensemble(line) or
        parse_exit(line) or
        parse_safeguard(line)
    )
    
    if parsed:
        return f"[{timestamp}] {parsed}"
    
    # Default formatting for other important lines
    if re.search(r'(ERROR|Error|Exception)', line, re.IGNORECASE):
        return f"[{timestamp}] {Colors.RED}[ERROR]{Colors.NC} {line.strip()}"
    elif re.search(r'(WARNING|Warning|⚠️)', line, re.IGNORECASE):
        return f"[{timestamp}] {Colors.YELLOW}[WARN]{Colors.NC} {line.strip()}"
    elif re.search(r'(Starting|Connected|Model loaded)', line, re.IGNORECASE):
        return f"[{timestamp}] {Colors.WHITE}[INFO]{Colors.NC} {line.strip()}"
    
    # Skip verbose lines by default (uncomment to show all)
    # return f"[{timestamp}] {line.strip()}"
    return None

def main():
    print(f"{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BOLD}║     🔍 MIKE AGENT REAL-TIME ACTIVITY MONITOR                 ║{Colors.NC}")
    print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()
    print("Monitoring: RL Actions | Symbol Selection | Safeguards | Trades")
    print("Press Ctrl+C to stop")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Start fly logs process
    try:
        process = subprocess.Popen(
            ['fly', 'logs', '--app', 'mike-agent-project'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Process lines in real-time
        for line in process.stdout:
            if not line.strip():
                continue
            
            formatted = format_line(line)
            if formatted:
                print(formatted)
                sys.stdout.flush()
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitor stopped by user{Colors.NC}")
        process.terminate()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.NC}")
        sys.exit(1)

if __name__ == "__main__":
    main()





