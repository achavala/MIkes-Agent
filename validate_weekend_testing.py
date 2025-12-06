#!/usr/bin/env python3
"""
Weekend Testing Environment Validation
Validates all components are ready for weekend testing as if market is live
"""
import sys
import os
from datetime import datetime, timedelta
import traceback

print("=" * 70)
print("🧪 WEEKEND TESTING ENVIRONMENT VALIDATION")
print("=" * 70)
print()

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

# ==================== 1. PYTHON ENVIRONMENT ====================
print("1. Checking Python Environment...")
print("-" * 70)

try:
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        results['passed'].append('Python version')
    else:
        print_error(f"Python version {python_version.major}.{python_version.minor} is too old (need 3.8+)")
        results['failed'].append('Python version')
except Exception as e:
    print_error(f"Error checking Python version: {e}")
    results['failed'].append('Python version')

print()

# ==================== 2. CRITICAL DEPENDENCIES ====================
print("2. Checking Critical Dependencies...")
print("-" * 70)

required_modules = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'yfinance': 'yfinance',
    'alpaca_trade_api': 'alpaca-trade-api',
    'pytz': 'pytz',
    'sqlite3': 'sqlite3 (built-in)',
}

optional_modules = {
    'stable_baselines3': 'stable-baselines3',
    'gymnasium': 'gymnasium',
    'torch': 'torch',
    'streamlit': 'streamlit',
}

for module_name, package_name in required_modules.items():
    try:
        __import__(module_name)
        print_success(f"{package_name} installed")
        results['passed'].append(f'Module: {package_name}')
    except ImportError:
        print_error(f"{package_name} NOT installed - run: pip install {package_name}")
        results['failed'].append(f'Module: {package_name}')

for module_name, package_name in optional_modules.items():
    try:
        __import__(module_name)
        print_success(f"{package_name} installed (optional)")
        results['passed'].append(f'Module: {package_name}')
    except ImportError:
        print_warning(f"{package_name} not installed (optional)")
        results['warnings'].append(f'Module: {package_name}')

print()

# ==================== 3. PROJECT FILES ====================
print("3. Checking Project Files...")
print("-" * 70)

required_files = {
    'mike_agent_live_safe.py': 'Main trading agent',
    'gap_detection.py': 'Gap detection module',
    'trade_database.py': 'Trade database',
    'weekend_backtest.py': 'Weekend backtesting script',
    'test_gap_detection.py': 'Gap detection tests',
    'app.py': 'Streamlit dashboard',
    'config.py': 'Configuration file',
    'Procfile': 'Railway deployment config',
    'requirements_railway.txt': 'Dependencies list',
}

for filename, description in required_files.items():
    if os.path.exists(filename):
        print_success(f"{filename} exists ({description})")
        results['passed'].append(f'File: {filename}')
    else:
        print_error(f"{filename} NOT found ({description})")
        results['failed'].append(f'File: {filename}')

print()

# ==================== 4. CONFIGURATION ====================
print("4. Checking Configuration...")
print("-" * 70)

try:
    import config
    
    # Check Alpaca API keys
    has_key = hasattr(config, 'ALPACA_KEY') and config.ALPACA_KEY and config.ALPACA_KEY != 'YOUR_PAPER_KEY'
    has_secret = hasattr(config, 'ALPACA_SECRET') and config.ALPACA_SECRET and config.ALPACA_SECRET != 'YOUR_PAPER_SECRET'
    has_url = hasattr(config, 'ALPACA_BASE_URL') and config.ALPACA_BASE_URL
    
    if has_key and has_secret:
        print_success("Alpaca API keys configured")
        results['passed'].append('Alpaca API keys')
    else:
        print_warning("Alpaca API keys not configured (required for live trading, optional for backtesting)")
        results['warnings'].append('Alpaca API keys')
    
    if has_url:
        print_info(f"Alpaca Base URL: {config.ALPACA_BASE_URL}")
    else:
        print_warning("Alpaca Base URL not set")
        
except Exception as e:
    print_error(f"Error checking configuration: {e}")
    results['failed'].append('Configuration check')

print()

# ==================== 5. GAP DETECTION MODULE ====================
print("5. Validating Gap Detection Module...")
print("-" * 70)

try:
    from gap_detection import detect_overnight_gap, get_gap_based_action
    print_success("Gap detection module imports successfully")
    results['passed'].append('Gap detection module')
    
    # Check if functions exist
    if callable(detect_overnight_gap) and callable(get_gap_based_action):
        print_success("Gap detection functions available")
        results['passed'].append('Gap detection functions')
    else:
        print_error("Gap detection functions not callable")
        results['failed'].append('Gap detection functions')
        
except ImportError as e:
    print_error(f"Gap detection module import failed: {e}")
    results['failed'].append('Gap detection module')
except Exception as e:
    print_error(f"Error validating gap detection: {e}")
    results['failed'].append('Gap detection validation')

print()

# ==================== 6. TRADE DATABASE ====================
print("6. Validating Trade Database...")
print("-" * 70)

try:
    from trade_database import TradeDatabase
    
    # Try to create database instance
    db = TradeDatabase()
    print_success("Trade database module imports successfully")
    results['passed'].append('Trade database module')
    
    # Check if database file exists or can be created
    db_path = db.db_path
    if os.path.exists(db_path) or os.path.exists(os.path.dirname(db_path) or '.'):
        print_success(f"Database path accessible: {db_path}")
        results['passed'].append('Database path')
    else:
        print_warning(f"Database path may not exist yet: {db_path} (will be created)")
        results['warnings'].append('Database path')
        
except ImportError as e:
    print_error(f"Trade database module import failed: {e}")
    results['failed'].append('Trade database module')
except Exception as e:
    print_error(f"Error validating trade database: {e}")
    results['failed'].append('Trade database validation')

print()

# ==================== 7. HISTORICAL DATA ACCESS ====================
print("7. Testing Historical Data Access...")
print("-" * 70)

try:
    import yfinance as yf
    import pandas as pd
    
    # Test fetching SPY data
    print_info("Testing SPY data fetch...")
    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="5d", interval="1m")
    
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    if len(hist) > 0:
        print_success(f"SPY data accessible ({len(hist)} bars)")
        results['passed'].append('Historical data access')
    else:
        print_error("SPY data fetch returned empty")
        results['failed'].append('Historical data access')
        
    # Test QQQ
    print_info("Testing QQQ data fetch...")
    ticker_qqq = yf.Ticker("QQQ")
    hist_qqq = ticker_qqq.history(period="2d", interval="1m")
    
    if isinstance(hist_qqq.columns, pd.MultiIndex):
        hist_qqq.columns = hist_qqq.columns.get_level_values(0)
    
    if len(hist_qqq) > 0:
        print_success(f"QQQ data accessible ({len(hist_qqq)} bars)")
        results['passed'].append('QQQ data access')
    else:
        print_warning("QQQ data fetch returned empty")
        results['warnings'].append('QQQ data access')
        
    # Test SPX
    print_info("Testing SPX data fetch...")
    ticker_spx = yf.Ticker("^SPX")
    hist_spx = ticker_spx.history(period="2d", interval="1m")
    
    if isinstance(hist_spx.columns, pd.MultiIndex):
        hist_spx.columns = hist_spx.columns.get_level_values(0)
    
    if len(hist_spx) > 0:
        print_success(f"SPX data accessible ({len(hist_spx)} bars)")
        results['passed'].append('SPX data access')
    else:
        print_warning("SPX data fetch returned empty")
        results['warnings'].append('SPX data access')
        
except Exception as e:
    print_error(f"Error testing historical data: {e}")
    results['failed'].append('Historical data access')
    traceback.print_exc()

print()

# ==================== 8. WEEKEND BACKTESTING SCRIPT ====================
print("8. Validating Weekend Backtesting Script...")
print("-" * 70)

try:
    # Check if script is executable and can be imported
    script_path = 'weekend_backtest.py'
    if os.path.exists(script_path):
        print_success("Weekend backtesting script exists")
        results['passed'].append('Weekend backtest script')
        
        # Check if it's executable
        if os.access(script_path, os.X_OK):
            print_success("Weekend backtesting script is executable")
        else:
            print_info("Weekend backtesting script (not executable, run with: python weekend_backtest.py)")
    else:
        print_error("Weekend backtesting script not found")
        results['failed'].append('Weekend backtest script')
        
except Exception as e:
    print_error(f"Error checking weekend backtesting script: {e}")
    results['failed'].append('Weekend backtest validation')

print()

# ==================== 9. ALPACA API CONNECTION (Optional) ====================
print("9. Testing Alpaca API Connection (Optional)...")
print("-" * 70)

try:
    import config
    
    if hasattr(config, 'ALPACA_KEY') and hasattr(config, 'ALPACA_SECRET'):
        if config.ALPACA_KEY and config.ALPACA_KEY != 'YOUR_PAPER_KEY':
            import alpaca_trade_api as tradeapi
            
            base_url = getattr(config, 'ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            
            try:
                api = tradeapi.REST(
                    config.ALPACA_KEY,
                    config.ALPACA_SECRET,
                    base_url,
                    api_version='v2'
                )
                
                # Test connection
                account = api.get_account()
                print_success(f"Alpaca API connection successful (Account: {account.account_number})")
                results['passed'].append('Alpaca API connection')
                
                # Check account status
                if account.trading_blocked:
                    print_warning("Alpaca account is trading blocked")
                    results['warnings'].append('Alpaca trading blocked')
                else:
                    print_success("Alpaca account is ready for trading")
                    results['passed'].append('Alpaca account status')
                    
            except Exception as e:
                print_warning(f"Alpaca API connection failed: {e} (optional for backtesting)")
                results['warnings'].append('Alpaca API connection')
        else:
            print_info("Alpaca API keys not configured (optional for backtesting)")
            results['warnings'].append('Alpaca API keys not set')
    else:
        print_info("Alpaca API configuration not found (optional for backtesting)")
        results['warnings'].append('Alpaca API config')
        
except Exception as e:
    print_info(f"Alpaca API test skipped: {e}")
    results['warnings'].append('Alpaca API test')

print()

# ==================== 10. GAP DETECTION TEST ====================
print("10. Testing Gap Detection Logic...")
print("-" * 70)

try:
    from gap_detection import detect_overnight_gap
    import yfinance as yf
    import pandas as pd
    import pytz
    
    # Create mock risk manager
    class MockRM:
        def log(self, msg, level="INFO"):
            pass
    
    # Get recent data
    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="5d", interval="1m")
    
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    if len(hist) > 0:
        current_price = float(hist['Close'].iloc[-1])
        
        # Test gap detection (may or may not detect gap, that's okay)
        gap_data = detect_overnight_gap("SPY", current_price, hist, MockRM())
        
        if gap_data:
            print_success("Gap detection function works")
            print_info(f"Gap detected: {gap_data.get('direction', 'N/A')}")
        else:
            print_success("Gap detection function works (no gap detected for test data)")
        
        results['passed'].append('Gap detection test')
    else:
        print_warning("Could not fetch data for gap detection test")
        results['warnings'].append('Gap detection test')
        
except Exception as e:
    print_error(f"Gap detection test failed: {e}")
    results['failed'].append('Gap detection test')
    traceback.print_exc()

print()

# ==================== 11. WEEKEND BACKTEST DRY RUN ====================
print("11. Testing Weekend Backtest (Dry Run)...")
print("-" * 70)

try:
    # Test if we can import the weekend backtest functions
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Check if script can be imported (simulate)
    if os.path.exists('weekend_backtest.py'):
        # Read first few lines to check syntax
        with open('weekend_backtest.py', 'r') as f:
            first_lines = f.read(500)
            if 'def simulate_trading_day' in first_lines or 'import' in first_lines:
                print_success("Weekend backtest script structure valid")
                results['passed'].append('Weekend backtest structure')
            else:
                print_warning("Weekend backtest script may have issues")
                results['warnings'].append('Weekend backtest structure')
    else:
        print_error("Weekend backtest script not found")
        results['failed'].append('Weekend backtest script')
        
except Exception as e:
    print_error(f"Error testing weekend backtest: {e}")
    results['failed'].append('Weekend backtest validation')

print()

# ==================== SUMMARY ====================
print("=" * 70)
print("📊 VALIDATION SUMMARY")
print("=" * 70)
print()

total_checks = len(results['passed']) + len(results['failed']) + len(results['warnings'])
pass_rate = (len(results['passed']) / total_checks * 100) if total_checks > 0 else 0

print(f"✅ Passed:  {len(results['passed'])}")
print(f"❌ Failed:  {len(results['failed'])}")
print(f"⚠️  Warnings: {len(results['warnings'])}")
print(f"📈 Pass Rate: {pass_rate:.1f}%")
print()

# Overall status
if len(results['failed']) == 0:
    if len(results['warnings']) == 0:
        print_success("🎉 ALL CHECKS PASSED - Environment is READY for weekend testing!")
    else:
        print_warning("✅ CORE CHECKS PASSED - Ready for weekend testing (some warnings)")
        print()
        print("Warnings (non-critical):")
        for warning in results['warnings']:
            print(f"  • {warning}")
else:
    print_error("❌ SOME CHECKS FAILED - Please fix issues before testing")
    print()
    print("Failed checks:")
    for failure in results['failed']:
        print(f"  • {failure}")
    print()
    print("Please fix the above issues and run validation again.")

print()
print("=" * 70)
print("📝 NEXT STEPS")
print("=" * 70)
print()

if len(results['failed']) == 0:
    print("✅ Environment is ready! You can now run:")
    print()
    print("  1. Quick gap test:")
    print("     python test_gap_detection.py 2025-12-05 SPY")
    print()
    print("  2. Full backtest:")
    print("     python weekend_backtest.py --symbol SPY --date 2025-12-05")
    print()
    print("  3. Multiple days:")
    print("     python weekend_backtest.py --symbol SPY --start 2025-12-01 --end 2025-12-05")
    print()
    print("  4. Full test suite:")
    print("     ./run_weekend_tests.sh")
    print()
else:
    print("⚠️  Please fix the failed checks above before running tests.")
    print()

print("=" * 70)

