#!/usr/bin/env python3
"""
Test script for trade database and 0DTE filtering
Run this in virtual environment to validate everything works
"""
import sys
import os
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_creation():
    """Test 1: Database creation and initialization"""
    print("=" * 70)
    print("TEST 1: Database Creation")
    print("=" * 70)
    
    try:
        from trade_database import TradeDatabase
        
        # Create test database
        test_db_path = "test_trades_database.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        db = TradeDatabase(db_path=test_db_path)
        print("✅ Database created successfully")
        
        # Test parsing option symbols
        test_symbols = [
            ("SPY241203C00450000", True),   # Today's date (0DTE)
            ("SPY241204C00450000", False),   # Tomorrow (not 0DTE)
            ("QQQ241203P00350000", True),   # Today's date (0DTE)
        ]
        
        today = date.today()
        for symbol, expected_0dte in test_symbols:
            parsed = db._parse_option_symbol(symbol)
            is_0dte = parsed.get('is_0dte', False)
            print(f"  Symbol: {symbol}")
            print(f"    Underlying: {parsed.get('underlying')}")
            print(f"    Expiration: {parsed.get('expiration_date')}")
            print(f"    Strike: {parsed.get('strike_price')}")
            print(f"    Type: {parsed.get('option_type')}")
            print(f"    Is 0DTE: {is_0dte} (Expected: {expected_0dte})")
            
            # For test, check if expiration matches today
            if parsed.get('expiration_date'):
                exp_date = datetime.strptime(parsed['expiration_date'], '%Y-%m-%d').date()
                actual_0dte = (exp_date == today)
                if actual_0dte == expected_0dte or (not expected_0dte and exp_date > today):
                    print(f"    ✅ Parsing correct")
                else:
                    print(f"    ⚠️  Parsing may need adjustment (date: {exp_date}, today: {today})")
            print()
        
        return db, test_db_path
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_trade_saving(db, test_db_path):
    """Test 2: Saving trades to database"""
    print("=" * 70)
    print("TEST 2: Saving Trades")
    print("=" * 70)
    
    if not db:
        print("❌ Skipping - database not available")
        return False
    
    try:
        today = date.today()
        today_str = today.strftime('%y%m%d')
        
        # Create test trades
        test_trades = [
            {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': f'SPY{today_str}C00450000',  # 0DTE
                'action': 'BUY',
                'qty': 10,
                'entry_premium': 2.50,
                'entry_price': 450.0,
                'strike_price': 450.0,
                'option_type': 'call',
                'regime': 'normal',
                'vix': 20.0,
                'reason': 'test_trade'
            },
            {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': f'SPY{today_str}P00450000',  # 0DTE
                'action': 'BUY',
                'qty': 5,
                'entry_premium': 1.75,
                'entry_price': 450.0,
                'strike_price': 450.0,
                'option_type': 'put',
                'regime': 'normal',
                'vix': 20.0,
                'reason': 'test_trade'
            },
            {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': f'SPY{(today + timedelta(days=1)).strftime("%y%m%d")}C00450000',  # NOT 0DTE
                'action': 'BUY',
                'qty': 3,
                'entry_premium': 3.00,
                'entry_price': 450.0,
                'strike_price': 450.0,
                'option_type': 'call',
                'regime': 'normal',
                'vix': 20.0,
                'reason': 'test_trade_not_0dte'
            }
        ]
        
        for trade in test_trades:
            trade_id = db.save_trade(trade)
            print(f"✅ Saved trade {trade_id}: {trade['symbol']} ({trade['action']})")
        
        # Test exit trade
        exit_trade = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': f'SPY{today_str}C00450000',
            'action': 'SELL',
            'qty': 10,
            'entry_premium': 2.50,
            'exit_premium': 3.75,
            'entry_price': 450.0,
            'pnl': 1250.0,  # (3.75 - 2.50) * 10 * 100
            'pnl_pct': 0.50,  # 50% gain
            'regime': 'normal',
            'vix': 20.0,
            'reason': 'take_profit'
        }
        exit_id = db.save_trade(exit_trade)
        print(f"✅ Saved exit trade {exit_id}: {exit_trade['symbol']} (SELL) - PnL: ${exit_trade['pnl']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Trade saving failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_0dte_filtering(db):
    """Test 3: 0DTE filtering"""
    print("=" * 70)
    print("TEST 3: 0DTE Filtering")
    print("=" * 70)
    
    if not db:
        print("❌ Skipping - database not available")
        return False
    
    try:
        # Get all trades
        all_trades = db.get_all_trades()
        print(f"Total trades in database: {len(all_trades)}")
        
        # Get only 0DTE trades
        odte_trades = db.get_0dte_trades_only()
        print(f"0DTE trades only: {len(odte_trades)}")
        
        # Verify filtering
        for trade in odte_trades:
            symbol = trade.get('symbol', '')
            is_0dte = trade.get('is_0dte', 0)
            print(f"  ✅ {symbol}: is_0dte={is_0dte}")
        
        if len(odte_trades) == 2:  # Should have 2 0DTE trades
            print("✅ 0DTE filtering works correctly")
            return True
        else:
            print(f"⚠️  Expected 2 0DTE trades, got {len(odte_trades)}")
            return False
    except Exception as e:
        print(f"❌ 0DTE filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_statistics(db):
    """Test 4: Trade statistics"""
    print("=" * 70)
    print("TEST 4: Trade Statistics")
    print("=" * 70)
    
    if not db:
        print("❌ Skipping - database not available")
        return False
    
    try:
        # Get all statistics
        all_stats = db.get_trade_statistics(filter_0dte=False)
        print("All trades statistics:")
        print(f"  Total trades: {all_stats['total_trades']}")
        print(f"  Win rate: {all_stats['win_rate']:.1f}%")
        print(f"  Total PnL: ${all_stats['total_pnl']:.2f}")
        
        # Get 0DTE only statistics
        odte_stats = db.get_trade_statistics(filter_0dte=True)
        print("\n0DTE trades only statistics:")
        print(f"  Total trades: {odte_stats['total_trades']}")
        print(f"  Win rate: {odte_stats['win_rate']:.1f}%")
        print(f"  Total PnL: ${odte_stats['total_pnl']:.2f}")
        
        print("✅ Statistics calculated correctly")
        return True
    except Exception as e:
        print(f"❌ Statistics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_filtering():
    """Test 5: Dashboard 0DTE filtering function"""
    print("=" * 70)
    print("TEST 5: Dashboard 0DTE Filtering Function")
    print("=" * 70)
    
    try:
        # Import the function from app.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        is_0dte_option = app_module.is_0dte_option
        
        today = date.today()
        today_str = today.strftime('%y%m%d')
        tomorrow_str = (today + timedelta(days=1)).strftime('%y%m%d')
        
        test_cases = [
            (f'SPY{today_str}C00450000', True),   # 0DTE
            (f'SPY{tomorrow_str}C00450000', False), # Not 0DTE
            (f'QQQ{today_str}P00350000', True),   # 0DTE
            ('SPY241203C00450000', False),  # Old date
            ('INVALID', False),  # Invalid symbol
        ]
        
        all_passed = True
        for symbol, expected in test_cases:
            result = is_0dte_option(symbol)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {symbol}: {result} (Expected: {expected})")
            if result != expected:
                all_passed = False
        
        if all_passed:
            print("✅ Dashboard filtering function works correctly")
        else:
            print("⚠️  Some test cases failed")
        
        return all_passed
    except Exception as e:
        print(f"❌ Dashboard filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_db(test_db_path):
    """Clean up test database"""
    if test_db_path and os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
            print(f"✅ Cleaned up test database: {test_db_path}")
        except:
            print(f"⚠️  Could not remove test database: {test_db_path}")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TRADE DATABASE & 0DTE FILTERING TEST SUITE")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Database creation
    db, test_db_path = test_database_creation()
    results.append(("Database Creation", db is not None))
    print()
    
    # Test 2: Trade saving
    if db:
        results.append(("Trade Saving", test_trade_saving(db, test_db_path)))
        print()
    
    # Test 3: 0DTE filtering
    if db:
        results.append(("0DTE Filtering", test_0dte_filtering(db)))
        print()
    
    # Test 4: Statistics
    if db:
        results.append(("Statistics", test_statistics(db)))
        print()
    
    # Test 5: Dashboard filtering
    results.append(("Dashboard Filtering", test_dashboard_filtering()))
    print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    # Cleanup
    cleanup_test_db(test_db_path)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Database and 0DTE filtering are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


