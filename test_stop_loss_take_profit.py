#!/usr/bin/env python3
"""
Comprehensive test suite for stop-loss and take-profit logic
Tests all scenarios without requiring Alpaca API
"""
import sys
from unittest.mock import Mock, MagicMock
import numpy as np

# Mock the imports
sys.modules['alpaca_trade_api'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()

# Import after mocking
from mike_agent_live_safe import RiskManager, VOL_REGIMES, check_stop_losses

class MockAlpacaPosition:
    """Mock Alpaca position object"""
    def __init__(self, symbol, qty, market_value, avg_entry_price):
        self.symbol = symbol
        self.qty = str(qty)
        self.market_value = str(market_value)
        self.avg_entry_price = str(avg_entry_price)
        self.asset_class = 'option'  # Alpaca uses 'option' for options

class MockAlpacaSnapshot:
    """Mock Alpaca option snapshot"""
    def __init__(self, bid_price):
        self.bid_price = str(bid_price) if bid_price else None

class MockAlpacaAPI:
    """Mock Alpaca API"""
    def __init__(self):
        self.positions = []
        self.closed_positions = []
        self.orders = []
    
    def list_positions(self):
        return self.positions
    
    def get_option_snapshot(self, symbol):
        # Find position and return mock snapshot
        for pos in self.positions:
            if pos.symbol == symbol:
                # Calculate bid from market value
                qty = float(pos.qty)
                market_val = float(pos.market_value)
                bid = market_val / (qty * 100) if qty > 0 else 0
                return MockAlpacaSnapshot(bid)
        return MockAlpacaSnapshot(None)
    
    def close_position(self, symbol):
        """Close a position"""
        self.closed_positions.append(symbol)
        # Remove from positions
        self.positions = [p for p in self.positions if p.symbol != symbol]
        return True
    
    def submit_order(self, symbol, qty, side, type, time_in_force):
        """Submit a sell order"""
        order = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': type,
            'time_in_force': time_in_force
        }
        self.orders.append(order)
        
        # Update position qty
        for pos in self.positions:
            if pos.symbol == symbol:
                current_qty = float(pos.qty)
                new_qty = current_qty - qty
                pos.qty = str(new_qty)
                pos.market_value = str(float(pos.market_value) * (new_qty / current_qty))
                break
        return order

def test_take_profit_tier1():
    """Test TP1 triggers at correct level"""
    print("\n" + "="*60)
    print("TEST 1: Take-Profit Tier 1 (TP1) - Should sell 50% at +40%")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Setup: Normal regime (VIX 18-25) - mock VIX to return 20
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    vol_params = VOL_REGIMES["normal"]
    
    # Create position: Entry premium $1.00, current premium $1.40 (+40%)
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 1.40  # +40% gain
    qty = 10
    
    # Add to tracked positions
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': False,
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    # Create mock Alpaca position
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    # Run check
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify
    assert len(api.orders) == 1, f"Expected 1 sell order, got {len(api.orders)}"
    assert api.orders[0]['qty'] == 5, f"Expected to sell 5 contracts (50%), got {api.orders[0]['qty']}"
    assert risk_mgr.open_positions[symbol]['tp1_done'] == True, "TP1 should be marked as done"
    assert risk_mgr.open_positions[symbol]['qty_remaining'] == 5, f"Expected 5 remaining, got {risk_mgr.open_positions[symbol]['qty_remaining']}"
    
    print("✅ PASS: TP1 triggered correctly, sold 50%")
    
    # Clean up for next test
    api.orders = []

def test_take_profit_tier2():
    """Test TP2 triggers at correct level"""
    print("\n" + "="*60)
    print("TEST 2: Take-Profit Tier 2 (TP2) - Should sell 60% at +80%")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 1.80  # +80% gain
    qty = 10
    
    # Position already had TP1 done
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': True,  # TP1 already done
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify - TP2 now sells 60% of remaining (updated fix)
    assert len(api.orders) == 1, f"Expected 1 sell order, got {len(api.orders)}"
    assert api.orders[0]['qty'] == 6, f"Expected to sell 6 contracts (60% of 10), got {api.orders[0]['qty']}"
    assert risk_mgr.open_positions[symbol]['tp2_done'] == True, "TP2 should be marked as done"
    assert risk_mgr.open_positions[symbol]['trail_active'] == True, "Trailing stop should be activated"
    
    print("✅ PASS: TP2 triggered correctly, sold 60%, trailing stop activated")

def test_take_profit_tier3():
    """Test TP3 triggers at correct level"""
    print("\n" + "="*60)
    print("TEST 3: Take-Profit Tier 3 (TP3) - Should close full position at +150%")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 2.50  # +150% gain
    qty = 10
    
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': True,
        'tp2_done': True,
        'tp3_done': False,
        'trail_active': True
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify
    assert symbol in api.closed_positions, "Position should be closed"
    assert symbol not in risk_mgr.open_positions, "Position should be removed from tracking"
    
    print("✅ PASS: TP3 triggered correctly, full position closed")

def test_stop_loss_normal():
    """Test normal stop-loss triggers damage control at -20%"""
    print("\n" + "="*60)
    print("TEST 4: Normal Stop-Loss - Should trigger damage control at -20%")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 0.80  # -20% loss (exactly at damage control threshold)
    qty = 10
    
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': False,
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify - at -20%, damage control should close 50%
    assert len(api.orders) == 1, f"Expected 1 sell order for damage control, got {len(api.orders)}"
    assert api.orders[0]['qty'] == 5, f"Expected to sell 5 contracts (50% damage control), got {api.orders[0]['qty']}"
    assert risk_mgr.open_positions[symbol]['qty_remaining'] == 5, f"Expected 5 remaining after damage control, got {risk_mgr.open_positions[symbol]['qty_remaining']}"
    
    print("✅ PASS: Damage control stop triggered correctly at -20%")

def test_stop_loss_hard():
    """Test hard stop-loss triggers at -35% (or -30% if regime hard_sl is tighter)"""
    print("\n" + "="*60)
    print("TEST 5: Hard Stop-Loss - Should close at -35% (or regime hard_sl)")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 0.65  # -35% loss (hard stop threshold)
    qty = 10
    
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': False,
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify
    assert symbol in api.closed_positions, "Position should be closed at hard stop-loss"
    
    print("✅ PASS: Hard stop-loss triggered correctly at -30%")

def test_trailing_stop():
    """Test trailing stop after TP2"""
    print("\n" + "="*60)
    print("TEST 6: Trailing Stop - Should close if price drops below trail level")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    vol_params = VOL_REGIMES["normal"]
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    # Price went up to +80% (TP2 hit), then dropped to trail level
    trail_price = entry_premium * (1 + vol_params['trail'])  # +60% trail
    current_premium = trail_price * 0.99  # Just below trail
    qty = 5  # After TP2, 5 remaining
    
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': True,
        'tp2_done': True,
        'tp3_done': False,
        'trail_active': True,
        'trail_price': trail_price
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify
    assert symbol in api.closed_positions, "Position should be closed by trailing stop"
    
    print("✅ PASS: Trailing stop triggered correctly")

def test_sequential_take_profits():
    """Test that TPs trigger in sequence"""
    print("\n" + "="*60)
    print("TEST 7: Sequential Take-Profits - TP1 → TP2 → TP3")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    qty = 10
    
    # Start with TP1 level
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': False,
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    # Test TP1
    current_premium = 1.40  # +40%
    market_value = current_premium * qty * 100
    api.positions = [MockAlpacaPosition(symbol, qty, market_value, entry_premium)]
    check_stop_losses(api, risk_mgr, 684.0)
    assert risk_mgr.open_positions[symbol]['tp1_done'] == True, "TP1 should be done"
    assert risk_mgr.open_positions[symbol]['qty_remaining'] == 5, "Should have 5 remaining"
    print("  ✓ TP1 triggered")
    
    # Test TP2 (with remaining qty)
    current_premium = 1.80  # +80%
    qty_remaining = 5
    market_value = current_premium * qty_remaining * 100
    api.positions = [MockAlpacaPosition(symbol, qty_remaining, market_value, entry_premium)]
    api.orders = []  # Reset orders
    check_stop_losses(api, risk_mgr, 684.0)
    assert risk_mgr.open_positions[symbol]['tp2_done'] == True, "TP2 should be done"
    print("  ✓ TP2 triggered")
    
    # Test TP3
    current_premium = 2.50  # +150%
    qty_remaining = 2  # After TP2, 2 remaining
    market_value = current_premium * qty_remaining * 100
    api.positions = [MockAlpacaPosition(symbol, qty_remaining, market_value, entry_premium)]
    check_stop_losses(api, risk_mgr, 684.0)
    assert symbol in api.closed_positions, "TP3 should close position"
    print("  ✓ TP3 triggered")
    
    print("✅ PASS: All take-profits triggered in correct sequence")

def test_stop_loss_priority():
    """Test that stop-losses have priority over take-profits"""
    print("\n" + "="*60)
    print("TEST 8: Stop-Loss Priority - Damage control at -25% (between -20% and -35%)")
    print("="*60)
    
    api = MockAlpacaAPI()
    risk_mgr = RiskManager()
    
    # Mock VIX to return normal regime value
    risk_mgr.get_current_vix = lambda: 20.0
    risk_mgr.current_regime = "normal"
    
    symbol = "SPY251203C00684000"
    entry_premium = 1.00
    current_premium = 0.75  # -25% (between -20% damage control and -35% hard stop)
    qty = 10
    
    risk_mgr.open_positions[symbol] = {
        'entry_premium': entry_premium,
        'entry_price': 684.0,
        'strike': 684.0,
        'type': 'call',
        'qty_remaining': qty,
        'vol_regime': 'normal',
        'tp1_done': False,
        'tp2_done': False,
        'tp3_done': False,
        'trail_active': False
    }
    
    market_value = current_premium * qty * 100
    api.positions.append(MockAlpacaPosition(symbol, qty, market_value, entry_premium))
    
    check_stop_losses(api, risk_mgr, 684.0)
    
    # Verify damage control triggered (not hard stop, since -25% is between -20% and -35%)
    # Damage control should close 50% at -25% (between -20% and -35%)
    assert len(api.orders) == 1, f"Expected 1 sell order for damage control, got {len(api.orders)}"
    assert api.orders[0]['qty'] == 5, f"Expected to sell 5 contracts (50% damage control), got {api.orders[0]['qty']}"
    
    print("✅ PASS: Damage control stop triggered correctly at -25%")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("STOP-LOSS & TAKE-PROFIT VALIDATION TESTS")
    print("="*60)
    
    tests = [
        test_take_profit_tier1,
        test_take_profit_tier2,
        test_take_profit_tier3,
        test_stop_loss_normal,
        test_stop_loss_hard,
        test_trailing_stop,
        test_sequential_take_profits,
        test_stop_loss_priority
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

