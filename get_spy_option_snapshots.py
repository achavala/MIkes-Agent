#!/usr/bin/env python3
"""Get detailed option snapshots for SPY options"""

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
import config
import os
from datetime import datetime, date

# Get credentials
api_key = getattr(config, 'ALPACA_KEY', None) or getattr(config, 'APCA_API_KEY_ID', None) or os.getenv('APCA_API_KEY_ID')
api_secret = getattr(config, 'ALPACA_SECRET', None) or getattr(config, 'APCA_API_SECRET_KEY', None) or os.getenv('APCA_API_SECRET_KEY')

client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)

print("="*80)
print("ATTEMPTING OPTION SNAPSHOT REQUEST FOR SPY")
print("="*80)

# Try the request as specified
try:
    req = OptionSnapshotRequest(symbol_or_symbols=["SPY"])
    snapshots = client.get_option_snapshot(req)
    print("\n✅ Success!")
    print(snapshots)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n" + "="*80)
    print("NOTE: API requires option symbols, not underlying symbols")
    print("="*80)
    print("\nFetching option snapshots for SPY 0DTE options...")
    
    # Generate option symbols for today's 0DTE options
    today = date.today()
    exp_date_str = today.strftime('%y%m%d')  # e.g., 251222
    
    # Try multiple strikes around current price (assuming ~$680)
    strikes = [670, 675, 680, 685, 690]
    option_symbols = []
    
    for strike in strikes:
        strike_str = f"{int(strike * 1000):08d}"
        # Calls
        option_symbols.append(f"SPY{exp_date_str}C{strike_str}")
        # Puts
        option_symbols.append(f"SPY{exp_date_str}P{strike_str}")
    
    print(f"\n📊 Requesting snapshots for {len(option_symbols)} option symbols...")
    print(f"Symbols: {option_symbols[:5]}... (showing first 5)")
    
    try:
        req = OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        snapshots = client.get_option_snapshot(req)
        
        print(f"\n✅ Retrieved {len(snapshots)} snapshots")
        print("\n" + "="*80)
        print("DETAILED OPTION SNAPSHOT DATA")
        print("="*80)
        
        for symbol, snapshot in snapshots.items():
            print(f"\n{'='*80}")
            print(f"OPTION: {symbol}")
            print(f"{'='*80}")
            
            print(f"\n📊 Symbol: {snapshot.symbol}")
            
            # Quote data
            if snapshot.latest_quote:
                quote = snapshot.latest_quote
                print(f"\n💰 LATEST QUOTE:")
                print(f"  Bid Price: ${quote.bid_price:.2f}")
                print(f"  Bid Size: {quote.bid_size:.0f} contracts")
                print(f"  Bid Exchange: {quote.bid_exchange}")
                print(f"  Ask Price: ${quote.ask_price:.2f}")
                print(f"  Ask Size: {quote.ask_size:.0f} contracts")
                print(f"  Ask Exchange: {quote.ask_exchange}")
                
                spread = quote.ask_price - quote.bid_price
                mid = (quote.bid_price + quote.ask_price) / 2
                spread_pct = (spread / mid) * 100 if mid > 0 else 0
                print(f"  Mid Price: ${mid:.2f}")
                print(f"  Spread: ${spread:.2f} ({spread_pct:.2f}%)")
                print(f"  Timestamp: {quote.timestamp}")
                print(f"  Conditions: {quote.conditions}")
                print(f"  Tape: {quote.tape}")
            else:
                print("\n💰 LATEST QUOTE: No data available")
            
            # Trade data
            if snapshot.latest_trade:
                trade = snapshot.latest_trade
                print(f"\n📈 LATEST TRADE:")
                print(f"  Price: ${trade.price:.2f}")
                print(f"  Size: {trade.size:.0f} contracts")
                print(f"  Exchange: {trade.exchange}")
                print(f"  Timestamp: {trade.timestamp}")
                print(f"  Conditions: {trade.conditions}")
                print(f"  Tape: {trade.tape}")
                print(f"  Trade ID: {trade.id}")
            else:
                print("\n📈 LATEST TRADE: No data available")
            
            # Greeks
            if snapshot.greeks:
                greeks = snapshot.greeks
                print(f"\n📐 GREEKS:")
                print(f"  Delta: {greeks.delta:.4f}" if hasattr(greeks, 'delta') else "  Delta: N/A")
                print(f"  Gamma: {greeks.gamma:.6f}" if hasattr(greeks, 'gamma') else "  Gamma: N/A")
                print(f"  Theta: {greeks.theta:.4f}" if hasattr(greeks, 'theta') else "  Theta: N/A")
                print(f"  Vega: {greeks.vega:.4f}" if hasattr(greeks, 'vega') else "  Vega: N/A")
                print(f"  Rho: {greeks.rho:.4f}" if hasattr(greeks, 'rho') else "  Rho: N/A")
            else:
                print(f"\n📐 GREEKS: None (not available)")
            
            # Implied Volatility
            if snapshot.implied_volatility is not None:
                print(f"\n📊 IMPLIED VOLATILITY: {snapshot.implied_volatility:.2%}")
            else:
                print(f"\n📊 IMPLIED VOLATILITY: None (not available)")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total snapshots retrieved: {len(snapshots)}")
        quotes_available = sum(1 for s in snapshots.values() if s.latest_quote)
        trades_available = sum(1 for s in snapshots.values() if s.latest_trade)
        greeks_available = sum(1 for s in snapshots.values() if s.greeks)
        iv_available = sum(1 for s in snapshots.values() if s.implied_volatility is not None)
        
        print(f"Snapshots with quotes: {quotes_available}/{len(snapshots)}")
        print(f"Snapshots with trades: {trades_available}/{len(snapshots)}")
        print(f"Snapshots with Greeks: {greeks_available}/{len(snapshots)}")
        print(f"Snapshots with IV: {iv_available}/{len(snapshots)}")
        
    except Exception as e2:
        print(f"\n❌ Error fetching option snapshots: {e2}")
        import traceback
        traceback.print_exc()


