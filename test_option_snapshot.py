#!/usr/bin/env python3
"""Test Alpaca Option Snapshot API"""

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
import config
import os

# Get credentials
api_key = getattr(config, 'ALPACA_KEY', None) or getattr(config, 'APCA_API_KEY_ID', None) or os.getenv('APCA_API_KEY_ID')
api_secret = getattr(config, 'ALPACA_SECRET', None) or getattr(config, 'APCA_API_SECRET_KEY', None) or os.getenv('APCA_API_SECRET_KEY')

client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)

# Example: SPY Dec 22, 2025 $680 Call
option_symbol = "SPY251222C00680000"

print("="*80)
print("OPTION SNAPSHOT DATA STRUCTURE")
print("="*80)

req = OptionSnapshotRequest(symbol_or_symbols=[option_symbol])
snapshots = client.get_option_snapshot(req)

snapshot = snapshots[option_symbol]

print(f"\n✅ Option Symbol: {snapshot.symbol}")
print(f"\n📊 Latest Quote:")
quote = snapshot.latest_quote
if quote:
    bid_price = quote.bid_price
    ask_price = quote.ask_price
    bid_size = quote.bid_size
    ask_size = quote.ask_size
    print(f"  Bid: ${bid_price:.2f} (Size: {bid_size:.0f})")
    print(f"  Ask: ${ask_price:.2f} (Size: {ask_size:.0f})")
    spread = ask_price - bid_price
    mid = (bid_price + ask_price) / 2
    spread_pct = (spread / mid) * 100 if mid > 0 else 0
    print(f"  Mid: ${mid:.2f}")
    print(f"  Spread: ${spread:.2f} ({spread_pct:.1f}%)")
else:
    print("  No quote data available")

print(f"\n💰 Latest Trade:")
trade = snapshot.latest_trade
if trade:
    print(f"  Price: ${trade.price:.2f}")
    print(f"  Size: {trade.size:.0f}")
    print(f"  Time: {trade.timestamp}")
else:
    print("  No trade data available")

print(f"\n📈 Greeks: {snapshot.greeks}")
print(f"📊 Implied Volatility: {snapshot.implied_volatility}")

print("\n" + "="*80)
print("USAGE IN PHASE 0 BACKTEST:")
print("="*80)
print("✅ Can get real bid/ask spreads for spread gate")
print("✅ Can get actual option prices (no estimates needed)")
print("✅ Can get Greeks if available")
print("✅ Can calculate spread % for gatekeeper")
print("="*80)

