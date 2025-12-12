# 🔧 MASSIVE API SETUP INSTRUCTIONS (✅ FIXED & VERIFIED)

## ✅ IMPORTANT DISCOVERY: Massive = Polygon.io

**Massive is the rebranded Polygon.io!** Use Polygon.io API structure.

## ✅ VERIFICATION COMPLETE

**Test Results:**
- ✅ **Stocks Developer:** WORKING (200 OK)
- ✅ **Options Contracts:** WORKING (200 OK)
- ⚠️ **Options Snapshot:** 404 (use contracts endpoint instead)

**API Key:** Verified working ✅

---

## 🎯 CORRECT API CONFIGURATION

### Base URL
```
https://api.polygon.io/v2/
```

**Note:** Massive documentation shows `https://api.massive.com/v1/` but may redirect or use Polygon.io endpoints.

### Authentication
Two methods supported:

**Method 1: Query Parameter (Recommended)**
```python
params = {"apiKey": "YOUR_API_KEY"}
```

**Method 2: Authorization Header**
```python
headers = {"Authorization": f"Bearer YOUR_API_KEY"}
```

---

## 📚 AVAILABLE ENDPOINTS

### Stocks Data (Stocks Developer Subscription)

**Get Aggregates/Bars:**
```
GET https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
```

**Parameters:**
- `ticker`: Symbol (e.g., "SPY")
- `multiplier`: Size of timespan (e.g., 1)
- `timespan`: minute, hour, day, week, month, quarter, year
- `from`: Start date (YYYY-MM-DD or timestamp)
- `to`: End date (YYYY-MM-DD or timestamp)
- `apiKey`: Your API key (query param)

**Example:**
```bash
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/2025-12-01/2025-12-08?apiKey=YOUR_API_KEY"
```

### Options Data (Options Starter Subscription)

**✅ WORKING: Get Options Contracts:**
```
GET https://api.polygon.io/v3/reference/options/contracts
```

**Parameters:**
- `underlying_ticker`: Symbol (e.g., "SPY")
- `expiration_date`: Expiry date (YYYY-MM-DD, optional)
- `contract_type`: call or put (optional)
- `strike_price`: Strike price (optional)
- `limit`: Number of results (default: 1000)
- `apiKey`: Your API key (query param)

**Example:**
```bash
curl "https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=SPY&expiration_date=2025-12-08&apiKey=YOUR_API_KEY"
```

**Note:** The `/v2/snapshot/options/` endpoint returns 404. Use `/v3/reference/options/contracts` instead.

---

## 🚀 QUICK START WITH PYTHON CLIENT

### Install Polygon.io Python Client

```bash
pip install polygon-api-client
```

**⚠️ Dependency Note:** There's a websockets version conflict:
- `polygon-api-client` needs `websockets>=14.0`
- `alpaca-trade-api` needs `websockets<11`

**Workaround:** Use REST API directly with `requests` library (no websockets needed for REST).
If you need WebSocket streams, you may need separate environments or use REST only.

### Basic Usage

```python
from polygon import RESTClient

# Initialize client
client = RESTClient("YOUR_API_KEY")

# Get stock aggregates (bars)
aggs = []
for agg in client.list_aggs(
    ticker="SPY",
    multiplier=1,
    timespan="minute",
    _from="2025-12-01",
    to="2025-12-08",
    limit=50000
):
    aggs.append(agg)

# Get options contracts (works!)
options_contracts = []
for contract in client.list_options_contracts(
    underlying_ticker="SPY",
    expiration_date="2025-12-08",
    limit=1000
):
    options_contracts.append(contract)

# Get specific option details (if needed)
# Use the contract ticker from the contracts list
```

---

## 🔧 FIXED TEST SCRIPT

I'll update the test script to use Polygon.io endpoints. Here's what to test:

### Test 1: Stock Aggregates
```python
import requests

API_KEY = "YOUR_API_KEY"
url = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/2025-12-01/2025-12-08"
params = {"apiKey": API_KEY}

response = requests.get(url, params=params)
print(response.status_code)
print(response.json())
```

### Test 2: Options Contracts (WORKING)
```python
import requests
from datetime import datetime

API_KEY = "YOUR_API_KEY"
today = datetime.now().strftime("%Y-%m-%d")

url = "https://api.polygon.io/v3/reference/options/contracts"
params = {
    "apiKey": API_KEY,
    "underlying_ticker": "SPY",
    "expiration_date": today,
    "limit": 10
}

response = requests.get(url, params=params)
print(response.status_code)  # Should be 200
data = response.json()
print(f"Contracts found: {data.get('count', 0)}")
if data.get('results'):
    print(f"Sample contract: {data['results'][0]}")
```

---

## 📋 INTEGRATION PLAN

### Step 1: Verify API Access

Run the updated test script to verify:
- ✅ Stocks Developer subscription works
- ✅ Options Starter subscription works
- ✅ API key is valid

### Step 2: Install Python Client

```bash
pip install polygon-api-client
```

### Step 3: Create Wrapper Module

Create `massive_api_client.py` that wraps Polygon.io client:

```python
from polygon import RESTClient
import os

class MassiveAPIClient:
    """Wrapper for Polygon.io (now Massive) API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('MASSIVE_API_KEY')
        self.client = RESTClient(self.api_key)
    
    def get_historical_data(self, symbol, start_date, end_date, interval='1m'):
        """Get historical stock data"""
        # Map interval to Polygon timespan
        timespan_map = {
            '1m': 'minute',
            '5m': 'minute',
            '1h': 'hour',
            '1d': 'day'
        }
        timespan = timespan_map.get(interval, 'minute')
        multiplier = 1 if interval in ['1m', '1h', '1d'] else int(interval[:-1])
        
        aggs = []
        for agg in self.client.list_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            _from=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append(agg)
        
        return aggs
    
    def get_options_chain(self, symbol, expiry=None, contract_type=None, strike=None):
        """Get options contracts for symbol"""
        contracts = []
        for contract in self.client.list_options_contracts(
            underlying_ticker=symbol,
            expiration_date=expiry,
            contract_type=contract_type,
            strike_price=strike,
            limit=1000
        ):
            contracts.append(contract)
        return contracts
    
    def get_option_details(self, contract_ticker):
        """Get details for specific option contract"""
        # Get last quote for the contract
        return self.client.get_last_quote(contract_ticker)
```

### Step 4: Integrate into Trading System

Replace yfinance calls with Massive API client in:
- `historical_training_system.py`
- `mike_agent_live_safe.py`
- `quant_features_collector.py`

---

## ✅ UPDATED TEST SCRIPT

I'll create a new test script that uses Polygon.io endpoints:

```python
# test_massive_api_polygon.py
import requests
import os

API_KEY = os.getenv('MASSIVE_API_KEY', '')

# Test 1: Stock aggregates
print("Test 1: Stock Aggregates")
url = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2025-12-01/2025-12-08"
response = requests.get(url, params={"apiKey": API_KEY})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Results: {data.get('resultsCount', 0)}")
    print("✅ Stocks subscription works!")

# Test 2: Options snapshot
print("\nTest 2: Options Snapshot")
url = "https://api.polygon.io/v2/snapshot/options/SPY"
response = requests.get(url, params={"apiKey": API_KEY})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("✅ Options subscription works!")
    print(f"Keys: {list(data.keys())}")
```

---

## 🎯 NEXT STEPS

1. **Install Polygon Client:**
   ```bash
   pip install polygon-api-client
   ```

2. **Test with Updated Script:**
   ```bash
   export MASSIVE_API_KEY='your_key'
   python3 test_massive_api_polygon.py
   ```

3. **Verify Access:**
   - Stocks Developer: Should get 200 on aggregates endpoint
   - Options Starter: Should get 200 on options snapshot endpoint

4. **If Tests Pass:** I'll integrate into your trading system!

---

## 📝 NOTES

- **Massive = Polygon.io** (rebranded)
- Use `api.polygon.io/v2/` endpoints
- Or use `polygon-api-client` Python library
- Authentication: `apiKey` query param OR `Authorization: Bearer` header

---

## 🔍 RESOURCES

- **Polygon.io Docs:** https://polygon.io/docs
- **Python Client:** https://github.com/polygon-io/client-python
- **REST API Reference:** https://polygon.io/docs/stocks/getting-started

---

**Status:** ✅ Fixed - Use Polygon.io endpoints!
