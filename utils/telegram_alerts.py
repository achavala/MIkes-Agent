"""
🔔 TELEGRAM TRADE ALERTS
Production-safe Telegram notifications for Mike Agent

Features:
- Entry alerts
- Exit alerts (TP/SL)
- Error alerts
- Safeguard blocks
- Daily summaries (optional)
- Rate limiting (prevents spam)

Safety:
- Never blocks trading
- Fire-and-forget
- Network failures ignored
- No secrets in code (uses env vars)
- Rate limiting prevents alert spam
"""

import os
import time
import requests
from datetime import datetime
from typing import Optional

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Rate limiting: prevent spam (last sent time per alert key)
_LAST_SENT: dict[str, float] = {}

# Rate limit windows (seconds)
RATE_LIMITS = {
    "ENTRY": 30,       # 30 seconds between entry alerts for same symbol (reduced from 5 min)
    "EXIT": 30,        # 30 seconds between exit alerts for same symbol (reduced from 1 min)
    "BLOCK": 300,      # 5 minutes between block alerts for same symbol (reduced from 10 min)
    "ERROR": 300,      # 5 minutes between error alerts
    "INFO": 60,        # 1 minute between info alerts
    "PNL": 3600,       # 1 hour between PnL summaries
}

# Emoji mapping for alert levels
EMOJI_MAP = {
    "ENTRY": "🟢",
    "EXIT": "🔴",
    "ERROR": "🚨",
    "INFO": "ℹ️",
    "BLOCK": "⛔",
    "PNL": "📊",
    "WARNING": "⚠️",
    "SUCCESS": "✅"
}


def _should_send(key: str, level: str = "INFO") -> bool:
    """
    Rate limiting: prevent spam by checking if enough time has passed
    since last alert of this type.
    
    Args:
        key: Unique key for this alert (e.g., "ENTRY_SPY", "ERROR_main_loop")
        level: Alert level (ENTRY, EXIT, etc.)
    
    Returns:
        True if should send, False if rate limited
    """
    now = time.time()
    limit_seconds = RATE_LIMITS.get(level, 60)  # Default 1 minute
    
    if key not in _LAST_SENT:
        _LAST_SENT[key] = now
        return True
    
    time_since_last = now - _LAST_SENT[key]
    if time_since_last >= limit_seconds:
        _LAST_SENT[key] = now
        return True
    
    # Rate limited - don't send
    return False


def send_telegram(message: str, level: str = "INFO", rate_limit_key: Optional[str] = None) -> bool:
    """
    Send Telegram alert (fire-and-forget, never blocks trading)
    
    Args:
        message: Alert message text
        level: Alert level (ENTRY, EXIT, ERROR, INFO, BLOCK, PNL, WARNING, SUCCESS)
        rate_limit_key: Optional unique key for rate limiting (e.g., "ENTRY_SPY")
                        If None, uses level as key
    
    Returns:
        True if sent successfully, False otherwise (never raises)
    """
    # Skip if not configured
    if not BOT_TOKEN or not CHAT_ID:
        return False
    
    # Rate limiting: check if we should send this alert
    key = rate_limit_key if rate_limit_key else level
    if not _should_send(key, level):
        print(f"⚠️ Telegram alert rate limited (level: {level}, key: {key}) - not sending")
        return False  # Rate limited - log it
    
    # Get emoji for level
    emoji = EMOJI_MAP.get(level, "📩")
    
    # Format message with timestamp
    text = f"""
{emoji} *MIKE AGENT ALERT*

{message}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            },
            timeout=5
        )
        response.raise_for_status()
        print(f"✅ Telegram alert sent (level: {level}, key: {key})")
        return True
    except requests.exceptions.HTTPError as e:
        # Log HTTP errors for debugging
        error_detail = response.text if hasattr(e, 'response') and e.response else str(e)
        print(f"❌ Telegram HTTP error (level: {level}, key: {key}): {e}")
        print(f"   Status Code: {e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'}")
        print(f"   Response: {error_detail}")
        return False
    except requests.exceptions.ConnectionError as conn_err:
        print(f"❌ Telegram Connection error (level: {level}, key: {key}): {conn_err}")
        return False
    except requests.exceptions.Timeout as timeout_err:
        print(f"❌ Telegram Timeout error (level: {level}, key: {key}): {timeout_err}")
        return False
    except Exception as e:
        print(f"❌ Telegram Unknown error (level: {level}, key: {key}): {e}")
        return False


def send_entry_alert(
    symbol: str,
    side: str,
    strike: float,
    expiry: str,
    fill_price: float,
    qty: int,
    confidence: Optional[float] = None,
    action_source: Optional[str] = None
) -> bool:
    """Send trade entry alert (rate limited: 30 seconds per symbol)"""
    print(f"📤 Attempting to send entry alert for {symbol} ({side})...")
    message = f"""
ENTERED {symbol}
Type: {side.upper()}
Strike: ${strike:.2f}
Expiry: {expiry}
Price: ${fill_price:.2f}
Size: {qty} contracts
"""
    if confidence is not None:
        message += f"Confidence: {confidence:.2%}\n"
    if action_source:
        message += f"Source: {action_source}\n"
    
    # Rate limit by symbol to prevent spam
    rate_key = f"ENTRY_{symbol}"
    result = send_telegram(message.strip(), level="ENTRY", rate_limit_key=rate_key)
    if result:
        print(f"✅ Entry alert sent successfully for {symbol}")
    else:
        print(f"❌ Entry alert failed for {symbol} (check rate limiting or API error above)")
    return result


def send_exit_alert(
    symbol: str,
    exit_reason: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    qty: int,
    pnl_dollar: Optional[float] = None
) -> bool:
    """Send trade exit alert (rate limited: 1 min per symbol)"""
    message = f"""
EXITED {symbol}
Reason: {exit_reason}
Entry: ${entry_price:.2f}
Exit: ${exit_price:.2f}
PnL: {pnl_pct:+.2f}%
Size: {qty} contracts
"""
    if pnl_dollar is not None:
        message += f"PnL: ${pnl_dollar:+.2f}\n"
    
    # Rate limit by symbol to prevent spam
    rate_key = f"EXIT_{symbol}"
    return send_telegram(message.strip(), level="EXIT", rate_limit_key=rate_key)


def send_block_alert(symbol: str, block_reason: str) -> bool:
    """Send safeguard block alert (rate limited: 10 min per symbol)"""
    message = f"""
TRADE BLOCKED
Symbol: {symbol}
Reason: {block_reason}
"""
    # Rate limit by symbol to prevent spam
    rate_key = f"BLOCK_{symbol}"
    return send_telegram(message.strip(), level="BLOCK", rate_limit_key=rate_key)


def send_error_alert(error_message: str, context: Optional[str] = None) -> bool:
    """Send critical error alert (rate limited: 5 min per context)"""
    message = f"""
CRITICAL ERROR
{error_message}
"""
    if context:
        message += f"\nContext: {context}"
    
    # Rate limit by context to prevent spam
    rate_key = f"ERROR_{context}" if context else "ERROR"
    return send_telegram(message.strip(), level="ERROR", rate_limit_key=rate_key)


def send_daily_summary(
    num_trades: int,
    win_rate: float,
    day_pnl: float,
    max_dd: float,
    total_pnl: Optional[float] = None
) -> bool:
    """Send daily summary at market close (rate limited: 1 hour)"""
    message = f"""
DAILY SUMMARY
Trades: {num_trades}
Win Rate: {win_rate:.1f}%
PnL: {day_pnl:+.2f}%
Max DD: {max_dd:.2f}%
"""
    if total_pnl is not None:
        message += f"Total PnL: ${total_pnl:+.2f}\n"
    
    # Rate limit to once per hour
    rate_key = "PNL_DAILY"
    return send_telegram(message.strip(), level="PNL", rate_limit_key=rate_key)


def send_info(message: str) -> bool:
    """Send informational alert"""
    return send_telegram(message, level="INFO")


def send_warning(message: str) -> bool:
    """Send warning alert"""
    return send_telegram(message, level="WARNING")


def is_configured() -> bool:
    """Check if Telegram alerts are configured"""
    return bool(BOT_TOKEN and CHAT_ID)


def test_telegram_alert() -> bool:
    """
    Test function to verify Telegram alerts are working.
    Call this once after setting up secrets to confirm everything works.
    
    Returns:
        True if test message sent successfully, False otherwise
    """
    if not is_configured():
        print("❌ Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    
    print("📤 Sending test Telegram alert...")
    print(f"   Bot Token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:] if len(BOT_TOKEN) > 15 else '***'}")
    print(f"   Chat ID: {CHAT_ID}")
    
    # Test API directly with better error reporting
    try:
        import requests
        test_message = "Telegram alert test successful 🚀\n\nIf you received this, your alerts are working correctly!"
        
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": test_message,
                "parse_mode": "Markdown"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Test alert sent! Check your Telegram.")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

