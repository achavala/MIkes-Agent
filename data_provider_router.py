"""
INSTITUTIONAL DATA PROVIDER ROUTER
Centralized data source priority system with guardrails
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Data provider enumeration"""
    MASSIVE = "massive"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YFINANCE = "yfinance"


# ==================== AUTHORITATIVE PRIORITY ORDER ====================
DATA_PROVIDER_PRIORITY = [
    DataProvider.MASSIVE,    # Priority 1: Options/Greeks/Indices
    DataProvider.ALPACA,     # Priority 2: Broker-aligned validation
    DataProvider.POLYGON,    # Priority 3: Historical bars fallback
    DataProvider.YFINANCE    # Priority 4: Last resort (never silent)
]


# ==================== PROVIDER CAPABILITY MAP ====================
PROVIDER_CAPABILITIES = {
    DataProvider.MASSIVE: {
        "minute_bars": True,
        "options": True,
        "greeks": True,
        "indices": True,
        "iv_data": True,
        "gamma_exposure": True,
        "options_chains": True
    },
    DataProvider.ALPACA: {
        "minute_bars": True,
        "options": False,
        "greeks": False,
        "indices": False,
        "iv_data": False,
        "gamma_exposure": False,
        "options_chains": False
    },
    DataProvider.POLYGON: {
        "minute_bars": True,
        "options": False,
        "greeks": False,
        "indices": True,
        "iv_data": False,
        "gamma_exposure": False,
        "options_chains": False
    },
    DataProvider.YFINANCE: {
        "minute_bars": True,
        "options": False,
        "greeks": False,
        "indices": True,
        "iv_data": False,
        "gamma_exposure": False,
        "options_chains": False
    }
}


# ==================== SYMBOL ROUTING MAP ====================
SYMBOL_MAP = {
    "SPX": {
        DataProvider.MASSIVE: "SPX",
        DataProvider.ALPACA: None,  # Alpaca doesn't support SPX
        DataProvider.POLYGON: "I:SPX",
        DataProvider.YFINANCE: "^GSPC"
    },
    "SPY": {
        DataProvider.MASSIVE: "SPY",
        DataProvider.ALPACA: "SPY",
        DataProvider.POLYGON: "SPY",
        DataProvider.YFINANCE: "SPY"
    },
    "QQQ": {
        DataProvider.MASSIVE: "QQQ",
        DataProvider.ALPACA: "QQQ",
        DataProvider.POLYGON: "QQQ",
        DataProvider.YFINANCE: "QQQ"
    }
}


# ==================== INSTITUTIONAL MODE ====================
INSTITUTIONAL_MODE = True  # Global switch


class DataProviderRouter:
    """
    Centralized data provider router with priority-based fallback
    
    Rules:
    - Massive: Primary for options/Greeks/indices
    - Alpaca: Secondary for broker-aligned validation
    - Polygon: Historical bars fallback
    - yfinance: Last resort (never silent, disabled in institutional mode)
    """
    
    def __init__(self, institutional_mode: bool = True):
        self.institutional_mode = institutional_mode
        self.provider_logs: List[Dict] = []
        self.provider_stats: Dict[str, int] = {
            "massive": 0,
            "alpaca": 0,
            "polygon": 0,
            "yfinance": 0
        }
    
    def can_provide(self, provider: DataProvider, data_type: str) -> bool:
        """Check if provider can satisfy the request"""
        capabilities = PROVIDER_CAPABILITIES.get(provider, {})
        return capabilities.get(data_type, False)
    
    def get_symbol_for_provider(self, symbol: str, provider: DataProvider) -> Optional[str]:
        """Get provider-specific symbol mapping"""
        symbol_info = SYMBOL_MAP.get(symbol, {})
        return symbol_info.get(provider)
    
    def log_provider_usage(
        self,
        symbol: str,
        data_type: str,
        provider_used: DataProvider,
        fallback_count: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log provider usage for analytics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "data_type": data_type,
            "provider_used": provider_used.value,
            "fallback_count": fallback_count,
            "institutional_mode": self.institutional_mode,
            "success": success,
            "error": error
        }
        self.provider_logs.append(log_entry)
        self.provider_stats[provider_used.value] += 1
        
        # Log to console
        if success:
            logger.info(f"Data fetch: {symbol} ({data_type}) from {provider_used.value}")
        else:
            logger.warning(f"Data fetch failed: {symbol} ({data_type}) from {provider_used.value}: {error}")
    
    def fetch_data(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Any], DataProvider, int]:
        """
        Fetch data using priority-based fallback
        
        Returns:
            (data, provider_used, fallback_count)
        """
        fallback_count = 0
        
        for provider in DATA_PROVIDER_PRIORITY:
            # Check if provider can satisfy request
            if not self.can_provide(provider, data_type):
                continue
            
            # Institutional mode: block yfinance
            if self.institutional_mode and provider == DataProvider.YFINANCE:
                logger.warning(f"yfinance blocked in institutional mode for {symbol} ({data_type})")
                continue
            
            # Get provider-specific symbol
            provider_symbol = self.get_symbol_for_provider(symbol, provider)
            if provider_symbol is None:
                continue
            
            # Attempt fetch
            try:
                data = self._fetch_from_provider(
                    provider=provider,
                    symbol=provider_symbol,
                    data_type=data_type,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                
                if data is not None:
                    self.log_provider_usage(
                        symbol=symbol,
                        data_type=data_type,
                        provider_used=provider,
                        fallback_count=fallback_count,
                        success=True
                    )
                    return data, provider, fallback_count
                else:
                    fallback_count += 1
                    self.log_provider_usage(
                        symbol=symbol,
                        data_type=data_type,
                        provider_used=provider,
                        fallback_count=fallback_count,
                        success=False,
                        error="No data returned"
                    )
            except Exception as e:
                fallback_count += 1
                self.log_provider_usage(
                    symbol=symbol,
                    data_type=data_type,
                    provider_used=provider,
                    fallback_count=fallback_count,
                    success=False,
                    error=str(e)
                )
                continue
        
        # All providers failed
        if self.institutional_mode:
            raise RuntimeError(
                f"Institutional mode: All data providers failed for {symbol} ({data_type}). "
                f"Attempted: {[p.value for p in DATA_PROVIDER_PRIORITY if self.can_provide(p, data_type)]}"
            )
        else:
            logger.error(f"All data providers failed for {symbol} ({data_type})")
            return None, DataProvider.YFINANCE, fallback_count
    
    def _fetch_from_provider(
        self,
        provider: DataProvider,
        symbol: str,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        """Fetch data from specific provider"""
        if provider == DataProvider.MASSIVE:
            return self._fetch_massive(symbol, data_type, start_date, end_date, **kwargs)
        elif provider == DataProvider.ALPACA:
            return self._fetch_alpaca(symbol, data_type, start_date, end_date, **kwargs)
        elif provider == DataProvider.POLYGON:
            return self._fetch_polygon(symbol, data_type, start_date, end_date, **kwargs)
        elif provider == DataProvider.YFINANCE:
            return self._fetch_yfinance(symbol, data_type, start_date, end_date, **kwargs)
        return None
    
    def _fetch_massive(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        **kwargs
    ) -> Optional[Any]:
        """Fetch from Massive API"""
        try:
            # TODO: Implement Massive API integration
            # For now, return None to trigger fallback
            logger.debug(f"Massive fetch not yet implemented: {symbol} ({data_type})")
            return None
        except Exception as e:
            logger.error(f"Massive fetch error: {e}")
            return None
    
    def _fetch_alpaca(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        **kwargs
    ) -> Optional[Any]:
        """Fetch from Alpaca API"""
        try:
            import alpaca_trade_api as tradeapi
            import config
            
            api = tradeapi.REST(
                config.ALPACA_KEY,
                config.ALPACA_SECRET,
                config.ALPACA_BASE_URL,
                api_version='v2'
            )
            
            if data_type == "minute_bars":
                # Fetch minute bars
                bars = api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start_date,
                    end=end_date
                ).df
                return bars
            else:
                return None
        except ImportError:
            logger.warning("Alpaca API not available")
            return None
        except Exception as e:
            logger.error(f"Alpaca fetch error: {e}")
            return None
    
    def _fetch_polygon(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        **kwargs
    ) -> Optional[Any]:
        """Fetch from Polygon API"""
        try:
            # TODO: Implement Polygon API integration
            logger.debug(f"Polygon fetch not yet implemented: {symbol} ({data_type})")
            return None
        except Exception as e:
            logger.error(f"Polygon fetch error: {e}")
            return None
    
    def _fetch_yfinance(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        **kwargs
    ) -> Optional[Any]:
        """Fetch from yfinance (last resort)"""
        try:
            import yfinance as yf
            
            if data_type == "minute_bars":
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1m')
                if len(data) > 0:
                    return data
            return None
        except ImportError:
            logger.warning("yfinance not available")
            return None
        except Exception as e:
            logger.error(f"yfinance fetch error: {e}")
            return None
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider usage statistics"""
        total = sum(self.provider_stats.values())
        return {
            "total_fetches": total,
            "by_provider": self.provider_stats,
            "percentages": {
                provider: (count / total * 100) if total > 0 else 0
                for provider, count in self.provider_stats.items()
            },
            "yfinance_usage": self.provider_stats.get("yfinance", 0),
            "yfinance_red_flag": self.provider_stats.get("yfinance", 0) > 0 and self.institutional_mode
        }
    
    def get_provider_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        provider: Optional[str] = None
    ) -> List[Dict]:
        """Get provider usage logs with filters"""
        logs = self.provider_logs
        
        if start_date:
            logs = [log for log in logs if log.get("timestamp", "") >= start_date]
        if end_date:
            logs = [log for log in logs if log.get("timestamp", "") <= end_date]
        if provider:
            logs = [log for log in logs if log.get("provider_used") == provider]
        
        return logs


# Global instance
_data_router: Optional[DataProviderRouter] = None


def initialize_data_router(institutional_mode: bool = True) -> DataProviderRouter:
    """Initialize global data router"""
    global _data_router
    _data_router = DataProviderRouter(institutional_mode=institutional_mode)
    return _data_router


def get_data_router() -> Optional[DataProviderRouter]:
    """Get global data router instance"""
    return _data_router





