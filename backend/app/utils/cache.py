"""Market data caching utilities"""
import time
import logging
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd

from ..core.config import settings
from .yfinance_helper import create_yf_ticker

logger = logging.getLogger(__name__)


class MarketDataCache:
    """Cache for market data to reduce API calls"""
    
    def __init__(self):
        self._market_data_cache = {}
        self._cache_timestamp = None
        self._last_api_call = 0
        self._realtime_cache = {}
        self._realtime_cache_timestamp = None
        self._last_cache_date = None
    
    def get_cached_market_data(self, period: str = "3mo") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get cached market data or fetch new data if cache is expired"""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        # Clear cache if date changed (new day started)
        if self._last_cache_date is not None and self._last_cache_date != today:
            logger.info(f"Date changed from {self._last_cache_date} to {today}, clearing cache")
            self._market_data_cache = {}
            self._cache_timestamp = None
            self._realtime_cache = {}
            self._realtime_cache_timestamp = None
            self._last_cache_date = today
        
        if (self._cache_timestamp is None or
            (now - self._cache_timestamp).total_seconds() > settings.cache_duration or
            not self._market_data_cache):
            
            # Rate limiting: wait if we made a call recently
            current_time = time.time()
            if current_time - self._last_api_call < settings.api_cooldown:
                time.sleep(settings.api_cooldown - (current_time - self._last_api_call))
            
            # Try multiple symbols for better reliability
            symbols_to_try = ["GC=F", "GOLD", "XAUUSD=X", "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
            hist = None
            
            for symbol in symbols_to_try:
                try:
                    self._last_api_call = time.time()
                    gold = create_yf_ticker(symbol)
                    hist = gold.history(period=period, interval="1d")
                    
                    if hist.empty:
                        logger.warning(f"Empty history data for {symbol}")
                        continue
                    
                    # Validate that we're getting a reasonable gold price
                    current_price = float(hist['Close'].iloc[-1])
                    
                    # Skip ETFs if they're giving prices too low for spot gold
                    etf_symbols = ["GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
                    if symbol in etf_symbols and current_price < 1000:
                        logger.warning(f"Skipping {symbol} ETF price: ${current_price:.2f} - too low for spot gold")
                        continue
                    
                    # Prefer spot gold symbols
                    if symbol in ["GC=F", "GOLD", "XAUUSD=X"] and current_price > 1000:
                        logger.info(f"Using spot gold price from {symbol}: ${current_price:.2f}")
                        self._market_data_cache = {'hist': hist, 'symbol': symbol}
                        self._cache_timestamp = now
                        self._last_cache_date = today
                        break
                    elif symbol in etf_symbols and current_price > 1000:
                        logger.info(f"Using {symbol} price: ${current_price:.2f}")
                        self._market_data_cache = {'hist': hist, 'symbol': symbol}
                        self._cache_timestamp = now
                        self._last_cache_date = today
                        break
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {type(e).__name__}: {e}")
                    if symbol == symbols_to_try[-1]:
                        logger.error(f"All {len(symbols_to_try)} gold data sources failed")
                    continue
            
            if hist is None or hist.empty:
                logger.error("No market data available from any symbol - returning None")
                return None, None
        
        return self._market_data_cache.get('hist'), self._market_data_cache.get('symbol')
    
    def get_realtime_price_data(self) -> Optional[dict]:
        """Get real-time price data with optimized caching"""
        now = datetime.now()
        if (self._realtime_cache_timestamp is None or
            (now - self._realtime_cache_timestamp).total_seconds() > settings.realtime_cache_duration or
            not self._realtime_cache):
            
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self._last_api_call < settings.api_cooldown:
                    time.sleep(settings.api_cooldown - (current_time - self._last_api_call))
                
                symbols_to_try = ["GC=F", "GOLD", "XAUUSD=X", "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
                
                for symbol in symbols_to_try:
                    try:
                        self._last_api_call = time.time()
                        gold = create_yf_ticker(symbol)
                        if symbol == "GC=F":
                            hist = gold.history(period="5d", interval="1d")
                        else:
                            hist = gold.history(period="1d", interval="1d")  # Use daily instead of 1m for speed
                        
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                            if len(hist) > 1:
                                prev_close = float(hist['Close'].iloc[-2])
                                price_change = current_price - prev_close
                                change_percentage = (price_change / prev_close) * 100
                            else:
                                price_change = 0
                                change_percentage = 0
                            
                            result = {
                                'current_price': round(current_price, 2),
                                'price_change': round(price_change, 2),
                                'change_percentage': round(change_percentage, 2),
                                'last_updated': hist.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': symbol,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            self._realtime_cache = result
                            self._realtime_cache_timestamp = now
                            return result
                    except Exception as e:
                        if symbol == symbols_to_try[-1]:
                            logger.error(f"All symbols failed for real-time data: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error getting real-time price: {e}")
        
        return self._realtime_cache if self._realtime_cache else None
    
    def clear_cache(self):
        """Clear all caches"""
        self._market_data_cache = {}
        self._cache_timestamp = None
        self._realtime_cache = {}
        self._realtime_cache_timestamp = None


# Global cache instance
market_data_cache = MarketDataCache()


