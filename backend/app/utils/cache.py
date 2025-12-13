"""Market data caching utilities"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import pandas as pd

from ..core.config import settings
from .yfinance_helper import create_yf_ticker

logger = logging.getLogger(__name__)

# Import yfinance exceptions for rate limit handling
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    # Fallback if yfinance version doesn't have this exception
    class YFRateLimitError(Exception):
        pass


class MarketDataCache:
    """Cache for market data to reduce API calls"""

    def __init__(self):
        self._market_data_cache = {}
        self._cache_timestamp = None
        self._last_api_call = 0
        self._realtime_cache = {}
        self._realtime_cache_timestamp = None
        self._last_cache_date = None
        self._rate_limit_until = 0  # Timestamp when rate limit expires
        # Start with configured initial backoff
        self._rate_limit_backoff = settings.rate_limit_initial_backoff
    
    def is_rate_limited(self) -> Tuple[bool, float]:
        """
        Check if currently rate limited without making API calls.
        Returns: (is_rate_limited, wait_seconds)
        """
        current_time = time.time()
        if current_time < self._rate_limit_until:
            wait_remaining = self._rate_limit_until - current_time
            return True, wait_remaining
        return False, 0.0

    def get_cached_market_data(self, period: str = "3mo") -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Dict]]:
        """
        Get cached market data or fetch new data if cache is expired.
        Returns: (hist_data, symbol, rate_limit_info)
        rate_limit_info: None if not rate limited, or dict with 'until' timestamp and 'wait_seconds'
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Clear cache if date changed (new day started)
        if self._last_cache_date is not None and self._last_cache_date != today:
            logger.debug(
                f"Date changed from {self._last_cache_date} to {today}, clearing cache")
            self._market_data_cache = {}
            self._cache_timestamp = None
            self._realtime_cache = {}
            self._realtime_cache_timestamp = None
            self._last_cache_date = today

            # Check if we're currently rate limited (before checking cache expiration)
        current_time = time.time()
        if current_time < self._rate_limit_until:
            wait_remaining = self._rate_limit_until - current_time
            if self._market_data_cache:
                logger.debug(f"Rate limited. Returning cached data (wait {wait_remaining:.1f}s for fresh data)")
                return self._market_data_cache.get('hist'), self._market_data_cache.get('symbol'), {
                    'rate_limited': True,
                    'until': self._rate_limit_until,
                    'wait_seconds': int(wait_remaining)
                }
            else:
                # When rate limited with no cache, extend cache timestamp to prevent repeated attempts
                # Set cache timestamp to expire when rate limit expires to prevent cache expiration check from triggering
                rate_limit_expiry = datetime.fromtimestamp(self._rate_limit_until)
                if self._cache_timestamp:
                    # Extend cache to expire when rate limit expires (or normal cache duration, whichever is longer)
                    normal_expiry = self._cache_timestamp + timedelta(seconds=settings.cache_duration)
                    self._cache_timestamp = max(rate_limit_expiry, normal_expiry)
                else:
                    # Set cache timestamp to expire when rate limit expires
                    self._cache_timestamp = rate_limit_expiry
                # Only log once per minute to reduce log spam
                if not hasattr(self, '_last_rate_limit_log') or (current_time - getattr(self, '_last_rate_limit_log', 0)) > 60:
                    logger.info(f"Rate limited with no cached data. Will retry after {wait_remaining:.1f}s")
                    self._last_rate_limit_log = current_time
                return None, None, {
                    'rate_limited': True,
                    'until': self._rate_limit_until,
                    'wait_seconds': int(wait_remaining)
                }

        # Check cache expiration - but skip if we just set cache_timestamp due to rate limit
        cache_expired = (
            self._cache_timestamp is None or
            (now - self._cache_timestamp).total_seconds() > settings.cache_duration or
            not self._market_data_cache
        )
        
        if cache_expired:

            # Rate limiting: wait if we made a call recently (but don't block for too long)
            wait_time = settings.api_cooldown - \
                (current_time - self._last_api_call)
            if wait_time > 0:
                # Only wait if it's a short wait, otherwise return cached data if available
                if wait_time <= 1.0:  # Only wait up to 1 second
                    time.sleep(wait_time)
                else:
                    # If we need to wait too long, return cached data if available
                    if self._market_data_cache:
                        logger.debug("API cooldown active, returning cached data")
                        return self._market_data_cache.get('hist'), self._market_data_cache.get('symbol'), None

            # Try multiple symbols for better reliability (prioritize faster ones)
            # Start with most reliable symbols first to reduce API calls
            symbols_to_try = ["GC=F", "XAUUSD=X",
                              "GOLD"]  # Reduced list for speed
            hist = None
            rate_limited = False

            for symbol in symbols_to_try:
                try:
                    # Add delay between symbol attempts to avoid rapid requests
                    if symbol != symbols_to_try[0]:
                        time.sleep(1)  # 1 second delay between symbols

                    self._last_api_call = time.time()
                    gold = create_yf_ticker(symbol)
                    # Fetch data - yfinance handles timeouts internally
                    # Use shorter period for faster response
                    hist = gold.history(period=period, interval="1d")

                    if hist.empty:
                        logger.warning(f"Empty history data for {symbol}")
                        continue

                    # Validate that we're getting a reasonable gold price
                    current_price = float(hist['Close'].iloc[-1])

                    # Prefer spot gold symbols - accept first valid one to speed up
                    if symbol in ["GC=F", "GOLD", "XAUUSD=X"] and current_price > 1000:
                        logger.debug(
                            f"Using spot gold price from {symbol}: ${current_price:.2f}")
                        self._market_data_cache = {
                            'hist': hist, 'symbol': symbol}
                        self._cache_timestamp = now
                        self._last_cache_date = today
                        # Reset rate limit backoff on successful fetch
                        self._rate_limit_backoff = settings.rate_limit_initial_backoff
                        break  # Found valid data, stop trying other symbols
                except YFRateLimitError as e:
                    rate_limited = True
                    # Exponential backoff: increase wait time each time we're rate limited
                    self._rate_limit_backoff = min(
                        self._rate_limit_backoff * 2, 3600)  # Max 1 hour
                    self._rate_limit_until = current_time + self._rate_limit_backoff
                    logger.warning(
                        f"Rate limited from {symbol}. Backing off for {self._rate_limit_backoff} seconds. Error: {e}")
                    # Don't try other symbols if rate limited - they'll likely fail too
                    break
                except Exception as e:
                    error_name = type(e).__name__
                    # Check if it's a rate limit error by message (for older yfinance versions)
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        rate_limited = True
                        self._rate_limit_backoff = min(
                            self._rate_limit_backoff * 2, 3600)
                        self._rate_limit_until = current_time + self._rate_limit_backoff
                        logger.warning(
                            f"Rate limited from {symbol} (detected by message). Backing off for {self._rate_limit_backoff} seconds. Error: {e}")
                        break
                    logger.warning(
                        f"Error fetching {symbol}: {error_name}: {e}")
                    # Continue to next symbol instead of failing immediately
                    continue

            # If rate limited, return cached data if available
            if rate_limited:
                if self._market_data_cache:
                    wait_remaining = max(0, self._rate_limit_until - current_time)
                    logger.info("Rate limited - returning cached data")
                    return self._market_data_cache.get('hist'), self._market_data_cache.get('symbol'), {
                        'rate_limited': True,
                        'until': self._rate_limit_until,
                        'wait_seconds': int(wait_remaining)
                    }
                # When rate limited with no cache, extend cache timestamp to prevent repeated attempts
                wait_remaining = max(0, self._rate_limit_until - current_time)
                # Extend cache timestamp to prevent repeated fetch attempts until rate limit expires
                rate_limit_expiry = datetime.fromtimestamp(self._rate_limit_until)
                if self._cache_timestamp:
                    # Extend cache to expire when rate limit expires (or normal cache duration, whichever is longer)
                    normal_expiry = self._cache_timestamp + timedelta(seconds=settings.cache_duration)
                    self._cache_timestamp = max(rate_limit_expiry, normal_expiry)
                else:
                    # Set cache timestamp to expire when rate limit expires
                    self._cache_timestamp = rate_limit_expiry
                # Only log once per minute to reduce log spam
                if not hasattr(self, '_last_rate_limit_log') or (current_time - getattr(self, '_last_rate_limit_log', 0)) > 60:
                    logger.info(f"Rate limited and no cached data available. Will retry after {wait_remaining:.1f}s")
                    self._last_rate_limit_log = current_time
                return None, None, {
                    'rate_limited': True,
                    'until': self._rate_limit_until,
                    'wait_seconds': int(wait_remaining)
                }

            if hist is None or hist.empty:
                logger.error(
                    "No market data available from any symbol - returning None")
                return None, None, None

        return self._market_data_cache.get('hist'), self._market_data_cache.get('symbol'), None

    def get_realtime_price_data(self) -> Optional[dict]:
        """Get real-time price data with optimized caching"""
        now = datetime.now()
        
        # Check if we're currently rate limited (before checking cache expiration)
        current_time = time.time()
        if current_time < self._rate_limit_until:
            wait_remaining = self._rate_limit_until - current_time
            if self._realtime_cache:
                logger.debug(f"Rate limited. Returning cached realtime data (wait {wait_remaining:.1f}s for fresh data)")
                return self._realtime_cache
            else:
                # Extend cache timestamp to prevent repeated checks
                rate_limit_expiry = datetime.fromtimestamp(self._rate_limit_until)
                if self._realtime_cache_timestamp:
                    normal_expiry = self._realtime_cache_timestamp + timedelta(seconds=settings.realtime_cache_duration)
                    self._realtime_cache_timestamp = max(rate_limit_expiry, normal_expiry)
                else:
                    self._realtime_cache_timestamp = rate_limit_expiry
                # Only log once per minute to reduce log spam
                if not hasattr(self, '_last_realtime_rate_limit_log') or (current_time - getattr(self, '_last_realtime_rate_limit_log', 0)) > 60:
                    logger.debug(f"Rate limited and no cached realtime data available. Wait {wait_remaining:.1f}s")
                    self._last_realtime_rate_limit_log = current_time
                return None
        
        if (self._realtime_cache_timestamp is None or
            (now - self._realtime_cache_timestamp).total_seconds() > settings.realtime_cache_duration or
                not self._realtime_cache):

            try:
                # Rate limiting (non-blocking for long waits)
                wait_time = settings.api_cooldown - \
                    (current_time - self._last_api_call)
                if wait_time > 0:
                    if wait_time <= 1.0:  # Only wait up to 1 second
                        time.sleep(wait_time)
                    else:
                        # Return cached data if available instead of waiting
                        if self._realtime_cache:
                            logger.debug("API cooldown active, returning cached realtime data")
                            return self._realtime_cache
                        return None

                symbols_to_try = ["GC=F", "XAUUSD=X",
                                  "GOLD"]  # Reduced for speed
                rate_limited = False

                for symbol in symbols_to_try:
                    try:
                        # Add delay between symbol attempts
                        if symbol != symbols_to_try[0]:
                            time.sleep(1)  # 1 second delay between symbols

                        self._last_api_call = time.time()
                        gold = create_yf_ticker(symbol)
                        # Use shorter period for faster response
                        if symbol == "GC=F":
                            hist = gold.history(period="5d", interval="1d")
                        else:
                            # Use daily instead of 1m for speed
                            hist = gold.history(period="1d", interval="1d")

                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                            if len(hist) > 1:
                                prev_close = float(hist['Close'].iloc[-2])
                                price_change = current_price - prev_close
                                change_percentage = (
                                    price_change / prev_close) * 100
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
                            # Reset rate limit backoff on successful fetch
                            self._rate_limit_backoff = settings.rate_limit_initial_backoff
                            return result
                    except YFRateLimitError as e:
                        rate_limited = True
                        # Exponential backoff
                        self._rate_limit_backoff = min(
                            self._rate_limit_backoff * 2, 3600)  # Max 1 hour
                        self._rate_limit_until = current_time + self._rate_limit_backoff
                        logger.warning(
                            f"Rate limited from {symbol} for realtime data. Backing off for {self._rate_limit_backoff} seconds. Error: {e}")
                        break
                    except Exception as e:
                        error_name = type(e).__name__
                        # Check if it's a rate limit error by message
                        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                            rate_limited = True
                            self._rate_limit_backoff = min(
                                self._rate_limit_backoff * 2, 3600)
                            self._rate_limit_until = current_time + self._rate_limit_backoff
                            logger.warning(
                                f"Rate limited from {symbol} for realtime data (detected by message). Backing off for {self._rate_limit_backoff} seconds. Error: {e}")
                            break
                        if symbol == symbols_to_try[-1]:
                            logger.error(
                                f"All symbols failed for real-time data: {error_name}: {e}")
                        continue

                # If rate limited, return cached data if available
                if rate_limited:
                    if self._realtime_cache:
                        logger.info("Rate limited - returning cached realtime data")
                        return self._realtime_cache
                    logger.warning("Rate limited and no cached realtime data available")
                    return None
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
