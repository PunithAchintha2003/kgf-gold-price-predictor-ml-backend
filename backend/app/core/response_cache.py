"""Response caching for API endpoints to improve performance"""
import time
import hashlib
import json
from typing import Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

# Lazy logger initialization to avoid circular dependency
def _get_logger():
    """Get logger instance (lazy initialization)"""
    try:
        from .logging_config import get_logger
        return get_logger(__name__)
    except Exception:
        import logging
        return logging.getLogger(__name__)


class ResponseCache:
    """Simple in-memory response cache with TTL"""
    
    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, ttl: int = 60) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                self._hits += 1
                # Log cache hit in debug mode
                try:
                    logger = _get_logger()
                    from .logging_config import log_cache_operation
                    log_cache_operation(logger, key, hit=True)
                except Exception:
                    pass  # Ignore logging errors
                return value
            else:
                # Expired, remove it
                del self._cache[key]
        
        self._misses += 1
        # Log cache miss in debug mode
        try:
            logger = _get_logger()
            from .logging_config import log_cache_operation
            log_cache_operation(logger, key, hit=False)
        except Exception:
            pass  # Ignore logging errors
        return None
    
    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        """Set cached value with TTL"""
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
        
        # Clean up expired entries periodically (every 100 operations)
        if len(self._cache) > 1000:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove expired entries - optimized"""
        current_time = time.time()
        # Use list comprehension for faster cleanup
        expired_keys = [key for key, (_, expiry) in self._cache.items() if current_time >= expiry]
        for key in expired_keys:
            try:
                del self._cache[key]
            except KeyError:
                pass  # Already deleted
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self._cache)
        }


# Global cache instance
response_cache = ResponseCache()


def cache_response(ttl: int = 60, key_func: Optional[Callable] = None):
    """
    Decorator to cache API responses
    
    Args:
        ttl: Time to live in seconds (default: 60)
        key_func: Optional function to generate cache key from request
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name + args + kwargs
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
                }
                key_str = json.dumps(key_data, sort_keys=True)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Check cache
            cached = response_cache.get(cache_key, ttl=ttl)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            response_cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_key_from_params(*param_names: str):
    """Generate cache key from specific function parameters"""
    def key_func(*args, **kwargs):
        key_parts = []
        # Get values from kwargs
        for name in param_names:
            if name in kwargs:
                key_parts.append(f"{name}:{kwargs[name]}")
        # If no kwargs, try args (less reliable but better than nothing)
        if not key_parts and args:
            key_parts.append(f"args:{str(args)}")
        
        key_str = "|".join(key_parts) if key_parts else "default"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    return key_func

