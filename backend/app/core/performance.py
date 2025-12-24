"""Performance optimization utilities"""
import asyncio
from typing import List, Callable, Any, TypeVar
from functools import wraps
import time

T = TypeVar('T')


def parallel_execute(*coros):
    """Execute multiple coroutines in parallel"""
    return asyncio.gather(*coros, return_exceptions=True)


def time_it(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > 0.5:  # Log if takes more than 500ms
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > 0.5:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration: float):
        """Record operation duration"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation"""
        if operation not in self.metrics:
            return {}
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()

