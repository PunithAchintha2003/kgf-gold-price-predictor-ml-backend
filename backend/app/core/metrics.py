"""Prometheus-style metrics collection"""
import time
import logging
from typing import Dict
from collections import defaultdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Simple metrics collector for application monitoring"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._request_count = defaultdict(int)
        self._request_duration = defaultdict(list)
        self._error_count = defaultdict(int)
        self._start_time = time.time()
    
    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record a request metric"""
        with self._lock:
            key = f"{method} {path}"
            self._request_count[key] += 1
            self._request_duration[key].append(duration)
            
            # Keep only last 1000 durations per endpoint
            if len(self._request_duration[key]) > 1000:
                self._request_duration[key] = self._request_duration[key][-1000:]
            
            if status_code >= 400:
                self._error_count[key] += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Calculate average durations
            avg_durations = {}
            for key, durations in self._request_duration.items():
                if durations:
                    avg_durations[key] = sum(durations) / len(durations)
            
            return {
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime),
                "request_counts": dict(self._request_count),
                "average_durations": avg_durations,
                "error_counts": dict(self._error_count),
                "total_requests": sum(self._request_count.values()),
                "total_errors": sum(self._error_count.values()),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human-readable string"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def reset(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self._request_count.clear()
            self._request_duration.clear()
            self._error_count.clear()
            self._start_time = time.time()


# Global metrics instance
metrics_collector = MetricsCollector()

