"""Enhanced health check with detailed system status"""
import logging
from typing import Dict, Optional
from datetime import datetime
import time
from functools import lru_cache

from .database import get_db_type, get_db_connection
from .config import settings

logger = logging.getLogger(__name__)

# Cache health check results for 5 seconds to avoid slow queries
_health_cache = {"timestamp": 0, "data": None}
_CACHE_DURATION = 5  # seconds


def check_database_health() -> Dict:
    """Check database connection health"""
    try:
        start_time = time.time()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            response_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "status": "healthy",
            "type": get_db_type(),
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "type": get_db_type(),
            "error": str(e)
        }


def check_disk_space() -> Dict:
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(settings.data_dir)
        free_percent = (free / total) * 100

        return {
            "status": "healthy" if free_percent > 10 else "warning",
            "free_gb": round(free / (1024**3), 2),
            "total_gb": round(total / (1024**3), 2),
            "free_percent": round(free_percent, 2),
            "used_gb": round(used / (1024**3), 2)
        }
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return {
            "status": "unknown",
            "error": str(e)
        }


def get_health_status() -> Dict:
    """Get comprehensive health status with caching"""
    global _health_cache

    # Use cached result if available and fresh
    current_time = time.time()
    if (_health_cache["data"] is not None and
            current_time - _health_cache["timestamp"] < _CACHE_DURATION):
        return _health_cache["data"]

    # Perform health checks
    db_health = check_database_health()
    disk_health = check_disk_space()

    # Determine overall status
    overall_status = "healthy"
    if db_health["status"] != "healthy":
        overall_status = "unhealthy"
    elif disk_health["status"] == "warning":
        overall_status = "degraded"

    result = {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "service": "XAU/USD Real-time Data API",
        "version": "1.0.0",
        "environment": settings.environment,
        "checks": {
            "database": db_health,
            "disk": disk_health
        },
        "configuration": {
            "log_level": settings.log_level,
            "cache_duration": settings.cache_duration,
            "api_cooldown": settings.api_cooldown,
            "database_type": get_db_type()
        }
    }

    # Cache the result
    _health_cache["timestamp"] = current_time
    _health_cache["data"] = result

    return result
