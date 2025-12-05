"""Health check routes"""
from fastapi import APIRouter, Depends
from datetime import datetime
from typing import Optional
from ....core.config import settings

router = APIRouter()

# Global task manager - will be set during app initialization
_task_manager: Optional[object] = None


def set_task_manager(task_manager):
    """Set global task manager instance"""
    global _task_manager
    _task_manager = task_manager


def get_task_manager():
    """Get task manager instance"""
    return _task_manager


@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint (Industry Standard)
    Includes service status, configuration, and background task health
    """
    # Get task health status if available
    tasks_health = {}
    unhealthy_tasks = []
    
    if _task_manager:
        tasks_health = _task_manager.get_all_tasks_health()
        unhealthy_tasks = [
            name for name, health in tasks_health.items()
            if health.get("status") == "error"
        ]

    # Determine overall health status
    overall_status = "healthy"
    if unhealthy_tasks:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "service": "XAU/USD Real-time Data API",
        "version": "1.0.0",
        "environment": settings.environment,
        "log_level": settings.log_level,
        "cache_duration": settings.cache_duration,
        "api_cooldown": settings.api_cooldown,
        "background_tasks": {
            "auto_update_enabled": settings.auto_update_enabled,
            "auto_update_interval": settings.auto_update_interval,
            "tasks": tasks_health
        },
        "unhealthy_tasks": unhealthy_tasks if unhealthy_tasks else None
    }


@router.head("/health")
async def health_check_head():
    """Handle HEAD requests to health endpoint"""
    from fastapi.responses import Response
    return Response(status_code=200)
