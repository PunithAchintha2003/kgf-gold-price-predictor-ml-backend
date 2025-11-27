"""Health check routes"""
from fastapi import APIRouter
from datetime import datetime
from ....core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "XAU/USD Real-time Data API",
        "version": "1.0.0",
        "environment": settings.environment,
        "log_level": settings.log_level,
        "cache_duration": settings.cache_duration,
        "api_cooldown": settings.api_cooldown
    }


@router.head("/health")
async def health_check_head():
    """Handle HEAD requests to health endpoint"""
    from fastapi.responses import Response
    return Response(status_code=200)


