"""API v1 routes"""
from fastapi import APIRouter
from .routes import health, xauusd, exchange

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["health"])
api_router.include_router(xauusd.router, prefix="/xauusd", tags=["xauusd"])
api_router.include_router(
    exchange.router, prefix="/exchange-rate", tags=["exchange"])


