"""API v1 routes"""
from fastapi import APIRouter
from .routes import health, xauusd, exchange

# Import spot trading routes
import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent  # backend/app/api/v1 -> backend
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from spot_trade.routes import router as spot_trade_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["health"])
api_router.include_router(
    xauusd.router, 
    prefix="/xauusd", 
    tags=["xauusd"]
)
api_router.include_router(
    exchange.router, 
    prefix="/exchange-rate", 
    tags=["exchange"]
)
api_router.include_router(
    spot_trade_router,
    prefix="/spot-trade",
    tags=["spot-trade"]
)
