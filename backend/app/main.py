"""
Main application entry point - Refactored to use modular structure
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List
from pathlib import Path
import sys

# Core imports
from .core.config import settings
from .core.logging_config import setup_logging, get_logger
from .core.database import (
    get_db_connection,
    get_db_type,
    get_date_function,
    init_database,
    init_backup_database,
    init_postgresql_pool
)

# Services
from .services.prediction_service import PredictionService
from .services.market_data_service import MarketDataService
from .services.exchange_service import ExchangeService

# Repositories
from .repositories.prediction_repository import PredictionRepository

# Utils
from .utils.cache import market_data_cache

# ML Models - import from parent backend directory
# Add backend directory to path before importing models
BACKEND_PARENT = Path(__file__).resolve().parent.parent
if str(BACKEND_PARENT) not in sys.path:
    sys.path.insert(0, str(BACKEND_PARENT))

# Import models after path is set
from models.news_prediction import NewsEnhancedLassoPredictor
from models.lasso_model import LassoGoldPredictor

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize ML models
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Initialize Lasso Regression predictor
lasso_predictor = LassoGoldPredictor()
try:
    lasso_predictor.load_model(
        str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
    logger.info("Lasso Regression model loaded successfully")
except Exception as e:
    logger.warning(f"Lasso Regression model not found: {e}")
    lasso_predictor = None

# Initialize News-Enhanced Lasso predictor
news_enhanced_predictor = NewsEnhancedLassoPredictor()
enhanced_model_path = BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'

if enhanced_model_path.exists():
    try:
        news_enhanced_predictor.load_enhanced_model(str(enhanced_model_path))
        logger.info("News-enhanced Lasso model loaded successfully")
    except Exception as e:
        logger.warning(f"News-enhanced model found but failed to load: {e}")
        news_enhanced_predictor = None
else:
    logger.info(
        "News-enhanced Lasso model not found - using regular Lasso model")

# Initialize services
prediction_service = PredictionService(
    lasso_predictor=lasso_predictor,
    news_enhanced_predictor=news_enhanced_predictor
)

market_data_service = MarketDataService(prediction_service=prediction_service)
exchange_service = ExchangeService()
prediction_repo = PredictionRepository()

# Initialize database
if settings.use_postgresql:
    if init_postgresql_pool():
        logger.info("✅ PostgreSQL enabled - using PostgreSQL database")
    else:
        logger.warning(
            "⚠️  PostgreSQL initialization failed - falling back to SQLite")
        settings.use_postgresql = False

init_database()
init_backup_database()

# Create FastAPI app
app = FastAPI(
    title="XAU/USD Real-time Data API",
    version="1.0.0",
    description=f"Gold price prediction API running in {settings.environment} environment"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")


manager = ConnectionManager()


# API Endpoints
@app.get("/health")
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


@app.head("/health")
async def health_check_head():
    """Handle HEAD requests to health endpoint"""
    return Response(status_code=200)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "XAU/USD Real-time Data API with News Sentiment Analysis",
        "status": "running"
    }


@app.head("/")
async def root_head():
    """Handle HEAD requests to root endpoint"""
    return Response(status_code=200)


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return Response(status_code=204)


@app.get("/xauusd")
async def get_daily_data(days: int = 90):
    """Get XAU/USD daily data"""
    return market_data_service.get_daily_data(days=days)


@app.get("/xauusd/realtime")
async def get_realtime_price():
    """Get real-time XAU/USD price"""
    return market_data_service.get_realtime_price()


@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def get_exchange_rate(from_currency: str, to_currency: str):
    """Get exchange rate between currencies"""
    return exchange_service.get_exchange_rate(from_currency, to_currency)


@app.websocket("/ws/xauusd")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates"""
    await manager.connect(websocket)
    try:
        last_sent_data = None
        while True:
            daily_data = market_data_service.get_daily_data()
            if daily_data != last_sent_data:
                await manager.send_personal_message(json.dumps(daily_data), websocket)
                last_sent_data = daily_data
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Background tasks
async def broadcast_daily_data():
    """Background task to broadcast daily data to all connected clients"""
    last_broadcast_data = None
    while True:
        if manager.active_connections:
            daily_data = market_data_service.get_daily_data()
            if daily_data != last_broadcast_data:
                await manager.broadcast(json.dumps(daily_data))
                last_broadcast_data = daily_data
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"Starting application in {settings.environment} environment")
    asyncio.create_task(broadcast_daily_data())
    logger.info("✅ All background tasks started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutdown")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
