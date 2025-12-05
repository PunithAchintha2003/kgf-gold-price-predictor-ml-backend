"""
Main application entry point - Optimized and modular structure
"""
import warnings
# Suppress SyntaxWarnings from third-party libraries (e.g., textblob)
# These warnings appear during import and don't affect functionality
warnings.filterwarnings('ignore', category=SyntaxWarning, module='textblob')

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from .core.logging_config import setup_logging, get_logger
from .core.config import settings
from .core.database import (
    init_database,
    init_backup_database,
    init_postgresql_pool
)
from .core.models import initialize_models
from .core.websocket import ConnectionManager
from .core.task_manager import BackgroundTaskManager
from .core.background_tasks import (
    broadcast_daily_data,
    auto_update_pending_predictions
)
from .core.dependencies import set_services
from .repositories.prediction_repository import PredictionRepository
from .services.exchange_service import ExchangeService
from .services.market_data_service import MarketDataService
from .services.prediction_service import PredictionService
from .api.v1 import api_router
from .api.v1.routes.health import set_task_manager

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize ML models
lasso_predictor, news_enhanced_predictor = initialize_models()

# Initialize services
prediction_service = PredictionService(
    lasso_predictor=lasso_predictor,
    news_enhanced_predictor=news_enhanced_predictor
)
market_data_service = MarketDataService(prediction_service=prediction_service)
exchange_service = ExchangeService()
prediction_repo = PredictionRepository()

# Set services for dependency injection
set_services(
    market_data_service=market_data_service,
    prediction_service=prediction_service,
    prediction_repo=prediction_repo,
    exchange_service=exchange_service
)

# Initialize database
if settings.use_postgresql:
    if init_postgresql_pool():
        logger.debug("PostgreSQL database connected")
    else:
        logger.warning("PostgreSQL failed - using SQLite")
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
# Note: If allow_origins=["*"], allow_credentials must be False
# For production, set specific origins in CORS_ORIGINS environment variable
cors_allow_credentials = "*" not in settings.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Initialize WebSocket connection manager
manager = ConnectionManager()

# Legacy health endpoint for backward compatibility


@app.get("/health")
async def health_check_legacy():
    """Health check endpoint (legacy - also available at /api/v1/health)"""
    from .api.v1.routes.health import health_check
    return await health_check()


@app.head("/health")
async def health_check_head_legacy():
    """Handle HEAD requests to health endpoint"""
    return Response(status_code=200)

# Initialize background task manager
task_manager = BackgroundTaskManager()
set_task_manager(task_manager)  # Set for health check endpoint


# Root endpoints
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


# WebSocket endpoint
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


# Legacy endpoints (for backward compatibility)
# These redirect to the new API routes
@app.get("/xauusd")
async def get_daily_data_legacy(days: int = 90):
    """Get XAU/USD daily data (legacy endpoint)"""
    return market_data_service.get_daily_data(days=days)


@app.get("/xauusd/realtime")
async def get_realtime_price_legacy():
    """Get real-time XAU/USD price (legacy endpoint)"""
    return market_data_service.get_realtime_price()


@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def get_exchange_rate_legacy(from_currency: str, to_currency: str):
    """Get exchange rate between currencies (legacy endpoint)"""
    return exchange_service.get_exchange_rate(from_currency, to_currency)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"🚀 Server starting in {settings.environment} mode")

    # Start background tasks with proper registration
    broadcast_task = asyncio.create_task(
        broadcast_daily_data(manager, market_data_service, task_manager)
    )
    task_manager.register_task("broadcast_daily_data", broadcast_task)

    if settings.auto_update_enabled:
        update_task = asyncio.create_task(
            auto_update_pending_predictions(
                market_data_service, prediction_repo, task_manager
            )
        )
        task_manager.register_task(
            "auto_update_pending_predictions", update_task
        )
        logger.info(
            f"🔄 Auto-update task started: pending predictions will be updated every {settings.auto_update_interval}s")
    else:
        logger.info("🔄 Auto-update task is disabled via configuration")

    logger.info("✅ Ready - API available at http://localhost:8001")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Server shutting down")
    await task_manager.shutdown()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
