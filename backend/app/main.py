"""
Main application entry point - Optimized with industry best practices
- Clean separation of concerns
- Comprehensive error handling
- Async database support
- Proper lifecycle management
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncio
import json
import warnings

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .api.v1.routes.health import set_task_manager
from .api.v1 import api_router
from .services.prediction_service import PredictionService
from .services.market_data_service import MarketDataService
from .services.exchange_service import ExchangeService
from .repositories.prediction_repository import PredictionRepository
from .core.dependencies import set_services
from .core.background_tasks import (
    broadcast_daily_data,
    auto_update_pending_predictions,
    auto_retrain_model,
    auto_generate_daily_prediction,
    auto_retrain_and_predict
)
from .core.task_manager import BackgroundTaskManager
from .core.websocket import ConnectionManager
from .core.models import initialize_models
from .core.database import (
    init_database,
    init_backup_database,
    init_postgresql_pool
)
from .core.database_async import (
    init_postgresql_pool_async,
    close_postgresql_pool_async,
    init_database_async
)
from .core.config import settings, get_settings
from .core.logging_config import setup_logging, get_logger
from .core.exceptions import (
    BaseAPIException,
    base_api_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from fastapi.exceptions import RequestValidationError
from .core.middleware import (
    SecurityHeadersMiddleware,
    CompressionMiddleware,
    TimingMiddleware,
    RequestSizeLimitMiddleware
)

# Suppress SyntaxWarnings from third-party libraries
warnings.filterwarnings('ignore', category=SyntaxWarning, module='textblob')

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager with comprehensive startup/shutdown

    Handles:
    - Database initialization (sync and async)
    - Service initialization
    - Background task startup
    - Graceful shutdown
    """
    # Startup
    try:
        logger.debug(f"🚀 Server starting in {settings.environment} mode")

        # Store settings in app state for access in exception handlers
        app.state.settings = settings

        # Add backend directory to sys.path for spot_trade imports (once at the start)
        import sys
        from pathlib import Path
        # backend/app/main.py -> backend
        backend_dir = Path(__file__).resolve().parent.parent
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

        # Initialize ML models
        lasso_predictor, news_enhanced_predictor = initialize_models()

        # Initialize services
        logger.debug("Initializing services...")
        prediction_service = PredictionService(
            lasso_predictor=lasso_predictor,
            news_enhanced_predictor=news_enhanced_predictor
        )
        market_data_service = MarketDataService(
            prediction_service=prediction_service)
        exchange_service = ExchangeService()
        prediction_repo = PredictionRepository()

        # Initialize spot trading service
        from spot_trade.service import SpotTradingService
        spot_trading_service = SpotTradingService(
            market_data_service=market_data_service,
            exchange_service=exchange_service
        )

        # Set services for dependency injection
        set_services(
            market_data_service=market_data_service,
            prediction_service=prediction_service,
            prediction_repo=prediction_repo,
            exchange_service=exchange_service,
            spot_trading_service=spot_trading_service
        )

        # Store services in app state
        app.state.prediction_service = prediction_service
        app.state.market_data_service = market_data_service
        app.state.exchange_service = exchange_service
        app.state.prediction_repo = prediction_repo

        # Initialize database (sync for backward compatibility)
        logger.debug("Initializing database...")
        try:
            # Use a local variable to track PostgreSQL status instead of modifying settings
            use_postgresql = settings.use_postgresql
            if use_postgresql:
                if init_postgresql_pool():
                    logger.debug("✅ PostgreSQL database connected")
                else:
                    logger.warning("PostgreSQL failed - using SQLite")
                    use_postgresql = False

            init_database()
            init_backup_database()

            # Initialize spot trading tables
            from spot_trade.models import init_spot_trade_tables
            init_spot_trade_tables()
            logger.debug("✅ Spot trading tables initialized")

            logger.debug("✅ Database initialized")
        except Exception as e:
            logger.error(
                f"❌ Database initialization failed: {e}", exc_info=True)
            raise

        # Initialize async database (if available)
        try:
            if settings.use_postgresql:
                if await init_postgresql_pool_async():
                    logger.debug(
                        "✅ Async PostgreSQL connection pool initialized")
                await init_database_async()
        except Exception as e:
            logger.debug(
                f"Async database initialization failed (non-critical): {e}")

        # Log essential system information only
        from .core.logging_config import Emojis
        logger.debug(
            f"{Emojis.CONFIG} Environment: {settings.environment} | Database: {'PostgreSQL' if settings.use_postgresql else 'SQLite'}")

        # Initialize WebSocket connection manager
        manager = ConnectionManager()
        app.state.websocket_manager = manager

        # Initialize background task manager
        task_manager = BackgroundTaskManager()
        app.state.task_manager = task_manager

        # Start background tasks
        logger.debug("Starting background tasks...")

        # Broadcast task
        broadcast_task = asyncio.create_task(
            broadcast_daily_data(manager, market_data_service, task_manager)
        )
        task_manager.register_task("broadcast_daily_data", broadcast_task)
        logger.debug("  ✓ Broadcast daily data task registered")

        # Auto-update task
        if settings.auto_update_enabled:
            update_task = asyncio.create_task(
                auto_update_pending_predictions(
                    market_data_service, prediction_repo, task_manager
                )
            )
            task_manager.register_task(
                "auto_update_pending_predictions", update_task)
            logger.debug(
                f"  ✓ Auto-update task started (interval: {settings.auto_update_interval}s)")
        else:
            logger.debug("  ⊘ Auto-update task disabled")

        # Combined retrain-and-predict task (runs at same time, sequentially)
        if settings.auto_retrain_enabled and settings.auto_predict_enabled:
            combined_task = asyncio.create_task(
                auto_retrain_and_predict(
                    market_data_service, prediction_service, prediction_repo, task_manager
                )
            )
            task_manager.register_task(
                "auto_retrain_and_predict", combined_task)
            logger.debug(
                f"  ✓ Combined retrain-and-predict task started (hour: {settings.auto_predict_hour}:00, "
                f"retrain first, then predict with Gemini reasons)")
        else:
            # Fallback to separate tasks if one is disabled
            # Auto-predict task
            if settings.auto_predict_enabled:
                predict_task = asyncio.create_task(
                    auto_generate_daily_prediction(
                        market_data_service, prediction_service, prediction_repo, task_manager
                    )
                )
                task_manager.register_task(
                    "auto_generate_daily_prediction", predict_task)
                logger.debug(
                    f"  ✓ Auto-predict task started (hour: {settings.auto_predict_hour}:00)")
            else:
                logger.debug("  ⊘ Auto-predict task disabled")

            # Auto-retrain task
            if settings.auto_retrain_enabled:
                retrain_task = asyncio.create_task(
                    auto_retrain_model(
                        prediction_service, prediction_repo, task_manager
                    )
                )
                task_manager.register_task("auto_retrain_model", retrain_task)
                logger.debug(
                    f"  ✓ Auto-retrain task started (hour: {settings.auto_retrain_hour}:00)")
            else:
                logger.debug("  ⊘ Auto-retrain task disabled")

        # Set task manager for health check endpoint
        set_task_manager(task_manager)

        logger.info("✅ Application startup complete")

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}", exc_info=True)
        raise

    # Yield control to FastAPI - this is where the app runs
    # If interrupted (Ctrl+C), this will raise CancelledError
    try:
        yield
    except asyncio.CancelledError:
        # Expected during KeyboardInterrupt - suppress the error
        logger.debug("Application interrupted (expected during shutdown)")
        raise  # Re-raise to allow proper cleanup

    # Shutdown
    try:
        logger.info("🛑 Server shutting down...")

        # Shutdown background tasks
        task_manager = getattr(app.state, 'task_manager', None)
        if task_manager:
            try:
                await asyncio.wait_for(task_manager.shutdown(), timeout=5.0)
                logger.info("  ✓ Background tasks stopped")
            except asyncio.TimeoutError:
                logger.warning(
                    "  ⚠ Task shutdown timed out, forcing cancellation")
            except asyncio.CancelledError:
                logger.debug(
                    "  ⊘ Task shutdown cancelled (expected during hot reload)")

        # Close WebSocket connections
        manager = getattr(app.state, 'websocket_manager', None)
        if manager:
            try:
                await asyncio.wait_for(manager.disconnect_all(), timeout=2.0)
                logger.info("  ✓ WebSocket connections closed")
            except asyncio.TimeoutError:
                logger.warning("  ⚠ WebSocket disconnect timed out")
            except asyncio.CancelledError:
                logger.debug(
                    "  ⊘ WebSocket disconnect cancelled (expected during hot reload)")

        # Close async database pool
        try:
            await close_postgresql_pool_async()
            logger.info("  ✓ Async database pool closed")
        except Exception as e:
            logger.debug(f"  ⊘ Async database pool close: {e}")

        logger.info("✅ Shutdown complete")

    except asyncio.CancelledError:
        logger.debug("Shutdown cancelled (expected during hot reload)")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI app with optimized configuration
app = FastAPI(
    title="XAU/USD Real-time Data API",
    version="2.0.0",
    description=(
        f"Gold price prediction API with ML models and news sentiment analysis. "
        f"Running in {settings.environment} environment."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add exception handlers
app.add_exception_handler(BaseAPIException, base_api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add middleware (order matters - last added is first executed)
# Security headers should be last (first to execute)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CompressionMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_size=settings.max_request_size
)

# CORS middleware
cors_allow_credentials = "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (api_router already has prefix="/api/v1")
app.include_router(api_router)


# Root endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "XAU/USD Real-time Data API with News Sentiment Analysis",
        "version": "2.0.0",
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.head("/")
async def root_head():
    """Handle HEAD requests to root endpoint"""
    return Response(status_code=200)


@app.get("/health", tags=["Health"])
async def health_check_legacy():
    """Legacy health check endpoint (also available at /api/v1/health)"""
    from .api.v1.routes.health import health_check
    return await health_check()


@app.head("/health")
async def health_check_head_legacy():
    """Handle HEAD requests to health endpoint"""
    return Response(status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests"""
    return Response(status_code=204)


# WebSocket endpoint
@app.websocket("/ws/xauusd")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates"""
    manager = getattr(app.state, 'websocket_manager', None)
    market_data_service = getattr(app.state, 'market_data_service', None)

    if not manager or not market_data_service:
        await websocket.close(code=1011, reason="Service unavailable")
        return

    await manager.connect(websocket)
    try:
        last_sent_data = None
        while True:
            daily_data = market_data_service.get_daily_data()

            # Handle rate limiting
            if daily_data.get('status') == 'rate_limited':
                wait_seconds = daily_data.get(
                    'rate_limit_info', {}).get('wait_seconds', 60)
                if daily_data != last_sent_data:
                    await manager.send_personal_message(json.dumps(daily_data), websocket)
                    last_sent_data = daily_data
                await asyncio.sleep(min(wait_seconds, 300))
            elif daily_data != last_sent_data:
                await manager.send_personal_message(json.dumps(daily_data), websocket)
                last_sent_data = daily_data
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            manager.disconnect(websocket)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level=settings.log_level.lower(),
        access_log=True
    )
