"""
Main application entry point - Optimized and modular structure
"""
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
    auto_retrain_model
)
from .core.task_manager import BackgroundTaskManager
from .core.websocket import ConnectionManager
from .core.models import initialize_models
from .core.database import (
    init_database,
    init_backup_database,
    init_postgresql_pool
)
from .core.config import settings
from .core.logging_config import setup_logging, get_logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
import asyncio
import warnings
# Suppress SyntaxWarnings from third-party libraries (e.g., textblob)
# These warnings appear during import and don't affect functionality
warnings.filterwarnings('ignore', category=SyntaxWarning, module='textblob')


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

            # Handle rate limiting - wait longer if rate limited
            if daily_data.get('status') == 'rate_limited':
                wait_seconds = daily_data.get(
                    'rate_limit_info', {}).get('wait_seconds', 60)
                # Send rate limit status to client
                if daily_data != last_sent_data:
                    await manager.send_personal_message(json.dumps(daily_data), websocket)
                    last_sent_data = daily_data
                # Wait for rate limit to expire (max 5 minutes)
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
        except Exception as disconnect_error:
            logger.error(
                f"Error disconnecting WebSocket: {disconnect_error}", exc_info=True)


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


@app.get("/xauusd/accuracy-visualization")
async def get_accuracy_visualization_legacy(days: int = 90):
    """Get accuracy visualization data (legacy endpoint)"""
    try:
        visualization_data = prediction_repo.get_accuracy_visualization_data(
            days=days)
        return {
            "status": "success",
            "data": visualization_data['data'],
            "statistics": visualization_data['statistics'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(
            f"Error getting accuracy visualization: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd/prediction-history")
async def get_prediction_history_legacy(days: int = 30):
    """Get prediction history (legacy endpoint)"""
    try:
        predictions = prediction_repo.get_historical_predictions(days=days)
        # Format to match frontend expectations
        formatted_predictions = []
        for pred in predictions:
            formatted_predictions.append({
                "date": pred['date'],
                "predicted_price": pred['predicted_price'],
                "actual_price": pred['actual_price'],
                "accuracy_percentage": pred.get('accuracy_percentage'),
                "status": "completed" if pred['actual_price'] is not None else "pending",
                "method": pred.get('method', 'Lasso Regression')
            })
        return {
            "status": "success",
            "predictions": formatted_predictions,
            "total": len(formatted_predictions)
        }
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/xauusd/enhanced-prediction")
async def get_enhanced_prediction_legacy():
    """Get enhanced prediction with model details (legacy endpoint)"""
    try:
        # Get current price
        current_price_data = market_data_service.get_realtime_price()
        current_price = current_price_data.get('current_price', 0.0)

        # Try to get enhanced prediction
        predicted_price = prediction_service.predict_next_day()

        if predicted_price is None:
            return {
                "status": "error",
                "message": "Unable to generate prediction",
                "timestamp": datetime.now().isoformat()
            }

        # Calculate change
        change = predicted_price - current_price
        change_percentage = (change / current_price *
                             100) if current_price > 0 else 0

        # Get method name and model information
        method = prediction_service.get_model_display_name()
        model_info = prediction_service.get_model_info()

        return {
            "status": "success",
            "prediction": {
                "next_day_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percentage": round(change_percentage, 2),
                "method": method
            },
            "model": {
                "name": model_info.get("active_model", "Unknown"),
                "type": model_info.get("model_type", "Unknown"),
                "r2_score": model_info.get("r2_score"),
                "features": {
                    "total": model_info.get("features_count"),
                    "selected": model_info.get("selected_features_count"),
                    "top_features": model_info.get("selected_features", [])[:5]
                },
                "fallback_available": model_info.get("fallback_available", False)
            },
            "sentiment": {
                "combined_sentiment": 0.0,
                "news_volume": 0,
                "sentiment_trend": 0.0
            },
            "top_features": model_info.get("selected_features", [])[:5],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting enhanced prediction: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd/model-info")
async def get_model_info_legacy():
    """Get detailed ML model information (legacy endpoint)"""
    try:
        model_info = prediction_service.get_model_info()
        
        # Add explanation for the R² scores
        r2_explanation = {
            "training_r2_score": "Static accuracy from model training (historical test data)",
            "live_r2_score": "Dynamic accuracy from real predictions vs actual market prices (updates automatically)",
            "r2_score": "Primary score shown to users (uses live if available)"
        }
        
        return {
            "status": "success",
            "model": model_info,
            "r2_explanation": r2_explanation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def get_exchange_rate_legacy(from_currency: str, to_currency: str):
    """Get exchange rate between currencies (legacy endpoint)"""
    return exchange_service.get_exchange_rate(from_currency, to_currency)


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler
    Industry-standard startup sequence with proper error handling
    """
    try:
        logger.info(f"🚀 Server starting in {settings.environment} mode")

        # Log system configuration
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Log Level: {settings.log_level.upper()}")
        logger.info(
            f"Database: {'PostgreSQL' if settings.use_postgresql else 'SQLite'}")
        logger.info(
            f"Auto-update: {'Enabled' if settings.auto_update_enabled else 'Disabled'}")

        # Log ML model information
        model_info = prediction_service.get_model_info()
        logger.info(
            f"🤖 ML Model: {model_info.get('active_model', 'No Model Available')}")
        
        # Show both R² scores clearly
        training_r2 = model_info.get('training_r2_score')
        live_r2 = model_info.get('live_r2_score')
        
        if training_r2 is not None:
            logger.info(f"📊 Training R² (from model): {training_r2:.4f} ({training_r2*100:.2f}%)")
        if live_r2 is not None:
            live_stats = model_info.get('live_accuracy_stats', {})
            eval_count = live_stats.get('evaluated_predictions', 0)
            logger.info(f"📈 Live R² (from {eval_count} predictions): {live_r2:.4f} ({live_r2*100:.2f}%)")
        
        selected_count = model_info.get('selected_features_count', 0)
        total_count = model_info.get(
            'total_features', model_info.get('features_count', 0))
        if total_count > 0:
            logger.info(f"🔧 Features: {selected_count}/{total_count} selected")
        if model_info.get('selected_features'):
            top_features = ', '.join(model_info['selected_features'][:3])
            logger.info(f"⭐ Top Features: {top_features}...")
        if model_info.get('fallback_available'):
            logger.info("🔄 Fallback model: Available (Lasso Regression)")

        # Start background tasks with proper registration
        broadcast_task = asyncio.create_task(
            broadcast_daily_data(manager, market_data_service, task_manager)
        )
        task_manager.register_task("broadcast_daily_data", broadcast_task)
        logger.debug("Background task 'broadcast_daily_data' registered")

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

        # Start auto-retrain task (daily model retraining)
        if settings.auto_retrain_enabled:
            retrain_task = asyncio.create_task(
                auto_retrain_model(
                    prediction_service, prediction_repo, task_manager
                )
            )
            task_manager.register_task(
                "auto_retrain_model", retrain_task
            )
            logger.info(
                f"🤖 Auto-retrain task started: model will retrain daily at {settings.auto_retrain_hour}:00")
        else:
            logger.info("🤖 Auto-retrain task is disabled via configuration")

        logger.info("✅ Ready - API available at http://localhost:8001")

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler
    Industry-standard graceful shutdown with resource cleanup
    """
    try:
        logger.info("🛑 Server shutting down")

        # Shutdown background tasks
        await task_manager.shutdown()

        # Close WebSocket connections
        await manager.disconnect_all()

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
