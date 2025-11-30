"""
Main application entry point - Refactored to use modular structure
"""
from models.lasso_model import LassoGoldPredictor
from models.news_prediction import NewsEnhancedLassoPredictor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict
import asyncio
import json
import logging
from datetime import datetime, timedelta
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
    logger.debug("Lasso Regression model loaded")
except Exception as e:
    logger.warning(f"Lasso model not found: {e}")
    lasso_predictor = None

# Initialize News-Enhanced Lasso predictor
news_enhanced_predictor = NewsEnhancedLassoPredictor()
enhanced_model_path = BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'

if enhanced_model_path.exists():
    try:
        news_enhanced_predictor.load_enhanced_model(str(enhanced_model_path))
        logger.debug("News-enhanced model loaded")
    except Exception as e:
        logger.warning(f"News-enhanced model failed to load: {e}")
        news_enhanced_predictor = None
else:
    logger.debug("Using regular Lasso model (enhanced model not found)")

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


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(
            f"WebSocket client connected ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.debug(
            f"WebSocket client disconnected ({len(self.active_connections)} total)")

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


@app.get("/xauusd/enhanced-prediction")
async def get_enhanced_prediction():
    """Get enhanced prediction with news sentiment analysis"""
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

        # Get method name
        method = prediction_service.get_model_display_name()

        return {
            "status": "success",
            "prediction": {
                "next_day_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percentage": round(change_percentage, 2),
                "method": method
            },
            "sentiment": {
                "combined_sentiment": 0.0,  # Placeholder - would need news analyzer
                "news_volume": 0,
                "sentiment_trend": 0.0
            },
            "top_features": [],  # Placeholder
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting enhanced prediction: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd/accuracy-visualization")
async def get_accuracy_visualization():
    """Get accuracy statistics for visualization"""
    try:
        stats = prediction_repo.get_accuracy_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting accuracy stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/xauusd/prediction-stats")
async def get_prediction_stats():
    """Get comprehensive prediction statistics (all time)"""
    try:
        stats = prediction_repo.get_comprehensive_stats()
        return {
            "status": "success",
            "data": {
                "total_predictions": stats['total_predictions'],
                "evaluated": {
                    "count": stats['evaluated_predictions'],
                    "with_results": stats['evaluated_predictions'],
                    "average_accuracy": stats['average_accuracy']
                },
                "pending": {
                    "count": stats['pending_predictions'],
                    "awaiting_market_results": stats['pending_predictions']
                },
                "r2_score": stats.get('r2_score'),
                "evaluation_rate_percent": stats['evaluation_rate']
            }
        }
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/xauusd/pending-predictions")
async def get_pending_predictions():
    """Get list of pending predictions awaiting market results"""
    try:
        pending = prediction_repo.get_pending_predictions()
        return {
            "status": "success",
            "data": {
                "pending_count": len(pending),
                "predictions": pending
            }
        }
    except Exception as e:
        logger.error(f"Error getting pending predictions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/xauusd/update-pending-predictions")
async def update_pending_predictions():
    """Update pending predictions with actual market prices"""
    try:
        result = market_data_service.update_pending_predictions()
        return result
    except Exception as e:
        logger.error(f"Error updating pending predictions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


class PriceUpdateItem(BaseModel):
    """Price update item"""
    date: str
    actual_price: float


class PriceUpdateRequest(BaseModel):
    """Price update request"""
    prices: List[PriceUpdateItem]


@app.post("/xauusd/update-actual-prices")
async def update_actual_prices(request: PriceUpdateRequest):
    """Manually update actual prices for specific dates"""
    try:
        updated_count = 0
        failed_count = 0
        updated_dates = []
        failed_dates = []

        for price_item in request.prices:
            date = price_item.date
            actual_price = price_item.actual_price

            # Validate date format
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                failed_count += 1
                failed_dates.append(
                    {"date": date, "error": "Invalid date format. Use YYYY-MM-DD"})
                continue

            # Update the prediction
            if prediction_repo.update_prediction_with_actual_price(date, actual_price):
                updated_count += 1
                updated_dates.append(date)
                logger.info(
                    f"Updated actual price for {date}: ${actual_price:.2f}")
            else:
                failed_count += 1
                failed_dates.append(
                    {"date": date, "error": "No prediction found for this date"})

        return {
            "status": "success",
            "message": f"Updated {updated_count} prices, {failed_count} failed",
            "updated_count": updated_count,
            "failed_count": failed_count,
            "updated_dates": updated_dates,
            "failed_dates": failed_dates
        }
    except Exception as e:
        logger.error(f"Error updating actual prices: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/xauusd/prediction-history")
async def get_prediction_history(days: int = 30):
    """Get historical predictions"""
    try:
        predictions = prediction_repo.get_historical_predictions(days=days)
        return {
            "status": "success",
            "data": predictions,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def get_exchange_rate(from_currency: str, to_currency: str):
    """Get exchange rate between currencies - supports both naming conventions"""
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
    logger.info(f"🚀 Server starting in {settings.environment} mode")
    asyncio.create_task(broadcast_daily_data())
    logger.info("✅ Ready - API available at http://localhost:8001")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Server shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
