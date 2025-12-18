"""XAU/USD routes"""
from fastapi import APIRouter, Depends
from typing import Optional
from datetime import datetime

from ....core.dependencies import (
    get_market_data_service,
    get_prediction_service,
    get_prediction_repo
)
from ....schemas.prediction import PriceUpdateRequest

router = APIRouter()


@router.get("")
async def get_daily_data(
    days: int = 90,
    market_data_service=Depends(get_market_data_service),
    prediction_service=Depends(get_prediction_service)
):
    """Get XAU/USD daily data with model information"""
    data = market_data_service.get_daily_data(days=days)
    # Add model info if not already present
    if isinstance(data, dict) and "model_info" not in data:
        data["model_info"] = prediction_service.get_model_info()
    return data


@router.get("/realtime")
async def get_realtime_price(
    market_data_service=Depends(get_market_data_service)
):
    """Get real-time XAU/USD price"""
    return market_data_service.get_realtime_price()


@router.get("/enhanced-prediction")
async def get_enhanced_prediction(
    market_data_service=Depends(get_market_data_service),
    prediction_service=Depends(get_prediction_service)
):
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
                "r2_score": model_info.get("r2_score"),  # Primary (live if available)
                "training_r2_score": model_info.get("training_r2_score"),  # Static from training
                "live_r2_score": model_info.get("live_r2_score"),  # Dynamic from predictions
                "features": {
                    "total": model_info.get("features_count"),
                    "selected": model_info.get("selected_features_count"),
                    "top_features": model_info.get("selected_features", [])[:5]  # Top 5 features
                },
                "fallback_available": model_info.get("fallback_available", False),
                "live_accuracy_stats": model_info.get("live_accuracy_stats")  # Full live stats
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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting enhanced prediction: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/accuracy-visualization")
async def get_accuracy_visualization(
    days: int = 90,
    prediction_repo=Depends(get_prediction_repo)
):
    """Get accuracy statistics for visualization"""
    try:
        visualization_data = prediction_repo.get_accuracy_visualization_data(days=days)
        return {
            "status": "success",
            "data": visualization_data['data'],
            "statistics": visualization_data['statistics'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting accuracy visualization: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/model-info")
async def get_model_info(
    prediction_service=Depends(get_prediction_service)
):
    """Get detailed information about the active ML model with live accuracy metrics
    
    Returns:
        - training_r2_score: Static R² from when model was trained (doesn't change)
        - live_r2_score: Dynamic R² calculated from actual predictions vs real prices (updates with each prediction)
        - r2_score: Primary score (live if available, otherwise training)
        - live_accuracy_stats: Full live accuracy statistics including average_accuracy, total_predictions, etc.
    """
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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting model info: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/prediction-stats")
async def get_prediction_stats(
    prediction_repo=Depends(get_prediction_repo)
):
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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting prediction stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/pending-predictions")
async def get_pending_predictions(
    prediction_repo=Depends(get_prediction_repo)
):
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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting pending predictions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/update-pending-predictions")
async def update_pending_predictions(
    market_data_service=Depends(get_market_data_service)
):
    """Update pending predictions with actual market prices"""
    try:
        result = market_data_service.update_pending_predictions()
        return result
    except Exception as e:
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error updating pending predictions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/update-actual-prices")
async def update_actual_prices(
    request: PriceUpdateRequest,
    prediction_repo=Depends(get_prediction_repo)
):
    """Manually update actual prices for specific dates"""
    try:
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)

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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error updating actual prices: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/prediction-history")
async def get_prediction_history(
    days: int = 30,
    prediction_repo=Depends(get_prediction_repo)
):
    """Get historical predictions"""
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
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting prediction history: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/retrain-model")
async def retrain_model_now(
    prediction_service=Depends(get_prediction_service),
    prediction_repo=Depends(get_prediction_repo)
):
    """Manually trigger model retraining
    
    This will:
    1. Fetch fresh market data
    2. Fetch news sentiment data
    3. Retrain the News-Enhanced Lasso model
    4. Reload the model in-memory
    
    Note: This can take several minutes to complete.
    """
    try:
        from ....core.logging_config import get_logger
        from ....core.background_tasks import _perform_model_retrain
        from ....core.config import settings
        
        logger = get_logger(__name__)
        logger.info("🔄 Manual model retrain triggered via API")
        
        # Perform retraining
        result = await _perform_model_retrain(
            prediction_service,
            settings.auto_retrain_news_days
        )
        
        if result["success"]:
            return {
                "status": "success",
                "message": "Both models retrained successfully",
                "data": {
                    "primary_model": {
                        "name": "News-Enhanced Lasso",
                        "r2_score": result.get("r2_score"),
                        "features_selected": result.get("features_selected"),
                        "total_features": result.get("total_features")
                    },
                    "fallback_model": {
                        "name": "Lasso Regression",
                        "r2_score": result.get("fallback_r2_score")
                    },
                    "models_trained": result.get("models_trained", []),
                    "trained_at": result.get("trained_at")
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Model retraining failed: {result.get('error')}",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error during manual retrain: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/retrain-status")
async def get_retrain_status():
    """Get the status of the auto-retrain background task"""
    try:
        from ....core.config import settings
        from ....api.v1.routes.health import get_task_manager
        
        task_manager = get_task_manager()
        
        if task_manager is None:
            return {
                "status": "error",
                "message": "Task manager not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get retrain task state
        retrain_state = task_manager.task_states.get("auto_retrain_model", {})
        
        return {
            "status": "success",
            "data": {
                "enabled": settings.auto_retrain_enabled,
                "scheduled_hour": settings.auto_retrain_hour,
                "interval_hours": settings.auto_retrain_interval // 3600,
                "task_status": retrain_state.get("status", "not_started"),
                "last_run": retrain_state.get("last_run"),
                "last_retrain": retrain_state.get("last_retrain"),
                "last_r2_score": retrain_state.get("last_r2_score"),
                "retrain_count": retrain_state.get("retrain_count", 0),
                "error_count": retrain_state.get("error_count", 0),
                "last_error": retrain_state.get("last_error")
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        from ....core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error getting retrain status: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
