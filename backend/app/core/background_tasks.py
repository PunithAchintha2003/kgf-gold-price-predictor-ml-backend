"""Background tasks for the application"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from pathlib import Path
import sys

from .config import settings
from .logging_config import get_logger
from .task_manager import BackgroundTaskManager

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService
    from ..services.prediction_service import PredictionService
    from ..repositories.prediction_repository import PredictionRepository
    from .websocket import ConnectionManager

logger = get_logger(__name__)


async def broadcast_daily_data(
    manager: "ConnectionManager",
    market_data_service: "MarketDataService",
    task_manager: BackgroundTaskManager
):
    """Background task to broadcast daily data to all connected clients"""
    task_name = "broadcast_daily_data"
    last_broadcast_data = None

    while not task_manager.shutdown_event.is_set():
        try:
            if manager.active_connections:
                # Check rate limit status first to avoid unnecessary service calls
                from ..utils.cache import market_data_cache
                is_rate_limited, wait_seconds = market_data_cache.is_rate_limited()
                if is_rate_limited:
                    logger.debug(
                        f"Skipping broadcast - rate limited. Waiting {wait_seconds:.0f}s")
                    # Wait longer when rate limited to avoid repeated attempts
                    try:
                        await asyncio.wait_for(
                            task_manager.shutdown_event.wait(),
                            # Max 5 minutes wait
                            timeout=min(wait_seconds, 300.0)
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Continue loop after wait

                # Only call service if not rate limited
                daily_data = market_data_service.get_daily_data()
                if daily_data != last_broadcast_data:
                    await manager.broadcast(json.dumps(daily_data))
                    last_broadcast_data = daily_data
                    # Wait normal interval after successful broadcast
                    try:
                        await asyncio.wait_for(
                            task_manager.shutdown_event.wait(),
                            timeout=5.0
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Continue loop
                else:
                    # No change in data - wait normal interval
                    try:
                        await asyncio.wait_for(
                            task_manager.shutdown_event.wait(),
                            timeout=5.0
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Continue loop
            else:
                # No active connections - wait normal interval
                try:
                    await asyncio.wait_for(
                        task_manager.shutdown_event.wait(),
                        timeout=5.0
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Continue loop
        except asyncio.CancelledError:
            logger.debug(f"Task {task_name} cancelled")
            break
        except Exception as e:
            logger.error(f"Error in {task_name}: {e}", exc_info=True)
            await asyncio.sleep(5)  # Brief pause before retry


async def auto_update_pending_predictions(
    market_data_service: "MarketDataService",
    prediction_repo: "PredictionRepository",
    task_manager: BackgroundTaskManager
):
    """
    Industry-standard background task to automatically update pending predictions.

    Features:
    - Configurable intervals via settings
    - Error handling with retries
    - Circuit breaker pattern
    - Health tracking
    - Graceful shutdown
    """
    task_name = "auto_update_pending_predictions"

    # Check if auto-update is enabled
    if not settings.auto_update_enabled:
        logger.info("🔄 Auto-update task is disabled via configuration")
        return

    # Wait for startup delay (configurable)
    try:
        await asyncio.wait_for(
            task_manager.shutdown_event.wait(),
            timeout=float(settings.auto_update_startup_delay)
        )
        return  # Shutdown requested during startup delay
    except asyncio.TimeoutError:
        pass  # Continue after delay

    consecutive_failures = 0
    max_consecutive_failures = 5  # Circuit breaker threshold
    _last_rate_limit_log = 0  # Throttle rate limit logs (once per minute)

    while not task_manager.shutdown_event.is_set():
        try:
            # Initialize/update task state
            if task_name not in task_manager.task_states:
                task_manager.task_states[task_name] = {
                    "status": "running",
                    "last_run": None,
                    "last_error": None,
                    "run_count": 0,
                    "error_count": 0
                }
            state = task_manager.task_states[task_name]
            state["status"] = "running"
            state["last_run"] = datetime.now().isoformat()

            # Check for pending predictions
            pending = prediction_repo.get_pending_predictions()

            if pending:
                logger.info(
                    f"🔄 Auto-updating {len(pending)} pending prediction(s)...")

                # Check if we're rate limited before attempting update
                # Check cache directly to avoid triggering service calls and logs
                from ..utils.cache import market_data_cache
                is_rate_limited, wait_seconds = market_data_cache.is_rate_limited()
                if is_rate_limited:
                    # Only log once per minute to reduce log spam
                    current_time = time.time()
                    if current_time - _last_rate_limit_log > 60:
                        logger.info(
                            f"⏸️ Skipping auto-update - rate limited. Will retry after {wait_seconds:.0f}s")
                        _last_rate_limit_log = current_time
                    else:
                        logger.debug(
                            f"⏸️ Rate limited. Will retry after {wait_seconds:.0f}s")
                    # Wait for rate limit to expire (or max interval)
                    try:
                        await asyncio.wait_for(
                            task_manager.shutdown_event.wait(),
                            timeout=min(wait_seconds, float(
                                settings.auto_update_interval))
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Continue loop after wait

                # Retry logic with exponential backoff
                result = None
                retry_count = 0

                while retry_count < settings.auto_update_max_retries:
                    try:
                        result = market_data_service.update_pending_predictions()

                        if result.get('status') == 'success':
                            consecutive_failures = 0  # Reset circuit breaker
                            break
                        elif result.get('status') == 'rate_limited':
                            # If rate limited during update, wait and skip
                            wait_seconds = result.get(
                                'rate_limit_info', {}).get('wait_seconds', 60)
                            # Only log once per minute to reduce log spam
                            current_time = time.time()
                            if current_time - _last_rate_limit_log > 60:
                                logger.info(
                                    f"⏸️ Rate limited during update. Skipping this cycle. Will retry after {wait_seconds}s")
                                _last_rate_limit_log = current_time
                            else:
                                logger.debug(
                                    f"⏸️ Rate limited. Will retry after {wait_seconds}s")
                            break  # Exit retry loop, will wait in main loop
                        else:
                            retry_count += 1
                            if retry_count < settings.auto_update_max_retries:
                                wait_time = settings.auto_update_retry_delay * retry_count
                                logger.warning(
                                    f"⚠️ Update failed, retrying in {wait_time}s (attempt {retry_count + 1}/{settings.auto_update_max_retries})")
                                await asyncio.sleep(wait_time)
                    except Exception as retry_error:
                        retry_count += 1
                        logger.error(
                            f"❌ Error during update attempt {retry_count}: {retry_error}",
                            exc_info=True)
                        if retry_count < settings.auto_update_max_retries:
                            wait_time = settings.auto_update_retry_delay * retry_count
                            await asyncio.sleep(wait_time)

                # Process results
                if result and result.get('status') == 'success':
                    updated = result.get('updated_count', 0)
                    failed = result.get('failed_count', 0)
                    skipped = result.get('skipped_count', 0)

                    state["run_count"] = state.get("run_count", 0) + 1

                    if updated > 0:
                        logger.info(
                            f"✅ Auto-updated {updated} prediction(s) with actual prices")
                    if failed > 0:
                        logger.warning(
                            f"⚠️ Failed to update {failed} prediction(s)")
                        state["error_count"] = state.get(
                            "error_count", 0) + failed
                    if skipped > 0:
                        logger.debug(
                            f"⏭️ Skipped {skipped} future prediction(s)")
                else:
                    consecutive_failures += 1
                    state["error_count"] = state.get("error_count", 0) + 1
                    state["last_error"] = result.get(
                        'message', 'Unknown error') if result else 'Max retries exceeded'

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"❌ Circuit breaker triggered: {consecutive_failures} consecutive failures. "
                            f"Pausing updates for {settings.auto_update_interval * 2} seconds.")
                        # Extended wait before retrying
                        try:
                            await asyncio.wait_for(
                                task_manager.shutdown_event.wait(),
                                timeout=float(
                                    settings.auto_update_interval * 2)
                            )
                            break
                        except asyncio.TimeoutError:
                            consecutive_failures = 0  # Reset after extended wait
                            continue
                    else:
                        logger.warning(
                            f"⚠️ Auto-update failed: {state['last_error']}")
            else:
                logger.debug("No pending predictions to update")
                state["run_count"] = state.get("run_count", 0) + 1

            # Update task state
            state["status"] = "idle"

        except asyncio.CancelledError:
            logger.debug(f"Task {task_name} cancelled")
            break
        except Exception as e:
            consecutive_failures += 1
            if task_name not in task_manager.task_states:
                task_manager.task_states[task_name] = {
                    "status": "error",
                    "last_run": None,
                    "last_error": None,
                    "run_count": 0,
                    "error_count": 0
                }
            state = task_manager.task_states[task_name]
            state["error_count"] = state.get("error_count", 0) + 1
            state["last_error"] = str(e)
            state["status"] = "error"
            logger.error(
                f"❌ Error in auto-update task: {e}", exc_info=True)

        # Wait for next interval (with cancellation support)
        try:
            await asyncio.wait_for(
                task_manager.shutdown_event.wait(),
                timeout=float(settings.auto_update_interval)
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue loop


async def auto_generate_daily_prediction(
    market_data_service: "MarketDataService",
    prediction_service: "PredictionService",
    prediction_repo: "PredictionRepository",
    task_manager: BackgroundTaskManager
):
    """
    Background task to automatically generate daily predictions when market opens.
    
    Features:
    - Runs daily at market open time (default: 8 AM)
    - Skips weekends (Saturday and Sunday)
    - Generates prediction for next trading day
    - Fetches news sentiment during generation
    - Saves prediction to database
    - Only generates once per day
    """
    task_name = "auto_generate_daily_prediction"
    
    # Check if auto-predict is enabled
    if not settings.auto_predict_enabled:
        logger.info("📊 Auto-prediction generation is disabled via configuration")
        return
    
    # Wait for startup delay
    try:
        await asyncio.wait_for(
            task_manager.shutdown_event.wait(),
            timeout=float(settings.auto_predict_startup_delay)
        )
        return  # Shutdown requested during startup delay
    except asyncio.TimeoutError:
        pass  # Continue after delay
    
    logger.info(
        f"📊 Auto-prediction task started - will generate predictions daily at {settings.auto_predict_hour}:00 (skips weekends)")
    
    max_consecutive_failures = 3
    consecutive_failures = 0
    last_prediction_date = None
    
    while not task_manager.shutdown_event.is_set():
        try:
            # Initialize/update task state
            if task_name not in task_manager.task_states:
                task_manager.task_states[task_name] = {
                    "status": "running",
                    "last_run": None,
                    "last_prediction_date": None,
                    "last_error": None,
                    "run_count": 0,
                    "prediction_count": 0,
                    "error_count": 0
                }
            state = task_manager.task_states[task_name]
            state["status"] = "running"
            state["last_run"] = datetime.now().isoformat()
            
            current_time = datetime.now()
            current_hour = current_time.hour
            current_date = current_time.date()
            
            # Check if it's a weekend (skip Saturday and Sunday)
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                # Calculate days until Monday
                if current_time.weekday() == 5:  # Saturday
                    days_until_monday = 2  # Wait 2 days until Monday
                else:  # Sunday
                    days_until_monday = 1  # Wait 1 day until Monday
                wait_seconds = days_until_monday * 86400  # Convert days to seconds
                logger.debug(f"⏸️ Weekend detected (day {current_time.weekday()}). Waiting {days_until_monday} day(s) until next Monday...")
                try:
                    await asyncio.wait_for(
                        task_manager.shutdown_event.wait(),
                        timeout=min(wait_seconds, 86400.0)  # Max 24 hours per wait
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Continue loop after wait
            
            # Check if it's time to generate prediction (at the configured hour, once per day)
            should_generate = (
                current_hour == settings.auto_predict_hour and
                last_prediction_date != current_date
            )
            
            if should_generate:
                logger.info(f"📊 Generating daily prediction for next trading day...")
                try:
                    # Get next trading day (skip weekends)
                    from ..services.market_data_service import get_next_trading_day
                    next_trading_day_dt = get_next_trading_day()
                    next_day = next_trading_day_dt.strftime("%Y-%m-%d")
                    
                    # Check if prediction already exists for this date
                    if prediction_repo.prediction_exists_for_date(next_day):
                        logger.info(f"✅ Prediction already exists for {next_day}, skipping generation")
                        last_prediction_date = current_date
                        state["last_prediction_date"] = next_day
                        # Wait until next day
                        await asyncio.sleep(3600)  # Wait 1 hour before checking again
                        continue
                    
                    # Generate prediction (this will fetch news sentiment)
                    logger.debug(f"Calling prediction_service.predict_next_day() for {next_day}...")
                    predicted_price = prediction_service.predict_next_day()
                    
                    if predicted_price is None:
                        # Check which models are available for better diagnostics
                        has_enhanced = prediction_service.news_enhanced_predictor is not None and prediction_service.news_enhanced_predictor.model is not None
                        has_lasso = prediction_service.lasso_predictor is not None and prediction_service.lasso_predictor.model is not None
                        
                        logger.error(
                            f"❌ Failed to generate prediction for {next_day} - service returned None. "
                            f"Enhanced model: {'available' if has_enhanced else 'unavailable'}, "
                            f"Lasso model: {'available' if has_lasso else 'unavailable'}. "
                            f"Check prediction service logs for details."
                        )
                        consecutive_failures += 1
                        state["error_count"] = state.get("error_count", 0) + 1
                        state["last_error"] = f"Prediction service returned None (Enhanced: {has_enhanced}, Lasso: {has_lasso})"
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), waiting before retry")
                            await asyncio.sleep(3600)  # Wait 1 hour
                            consecutive_failures = 0
                        continue
                    
                    # Get prediction method
                    prediction_method = prediction_service.get_model_display_name()
                    
                    # Generate prediction reasons using AI (only when new prediction is created)
                    prediction_reasons = None
                    try:
                        from ai.services.prediction_reason_service import PredictionReasonService
                        from ai.services.gemini_service import GeminiService
                        from ai.config import ai_config
                        
                        if ai_config.is_configured():
                            logger.info(f"🤖 Generating AI prediction reasons for {next_day}...")
                            
                            # Get current price
                            current_price_data = market_data_service.get_realtime_price()
                            current_price = current_price_data.get('current_price', 0.0)
                            
                            if current_price > 0:
                                # Get last 10 days of predictions
                                historical_predictions = prediction_repo.get_historical_predictions(days=10)
                                formatted_predictions = []
                                for pred in historical_predictions:
                                    formatted_predictions.append({
                                        "date": pred['date'],
                                        "predicted_price": pred['predicted_price'],
                                        "actual_price": pred.get('actual_price'),
                                        "accuracy_percentage": pred.get('accuracy_percentage'),
                                        "method": pred.get('method', 'Lasso Regression')
                                    })
                                
                                # Get aggregated news sentiment (excluding Alpha Vantage data)
                                news_info = None
                                try:
                                    from models.news_prediction import NewsSentimentAnalyzer
                                    news_analyzer = NewsSentimentAnalyzer()
                                    # Get aggregated sentiment metrics and top headlines (last 7 days)
                                    news_info = news_analyzer.get_aggregated_sentiment_for_gemini(days_back=7)
                                    
                                    if news_info.get('news_volume', 0) > 0:
                                        logger.debug(f"📰 Fetched news sentiment: {news_info.get('news_volume')} articles, sentiment: {news_info.get('combined_sentiment', 0):.2f}")
                                    else:
                                        logger.debug("📰 No news data available for Gemini")
                                        news_info = None
                                except Exception as news_error:
                                    logger.warning(f"⚠️ Error fetching news for Gemini: {news_error}")
                                    news_info = None  # Continue without news
                                
                                # Generate reasons using AI
                                gemini_service = GeminiService()
                                reason_service = PredictionReasonService(gemini_service)
                                
                                prediction_reasons = reason_service.generate_prediction_reasons(
                                    current_price=current_price,
                                    predicted_price=predicted_price,
                                    prediction_date=next_day,
                                    prediction_method=prediction_method,
                                    historical_predictions=formatted_predictions,
                                    news_info=news_info  # Aggregated sentiment metrics (excluding Alpha Vantage)
                                )
                                
                                if prediction_reasons:
                                    logger.info(f"✅ Generated prediction reasons for {next_day}")
                                else:
                                    logger.warning(f"⚠️ Failed to generate prediction reasons for {next_day}")
                        else:
                            logger.debug("Gemini API not configured, skipping prediction reasons generation")
                    except Exception as reason_error:
                        logger.warning(f"⚠️ Error generating prediction reasons: {reason_error}")
                        # Continue without reasons - prediction is still saved
                    
                    # Save prediction to database with reasons
                    prediction_repo.save_prediction(
                        next_day, 
                        predicted_price, 
                        prediction_method=prediction_method,
                        prediction_reasons=prediction_reasons
                    )
                    
                    logger.info(
                        f"✅ Generated and saved prediction for {next_day}: ${predicted_price:.2f} (Method: {prediction_method})"
                        f"{' with AI reasons' if prediction_reasons else ''}")
                    
                    last_prediction_date = current_date
                    state["last_prediction_date"] = next_day
                    state["prediction_count"] = state.get("prediction_count", 0) + 1
                    consecutive_failures = 0  # Reset on success
                    
                except Exception as e:
                    logger.error(f"❌ Error generating daily prediction: {e}", exc_info=True)
                    consecutive_failures += 1
                    state["error_count"] = state.get("error_count", 0) + 1
                    state["last_error"] = str(e)
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), waiting before retry")
                        await asyncio.sleep(3600)  # Wait 1 hour
                        consecutive_failures = 0
            else:
                # Not time yet, wait and check again
                # Calculate seconds until next check (check every hour)
                await asyncio.sleep(3600)  # Check every hour
                
        except asyncio.CancelledError:
            logger.debug(f"Task {task_name} cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in {task_name}: {e}", exc_info=True)
            await asyncio.sleep(60)  # Brief pause before retry
    
    logger.info(f"🛑 {task_name} task stopped")


async def auto_retrain_model(
    prediction_service: "PredictionService",
    prediction_repo: "PredictionRepository",
    task_manager: BackgroundTaskManager
):
    """
    Background task to automatically retrain the ML model daily.

    Features:
    - Runs daily at a configurable hour (default: 2 AM)
    - Fetches fresh market data and news sentiment
    - Retrains the News-Enhanced Lasso model
    - Updates the model in-memory without restart
    - Tracks training history and performance
    """
    task_name = "auto_retrain_model"

    # Check if auto-retrain is enabled
    if not settings.auto_retrain_enabled:
        logger.info("🔄 Auto-retrain task is disabled via configuration")
        return

    # Wait for startup delay
    try:
        await asyncio.wait_for(
            task_manager.shutdown_event.wait(),
            timeout=float(settings.auto_retrain_startup_delay)
        )
        return  # Shutdown requested during startup delay
    except asyncio.TimeoutError:
        pass  # Continue after delay

    logger.info(
        f"🤖 Auto-retrain task started - will retrain daily at {settings.auto_retrain_hour}:00")

    consecutive_failures = 0
    max_consecutive_failures = 3
    last_retrain_date = None

    while not task_manager.shutdown_event.is_set():
        try:
            # Initialize/update task state
            if task_name not in task_manager.task_states:
                task_manager.task_states[task_name] = {
                    "status": "running",
                    "last_run": None,
                    "last_retrain": None,
                    "last_error": None,
                    "run_count": 0,
                    "retrain_count": 0,
                    "error_count": 0,
                    "last_r2_score": None
                }
            state = task_manager.task_states[task_name]
            state["status"] = "running"
            state["last_run"] = datetime.now().isoformat()

            current_time = datetime.now()
            current_hour = current_time.hour
            current_date = current_time.date()

            # Check if it's time to retrain (at the configured hour, once per day)
            should_retrain = (
                current_hour == settings.auto_retrain_hour and
                last_retrain_date != current_date
            )

            if should_retrain:
                logger.info("🔄 Starting daily model retraining...")

                # Check if we have enough predictions to justify retraining
                try:
                    stats = prediction_repo.get_comprehensive_stats()
                    evaluated_count = stats.get('evaluated_predictions', 0)

                    if evaluated_count < settings.auto_retrain_min_predictions:
                        logger.info(
                            f"⏭️ Skipping retrain - only {evaluated_count} evaluated predictions "
                            f"(need {settings.auto_retrain_min_predictions})"
                        )
                        last_retrain_date = current_date
                        state["run_count"] = state.get("run_count", 0) + 1
                    else:
                        # Perform the retraining
                        retrain_result = await _perform_model_retrain(
                            prediction_service,
                            settings.auto_retrain_news_days
                        )

                        if retrain_result["success"]:
                            consecutive_failures = 0
                            last_retrain_date = current_date
                            state["retrain_count"] = state.get(
                                "retrain_count", 0) + 1
                            state["last_retrain"] = datetime.now().isoformat()
                            state["last_r2_score"] = retrain_result.get(
                                "r2_score")

                            logger.info(
                                f"✅ Model retrained successfully! "
                                f"New R² score: {retrain_result.get('r2_score', 'N/A')}"
                            )
                        else:
                            consecutive_failures += 1
                            state["error_count"] = state.get(
                                "error_count", 0) + 1
                            state["last_error"] = retrain_result.get(
                                "error", "Unknown error")

                            logger.error(
                                f"❌ Model retraining failed: {retrain_result.get('error')}")

                            if consecutive_failures >= max_consecutive_failures:
                                logger.error(
                                    f"❌ Circuit breaker: {consecutive_failures} consecutive failures. "
                                    f"Pausing retraining for extended period."
                                )
                except Exception as e:
                    consecutive_failures += 1
                    state["error_count"] = state.get("error_count", 0) + 1
                    state["last_error"] = str(e)
                    logger.error(
                        f"❌ Error during retrain check: {e}", exc_info=True)

            state["status"] = "idle"
            state["run_count"] = state.get("run_count", 0) + 1

        except asyncio.CancelledError:
            logger.debug(f"Task {task_name} cancelled")
            break
        except Exception as e:
            if task_name in task_manager.task_states:
                state = task_manager.task_states[task_name]
                state["error_count"] = state.get("error_count", 0) + 1
                state["last_error"] = str(e)
                state["status"] = "error"
            logger.error(f"❌ Error in auto-retrain task: {e}", exc_info=True)

        # Wait for next check interval (check every hour)
        try:
            await asyncio.wait_for(
                task_manager.shutdown_event.wait(),
                timeout=3600.0  # Check every hour
            )
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue loop


async def _perform_model_retrain(
    prediction_service: "PredictionService",
    news_days: int = 30
) -> dict:
    """
    Perform the actual model retraining for BOTH models:
    1. News-Enhanced Lasso (primary model)
    2. Basic Lasso Regression (fallback model)

    This runs in a thread pool to avoid blocking the event loop
    since training can take several minutes.
    """
    import concurrent.futures

    def _retrain_sync():
        """Synchronous retraining function to run in thread pool"""
        fallback_r2 = None

        try:
            # Get the paths
            backend_dir = Path(__file__).resolve().parent.parent.parent
            models_dir = backend_dir / "models"

            # Add to sys.path if needed
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))

            # Import the model classes
            from models.lasso_model import LassoGoldPredictor
            from models.news_prediction import NewsEnhancedLassoPredictor

            logger.info("📊 Fetching fresh market data...")

            # Create new predictor instance for training
            base_predictor = LassoGoldPredictor()
            market_data = base_predictor.fetch_market_data()

            if market_data is None:
                return {"success": False, "error": "Failed to fetch market data"}

            # ============================================
            # TRAIN FALLBACK MODEL (Basic Lasso Regression)
            # ============================================
            logger.info("🏋️ Training fallback Lasso model...")

            try:
                # Create features for basic model
                base_features = base_predictor.create_fundamental_features(
                    market_data)

                if not base_features.empty:
                    # Prepare X and y for training
                    target_col = 'gold_close'
                    feature_cols = [
                        col for col in base_features.columns if col != target_col]

                    X = base_features[feature_cols].dropna()
                    y = base_features.loc[X.index, target_col]

                    # Store feature columns for later use
                    base_predictor.feature_columns = feature_cols

                    # Train the basic Lasso model
                    base_training_results = base_predictor.train_model(X, y)

                    if base_training_results:
                        # Save fallback model
                        fallback_model_path = models_dir / "lasso_gold_model.pkl"
                        base_predictor.save_model(str(fallback_model_path))

                        # Note: key is 'lasso_model' not 'lasso_regression'
                        fallback_r2 = base_training_results.get(
                            'lasso_model', {}).get('r2', None)
                        if fallback_r2:
                            logger.info(
                                f"✅ Fallback Lasso model trained - R²: {fallback_r2:.4f}")
                        else:
                            logger.info("✅ Fallback model trained")

                        # Reload fallback model in prediction service
                        if prediction_service.lasso_predictor is not None:
                            prediction_service.lasso_predictor.load_model(
                                str(fallback_model_path))
                            logger.info(
                                "✅ Fallback model reloaded in prediction service")
                    else:
                        logger.warning(
                            "⚠️ Fallback model training returned no results")
                else:
                    logger.warning("⚠️ No features created for fallback model")
            except Exception as fallback_error:
                logger.warning(
                    f"⚠️ Fallback model training failed: {fallback_error}")
                import traceback
                logger.debug(traceback.format_exc())
                # Continue with enhanced model even if fallback fails

            # ============================================
            # TRAIN PRIMARY MODEL (News-Enhanced Lasso)
            # ============================================
            logger.info(f"📰 Fetching news data for last {news_days} days...")

            # Create enhanced predictor and train
            enhanced_predictor = NewsEnhancedLassoPredictor()
            sentiment_features = enhanced_predictor.fetch_and_analyze_news(
                days_back=news_days)

            logger.info("🔧 Creating enhanced features...")
            enhanced_features = enhanced_predictor.create_enhanced_features(
                market_data, sentiment_features
            )

            if enhanced_features.empty:
                return {"success": False, "error": "No enhanced features created"}

            logger.info("🏋️ Training News-Enhanced Lasso model...")
            training_results = enhanced_predictor.train_enhanced_model(
                enhanced_features)

            if training_results is None:
                return {"success": False, "error": "Training failed - no results returned"}

            # Get the new R² score
            new_r2_score = training_results.get(
                'enhanced_lasso_model', {}).get('r2', None)

            logger.info("💾 Saving enhanced model...")
            model_path = models_dir / "enhanced_lasso_gold_model.pkl"
            enhanced_predictor.save_enhanced_model(str(model_path))

            # Reload the model in the prediction service
            logger.info("🔄 Reloading enhanced model in prediction service...")
            if prediction_service.news_enhanced_predictor is not None:
                prediction_service.news_enhanced_predictor.load_enhanced_model(
                    str(model_path))
                logger.info("✅ Enhanced model reloaded successfully")

            return {
                "success": True,
                "r2_score": round(new_r2_score, 4) if new_r2_score else None,
                "fallback_r2_score": round(fallback_r2, 4) if fallback_r2 else None,
                "features_selected": len(enhanced_predictor.selected_features),
                "total_features": len(enhanced_predictor.feature_columns),
                "models_trained": ["News-Enhanced Lasso", "Lasso Regression (Fallback)"],
                "trained_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during model retraining: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # Run the synchronous training in a thread pool
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(executor, _retrain_sync)

    return result
