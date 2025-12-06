"""Background tasks for the application"""
import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING

from .config import settings
from .logging_config import get_logger
from .task_manager import BackgroundTaskManager

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService
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
                daily_data = market_data_service.get_daily_data()

                # Skip broadcasting if rate limited (to avoid spamming clients)
                if daily_data.get('status') == 'rate_limited':
                    wait_seconds = daily_data.get(
                        'rate_limit_info', {}).get('wait_seconds', 60)
                    logger.debug(
                        f"Skipping broadcast - rate limited. Waiting {wait_seconds}s")
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
                elif daily_data != last_broadcast_data:
                    await manager.broadcast(json.dumps(daily_data))
                    last_broadcast_data = daily_data

            # Wait with cancellation support
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
                # Get daily data to check rate limit status
                daily_data_check = market_data_service.get_daily_data()
                if daily_data_check.get('status') == 'rate_limited':
                    wait_seconds = daily_data_check.get(
                        'rate_limit_info', {}).get('wait_seconds', 60)
                    logger.info(
                        f"⏸️ Skipping auto-update - rate limited. Will retry after {wait_seconds}s")
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
                            logger.info(
                                f"⏸️ Rate limited during update. Skipping this cycle. Will retry after {wait_seconds}s")
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
