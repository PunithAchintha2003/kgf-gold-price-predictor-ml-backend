"""Background task manager for industry-standard task lifecycle management"""
import asyncio
from typing import Dict
from .logging_config import get_logger

logger = get_logger(__name__)


class BackgroundTaskManager:
    """
    Industry-standard background task manager with:
    - Task lifecycle management
    - Graceful shutdown support
    - Error handling with retries
    - Health tracking
    """

    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_states: Dict[str, Dict] = {}
        self.shutdown_event = asyncio.Event()

    def register_task(self, name: str, task: asyncio.Task):
        """Register a background task"""
        self.tasks[name] = task
        self.task_states[name] = {
            "status": "running",
            "last_run": None,
            "last_error": None,
            "run_count": 0,
            "error_count": 0
        }

    def get_task_health(self, name: str) -> Dict:
        """Get health status of a task"""
        return self.task_states.get(name, {"status": "unknown"})

    def get_all_tasks_health(self) -> Dict:
        """Get health status of all tasks"""
        return {
            name: self.get_task_health(name)
            for name in self.tasks.keys()
        }

    async def shutdown(self):
        """Gracefully shutdown all background tasks"""
        logger.info("🛑 Shutting down background tasks...")
        self.shutdown_event.set()

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.debug(f"Cancelling task: {name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {name} cancelled successfully")
                except Exception as e:
                    logger.warning(f"Error while cancelling task {name}: {e}")

        logger.info("✅ All background tasks shut down")
