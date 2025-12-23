"""WebSocket connection manager"""
from typing import List
from fastapi import WebSocket
from .logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time data broadcasting"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(
            f"WebSocket client connected ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.debug(
            f"WebSocket client disconnected ({len(self.active_connections)} total)")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSocket clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")

    async def disconnect_all(self):
        """Disconnect all active WebSocket connections (graceful shutdown)"""
        disconnected_count = 0
        for connection in list(self.active_connections):  # Create copy to iterate safely
            try:
                await connection.close()
                self.active_connections.remove(connection)
                disconnected_count += 1
            except asyncio.CancelledError:
                # Expected during shutdown, just remove from list
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
                # Don't re-raise - allow graceful shutdown
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")
                # Remove from list even if close failed
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
        
        if disconnected_count > 0:
            logger.info(f"Disconnected {disconnected_count} WebSocket client(s)")
