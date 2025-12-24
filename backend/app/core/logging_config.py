"""
Industry-standard logging configuration
Follows best practices for production applications:
- Structured logging with consistent format
- Proper log levels and filtering
- Request/response correlation IDs
- Performance metrics
"""
import logging
import sys
from typing import Optional
from datetime import datetime
from .config import settings


class CancelledErrorFilter(logging.Filter):
    """
    Filter to suppress CancelledError exceptions during shutdown.
    These are expected when the server is interrupted (Ctrl+C) and shouldn't be logged as errors.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress CancelledError from Starlette/uvicorn during shutdown
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type and exc_type.__name__ == 'CancelledError':
                # Check if it's from Starlette/uvicorn lifespan handling (expected during shutdown)
                pathname_lower = str(record.pathname).lower()
                message_lower = str(record.getMessage()).lower()
                if ('starlette' in pathname_lower or 
                    'uvicorn' in pathname_lower or
                    'lifespan' in pathname_lower or
                    'lifespan' in message_lower or
                    'receive_queue' in message_lower):
                    return False  # Suppress this log record
        
        # Also check message content for CancelledError patterns
        message_lower = str(record.getMessage()).lower()
        if ('cancellederror' in message_lower and 
            ('lifespan' in message_lower or 'receive' in message_lower)):
            return False  # Suppress this log record
            
        return True  # Allow all other log records


class StructuredFormatter(logging.Formatter):
    """Industry-standard structured log formatter"""
    
    def __init__(self, include_timestamp: bool = True, include_module: bool = True):
        self.include_timestamp = include_timestamp
        self.include_module = include_module
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured output"""
        # Build log message components
        parts = []
        
        # Timestamp (ISO 8601 format - industry standard)
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            parts.append(f"[{timestamp}]")
        
        # Log level (padded for alignment)
        level = f"{record.levelname:8s}"
        parts.append(level)
        
        # Module name (optional, useful for debugging)
        if self.include_module and record.name != 'root':
            module = record.name.split('.')[-1]
            parts.append(f"[{module:20s}]")
        
        # Message
        parts.append(record.getMessage())
        
        # Exception info (if present)
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        
        return " ".join(parts)


def setup_logging():
    """
    Setup industry-standard application logging
    
    Features:
    - Structured logging format
    - Environment-based log levels
    - Proper third-party library filtering
    - Performance-optimized formatting
    """
    import warnings

    # Get log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.WARNING)

    # Suppress SyntaxWarnings from third-party libraries (e.g., textblob)
    warnings.filterwarnings(
        'ignore', category=SyntaxWarning, module='textblob')

    # Industry-standard log format
    # Format: [TIMESTAMP] LEVEL    [MODULE] Message
    formatter = StructuredFormatter(
        include_timestamp=(log_level <= logging.INFO),
        include_module=(log_level <= logging.DEBUG)
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    # Add filter to suppress expected CancelledError during shutdown
    console_handler.addFilter(CancelledErrorFilter())
    root_logger.addHandler(console_handler)

    # Configure third-party loggers
    # Suppress verbose uvicorn access logs (HTTP requests) - only show errors
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(logging.WARNING)
    uvicorn_access_logger.addFilter(CancelledErrorFilter())
    
    # Suppress uvicorn startup/shutdown messages (we'll show our own)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.WARNING)
    uvicorn_logger.addFilter(CancelledErrorFilter())
    
    # Suppress Starlette lifespan CancelledError during shutdown
    starlette_logger = logging.getLogger("starlette")
    starlette_logger.addFilter(CancelledErrorFilter())
    
    # Suppress noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    # Log configuration at startup (INFO level or below)
    if log_level <= logging.INFO:
        logger.info(f"Logging configured: level={settings.log_level.upper()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with industry-standard configuration
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
