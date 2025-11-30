"""Logging configuration"""
import logging
from .config import settings


def setup_logging():
    """Setup application logging"""
    log_level = getattr(logging, settings.log_level.upper(), logging.WARNING)
    
    # More user-friendly log format - cleaner and less verbose
    if log_level >= logging.INFO:
        # For INFO and above, use simple format (no timestamps)
        log_format = '%(levelname)s: %(message)s'
        datefmt = None
    else:
        # For DEBUG, include more details
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        datefmt = '%H:%M:%S'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=datefmt
    )
    
    # Suppress verbose uvicorn access logs (HTTP requests)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Suppress uvicorn startup/shutdown messages (we'll show our own)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    # Only log startup message at INFO level or below
    if log_level <= logging.INFO:
        logger.info(f"Logging: {settings.log_level}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)



