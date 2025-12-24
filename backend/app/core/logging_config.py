"""
Futuristic, user-friendly logging configuration with emojis and colors
Industry-standard structured logging with enhanced visual formatting
"""
import logging
import sys
import os
from typing import Optional
from datetime import datetime
from .config import settings


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful terminal output"""
    # Reset
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


# Emoji mapping for log levels and operations
class Emojis:
    """Emojis for different log types"""
    # Log levels
    DEBUG = "🔍"
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    
    # Operations
    STARTUP = "🚀"
    SHUTDOWN = "🛑"
    DATABASE = "💾"
    API = "🌐"
    CACHE = "⚡"
    MODEL = "🤖"
    PREDICTION = "📊"
    PERFORMANCE = "⚡"
    SUCCESS = "✅"
    FAILURE = "❌"
    LOADING = "⏳"
    COMPLETE = "✨"
    CONNECTION = "🔌"
    REQUEST = "📥"
    RESPONSE = "📤"
    SECURITY = "🔒"
    CONFIG = "⚙️"


def should_use_colors() -> bool:
    """Check if terminal supports colors"""
    # Check if we're in a TTY and not in CI
    if not sys.stdout.isatty():
        return False
    # Check for common CI environment variables
    ci_envs = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS']
    if any(os.getenv(env) for env in ci_envs):
        return False
    return True


USE_COLORS = should_use_colors()


class CancelledErrorFilter(logging.Filter):
    """
    Filter to suppress CancelledError exceptions during shutdown.
    These are expected when the server is interrupted (Ctrl+C) and shouldn't be logged as errors.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type and exc_type.__name__ == 'CancelledError':
                pathname_lower = str(record.pathname).lower()
                message_lower = str(record.getMessage()).lower()
                if ('starlette' in pathname_lower or 
                    'uvicorn' in pathname_lower or
                    'lifespan' in pathname_lower or
                    'lifespan' in message_lower or
                    'receive_queue' in message_lower):
                    return False
        message_lower = str(record.getMessage()).lower()
        if ('cancellederror' in message_lower and 
            ('lifespan' in message_lower or 'receive' in message_lower)):
            return False
        return True


class FuturisticFormatter(logging.Formatter):
    """
    Futuristic, user-friendly log formatter with emojis and colors
    Industry-standard structured logging with enhanced visual formatting
    """
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        'DEBUG': Colors.DIM + Colors.CYAN,
        'INFO': Colors.BRIGHT_BLUE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BOLD + Colors.BRIGHT_RED + Colors.BG_YELLOW,
    }
    
    # Emoji mapping for log levels
    LEVEL_EMOJIS = {
        'DEBUG': Emojis.DEBUG,
        'INFO': Emojis.INFO,
        'WARNING': Emojis.WARNING,
        'ERROR': Emojis.ERROR,
        'CRITICAL': Emojis.CRITICAL,
    }
    
    def __init__(self, include_timestamp: bool = True, include_module: bool = True):
        self.include_timestamp = include_timestamp
        self.include_module = include_module
        super().__init__()
    
    def _get_color(self, level: str) -> str:
        """Get color code for log level"""
        if not USE_COLORS:
            return ""
        return self.LEVEL_COLORS.get(level, Colors.RESET)
    
    def _get_emoji(self, level: str, message: str) -> str:
        """Get emoji for log level, with smart detection for specific operations"""
        # Check message for specific operation emojis
        msg_lower = message.lower()
        
        if 'startup' in msg_lower or 'starting' in msg_lower or 'initializing' in msg_lower:
            return Emojis.STARTUP
        elif 'shutdown' in msg_lower or 'stopping' in msg_lower:
            return Emojis.SHUTDOWN
        elif 'database' in msg_lower or 'db' in msg_lower or 'sql' in msg_lower:
            return Emojis.DATABASE
        elif 'api' in msg_lower or 'endpoint' in msg_lower or 'route' in msg_lower:
            return Emojis.API
        elif 'cache' in msg_lower or 'cached' in msg_lower:
            return Emojis.CACHE
        elif 'model' in msg_lower or 'ml' in msg_lower or 'prediction' in msg_lower:
            return Emojis.MODEL
        elif 'prediction' in msg_lower or 'predict' in msg_lower:
            return Emojis.PREDICTION
        elif 'performance' in msg_lower or 'slow' in msg_lower or 'took' in msg_lower:
            return Emojis.PERFORMANCE
        elif 'success' in msg_lower or 'saved' in msg_lower or 'loaded' in msg_lower or '✅' in message:
            return Emojis.SUCCESS
        elif 'error' in msg_lower or 'failed' in msg_lower or '❌' in message:
            return Emojis.ERROR
        elif 'connection' in msg_lower or 'connect' in msg_lower:
            return Emojis.CONNECTION
        elif 'request' in msg_lower:
            return Emojis.REQUEST
        elif 'response' in msg_lower:
            return Emojis.RESPONSE
        elif 'security' in msg_lower or 'auth' in msg_lower:
            return Emojis.SECURITY
        elif 'config' in msg_lower or 'setting' in msg_lower:
            return Emojis.CONFIG
        
        # Default to level emoji
        return self.LEVEL_EMOJIS.get(level, "📝")
    
    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """Format timestamp in a user-friendly way"""
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%H:%M:%S")
    
    def _format_module(self, record: logging.LogRecord) -> str:
        """Format module name in a user-friendly way"""
        if record.name == 'root':
            return ""
        # Get the last part of the module name
        parts = record.name.split('.')
        module = parts[-1]
        # Truncate if too long
        if len(module) > 25:
            module = module[:22] + "..."
        return f"{Colors.DIM}[{module}]{Colors.RESET}" if USE_COLORS else f"[{module}]"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with futuristic, user-friendly output"""
        # Get emoji and color
        emoji = self._get_emoji(record.levelname, record.getMessage())
        color = self._get_color(record.levelname)
        reset = Colors.RESET if USE_COLORS else ""
        
        # Build log message components
        parts = []
        
        # Timestamp (compact format)
        if self.include_timestamp:
            timestamp = self._format_timestamp(record)
            parts.append(f"{Colors.DIM}{timestamp}{reset}")
        
        # Emoji + Level (colored and bold)
        level_name = record.levelname
        if level_name == "WARNING":
            level_display = "WARN"
        elif level_name == "CRITICAL":
            level_display = "CRIT"
        else:
            level_display = level_name
        
        level_str = f"{color}{Colors.BOLD}{level_display:5s}{reset}"
        parts.append(f"{emoji} {level_str}")
        
        # Module name (dimmed)
        if self.include_module and record.name != 'root':
            module_str = self._format_module(record)
            if module_str:
                parts.append(module_str)
        
        # Message (with smart formatting)
        message = record.getMessage()
        
        # Format performance metrics (e.g., "took 5.23s")
        if "took" in message.lower() and "s" in message:
            # Highlight performance metrics
            if USE_COLORS:
                import re
                # Find "took X.XXs" pattern and highlight it
                message = re.sub(
                    r'(took\s+)([\d.]+)(s)',
                    rf'\1{Colors.BRIGHT_CYAN}{Colors.BOLD}\2\3{reset}',
                    message,
                    flags=re.IGNORECASE
                )
        
        parts.append(message)
        
        # Exception info (if present) - formatted nicely
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            # Indent exception traceback
            exc_lines = exc_text.split('\n')
            formatted_exc = '\n'.join([f"{Colors.DIM}  {line}{reset}" for line in exc_lines])
            parts.append(f"\n{formatted_exc}")
        
        return " ".join(parts)


def setup_logging():
    """
    Setup futuristic, user-friendly application logging
    
    Features:
    - Beautiful color-coded output
    - Emoji indicators for different operations
    - Structured logging format
    - Environment-based log levels
    - Performance-optimized formatting
    """
    import warnings

    # Get log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.WARNING)

    # Suppress SyntaxWarnings from third-party libraries
    warnings.filterwarnings('ignore', category=SyntaxWarning, module='textblob')

    # Futuristic log format
    formatter = FuturisticFormatter(
        include_timestamp=(log_level <= logging.INFO),
        include_module=(log_level <= logging.DEBUG)
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with futuristic formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CancelledErrorFilter())
    root_logger.addHandler(console_handler)

    # Configure third-party loggers
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(logging.WARNING)
    uvicorn_access_logger.addFilter(CancelledErrorFilter())
    
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.WARNING)
    uvicorn_logger.addFilter(CancelledErrorFilter())
    
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
        logger.info(f"{Emojis.CONFIG} Logging configured: level={settings.log_level.upper()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with futuristic, user-friendly configuration
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience functions for common log patterns
def log_performance(logger: logging.Logger, operation: str, duration: float, threshold: float = 1.0):
    """Log performance metrics in a user-friendly way"""
    if duration > threshold:
        logger.warning(
            f"{Emojis.PERFORMANCE} {operation} took {duration:.4f}s "
            f"(threshold: {threshold:.1f}s)"
        )
    else:
        logger.debug(f"{Emojis.PERFORMANCE} {operation} took {duration:.4f}s")


def log_request(logger: logging.Logger, method: str, path: str, status: int, duration: float):
    """Log HTTP requests in a user-friendly way (only for very slow requests)"""
    status_emoji = Emojis.SUCCESS if 200 <= status < 300 else Emojis.WARNING if 300 <= status < 400 else Emojis.ERROR
    # Only log at INFO level if request is very slow (>5s), otherwise debug
    if duration > 5.0:
        logger.info(
            f"{status_emoji} {method} {path} → {status} ({duration:.3f}s)"
        )
    else:
        logger.debug(
            f"{status_emoji} {method} {path} → {status} ({duration:.3f}s)"
        )


def log_database_operation(logger: logging.Logger, operation: str, success: bool = True):
    """Log database operations in a user-friendly way"""
    emoji = Emojis.SUCCESS if success else Emojis.ERROR
    status = "completed" if success else "failed"
    logger.debug(f"{emoji} {Emojis.DATABASE} {operation} {status}")


def log_cache_operation(logger: logging.Logger, operation: str, hit: bool = True):
    """Log cache operations in a user-friendly way"""
    status = "HIT" if hit else "MISS"
    emoji = Emojis.SUCCESS if hit else Emojis.LOADING
    logger.debug(f"{emoji} {Emojis.CACHE} Cache {status}: {operation}")
