"""AI Configuration"""
import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
# Use the same robust .env finding logic as the main config
def _find_env_file() -> Optional[str]:
    """Find .env file in project root or current directory"""
    try:
        # Try project root first (4 levels up from this file: backend/ai/config.py -> project root)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists() and env_path.is_file():
            return str(env_path)
        
        # Try current working directory
        env_path = Path.cwd() / ".env"
        if env_path.exists() and env_path.is_file():
            return str(env_path)
        
        # Try relative to current directory
        env_path = Path(".env")
        if env_path.exists() and env_path.is_file():
            return str(env_path.resolve())
    except Exception:
        pass
    return None

# Load .env file if it exists
# Load before AIConfig is instantiated to ensure env vars are available
_env_loaded = False
try:
    from dotenv import load_dotenv
    env_file_path = _find_env_file()
    if env_file_path:
        # Load .env file
        # First, try to load without override to respect existing env vars
        # Then, if key variables are missing, reload with override=True
        load_dotenv(env_file_path, override=False)
        _env_loaded = True
        
        # Check if critical variables are missing and reload if needed
        if not os.getenv("GEMINI_API_KEY"):
            # Reload with override to ensure .env values are used
            load_dotenv(env_file_path, override=True)
    else:
        # Try loading from current directory as fallback
        load_dotenv(override=False)
        _env_loaded = True
except ImportError:
    # python-dotenv not installed, skip .env loading
    # os.getenv() will still work for system environment variables
    pass
except Exception as e:
    # If .env loading fails, continue - os.getenv() will still work for system env vars
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Could not load .env file: {e}")
    pass


class AIConfig:
    """Configuration for AI services"""

    def __init__(self):
        # Gemini API Configuration
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        
        # Debug: Log if API key is found (without exposing the key)
        if self.gemini_api_key:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Gemini API key loaded (length: {len(self.gemini_api_key)})")
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("GEMINI_API_KEY not found in environment variables")

        # Model Configuration
        # For v1 API: "gemini-pro" or "gemini-1.5-flash"
        # For v1beta API: "gemini-1.5-pro-latest" or "gemini-1.5-flash-latest"
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.gemini_temperature: float = float(
            os.getenv("GEMINI_TEMPERATURE", "0.5"))
        self.gemini_max_tokens: int = int(
            os.getenv("GEMINI_MAX_TOKENS", "8192"))

        # API Configuration
        # v1 API is more stable and supports gemini-pro and gemini-1.5-flash
        # v1beta supports systemInstruction and latest model versions
        self.gemini_api_base: str = os.getenv(
            "GEMINI_API_BASE",
            "https://generativelanguage.googleapis.com/v1"
        )

        # Request Configuration
        self.request_timeout: int = int(
            os.getenv("GEMINI_REQUEST_TIMEOUT", "30"))
        self.max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

    def is_configured(self) -> bool:
        """Check if Gemini API is properly configured"""
        return self.gemini_api_key is not None and len(self.gemini_api_key.strip()) > 0

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.is_configured():
            return False
        if self.gemini_temperature < 0 or self.gemini_temperature > 2:
            return False
        if self.gemini_max_tokens < 1 or self.gemini_max_tokens > 8192:
            return False
        return True


# Global config instance
ai_config = AIConfig()
