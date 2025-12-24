"""AI Configuration"""
import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


class AIConfig:
    """Configuration for AI services"""

    def __init__(self):
        # Gemini API Configuration
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

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
