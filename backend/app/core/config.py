"""Application configuration"""
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


class Settings:
    """Application settings"""

    def __init__(self):
        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.log_level: str = os.getenv("LOG_LEVEL", "WARNING")

        # Cache settings
        self.cache_duration: int = int(os.getenv("CACHE_DURATION", "300"))
        self.api_cooldown: int = int(os.getenv("API_COOLDOWN", "2"))
        self.realtime_cache_duration: int = int(
            os.getenv("REALTIME_CACHE_DURATION", "60"))

        # Database settings
        self.use_postgresql: bool = os.getenv(
            "USE_POSTGRESQL", "true").lower() == "true"
        self.postgresql_host: str = os.getenv("POSTGRESQL_HOST", "localhost")
        self.postgresql_port: str = os.getenv("POSTGRESQL_PORT", "5432")
        self.postgresql_database: str = os.getenv(
            "POSTGRESQL_DATABASE", "gold_predictor")
        self.postgresql_user: str = os.getenv("POSTGRESQL_USER", "postgres")
        self.postgresql_password: Optional[str] = os.getenv(
            "POSTGRESQL_PASSWORD")

        # Paths
        self.backend_dir: Path = Path(__file__).resolve().parent.parent.parent
        self.data_dir: Path = self.backend_dir / "data"
        self.db_path: str = str(self.data_dir / "gold_predictions.db")
        self.backup_db_path: str = str(
            self.data_dir / "gold_predictions_backup.db")

        # CORS - Allow specific origins or all if not specified
        cors_origins_env = os.getenv("CORS_ORIGINS")
        if cors_origins_env:
            # Parse comma-separated list from environment variable
            self.cors_origins: list = [origin.strip() for origin in cors_origins_env.split(",")]
        else:
            # Default: Allow all origins (for development)
            # In production, set CORS_ORIGINS environment variable
            self.cors_origins: list = ["*"]


# Ensure data directory exists
settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
