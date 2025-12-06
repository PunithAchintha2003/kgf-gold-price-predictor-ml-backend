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
        self.api_cooldown: int = int(os.getenv("API_COOLDOWN", "5"))  # Increased from 2 to 5 seconds
        self.realtime_cache_duration: int = int(
            os.getenv("REALTIME_CACHE_DURATION", "60"))
        self.rate_limit_initial_backoff: int = int(
            os.getenv("RATE_LIMIT_INITIAL_BACKOFF", "60"))  # Initial backoff: 60 seconds

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
            self.cors_origins: list = [origin.strip()
                                       for origin in cors_origins_env.split(",")]
        else:
            # Default behavior based on environment
            env = os.getenv("ENVIRONMENT", "development")
            if env == "production":
                # In production, default to empty list (no CORS) for security
                # Set CORS_ORIGINS environment variable to allow specific origins
                self.cors_origins: list = []
            else:
                # In development/staging, allow all origins by default
                self.cors_origins: list = ["*"]

        # Request size limits
        self.max_request_size: int = int(
            # 10MB default
            os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))

        # Background task settings
        self.auto_update_enabled: bool = os.getenv(
            "AUTO_UPDATE_ENABLED", "true").lower() == "true"
        self.auto_update_interval: int = int(
            os.getenv("AUTO_UPDATE_INTERVAL", "3600"))  # Default: 1 hour
        self.auto_update_startup_delay: int = int(
            os.getenv("AUTO_UPDATE_STARTUP_DELAY", "60"))  # Default: 1 minute
        self.auto_update_max_retries: int = int(
            os.getenv("AUTO_UPDATE_MAX_RETRIES", "3"))
        self.auto_update_retry_delay: int = int(
            os.getenv("AUTO_UPDATE_RETRY_DELAY", "300"))  # 5 minutes

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration settings"""
        errors = []

        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            errors.append(
                f"Invalid environment: {self.environment}. Must be development, staging, or production")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")

        # Validate cache duration (must be positive)
        if self.cache_duration < 0:
            errors.append(
                f"cache_duration must be non-negative, got {self.cache_duration}")

        # Validate PostgreSQL settings if using PostgreSQL
        if self.use_postgresql:
            if not self.postgresql_password:
                errors.append(
                    "POSTGRESQL_PASSWORD is required when USE_POSTGRESQL=true")
            if not self.postgresql_host:
                errors.append("POSTGRESQL_HOST cannot be empty")
            if not self.postgresql_database:
                errors.append("POSTGRESQL_DATABASE cannot be empty")

        # Validate request size limit
        if self.max_request_size < 1024:  # At least 1KB
            errors.append(
                f"MAX_REQUEST_SIZE must be at least 1024 bytes, got {self.max_request_size}")

        # Validate background task settings
        if self.auto_update_interval < 60:  # At least 1 minute
            errors.append(
                f"AUTO_UPDATE_INTERVAL must be at least 60 seconds, got {self.auto_update_interval}")
        if self.auto_update_startup_delay < 0:
            errors.append(
                f"AUTO_UPDATE_STARTUP_DELAY must be non-negative, got {self.auto_update_startup_delay}")
        if self.auto_update_max_retries < 0:
            errors.append(
                f"AUTO_UPDATE_MAX_RETRIES must be non-negative, got {self.auto_update_max_retries}")

        # Warn about production settings
        if self.environment == "production":
            if "*" in self.cors_origins:
                errors.append(
                    "CORS_ORIGINS should not be '*' in production. "
                    "Set specific origins via CORS_ORIGINS environment variable (e.g., 'https://yourdomain.com,https://www.yourdomain.com'). "
                    "If no CORS is needed, leave CORS_ORIGINS unset (defaults to no CORS)."
                )
            if self.log_level.upper() == "DEBUG":
                errors.append("LOG_LEVEL should not be DEBUG in production")

        if errors:
            error_msg = "Configuration validation failed:\n" + \
                "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)


# Ensure data directory exists
settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
