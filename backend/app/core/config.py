"""Application configuration using Pydantic Settings for industry best practices"""
from pathlib import Path
from typing import Optional, List, Literal
from functools import lru_cache

from pydantic import Field, field_validator, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_env_file() -> Optional[str]:
    """Find .env file in project root or current directory"""
    try:
        # Try project root first (4 levels up from this file: backend/app/core/config.py -> project root)
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


# Determine .env file path
_env_file_path = _find_env_file()


class Settings(BaseSettings):
    """Application settings with Pydantic validation and type safety"""
    
    model_config = SettingsConfigDict(
        env_file=_env_file_path if _env_file_path else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        env_ignore_empty=True,  # Ignore empty environment variables
    )
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        description="Logging level"
    )

    # Cache settings
    cache_duration: int = Field(
        default=300,
        ge=0,
        description="Cache duration in seconds"
    )
    api_cooldown: int = Field(
        default=5,
        ge=0,
        description="API cooldown period in seconds"
    )
    realtime_cache_duration: int = Field(
        default=60,
        ge=0,
        description="Real-time cache duration in seconds"
    )
    rate_limit_initial_backoff: int = Field(
        default=60,
        ge=0,
        description="Initial rate limit backoff in seconds"
    )
    rate_limit_max_backoff: int = Field(
        default=1800,
        ge=0,
        description="Maximum rate limit backoff in seconds"
    )

    # Database settings
    use_postgresql: bool = Field(
        default=False,  # Default to False (SQLite) for better compatibility
        description="Use PostgreSQL instead of SQLite"
    )
    postgresql_host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    postgresql_port: int = Field(
        default=5432,
        gt=0,
        le=65535,
        description="PostgreSQL port"
    )
    postgresql_database: str = Field(
        default="gold_predictor",
        description="PostgreSQL database name"
    )
    postgresql_user: str = Field(
        default="postgres",
        description="PostgreSQL user"
    )
    postgresql_password: Optional[str] = Field(
        default=None,
        description="PostgreSQL password"
    )
    postgresql_pool_min_size: int = Field(
        default=5,
        ge=1,
        description="PostgreSQL connection pool minimum size"
    )
    postgresql_pool_max_size: int = Field(
        default=50,
        ge=1,
        description="PostgreSQL connection pool maximum size"
    )

    # AI Configuration
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key"
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins"
    )

    # Request size limits
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum request size in bytes"
    )

    # Background task settings
    auto_update_enabled: bool = Field(
        default=True,
        description="Enable automatic prediction updates"
    )
    auto_update_interval: int = Field(
        default=3600,
        ge=60,
        description="Auto-update interval in seconds"
    )
    auto_update_startup_delay: int = Field(
        default=60,
        ge=0,
        description="Auto-update startup delay in seconds"
    )
    auto_update_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries for auto-update"
    )
    auto_update_retry_delay: int = Field(
        default=300,
        ge=0,
        description="Auto-update retry delay in seconds"
    )
    
    # Auto-retrain settings
    auto_retrain_enabled: bool = Field(
        default=True,
        description="Enable automatic model retraining"
    )
    auto_retrain_interval: int = Field(
        default=86400,
        ge=3600,
        description="Auto-retrain interval in seconds"
    )
    auto_retrain_hour: int = Field(
        default=2,
        ge=0,
        le=23,
        description="Hour of day for auto-retrain (0-23)"
    )
    auto_retrain_startup_delay: int = Field(
        default=120,
        ge=0,
        description="Auto-retrain startup delay in seconds"
    )
    auto_retrain_min_predictions: int = Field(
        default=10,
        ge=1,
        description="Minimum predictions required before retrain"
    )
    auto_retrain_news_days: int = Field(
        default=30,
        ge=1,
        description="Days of news data for training"
    )
    
    # Auto-predict settings
    auto_predict_enabled: bool = Field(
        default=True,
        description="Enable automatic daily predictions"
    )
    auto_predict_hour: int = Field(
        default=8,
        ge=0,
        le=23,
        description="Hour of day for auto-predict (0-23)"
    )
    auto_predict_startup_delay: int = Field(
        default=60,
        ge=0,
        description="Auto-predict startup delay in seconds"
    )

    @computed_field
    @property
    def backend_dir(self) -> Path:
        """Backend directory path"""
        return Path(__file__).resolve().parent.parent.parent

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Data directory path"""
        return self.backend_dir / "data"

    @computed_field
    @property
    def db_path(self) -> str:
        """SQLite database path"""
        return str(self.data_dir / "gold_predictions.db")

    @computed_field
    @property
    def backup_db_path(self) -> str:
        """Backup database path"""
        return str(self.data_dir / "gold_predictions_backup.db")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return ["*"]

    @field_validator("postgresql_port", mode="before")
    @classmethod
    def parse_port(cls, v):
        """Parse port as integer"""
        if isinstance(v, str):
            return int(v)
        return v

    @field_validator("use_postgresql", mode="before")
    @classmethod
    def parse_bool(cls, v):
        """Parse boolean from string"""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("environment", mode="after")
    @classmethod
    def validate_environment(cls, v, info):
        """Validate environment-specific settings"""
        if v == "production":
            # Production-specific validations
            if info.data.get("cors_origins") == ["*"]:
                raise ValueError(
                    "CORS_ORIGINS should not be '*' in production. "
                    "Set specific origins via CORS_ORIGINS environment variable."
                )
            if info.data.get("log_level") == "DEBUG":
                raise ValueError("LOG_LEVEL should not be DEBUG in production")
        
        # Set default CORS based on environment
        if v == "production" and not info.data.get("cors_origins"):
            info.data["cors_origins"] = []
        elif v != "production" and not info.data.get("cors_origins"):
            info.data["cors_origins"] = ["*"]
        
        return v

    @model_validator(mode="after")
    def validate_postgresql_settings(self):
        """Validate PostgreSQL settings if enabled"""
        if self.use_postgresql:
            # Check if password is set (not None and not empty string)
            if not self.postgresql_password or (isinstance(self.postgresql_password, str) and not self.postgresql_password.strip()):
                raise ValueError(
                    "POSTGRESQL_PASSWORD is required when USE_POSTGRESQL=true. "
                    "Please set POSTGRESQL_PASSWORD in your .env file or environment variables."
                )
            # Check if host is set
            if not self.postgresql_host or (isinstance(self.postgresql_host, str) and not self.postgresql_host.strip()):
                raise ValueError("POSTGRESQL_HOST cannot be empty when USE_POSTGRESQL=true")
            # Check if database is set
            if not self.postgresql_database or (isinstance(self.postgresql_database, str) and not self.postgresql_database.strip()):
                raise ValueError("POSTGRESQL_DATABASE cannot be empty when USE_POSTGRESQL=true")
        return self

    def __init__(self, **kwargs):
        """Initialize settings and ensure data directory exists"""
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)"""
    return Settings()


# Global settings instance for backward compatibility
settings = get_settings()
