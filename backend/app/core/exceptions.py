"""Custom exception classes for better error handling with comprehensive type hints"""
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)


class BaseAPIException(HTTPException):
    """Base exception for API errors with enhanced error handling"""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "status": "error",
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "details": self.details
        }


class ValidationError(BaseAPIException):
    """Validation error (400)"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(BaseAPIException):
    """Resource not found (404)"""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            error_code="NOT_FOUND",
            details=details
        )


class InternalServerError(BaseAPIException):
    """Internal server error (500)"""
    
    def __init__(self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="INTERNAL_ERROR",
            details=details
        )


class ServiceUnavailableError(BaseAPIException):
    """Service unavailable (503)"""
    
    def __init__(self, message: str = "Service temporarily unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details=details
        )


class DatabaseError(BaseAPIException):
    """Database operation error (500)"""
    
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )


class RateLimitError(BaseAPIException):
    """Rate limit exceeded (429)"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """Global exception handler for BaseAPIException"""
    logger.error(
        f"API Error: {exc.error_code} - {exc.message}",
        extra={"error_code": exc.error_code, "details": exc.details}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Global exception handler for Pydantic validation errors"""
    errors = exc.errors()
    logger.warning(f"Validation error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.now().isoformat(),
            "details": {"errors": errors}
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions"""
    logger.exception(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    
    # Check if settings are available
    is_development = False
    if hasattr(request.app.state, 'settings'):
        is_development = request.app.state.settings.environment == "development"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat(),
            "details": (
                {"error": str(exc)} 
                if is_development
                else {}
            )
        }
    )

