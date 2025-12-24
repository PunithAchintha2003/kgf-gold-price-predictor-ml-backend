"""Custom middleware for security, compression, and monitoring"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import gzip
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_metrics_collector = None

def _get_metrics_collector():
    """Lazy import of metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        from .metrics import metrics_collector as mc
        _metrics_collector = mc
    return _metrics_collector


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS (only in production with HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy (adjust based on your needs)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
        
        # Remove server header (security through obscurity)
        # MutableHeaders doesn't have pop(), so we set it to empty string or check first
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Compress responses using gzip"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Only compress if client accepts gzip and response is successful
        accept_encoding = request.headers.get("Accept-Encoding", "")
        
        if "gzip" in accept_encoding and 200 <= response.status_code < 300:
            # Skip compression for streaming responses or already compressed
            if response.headers.get("Content-Encoding"):
                return response
            
            # Get response body
            try:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Only compress if body is larger than 1KB
                if len(body) > 1024:
                    compressed_body = gzip.compress(body, compresslevel=6)
                    response = Response(
                        content=compressed_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                    response.headers["Content-Encoding"] = "gzip"
                    response.headers["Content-Length"] = str(len(compressed_body))
                else:
                    response = Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
            except Exception as e:
                # If compression fails, log and return original response
                logger.debug(f"Compression failed: {e}")
                # Re-raise to let FastAPI handle it
                return response
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS attacks"""
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("Content-Length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    return Response(
                        content='{"error": "Request body too large"}',
                        status_code=413,
                        media_type="application/json"
                    )
            except ValueError:
                pass
        
        response = await call_next(request)
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add timing headers and log request processing time"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Record metrics
        try:
            metrics_collector = _get_metrics_collector()
            metrics_collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=process_time
            )
        except Exception as e:
            logger.debug(f"Failed to record metrics: {e}")
        
        # Log requests with user-friendly formatting (only log, no slow request warnings)
        from .logging_config import Emojis, log_request
        
        # Log requests at debug level only (reduce noise)
        # Suppress all request logging to reduce verbosity
        logger.debug(
            f"{Emojis.REQUEST} {request.method} {request.url.path} → {response.status_code} ({process_time:.3f}s)"
        )
        
        return response

