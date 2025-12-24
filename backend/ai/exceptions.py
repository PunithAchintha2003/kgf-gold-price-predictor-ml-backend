"""AI Service Exceptions"""


class AIServiceError(Exception):
    """Base exception for AI services"""
    pass


class GeminiAPIError(AIServiceError):
    """Exception for Gemini API errors"""
    pass


class ConfigurationError(AIServiceError):
    """Exception for configuration errors"""
    pass

