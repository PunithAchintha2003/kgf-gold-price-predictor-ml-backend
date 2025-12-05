"""Dependency injection for FastAPI routes"""
from typing import Optional

# Global service instances - will be set during app initialization
_market_data_service: Optional[object] = None
_prediction_service: Optional[object] = None
_prediction_repo: Optional[object] = None
_exchange_service: Optional[object] = None


def set_services(
    market_data_service=None,
    prediction_service=None,
    prediction_repo=None,
    exchange_service=None
):
    """Set global service instances"""
    global _market_data_service, _prediction_service, _prediction_repo, _exchange_service
    _market_data_service = market_data_service
    _prediction_service = prediction_service
    _prediction_repo = prediction_repo
    _exchange_service = exchange_service


def get_market_data_service():
    """Get market data service instance"""
    if _market_data_service is None:
        raise RuntimeError("Market data service not initialized")
    return _market_data_service


def get_prediction_service():
    """Get prediction service instance"""
    if _prediction_service is None:
        raise RuntimeError("Prediction service not initialized")
    return _prediction_service


def get_prediction_repo():
    """Get prediction repository instance"""
    if _prediction_repo is None:
        raise RuntimeError("Prediction repository not initialized")
    return _prediction_repo


def get_exchange_service():
    """Get exchange service instance"""
    if _exchange_service is None:
        raise RuntimeError("Exchange service not initialized")
    return _exchange_service
