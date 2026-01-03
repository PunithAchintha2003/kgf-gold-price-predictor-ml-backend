"""Dependency injection for FastAPI routes with proper type hints and lifecycle management"""
from typing import Optional, Annotated
from functools import lru_cache

from fastapi import Depends

# Type imports for better IDE support and type checking
from ..services.market_data_service import MarketDataService
from ..services.prediction_service import PredictionService
from ..services.exchange_service import ExchangeService
from ..repositories.prediction_repository import PredictionRepository

# Import spot trading service
# Note: spot_trade is in backend/spot_trade, so we need to add backend to path
import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent  # backend/app/core -> backend
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from spot_trade.service import SpotTradingService

# Global service instances - will be set during app initialization
_market_data_service: Optional[MarketDataService] = None
_prediction_service: Optional[PredictionService] = None
_prediction_repo: Optional[PredictionRepository] = None
_exchange_service: Optional[ExchangeService] = None
_spot_trading_service: Optional[SpotTradingService] = None


def set_services(
    market_data_service: Optional[MarketDataService] = None,
    prediction_service: Optional[PredictionService] = None,
    prediction_repo: Optional[PredictionRepository] = None,
    exchange_service: Optional[ExchangeService] = None,
    spot_trading_service: Optional[SpotTradingService] = None
) -> None:
    """Set global service instances during application startup
    
    Args:
        market_data_service: Market data service instance
        prediction_service: Prediction service instance
        prediction_repo: Prediction repository instance
        exchange_service: Exchange service instance
        spot_trading_service: Spot trading service instance
    """
    global _market_data_service, _prediction_service, _prediction_repo, _exchange_service, _spot_trading_service
    _market_data_service = market_data_service
    _prediction_service = prediction_service
    _prediction_repo = prediction_repo
    _exchange_service = exchange_service
    _spot_trading_service = spot_trading_service


async def get_market_data_service() -> MarketDataService:
    """Get market data service instance
    
    Returns:
        MarketDataService instance
        
    Raises:
        RuntimeError: If service is not initialized
    """
    if _market_data_service is None:
        raise RuntimeError("Market data service not initialized")
    return _market_data_service


async def get_prediction_service() -> PredictionService:
    """Get prediction service instance
    
    Returns:
        PredictionService instance
        
    Raises:
        RuntimeError: If service is not initialized
    """
    if _prediction_service is None:
        raise RuntimeError("Prediction service not initialized")
    return _prediction_service


async def get_prediction_repo() -> PredictionRepository:
    """Get prediction repository instance
    
    Returns:
        PredictionRepository instance
        
    Raises:
        RuntimeError: If repository is not initialized
    """
    if _prediction_repo is None:
        raise RuntimeError("Prediction repository not initialized")
    return _prediction_repo


async def get_exchange_service() -> ExchangeService:
    """Get exchange service instance
    
    Returns:
        ExchangeService instance
        
    Raises:
        RuntimeError: If service is not initialized
    """
    if _exchange_service is None:
        raise RuntimeError("Exchange service not initialized")
    return _exchange_service


async def get_spot_trading_service() -> SpotTradingService:
    """Get spot trading service instance
    
    Returns:
        SpotTradingService instance
        
    Raises:
        RuntimeError: If service is not initialized
    """
    if _spot_trading_service is None:
        raise RuntimeError("Spot trading service not initialized")
    return _spot_trading_service


# Type-annotated dependencies for cleaner route definitions
MarketDataServiceDep = Annotated[MarketDataService, Depends(get_market_data_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
PredictionRepoDep = Annotated[PredictionRepository, Depends(get_prediction_repo)]
ExchangeServiceDep = Annotated[ExchangeService, Depends(get_exchange_service)]
SpotTradingServiceDep = Annotated[SpotTradingService, Depends(get_spot_trading_service)]
