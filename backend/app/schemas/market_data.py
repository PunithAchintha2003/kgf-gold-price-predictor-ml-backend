"""Market data schemas"""
from pydantic import BaseModel
from typing import Optional, List
from .prediction import Prediction, PredictionHistory, AccuracyStats


class DailyDataPoint(BaseModel):
    """Daily market data point"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    predicted_price: Optional[float] = None
    actual_price: Optional[float] = None


class DailyDataResponse(BaseModel):
    """Daily data response"""
    symbol: str
    timeframe: str
    data: List[DailyDataPoint]
    historical_predictions: List[dict]
    accuracy_stats: AccuracyStats
    current_price: float
    prediction: Optional[Prediction] = None
    timestamp: str
    status: str
    message: Optional[str] = None


class RealtimePriceResponse(BaseModel):
    """Real-time price response"""
    symbol: str
    current_price: float
    timestamp: str
    status: str
    message: Optional[str] = None



