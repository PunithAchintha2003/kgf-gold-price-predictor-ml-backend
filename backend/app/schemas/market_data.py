"""Market data schemas with comprehensive validation"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from .prediction import Prediction, PredictionHistory, AccuracyStats, ModelInfo


class DailyDataPoint(BaseModel):
    """Daily market data point with validation"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    predicted_price: Optional[float] = Field(None, gt=0, description="Predicted price")
    actual_price: Optional[float] = Field(None, gt=0, description="Actual price")
    
    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        """Validate high is >= open and close"""
        if "open" in info.data and v < info.data["open"]:
            raise ValueError("High must be >= open")
        if "close" in info.data and v < info.data["close"]:
            raise ValueError("High must be >= close")
        return v
    
    @field_validator("low")
    @classmethod
    def validate_low(cls, v: float, info) -> float:
        """Validate low is <= open and close"""
        if "open" in info.data and v > info.data["open"]:
            raise ValueError("Low must be <= open")
        if "close" in info.data and v > info.data["close"]:
            raise ValueError("Low must be <= close")
        return v


class DailyDataResponse(BaseModel):
    """Daily data response with comprehensive information"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    data: List[DailyDataPoint] = Field(..., description="Daily data points")
    historical_predictions: List[dict] = Field(default_factory=list, description="Historical predictions")
    accuracy_stats: Optional[AccuracyStats] = Field(None, description="Accuracy statistics")
    model_info: Optional[ModelInfo] = Field(None, description="Model information")
    current_price: float = Field(..., gt=0, description="Current market price")
    prediction: Optional[Prediction] = Field(None, description="Next day prediction")
    timestamp: str = Field(..., description="Response timestamp")
    status: Literal["success", "error", "rate_limited"] = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Optional message")
    rate_limit_info: Optional[dict] = Field(None, description="Rate limit information if applicable")


class RealtimePriceResponse(BaseModel):
    """Real-time price response with validation"""
    symbol: str = Field(..., description="Trading symbol")
    current_price: float = Field(..., gt=0, description="Current market price")
    price_change: Optional[float] = Field(None, description="Price change from previous close")
    change_percentage: Optional[float] = Field(None, description="Price change percentage")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    timestamp: str = Field(..., description="Response timestamp")
    status: Literal["success", "error", "rate_limited"] = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Optional message")
    rate_limit_info: Optional[dict] = Field(None, description="Rate limit information if applicable")


class ErrorResponse(BaseModel):
    """Standard error response"""
    status: Literal["error"] = Field(..., description="Error status")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")
    details: Optional[dict] = Field(None, description="Additional error details")



