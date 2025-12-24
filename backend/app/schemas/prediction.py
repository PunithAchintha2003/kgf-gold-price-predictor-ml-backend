"""Prediction schemas with comprehensive type hints and validation"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class PredictionStatus(str, Enum):
    """Prediction status enumeration"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Prediction(BaseModel):
    """Prediction model with validation"""
    next_day: str = Field(..., description="Next trading day in YYYY-MM-DD format")
    predicted_price: float = Field(..., gt=0, description="Predicted price in USD")
    current_price: float = Field(..., gt=0, description="Current market price in USD")
    prediction_method: str = Field(..., description="ML model used for prediction")
    warning: Optional[str] = Field(None, description="Optional warning message")
    change: Optional[float] = Field(None, description="Predicted price change")
    change_percentage: Optional[float] = Field(None, description="Predicted price change percentage")
    
    @field_validator("next_day")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class PredictionHistoryItem(BaseModel):
    """Historical prediction item with validation"""
    date: str = Field(..., description="Prediction date in YYYY-MM-DD format")
    predicted_price: float = Field(..., gt=0, description="Predicted price")
    actual_price: Optional[float] = Field(None, gt=0, description="Actual market price")
    accuracy_percentage: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Prediction accuracy percentage"
    )
    status: Literal["pending", "completed"] = Field(..., description="Prediction status")
    method: str = Field(..., description="Prediction method used")
    prediction_reasons: Optional[str] = Field(None, description="AI-generated prediction reasons")


class PredictionHistory(BaseModel):
    """Prediction history response"""
    status: Literal["success", "error"] = Field(..., description="Response status")
    predictions: list[PredictionHistoryItem] = Field(..., description="List of predictions")
    total: int = Field(..., ge=0, description="Total number of predictions")
    message: Optional[str] = Field(None, description="Optional message")


class AccuracyStats(BaseModel):
    """Accuracy statistics with validation"""
    average_accuracy: float = Field(..., ge=0, le=100, description="Average accuracy percentage")
    r2_score: Optional[float] = Field(None, ge=-1, le=1, description="R² score")
    training_r2_score: Optional[float] = Field(None, ge=-1, le=1, description="Training R² score")
    live_r2_score: Optional[float] = Field(None, ge=-1, le=1, description="Live R² score")
    total_predictions: int = Field(..., ge=0, description="Total predictions")
    evaluated_predictions: int = Field(..., ge=0, description="Evaluated predictions")
    pending_predictions: int = Field(..., ge=0, description="Pending predictions")
    evaluation_rate: Optional[float] = Field(None, ge=0, le=100, description="Evaluation rate percentage")


class ModelInfo(BaseModel):
    """ML model information"""
    active_model: Optional[str] = Field(None, description="Active model name")
    model_type: Optional[str] = Field(None, description="Model type")
    r2_score: Optional[float] = Field(None, ge=-1, le=1, description="Primary R² score")
    training_r2_score: Optional[float] = Field(None, ge=-1, le=1, description="Training R² score")
    live_r2_score: Optional[float] = Field(None, ge=-1, le=1, description="Live R² score")
    features_count: Optional[int] = Field(None, ge=0, description="Total features")
    selected_features_count: Optional[int] = Field(None, ge=0, description="Selected features count")
    selected_features: Optional[list[str]] = Field(None, description="Selected feature names")
    fallback_available: bool = Field(False, description="Whether fallback model is available")
    live_accuracy_stats: Optional[dict] = Field(None, description="Live accuracy statistics")


class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction response with comprehensive model information"""
    status: Literal["success", "error"] = Field(..., description="Response status")
    prediction: Optional[Prediction] = Field(None, description="Prediction details")
    model: Optional[ModelInfo] = Field(None, description="Model information")
    sentiment: Optional[dict] = Field(None, description="News sentiment analysis")
    top_features: Optional[list[str]] = Field(None, description="Top features used")
    timestamp: str = Field(..., description="Response timestamp")
    message: Optional[str] = Field(None, description="Optional message")


class PriceUpdateItem(BaseModel):
    """Price update item with validation"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    actual_price: float = Field(..., gt=0, description="Actual market price")
    
    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class PriceUpdateRequest(BaseModel):
    """Price update request"""
    prices: list[PriceUpdateItem] = Field(..., min_length=1, description="List of price updates")


class PriceUpdateResponse(BaseModel):
    """Price update response"""
    status: Literal["success", "error"] = Field(..., description="Response status")
    updated_count: int = Field(..., ge=0, description="Number of predictions updated")
    message: Optional[str] = Field(None, description="Optional message")
