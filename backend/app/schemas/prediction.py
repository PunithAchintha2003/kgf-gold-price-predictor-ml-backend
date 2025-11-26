"""Prediction schemas"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Prediction(BaseModel):
    """Prediction model"""
    next_day: str
    predicted_price: float
    current_price: float
    prediction_method: str
    warning: Optional[str] = None


class PredictionHistoryItem(BaseModel):
    """Historical prediction item"""
    date: str
    predicted_price: float
    actual_price: Optional[float] = None
    accuracy_percentage: Optional[float] = None
    status: str  # 'pending' | 'completed'
    method: str


class PredictionHistory(BaseModel):
    """Prediction history response"""
    status: str
    predictions: list[PredictionHistoryItem]
    total: int


class AccuracyStats(BaseModel):
    """Accuracy statistics"""
    average_accuracy: float
    r2_score: float
    total_predictions: int
    evaluated_predictions: int


