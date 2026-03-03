"""Prediction validation utilities for gold price predictions"""
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of prediction validation"""
    is_valid: bool
    predicted_price: float
    current_price: float
    change: float
    change_percentage: float
    error_reason: Optional[str] = None
    warning_flags: list = None
    
    def __post_init__(self):
        if self.warning_flags is None:
            self.warning_flags = []


def validate_prediction(
    predicted_price: float,
    current_price: float,
    min_abs_price: float = 500.0,
    max_abs_price: float = 50000.0,
    max_daily_percent_move: float = 20.0
) -> ValidationResult:
    """
    Validate a gold price prediction using industry-standard checks.
    
    This function implements a two-tier validation approach:
    1. Absolute bounds: Filters obviously impossible prices
    2. Relative change guard: Flags extreme daily moves
    
    Args:
        predicted_price: The predicted gold price
        current_price: The current gold price
        min_abs_price: Minimum reasonable absolute price (default: $500)
        max_abs_price: Maximum reasonable absolute price (default: $50,000)
        max_daily_percent_move: Maximum allowed daily percent change (default: 20%)
    
    Returns:
        ValidationResult containing validation status and details
    """
    # Calculate change metrics
    change = predicted_price - current_price
    change_percentage = (change / current_price * 100) if current_price > 0 else 0
    
    # Initialize result
    result = ValidationResult(
        is_valid=True,
        predicted_price=predicted_price,
        current_price=current_price,
        change=change,
        change_percentage=change_percentage,
        warning_flags=[]
    )
    
    # Validation 1: Absolute price bounds
    if predicted_price < min_abs_price:
        result.is_valid = False
        result.error_reason = (
            f"Predicted price ${predicted_price:.2f} is below minimum "
            f"reasonable price ${min_abs_price:.2f}"
        )
        logger.error(
            f"PredictionValidation: {result.error_reason} | "
            f"Current: ${current_price:.2f}, Change: {change_percentage:+.2f}%"
        )
        return result
    
    if predicted_price > max_abs_price:
        result.is_valid = False
        result.error_reason = (
            f"Predicted price ${predicted_price:.2f} exceeds maximum "
            f"reasonable price ${max_abs_price:.2f}"
        )
        logger.error(
            f"PredictionValidation: {result.error_reason} | "
            f"Current: ${current_price:.2f}, Change: {change_percentage:+.2f}%"
        )
        return result
    
    # Validation 2: Relative percent change
    if abs(change_percentage) > max_daily_percent_move:
        result.is_valid = False
        result.error_reason = (
            f"Predicted change of {change_percentage:+.2f}% exceeds maximum "
            f"allowed daily move of ±{max_daily_percent_move:.2f}%"
        )
        logger.error(
            f"PredictionValidation: {result.error_reason} | "
            f"Current: ${current_price:.2f}, Predicted: ${predicted_price:.2f}, "
            f"Change: ${change:+.2f}"
        )
        return result
    
    # Add warning flags for notable (but valid) conditions
    if abs(change_percentage) > max_daily_percent_move * 0.5:
        result.warning_flags.append("large_move")
        logger.warning(
            f"PredictionValidation: Large (but valid) predicted move of "
            f"{change_percentage:+.2f}% | Current: ${current_price:.2f}, "
            f"Predicted: ${predicted_price:.2f}"
        )
    
    # Success
    logger.info(
        f"PredictionValidation: ✓ Valid prediction | "
        f"Current: ${current_price:.2f}, Predicted: ${predicted_price:.2f}, "
        f"Change: {change_percentage:+.2f}%"
    )
    
    return result


def format_validation_error(result: ValidationResult) -> str:
    """
    Format a validation error for API responses.
    
    Args:
        result: ValidationResult from validate_prediction
    
    Returns:
        Formatted error message
    """
    if result.is_valid:
        return ""
    
    return (
        f"{result.error_reason} (Current: ${result.current_price:.2f}, "
        f"Predicted: ${result.predicted_price:.2f}, "
        f"Change: {result.change_percentage:+.2f}%)"
    )
