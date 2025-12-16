"""Prediction service"""
import logging
from typing import Optional
from pathlib import Path

# Import ML models - these will be initialized in main.py
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for prediction operations"""

    def __init__(self, lasso_predictor=None, news_enhanced_predictor=None):
        self.lasso_predictor = lasso_predictor
        self.news_enhanced_predictor = news_enhanced_predictor

    def predict_next_day(self) -> Optional[float]:
        """Predict next day price using available models"""
        try:
            # Try News-Enhanced Lasso model first
            if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
                try:
                    # Get fresh market data
                    market_data = self.lasso_predictor.fetch_market_data()
                    if not market_data:
                        raise ValueError("Failed to fetch market data")

                    # Fetch news sentiment
                    sentiment_features = self.news_enhanced_predictor.fetch_and_analyze_news(
                        days_back=7)

                    # Create enhanced features
                    enhanced_features = self.news_enhanced_predictor.create_enhanced_features(
                        market_data, sentiment_features)

                    if enhanced_features.empty:
                        raise ValueError("No enhanced features created")

                    # Make prediction using News-Enhanced model
                    prediction = self.news_enhanced_predictor.predict_with_news(
                        enhanced_features)

                    if prediction is not None:
                        logger.info(
                            "Using News-Enhanced Lasso model for prediction")
                        return float(prediction)
                except Exception as e:
                    logger.warning(f"News-Enhanced prediction failed: {e}")

            # Fallback to Lasso Regression
            if self.lasso_predictor and self.lasso_predictor.model is not None:
                try:
                    # Get fresh market data
                    market_data = self.lasso_predictor.fetch_market_data()
                    if not market_data:
                        raise ValueError("Failed to fetch market data")

                    # Create features
                    features_df = self.lasso_predictor.create_fundamental_features(
                        market_data)

                    if features_df.empty:
                        raise ValueError("No features created")

                    # Make prediction using Lasso Regression
                    prediction = self.lasso_predictor.predict_next_price(
                        features_df)

                    if prediction is not None:
                        logger.info(
                            "Using Lasso Regression model for prediction")
                        return float(prediction)
                except Exception as e:
                    logger.warning(f"Lasso prediction failed: {e}")

            logger.error("No prediction model available")
            return None
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return None

    def get_model_display_name(self) -> str:
        """Get the display name for the current ML model"""
        if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
            return "News-Enhanced Lasso Regression"
        elif self.lasso_predictor and self.lasso_predictor.model is not None:
            return "Lasso Regression (Fallback)"
        else:
            return "No Model Available"
    
    def get_model_info(self) -> dict:
        """Get detailed information about the active model"""
        model_info = {
            "active_model": None,
            "model_type": None,
            "r2_score": None,
            "features_count": None,
            "selected_features_count": None,
            "fallback_available": False
        }
        
        # Check for News-Enhanced model
        if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
            model_info["active_model"] = "News-Enhanced Lasso Regression"
            model_info["model_type"] = "News-Enhanced Lasso"
            model_info["r2_score"] = round(self.news_enhanced_predictor.best_score, 4) if hasattr(self.news_enhanced_predictor, 'best_score') and self.news_enhanced_predictor.best_score else None
            model_info["features_count"] = len(self.news_enhanced_predictor.feature_columns) if hasattr(self.news_enhanced_predictor, 'feature_columns') else None
            model_info["selected_features_count"] = len(self.news_enhanced_predictor.selected_features) if hasattr(self.news_enhanced_predictor, 'selected_features') else None
            model_info["fallback_available"] = self.lasso_predictor is not None and self.lasso_predictor.model is not None
            
            # Add feature details if available
            if hasattr(self.news_enhanced_predictor, 'selected_features') and self.news_enhanced_predictor.selected_features:
                model_info["selected_features"] = self.news_enhanced_predictor.selected_features[:10]  # First 10 features
                model_info["total_features"] = len(self.news_enhanced_predictor.feature_columns) if hasattr(self.news_enhanced_predictor, 'feature_columns') else None
            
        # Check for Lasso Regression fallback
        elif self.lasso_predictor and self.lasso_predictor.model is not None:
            model_info["active_model"] = "Lasso Regression"
            model_info["model_type"] = "Lasso Regression"
            model_info["r2_score"] = round(self.lasso_predictor.best_score, 4) if hasattr(self.lasso_predictor, 'best_score') and self.lasso_predictor.best_score else None
            model_info["features_count"] = len(self.lasso_predictor.feature_columns) if hasattr(self.lasso_predictor, 'feature_columns') else None
            model_info["selected_features_count"] = len(self.lasso_predictor.selected_features) if hasattr(self.lasso_predictor, 'selected_features') else None
            model_info["fallback_available"] = False
            
            # Add feature details if available
            if hasattr(self.lasso_predictor, 'selected_features') and self.lasso_predictor.selected_features:
                model_info["selected_features"] = self.lasso_predictor.selected_features[:10]  # First 10 features
                model_info["total_features"] = len(self.lasso_predictor.feature_columns) if hasattr(self.lasso_predictor, 'feature_columns') else None
        
        return model_info