"""Prediction service"""
import logging
from typing import Optional
from pathlib import Path

from ..repositories.prediction_repository import PredictionRepository

# Import ML models - these will be initialized in main.py
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for prediction operations"""

    def __init__(self, lasso_predictor=None, news_enhanced_predictor=None):
        self.lasso_predictor = lasso_predictor
        self.news_enhanced_predictor = news_enhanced_predictor

    def predict_next_day(self) -> Optional[float]:
        """
        Predict next day price using available models.
        Primary: News-Enhanced Lasso Regression
        Fallback: Basic Lasso Regression (if enhanced model fails or unavailable)
        """
        try:
            # PRIMARY: Try News-Enhanced Lasso model first
            if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
                try:
                    logger.info("🔄 Attempting prediction with News-Enhanced Lasso (Primary)...")
                    
                    # Get fresh market data (can use either predictor for this)
                    market_data = None
                    if self.lasso_predictor and self.lasso_predictor.model is not None:
                        market_data = self.lasso_predictor.fetch_market_data()
                    else:
                        # If lasso_predictor not available, create a temporary one just for fetching data
                        from models.lasso_model import LassoGoldPredictor
                        temp_predictor = LassoGoldPredictor()
                        market_data = temp_predictor.fetch_market_data()
                    
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
                            "✅ Successfully used News-Enhanced Lasso model (Primary) for prediction")
                        return float(prediction)
                except Exception as e:
                    logger.warning(
                        f"⚠️  News-Enhanced Lasso (Primary) prediction failed: {e}")
                    logger.info("🔄 Falling back to Basic Lasso Regression...")

            # FALLBACK: Use Basic Lasso Regression if News-Enhanced fails or unavailable
            if self.lasso_predictor and self.lasso_predictor.model is not None:
                try:
                    logger.info("🔄 Attempting prediction with Basic Lasso Regression (Fallback)...")
                    
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
                            "✅ Using Basic Lasso Regression (Fallback) for prediction")
                        return float(prediction)
                except Exception as e:
                    logger.warning(f"⚠️  Basic Lasso Regression (Fallback) prediction failed: {e}")

            logger.error("❌ No prediction model available - both News-Enhanced and Basic Lasso failed")
            return None
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}", exc_info=True)
            return None

    def get_model_display_name(self) -> str:
        """Get the display name for the current ML model"""
        if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
            return "News-Enhanced Lasso Regression (Primary)"
        elif self.lasso_predictor and self.lasso_predictor.model is not None:
            return "Lasso Regression (Fallback)"
        else:
            return "No Model Available"
    
    def get_model_info(self) -> dict:
        """Get detailed information about the active model with live accuracy metrics"""
        model_info = {
            "active_model": None,
            "model_type": None,
            "training_r2_score": None,  # Static R² from model training
            "live_r2_score": None,      # Dynamic R² from actual predictions
            "r2_score": None,           # Primary R² score (live if available, else training)
            "features_count": None,
            "selected_features_count": None,
            "fallback_available": False,
            "live_accuracy_stats": None  # Full live accuracy statistics
        }
        
        # Get live accuracy stats from database (dynamic R² based on predictions vs actual prices)
        try:
            prediction_repo = PredictionRepository()
            live_stats = prediction_repo.get_accuracy_stats()
            model_info["live_accuracy_stats"] = live_stats
            model_info["live_r2_score"] = live_stats.get('r2_score')
        except Exception as e:
            logger.warning(f"Could not get live accuracy stats: {e}")
        
        # Check for News-Enhanced model (Primary)
        if self.news_enhanced_predictor and self.news_enhanced_predictor.model is not None:
            model_info["active_model"] = "News-Enhanced Lasso Regression (Primary)"
            model_info["model_type"] = "News-Enhanced Lasso"
            training_r2 = round(self.news_enhanced_predictor.best_score, 4) if hasattr(self.news_enhanced_predictor, 'best_score') and self.news_enhanced_predictor.best_score else None
            model_info["training_r2_score"] = training_r2
            model_info["features_count"] = len(self.news_enhanced_predictor.feature_columns) if hasattr(self.news_enhanced_predictor, 'feature_columns') else None
            model_info["selected_features_count"] = len(self.news_enhanced_predictor.selected_features) if hasattr(self.news_enhanced_predictor, 'selected_features') else None
            model_info["fallback_available"] = self.lasso_predictor is not None and self.lasso_predictor.model is not None
            
            # Primary R² score: use live if available and valid, otherwise training
            if model_info["live_r2_score"] is not None:
                model_info["r2_score"] = model_info["live_r2_score"]
            else:
                model_info["r2_score"] = training_r2
            
            # Add feature details if available
            if hasattr(self.news_enhanced_predictor, 'selected_features') and self.news_enhanced_predictor.selected_features:
                model_info["selected_features"] = self.news_enhanced_predictor.selected_features[:10]  # First 10 features
                model_info["total_features"] = len(self.news_enhanced_predictor.feature_columns) if hasattr(self.news_enhanced_predictor, 'feature_columns') else None
            
        # Check for Lasso Regression fallback
        elif self.lasso_predictor and self.lasso_predictor.model is not None:
            model_info["active_model"] = "Lasso Regression (Fallback)"
            model_info["model_type"] = "Lasso Regression"
            training_r2 = round(self.lasso_predictor.best_score, 4) if hasattr(self.lasso_predictor, 'best_score') and self.lasso_predictor.best_score else None
            model_info["training_r2_score"] = training_r2
            model_info["features_count"] = len(self.lasso_predictor.feature_columns) if hasattr(self.lasso_predictor, 'feature_columns') else None
            model_info["selected_features_count"] = len(self.lasso_predictor.selected_features) if hasattr(self.lasso_predictor, 'selected_features') else None
            model_info["fallback_available"] = False
            
            # Primary R² score: use live if available and valid, otherwise training
            if model_info["live_r2_score"] is not None:
                model_info["r2_score"] = model_info["live_r2_score"]
            else:
                model_info["r2_score"] = training_r2
            
            # Add feature details if available
            if hasattr(self.lasso_predictor, 'selected_features') and self.lasso_predictor.selected_features:
                model_info["selected_features"] = self.lasso_predictor.selected_features[:10]  # First 10 features
                model_info["total_features"] = len(self.lasso_predictor.feature_columns) if hasattr(self.lasso_predictor, 'feature_columns') else None
        
        return model_info