"""ML Model initialization and management"""
from pathlib import Path
import sys
from typing import Optional

from models.news_prediction import NewsEnhancedLassoPredictor
from models.lasso_model import LassoGoldPredictor
from .logging_config import get_logger

logger = get_logger(__name__)

# Add backend directory to path before importing models
BACKEND_PARENT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_PARENT) not in sys.path:
    sys.path.insert(0, str(BACKEND_PARENT))


def initialize_models():
    """
    Initialize ML models for prediction.
    Returns tuple of (lasso_predictor, news_enhanced_predictor)
    """
    BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

    # Initialize Lasso Regression predictor
    lasso_predictor = LassoGoldPredictor()
    try:
        lasso_predictor.load_model(
            str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
        logger.info("✅ Lasso Regression model loaded successfully")
    except Exception as e:
        logger.warning(f"Lasso model not found: {e}")
        lasso_predictor = None

    # Initialize News-Enhanced Lasso predictor
    news_enhanced_predictor = NewsEnhancedLassoPredictor()
    enhanced_model_path = BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'

    if enhanced_model_path.exists():
        try:
            news_enhanced_predictor.load_enhanced_model(
                str(enhanced_model_path))
            logger.info("✅ News-Enhanced Lasso model loaded successfully")
            logger.info(
                f"   Model accuracy (R²): {news_enhanced_predictor.best_score:.4f}")
            logger.info(
                f"   Selected features: {len(news_enhanced_predictor.selected_features)}")
        except Exception as e:
            logger.warning(f"News-enhanced model failed to load: {e}")
            news_enhanced_predictor = None
    else:
        logger.info(
            "ℹ️  News-Enhanced model not found - using Lasso Regression only")
        logger.info(f"   Enhanced model path: {enhanced_model_path}")

    return lasso_predictor, news_enhanced_predictor
