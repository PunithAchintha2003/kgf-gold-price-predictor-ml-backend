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
    Primary: News-Enhanced Lasso Regression
    Fallback: Basic Lasso Regression
    
    Returns tuple of (lasso_predictor, news_enhanced_predictor)
    """
    BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

    # Initialize News-Enhanced Lasso predictor (PRIMARY)
    logger.debug("🔄 Initializing ML models...")
    news_enhanced_predictor = NewsEnhancedLassoPredictor()
    enhanced_model_path = BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'

    if enhanced_model_path.exists():
        try:
            news_enhanced_predictor.load_enhanced_model(
                str(enhanced_model_path))
            logger.debug(f"✅ News-Enhanced Lasso model loaded (R²: {news_enhanced_predictor.best_score:.4f}, Features: {len(news_enhanced_predictor.selected_features)})")
        except Exception as e:
            logger.warning(f"⚠️  News-Enhanced model failed to load: {e}")
            news_enhanced_predictor = None
    else:
        logger.warning(f"⚠️  News-Enhanced model not found at: {enhanced_model_path}")
        news_enhanced_predictor = None

    # Initialize Basic Lasso Regression predictor (FALLBACK)
    lasso_predictor = LassoGoldPredictor()
    try:
        lasso_predictor.load_model(
            str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
        logger.debug("✅ Basic Lasso Regression model loaded")
    except Exception as e:
        logger.debug(f"Basic Lasso model not found: {e}")
        lasso_predictor = None

    # Summary - only log if there's an issue
    if news_enhanced_predictor and news_enhanced_predictor.model is not None:
        logger.info("🤖 ML Model: News-Enhanced Lasso Regression (PRIMARY)")
    elif lasso_predictor and lasso_predictor.model is not None:
        logger.info("🤖 ML Model: Basic Lasso Regression (FALLBACK)")
    else:
        logger.error("❌ No ML models available!")

    return lasso_predictor, news_enhanced_predictor
