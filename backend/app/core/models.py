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
    logger.info("🔄 Initializing ML models...")
    logger.info("📊 Primary Model: News-Enhanced Lasso Regression")
    news_enhanced_predictor = NewsEnhancedLassoPredictor()
    enhanced_model_path = BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'

    if enhanced_model_path.exists():
        try:
            news_enhanced_predictor.load_enhanced_model(
                str(enhanced_model_path))
            logger.info("✅ News-Enhanced Lasso model (PRIMARY) loaded successfully")
            logger.info(
                f"   Model accuracy (R²): {news_enhanced_predictor.best_score:.4f}")
            logger.info(
                f"   Selected features: {len(news_enhanced_predictor.selected_features)}")
        except Exception as e:
            logger.warning(f"⚠️  News-Enhanced model (PRIMARY) failed to load: {e}")
            news_enhanced_predictor = None
    else:
        logger.warning(
            f"⚠️  News-Enhanced model (PRIMARY) not found at: {enhanced_model_path}")
        logger.info("   Will use Basic Lasso Regression as fallback if available")

    # Initialize Basic Lasso Regression predictor (FALLBACK)
    logger.info("📊 Fallback Model: Basic Lasso Regression")
    lasso_predictor = LassoGoldPredictor()
    try:
        lasso_predictor.load_model(
            str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
        logger.info("✅ Basic Lasso Regression model (FALLBACK) loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Basic Lasso model (FALLBACK) not found: {e}")
        lasso_predictor = None

    # Summary
    if news_enhanced_predictor and news_enhanced_predictor.model is not None:
        logger.info("🤖 ML Model: News-Enhanced Lasso Regression (PRIMARY)")
    elif lasso_predictor and lasso_predictor.model is not None:
        logger.info("🤖 ML Model: Basic Lasso Regression (FALLBACK)")
    else:
        logger.error("❌ No ML models available!")

    return lasso_predictor, news_enhanced_predictor
