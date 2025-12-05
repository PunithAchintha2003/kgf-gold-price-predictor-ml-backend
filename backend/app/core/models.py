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
        logger.debug("Lasso Regression model loaded")
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
            logger.debug("News-enhanced model loaded")
        except Exception as e:
            logger.warning(f"News-enhanced model failed to load: {e}")
            news_enhanced_predictor = None
    else:
        logger.debug("Using regular Lasso model (enhanced model not found)")

    return lasso_predictor, news_enhanced_predictor
