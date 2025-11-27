"""Market data service"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from ..repositories.prediction_repository import PredictionRepository
from ..utils.cache import market_data_cache
from ..schemas.market_data import DailyDataResponse, DailyDataPoint
from ..schemas.prediction import Prediction, AccuracyStats

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for market data operations"""
    
    def __init__(self, prediction_service):
        self.prediction_repo = PredictionRepository()
        self.prediction_service = prediction_service
    
    def get_daily_data(self, days: int = 90) -> Dict:
        """Get daily market data with predictions"""
        try:
            period = f"{max(days, 90)}d" if isinstance(days, int) else "3mo"
            hist, symbol_used = market_data_cache.get_cached_market_data(period=period)
            
            if hist is None or hist.empty:
                logger.error("All gold data sources failed")
                return {
                    "symbol": "XAUUSD",
                    "timeframe": "daily",
                    "data": [],
                    "current_price": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "message": "Unable to fetch gold price data"
                }
            
            # Get or create prediction for next day
            next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            predicted_price = None
            prediction_method = None
            
            try:
                if not self.prediction_repo.prediction_exists_for_date(next_day):
                    predicted_price = self.prediction_service.predict_next_day()
                    if predicted_price:
                        prediction_method = self.prediction_service.get_model_display_name()
                        self.prediction_repo.save_prediction(
                            next_day, predicted_price, prediction_method=prediction_method
                        )
                else:
                    predicted_price = self.prediction_repo.get_prediction_for_date(next_day)
            except Exception as e:
                logger.warning(f"Prediction check failed: {e}")
            
            # Get historical predictions and accuracy stats
            all_historical_predictions = self.prediction_repo.get_historical_predictions(days)
            accuracy_stats = self.prediction_repo.get_accuracy_stats()
            
            # Convert to daily data points
            daily_data = []
            data_dates = set()
            for date, row in hist.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                data_dates.add(date_str)
                daily_data.append({
                    "date": date_str,
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })
            
            # Add predictions to data points
            predictions_by_date = {p['date']: p for p in all_historical_predictions}
            for data_point in daily_data:
                date = data_point['date']
                if date in predictions_by_date:
                    pred = predictions_by_date[date]
                    data_point['predicted_price'] = pred.get('predicted_price')
                    data_point['actual_price'] = pred.get('actual_price')
            
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            
            # Build prediction object
            prediction_obj = None
            if predicted_price:
                prediction_obj = {
                    "next_day": next_day,
                    "predicted_price": round(predicted_price, 2),
                    "current_price": current_price,
                    "prediction_method": prediction_method or "Lasso Regression"
                }
            
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": daily_data,
                "historical_predictions": all_historical_predictions,
                "accuracy_stats": accuracy_stats,
                "current_price": current_price,
                "prediction": prediction_obj,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting daily data: {e}", exc_info=True)
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": [],
                "current_price": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(e)
            }
    
    def get_realtime_price(self) -> Dict:
        """Get real-time price data"""
        realtime_data = market_data_cache.get_realtime_price_data()
        if realtime_data:
            return {
                "symbol": realtime_data.get('symbol', 'XAUUSD'),
                "current_price": realtime_data.get('current_price', 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        else:
            # Fallback to daily data
            daily_data = self.get_daily_data()
            return {
                "symbol": "XAUUSD",
                "current_price": daily_data.get('current_price', 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }



