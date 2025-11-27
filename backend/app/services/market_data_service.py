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
            
            # Add predictions to data points - ensure all data points have prediction fields
            predictions_by_date = {p['date']: p for p in all_historical_predictions}
            data_with_predictions = 0
            for data_point in daily_data:
                date = data_point['date']
                if date in predictions_by_date:
                    pred = predictions_by_date[date]
                    data_point['predicted_price'] = pred.get('predicted_price')
                    data_point['actual_price'] = pred.get('actual_price')
                    if data_point.get('predicted_price') is not None:
                        data_with_predictions += 1
                else:
                    # Ensure consistent structure - set to None if no prediction
                    data_point['predicted_price'] = None
                    data_point['actual_price'] = None
            
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
            
            # Calculate metadata for frontend
            if daily_data:
                market_data_range = f"{daily_data[0]['date']} to {daily_data[-1]['date']}"
            else:
                market_data_range = "N/A"
            
            if all_historical_predictions:
                prediction_range = f"{all_historical_predictions[0]['date']} to {all_historical_predictions[-1]['date']}"
            else:
                prediction_range = "N/A"
            
            # Calculate date ranges
            full_date_range = market_data_range
            if all_historical_predictions:
                all_dates = [d['date'] for d in daily_data] + [p['date'] for p in all_historical_predictions]
                if all_dates:
                    full_date_range = f"{min(all_dates)} to {max(all_dates)}"
            
            # Count predictions before Oct 6, 2025
            predictions_before_oct6 = len([p for p in all_historical_predictions if p['date'] < '2025-10-06'])
            data_before_oct6 = len([d for d in daily_data if d['date'] < '2025-10-06'])
            
            # Build metadata object
            metadata = {
                "totalDataPoints": len(daily_data),
                "totalPredictions": len(all_historical_predictions),
                "dataWithPredictedPrice": data_with_predictions,
                "dataBeforeOct6": data_before_oct6,
                "predictionsBeforeOct6": predictions_before_oct6,
                "fullDateRange": full_date_range,
                "marketDataRange": market_data_range,
                "predictionRange": prediction_range,
                "note": "Backend includes all available market data and predictions"
            }
            
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": daily_data,
                "historical_predictions": all_historical_predictions,
                "accuracy_stats": accuracy_stats,
                "current_price": current_price,
                "prediction": prediction_obj,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting daily data: {e}", exc_info=True)
            # Return consistent structure even on error
            try:
                all_historical_predictions = self.prediction_repo.get_historical_predictions(days)
                accuracy_stats = self.prediction_repo.get_accuracy_stats()
            except:
                all_historical_predictions = []
                accuracy_stats = {
                    'average_accuracy': 0.0,
                    'r2_score': 0.0,
                    'total_predictions': 0,
                    'evaluated_predictions': 0
                }
            
            # Empty metadata for error case
            metadata = {
                "totalDataPoints": 0,
                "totalPredictions": len(all_historical_predictions),
                "dataWithPredictedPrice": 0,
                "dataBeforeOct6": 0,
                "predictionsBeforeOct6": 0,
                "fullDateRange": "N/A",
                "marketDataRange": "N/A",
                "predictionRange": "N/A" if not all_historical_predictions else f"{all_historical_predictions[0]['date']} to {all_historical_predictions[-1]['date']}",
                "note": "Error occurred while fetching data"
            }
            
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": [],
                "historical_predictions": all_historical_predictions,
                "accuracy_stats": accuracy_stats,
                "current_price": 0.0,
                "prediction": None,
                "metadata": metadata,
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



