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


def is_weekend(date: datetime) -> bool:
    """Check if a date is Saturday (5) or Sunday (6)"""
    return date.weekday() >= 5


def get_next_trading_day(start_date: datetime = None) -> datetime:
    """Get the next trading day (skip weekends)"""
    if start_date is None:
        start_date = datetime.now()

    next_day = start_date + timedelta(days=1)
    # Skip weekends - if it's Saturday, go to Monday; if Sunday, go to Monday
    while is_weekend(next_day):
        days_to_add = 7 - next_day.weekday()  # Days until Monday
        next_day = next_day + timedelta(days=days_to_add)

    return next_day


class MarketDataService:
    """Service for market data operations"""

    def __init__(self, prediction_service):
        self.prediction_repo = PredictionRepository()
        self.prediction_service = prediction_service

    def get_daily_data(self, days: int = 90, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """Get daily market data with predictions
        
        Args:
            days: Number of days to fetch (default: 90)
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Note: If start_date/end_date are provided, they take precedence over days parameter
        """
        try:
            period = f"{max(days, 90)}d" if isinstance(days, int) else "3mo"
            hist, symbol_used, rate_limit_info = market_data_cache.get_cached_market_data(
                period=period)

            if hist is None or hist.empty:
                # Check if we're rate limited
                if rate_limit_info and rate_limit_info.get('rate_limited'):
                    wait_seconds = int(rate_limit_info.get('wait_seconds', 0))
                    # Only log once per minute to reduce log spam
                    if not hasattr(self, '_last_rate_limit_log') or (datetime.now().timestamp() - getattr(self, '_last_rate_limit_log', 0)) > 60:
                        logger.info(f"Rate limited by data provider. Retry after {wait_seconds} seconds")
                        self._last_rate_limit_log = datetime.now().timestamp()
                    # Try to get historical predictions even when rate limited
                    try:
                        all_historical_predictions = self.prediction_repo.get_historical_predictions(days)
                        accuracy_stats = self.prediction_repo.get_accuracy_stats()
                    except Exception as e:
                        logger.debug(f"Error getting historical predictions during rate limit: {e}")
                        all_historical_predictions = []
                        accuracy_stats = {
                            'average_accuracy': 0.0,
                            'r2_score': 0.0,
                            'total_predictions': 0,
                            'evaluated_predictions': 0
                        }
                    
                    return {
                        "symbol": "XAUUSD",
                        "timeframe": "daily",
                        "data": [],
                        "historical_predictions": all_historical_predictions,
                        "accuracy_stats": accuracy_stats,
                        "current_price": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "status": "rate_limited",
                        "message": f"Data provider rate limit. Please retry after {wait_seconds} seconds.",
                        "rate_limit_info": {
                            "wait_seconds": wait_seconds,
                            "retry_after": datetime.fromtimestamp(rate_limit_info.get('until', 0)).isoformat() if rate_limit_info.get('until') else None
                        }
                    }
                else:
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

            # Get or create prediction for next trading day (skip weekends)
            next_trading_day_dt = get_next_trading_day()
            next_day = next_trading_day_dt.strftime("%Y-%m-%d")
            predicted_price = None
            prediction_method = None

            try:
                # Always use stored prediction from database (generated by background task)
                # Do not generate new predictions on API calls
                if self.prediction_repo.prediction_exists_for_date(next_day):
                    predicted_price = self.prediction_repo.get_prediction_for_date(next_day)
                    stored_prediction = self.prediction_repo.get_prediction_details_for_date(next_day)
                    if stored_prediction and stored_prediction.get('method'):
                        prediction_method = stored_prediction.get('method')
                    else:
                        prediction_method = self.prediction_service.get_model_display_name()
                    logger.debug(
                        f"Using stored prediction for next trading day: {next_day} = ${predicted_price:.2f} (Method: {prediction_method})")
                else:
                    # No stored prediction available - will be generated on-demand by API route if needed
                    logger.debug(
                        f"No stored prediction found for {next_day}. Will be generated on-demand if requested.")
                    predicted_price = None
                    prediction_method = None
            except Exception as e:
                logger.warning(f"Prediction check failed: {e}")
                # Fallback: try to get stored prediction if available
                try:
                    if self.prediction_repo.prediction_exists_for_date(next_day):
                        predicted_price = self.prediction_repo.get_prediction_for_date(next_day)
                        prediction_method = self.prediction_service.get_model_display_name()
                except Exception as e:
                    logger.debug(f"Error checking existing prediction: {e}")
                    pass

            # Get historical predictions and accuracy stats in parallel (if possible)
            # For now, sequential but optimized with caching
            all_historical_predictions = self.prediction_repo.get_historical_predictions(days)
            # Filter out weekend predictions (Saturday/Sunday) - we don't want to show them
            all_historical_predictions = [
                p for p in all_historical_predictions
                if not is_weekend(datetime.strptime(p['date'], "%Y-%m-%d"))
            ]
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
            predictions_by_date = {
                p['date']: p for p in all_historical_predictions}
            data_with_predictions = 0
            for data_point in daily_data:
                date = data_point['date']
                if date in predictions_by_date:
                    pred = predictions_by_date[date]
                    data_point['predicted_price'] = pred.get('predicted_price')
                    actual_price = pred.get('actual_price')
                    data_point['actual_price'] = actual_price
                    # Update close price with actual_price if available (manual entry takes precedence)
                    if actual_price is not None:
                        data_point['close'] = round(float(actual_price), 2)
                        # Also update high/low/open if they're the same (likely manual entry)
                        if data_point.get('high') == data_point.get('low') == data_point.get('open'):
                            data_point['high'] = data_point['close']
                            data_point['low'] = data_point['close']
                            data_point['open'] = data_point['close']
                    if data_point.get('predicted_price') is not None:
                        data_with_predictions += 1
                else:
                    # Ensure consistent structure - set to None if no prediction
                    data_point['predicted_price'] = None
                    data_point['actual_price'] = None

            # Add synthetic data points for predictions that have actual prices but aren't in market data
            # BUT skip weekends - we don't want to show weekend data points
            # Only show manual entries for weekdays that aren't in market data (holidays, etc.)
            for pred_date, pred in predictions_by_date.items():
                if pred_date not in data_dates and pred.get('actual_price') is not None:
                    # Skip weekends - don't create synthetic data points for Saturday/Sunday
                    try:
                        pred_date_dt = datetime.strptime(pred_date, "%Y-%m-%d")
                        if is_weekend(pred_date_dt):
                            # Skip weekends - don't add synthetic data points for them
                            logger.debug(
                                f"Skipping weekend date {pred_date} from synthetic data points")
                            continue
                    except ValueError:
                        # Invalid date format, skip
                        logger.warning(
                            f"Invalid date format in prediction: {pred_date}")
                        continue

                    # Create a synthetic data point for this date (only weekdays)
                    actual_price = float(pred.get('actual_price'))
                    synthetic_point = {
                        "date": pred_date,
                        "open": round(actual_price, 2),
                        "high": round(actual_price, 2),
                        "low": round(actual_price, 2),
                        "close": round(actual_price, 2),
                        "volume": 0,  # No volume for non-trading days
                        "predicted_price": pred.get('predicted_price'),
                        "actual_price": actual_price
                    }
                    daily_data.append(synthetic_point)
                    data_dates.add(pred_date)
                    if synthetic_point.get('predicted_price') is not None:
                        data_with_predictions += 1

            # Sort daily_data by date to maintain chronological order
            daily_data.sort(key=lambda x: x['date'])
            
            # Filter by date range if start_date or end_date provided
            if start_date or end_date:
                filtered_data = []
                for data_point in daily_data:
                    date_str = data_point['date']
                    # Check if date is within the specified range
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                    filtered_data.append(data_point)
                daily_data = filtered_data
                
                # Also filter historical predictions to match the date range
                filtered_predictions = []
                for pred in all_historical_predictions:
                    pred_date = pred['date']
                    if start_date and pred_date < start_date:
                        continue
                    if end_date and pred_date > end_date:
                        continue
                    filtered_predictions.append(pred)
                all_historical_predictions = filtered_predictions
                
                logger.debug(f"Filtered data by date range: {start_date or 'beginning'} to {end_date or 'end'}, "
                           f"result: {len(daily_data)} data points, {len(all_historical_predictions)} predictions")

            # Get current price - use last ACTUAL trading day's closing price (not weekend)
            # Find the last trading day from the original market data (hist), not synthetic weekend points
            # Market data from yfinance excludes weekends, so the last row is the last trading day
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            last_trading_date = hist.index[-1].strftime("%Y-%m-%d")

            # Verify we're not using a weekend - if the last market data point is somehow a weekend,
            # find the last actual trading day
            last_trading_day_dt = hist.index[-1]
            if is_weekend(last_trading_day_dt):
                # This shouldn't happen with yfinance data, but handle it just in case
                # Go backwards through the market data to find the last weekday
                for i in range(len(hist) - 1, -1, -1):
                    if not is_weekend(hist.index[i]):
                        current_price = round(float(hist['Close'].iloc[i]), 2)
                        last_trading_date = hist.index[i].strftime("%Y-%m-%d")
                        logger.debug(
                            f"Found last trading day: {last_trading_date} (${current_price:.2f})")
                        break

            # Only log if it's a weekend (more informative)
            if is_weekend(datetime.now()):
                logger.debug(
                    f"Using last trading day price ({last_trading_date}): ${current_price:.2f}")

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
                all_dates = [d['date'] for d in daily_data] + \
                    [p['date'] for p in all_historical_predictions]
                if all_dates:
                    full_date_range = f"{min(all_dates)} to {max(all_dates)}"

            # Count predictions before Oct 6, 2025
            predictions_before_oct6 = len(
                [p for p in all_historical_predictions if p['date'] < '2025-10-06'])
            data_before_oct6 = len(
                [d for d in daily_data if d['date'] < '2025-10-06'])

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

            # Get model information
            model_info = None
            if hasattr(self.prediction_service, 'get_model_info'):
                try:
                    model_info = self.prediction_service.get_model_info()
                except Exception as e:
                    logger.debug(f"Could not get model info: {e}")

            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": daily_data,
                "historical_predictions": all_historical_predictions,
                "accuracy_stats": accuracy_stats,
                "current_price": current_price,
                "prediction": prediction_obj,
                "model_info": model_info,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting daily data: {e}", exc_info=True)
            # Store the outer exception message before inner try block
            error_message = str(e)
            # Return consistent structure even on error
            try:
                all_historical_predictions = self.prediction_repo.get_historical_predictions(
                    days)
                accuracy_stats = self.prediction_repo.get_accuracy_stats()
            except Exception as inner_e:
                logger.debug(f"Error getting historical predictions: {inner_e}")
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
                "message": error_message
            }

    def get_realtime_price(self) -> Dict:
        """Get real-time price data - uses last trading day's closing price on weekends"""
        # If today is a weekend, use last trading day's price from market data
        if is_weekend(datetime.now()):
            # Get daily data which already has the last trading day's price
            daily_data = self.get_daily_data()
            # Handle rate limit case
            if daily_data.get('status') == 'rate_limited':
                return {
                    "symbol": "XAUUSD",
                    "current_price": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "rate_limited",
                    "message": daily_data.get('message', 'Data provider rate limit'),
                    "rate_limit_info": daily_data.get('rate_limit_info', {}),
                    "note": "Using last trading day's closing price (market closed on weekends)"
                }
            return {
                "symbol": "XAUUSD",
                "current_price": daily_data.get('current_price', 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": daily_data.get('status', 'success'),
                "note": "Using last trading day's closing price (market closed on weekends)"
            }

        # On weekdays, try to get real-time data
        realtime_data = market_data_cache.get_realtime_price_data()
        if realtime_data:
            return {
                "symbol": realtime_data.get('symbol', 'XAUUSD'),
                "current_price": realtime_data.get('current_price', 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        else:
            # Fallback to daily data (last trading day)
            daily_data = self.get_daily_data()
            # Handle rate limit case
            if daily_data.get('status') == 'rate_limited':
                return {
                    "symbol": "XAUUSD",
                    "current_price": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "rate_limited",
                    "message": daily_data.get('message', 'Data provider rate limit'),
                    "rate_limit_info": daily_data.get('rate_limit_info', {})
                }
            return {
                "symbol": "XAUUSD",
                "current_price": daily_data.get('current_price', 0.0),
                "timestamp": datetime.now().isoformat(),
                "status": daily_data.get('status', 'success')
            }

    def update_pending_predictions(self) -> Dict:
        """Update pending predictions with actual market prices"""
        try:
            # Get all pending predictions
            pending = self.prediction_repo.get_pending_predictions()

            if not pending:
                return {
                    "status": "success",
                    "message": "No pending predictions to update",
                    "updated_count": 0,
                    "failed_count": 0
                }

            # Fetch market data for a period that covers all pending predictions
            # Get the oldest pending prediction date
            oldest_date = min([p['date'] for p in pending])
            oldest_dt = datetime.strptime(oldest_date, "%Y-%m-%d")
            days_back = (datetime.now() - oldest_dt).days + 10  # Add buffer

            period = f"{max(days_back, 90)}d"
            hist, symbol_used, rate_limit_info = market_data_cache.get_cached_market_data(
                period=period)

            if hist is None or hist.empty:
                # Check if we're rate limited
                if rate_limit_info and rate_limit_info.get('rate_limited'):
                    wait_seconds = int(rate_limit_info.get('wait_seconds', 0))
                    return {
                        "status": "rate_limited",
                        "message": f"Data provider rate limit. Please retry after {wait_seconds} seconds.",
                        "updated_count": 0,
                        "failed_count": len(pending),
                        "rate_limit_info": {
                            "wait_seconds": wait_seconds,
                            "retry_after": datetime.fromtimestamp(rate_limit_info.get('until', 0)).isoformat() if rate_limit_info.get('until') else None
                        }
                    }
                return {
                    "status": "error",
                    "message": "Unable to fetch market data",
                    "updated_count": 0,
                    "failed_count": len(pending)
                }

            # Create a date-to-price mapping
            price_map = {}
            for date, row in hist.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                price_map[date_str] = float(row['Close'])

            # Update each pending prediction
            updated_count = 0
            failed_count = 0
            updated_dates = []
            failed_dates = []
            skipped_dates = []

            # Get sorted list of available dates for finding nearest trading day
            available_dates = sorted(
                [datetime.strptime(d, "%Y-%m-%d") for d in price_map.keys()])

            for pred in pending:
                pred_date = pred['date']
                pred_dt = datetime.strptime(pred_date, "%Y-%m-%d")

                if pred_date in price_map:
                    # Direct match - update with actual price
                    actual_price = price_map[pred_date]
                    if self.prediction_repo.update_prediction_with_actual_price(pred_date, actual_price):
                        updated_count += 1
                        updated_dates.append(pred_date)
                        logger.debug(
                            f"Updated prediction for {pred_date} with actual price ${actual_price:.2f}")
                    else:
                        failed_count += 1
                        failed_dates.append(pred_date)
                elif pred_dt > datetime.now():
                    # Future date - can't have actual price yet
                    skipped_dates.append(pred_date)
                    logger.debug(
                        f"Prediction date {pred_date} is in the future, skipping")
                else:
                    # Date not in price_map - might be weekend/holiday
                    # Try to find the nearest previous trading day
                    actual_price = None
                    for available_dt in reversed(available_dates):
                        if available_dt <= pred_dt:
                            # Use the last trading day's price
                            last_trading_date = available_dt.strftime(
                                "%Y-%m-%d")
                            actual_price = price_map.get(last_trading_date)
                            if actual_price:
                                logger.debug(
                                    f"Using last trading day ({last_trading_date}) price for {pred_date} (likely weekend/holiday)")
                                break

                    if actual_price:
                        if self.prediction_repo.update_prediction_with_actual_price(pred_date, actual_price):
                            updated_count += 1
                            updated_dates.append(pred_date)
                            logger.debug(
                                f"Updated prediction for {pred_date} with last trading day price ${actual_price:.2f}")
                        else:
                            failed_count += 1
                            failed_dates.append(pred_date)
                    else:
                        failed_count += 1
                        failed_dates.append(pred_date)
                        logger.warning(
                            f"No market data found for date {pred_date} and no previous trading day available")

            return {
                "status": "success",
                "message": f"Updated {updated_count} predictions, {failed_count} failed, {len(skipped_dates)} skipped (future dates)",
                "updated_count": updated_count,
                "failed_count": failed_count,
                "skipped_count": len(skipped_dates),
                "updated_dates": updated_dates,
                "failed_dates": failed_dates,
                "skipped_dates": skipped_dates,
                "total_pending": len(pending)
            }
        except Exception as e:
            logger.error(
                f"Error updating pending predictions: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "updated_count": 0,
                "failed_count": 0
            }
