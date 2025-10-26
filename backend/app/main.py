from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import uvicorn
import sqlite3
import os
import time
from contextlib import contextmanager
from models.lasso_model import LassoGoldPredictor
from models.news_prediction import NewsEnhancedLassoPredictor, NewsSentimentAnalyzer

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "300"))
API_COOLDOWN = int(os.getenv("API_COOLDOWN", "2"))
REALTIME_CACHE_DURATION = int(os.getenv("REALTIME_CACHE_DURATION", "60"))

# Set up logging based on environment
log_level = getattr(logging, LOG_LEVEL.upper(), logging.WARNING)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Log environment info
logger.info(f"Starting app in {ENVIRONMENT} environment with log level {LOG_LEVEL}")


# Paths relative to backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
# Database setup - Optimized
DB_PATH = str(BACKEND_DIR / "data/gold_predictions.db")
BACKUP_DB_PATH = str(BACKEND_DIR / "data/gold_predictions_backup.db")

# Database connection context manager for better performance


@contextmanager
def get_db_connection(db_path=DB_PATH):
    """Context manager for database connections - Optimized"""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better performance
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


# Optimized cache for market data to reduce API calls
_market_data_cache = {}
_cache_timestamp = None
# CACHE_DURATION is now set from environment variables above
_last_api_call = 0
# API_COOLDOWN is now set from environment variables above
_realtime_cache = {}
_realtime_cache_timestamp = None
# REALTIME_CACHE_DURATION is now set from environment variables above


def get_cached_market_data():
    """Get cached market data or fetch new data if cache is expired - Optimized"""
    global _market_data_cache, _cache_timestamp, _last_api_call

    now = datetime.now()
    if (_cache_timestamp is None or
        (now - _cache_timestamp).total_seconds() > CACHE_DURATION or
            not _market_data_cache):

        # Rate limiting: wait if we made a call recently
        current_time = time.time()
        if current_time - _last_api_call < API_COOLDOWN:
            time.sleep(API_COOLDOWN - (current_time - _last_api_call))

        # Try multiple symbols for better reliability - Prioritize spot price over ETF
        symbols_to_try = ["GC=F", "GOLD", "GLD"]
        hist = None

        for symbol in symbols_to_try:
            try:
                _last_api_call = time.time()
                gold = yf.Ticker(symbol)
                hist = gold.history(period="1mo", interval="1d")
                if not hist.empty:
                    # Validate that we're getting a reasonable gold price (not ETF price)
                    current_price = float(hist['Close'].iloc[-1])
                    
                    # Skip GLD if it's giving ETF prices (too low)
                    if symbol == "GLD" and current_price < 1000:
                        logger.warning(f"Skipping GLD ETF price: ${current_price:.2f} - too low for spot gold")
                        continue
                    
                    # Prefer spot gold symbols
                    if symbol in ["GC=F", "GOLD"] and current_price > 1000:
                        logger.info(f"Using spot gold price from {symbol}: ${current_price:.2f}")
                        _market_data_cache = {'hist': hist, 'symbol': symbol}
                        _cache_timestamp = now
                        break
                    elif symbol == "GLD" and current_price > 1000:
                        # Only use GLD if spot symbols fail and GLD gives reasonable price
                        logger.info(f"Using GLD price: ${current_price:.2f}")
                        _market_data_cache = {'hist': hist, 'symbol': symbol}
                        _cache_timestamp = now
                        break
            except Exception as e:
                # Only log errors, not warnings for better performance
                if symbol == symbols_to_try[-1]:  # Only log on last attempt
                    logger.error(f"All gold data sources failed: {e}")
                continue

        if hist is None or hist.empty:
            return None, None

    return _market_data_cache.get('hist'), _market_data_cache.get('symbol')


def get_realtime_price_data():
    """Get real-time price data with optimized caching - Performance optimized"""
    global _last_api_call, _realtime_cache, _realtime_cache_timestamp

    # Check cache first
    now = datetime.now()
    if (_realtime_cache_timestamp is None or
        (now - _realtime_cache_timestamp).total_seconds() > REALTIME_CACHE_DURATION or
            not _realtime_cache):

        try:
            # Rate limiting: wait if we made a call recently
            current_time = time.time()
            if current_time - _last_api_call < API_COOLDOWN:
                time.sleep(API_COOLDOWN - (current_time - _last_api_call))

            # Try multiple symbols for better reliability - Prioritize spot price over ETF
            symbols_to_try = ["GC=F", "GOLD", "GLD"]

            for symbol in symbols_to_try:
                try:
                    _last_api_call = time.time()
                    gold = yf.Ticker(symbol)
                    # Get very recent data with 1-minute intervals for real-time feel
                    hist = gold.history(period="1d", interval="1m")

                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        # Calculate price change from previous close
                        if len(hist) > 1:
                            prev_close = float(hist['Close'].iloc[-2])
                            price_change = current_price - prev_close
                            change_percentage = (
                                price_change / prev_close) * 100
                        else:
                            price_change = 0
                            change_percentage = 0

                        result = {
                            'current_price': round(current_price, 2),
                            'price_change': round(price_change, 2),
                            'change_percentage': round(change_percentage, 2),
                            'last_updated': hist.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat()
                        }

                        # Cache the result
                        _realtime_cache = result
                        _realtime_cache_timestamp = now
                        return result

                except Exception as e:
                    # Only log on last attempt for performance
                    if symbol == symbols_to_try[-1]:
                        logger.error(f"Failed to fetch real-time data: {e}")
                    continue

            # Fallback to cached data if real-time fails
            hist, symbol = get_cached_market_data()
            if hist is not None and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                result = {
                    'current_price': round(current_price, 2),
                    'price_change': None,
                    'change_percentage': None,
                    'last_updated': None,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
                _realtime_cache = result
                _realtime_cache_timestamp = now
                return result

            return None

        except Exception as e:
            logger.error(f"Error fetching real-time price data: {e}")
            return None
    else:
        # Return cached data
        return _realtime_cache


# Initialize Lasso Regression predictor (primary model)
lasso_predictor = LassoGoldPredictor()
try:
    lasso_predictor.load_model(
        str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
    logger.info("Lasso Regression model loaded successfully")
except:
    logger.warning("Lasso Regression model not found, will train new model")
    # Train new Lasso model
    market_data = lasso_predictor.fetch_market_data()
    if market_data:
        features_df = lasso_predictor.create_fundamental_features(market_data)
        X, y = lasso_predictor.prepare_training_data(features_df)
        if not X.empty:
            lasso_predictor.train_model(X, y)
            lasso_predictor.save_model(
                str(BACKEND_DIR / 'models/lasso_gold_model.pkl'))
            logger.info("New Lasso Regression model trained and saved")
        else:
            logger.error("Failed to prepare training data for Lasso model")
    else:
        logger.error("Failed to fetch market data for Lasso model")


# Initialize News-Enhanced Lasso predictor
news_enhanced_predictor = NewsEnhancedLassoPredictor()
try:
    news_enhanced_predictor.load_enhanced_model(
        str(BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'))
    logger.info("News-enhanced Lasso model loaded successfully")
except:
    logger.warning("News-enhanced Lasso model not found, will train new model")
    # Train new enhanced model
    market_data = news_enhanced_predictor.fetch_market_data() if hasattr(news_enhanced_predictor, 'fetch_market_data') else lasso_predictor.fetch_market_data()
    if market_data:
        # Fetch news sentiment data
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(days_back=30)
        # Create enhanced features
        enhanced_features = news_enhanced_predictor.create_enhanced_features(market_data, sentiment_features)
        if not enhanced_features.empty:
            news_enhanced_predictor.train_enhanced_model(enhanced_features)
            news_enhanced_predictor.save_enhanced_model(
                str(BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl'))
            logger.info("New news-enhanced Lasso model trained and saved")
        else:
            logger.error("Failed to prepare enhanced training data")
    else:
        logger.error("Failed to fetch market data for news-enhanced model")


def init_database():
    """Initialize SQLite database for storing predictions - Optimized"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create predictions table with optimized indexes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                accuracy_percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create historical data table for ghost line
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prediction_date ON predictions(prediction_date)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_actual_price ON predictions(actual_price)
        ''')

        conn.commit()


def init_backup_database():
    """Initialize backup SQLite database for storing predictions - Optimized"""
    with get_db_connection(BACKUP_DB_PATH) as conn:
        cursor = conn.cursor()

        # Create predictions table with same structure plus backup timestamp
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                accuracy_percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                backup_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for backup database too
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_backup_prediction_date ON predictions(prediction_date)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_backup_created_at ON predictions(created_at)
        ''')

        conn.commit()


def backup_predictions():
    """Backup all predictions to backup database"""
    try:
        # Connect to both databases
        main_conn = sqlite3.connect(DB_PATH)
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)

        main_cursor = main_conn.cursor()
        backup_cursor = backup_conn.cursor()

        # Get all predictions from main database
        main_cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at
            FROM predictions
            ORDER BY created_at
        ''')

        predictions = main_cursor.fetchall()

        # Clear existing backup data
        backup_cursor.execute('DELETE FROM predictions')

        # Insert all predictions into backup database
        for pred in predictions:
            backup_cursor.execute('''
                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at, backup_created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', pred)

        backup_conn.commit()

        main_conn.close()
        backup_conn.close()

        logger.info(
            f"Successfully backed up {len(predictions)} predictions to backup database")
        return True

    except Exception as e:
        logger.error(f"Error backing up predictions: {e}")
        return False


def restore_from_backup():
    """Restore predictions from backup database"""
    try:
        # Connect to both databases
        main_conn = sqlite3.connect(DB_PATH)
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)

        main_cursor = main_conn.cursor()
        backup_cursor = backup_conn.cursor()

        # Get all predictions from backup database
        backup_cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at
            FROM predictions
            ORDER BY created_at
        ''')

        predictions = backup_cursor.fetchall()

        # Clear existing main data
        main_cursor.execute('DELETE FROM predictions')

        # Insert all predictions into main database
        for pred in predictions:
            main_cursor.execute('''
                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', pred)

        main_conn.commit()

        main_conn.close()
        backup_conn.close()

        logger.info(
            f"Successfully restored {len(predictions)} predictions from backup database")
        return True

    except Exception as e:
        logger.error(f"Error restoring from backup: {e}")
        return False


def save_prediction(prediction_date, predicted_price, actual_price=None):
    """Save prediction to database with correct accuracy calculation - Optimized"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Calculate accuracy if actual price is available (accuracy = 100 - error_percentage)
        accuracy = None
        if actual_price:
            error_percentage = abs(
                predicted_price - actual_price) / actual_price * 100
            # Accuracy - higher is better
            accuracy = max(0, 100 - error_percentage)

        cursor.execute('''
            INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage)
            VALUES (?, ?, ?, ?)
        ''', (prediction_date, predicted_price, actual_price, accuracy))

        conn.commit()

    # Automatically backup after saving
    backup_predictions()


def get_historical_predictions(days=30):
    """Get historical predictions for ghost line - all predictions including future ones - Optimized"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price
            FROM predictions p1
            WHERE prediction_date >= date('now', '-{} days')
            AND p1.created_at = (
                SELECT MAX(p2.created_at)
                FROM predictions p2
                WHERE p2.prediction_date = p1.prediction_date
            )
            ORDER BY prediction_date
        '''.format(days))

        results = cursor.fetchall()

    return [{
        'date': row[0],
        'predicted_price': row[1],
        'actual_price': row[2]
    } for row in results]


def get_ml_model_display_name():
    """Get the display name for the current ML model"""
    if lasso_predictor.model is not None:
        return "Lasso Regression"
    elif False:  # GRU model removed
        return "Legacy MLP Neural Network"
    else:
        return "No Model Available"


def get_accuracy_stats():
    """Get accuracy statistics for SMC method with real-time updates - only unique dates"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get accuracy for unique predictions with actual prices (matching chart display)
    cursor.execute('''
        SELECT AVG(accuracy_percentage), COUNT(DISTINCT prediction_date)
        FROM predictions p1
        WHERE accuracy_percentage IS NOT NULL
        AND prediction_date >= date('now', '-30 days')
        AND prediction_date != '2025-10-13'
        AND p1.created_at = (
            SELECT MAX(p2.created_at)
            FROM predictions p2
            WHERE p2.prediction_date = p1.prediction_date
        )
    ''')

    accuracy_result = cursor.fetchone()

    # Get total unique prediction dates (matching chart display)
    cursor.execute('''
        SELECT COUNT(DISTINCT prediction_date)
        FROM predictions p1
        WHERE prediction_date >= date('now', '-30 days')
        AND prediction_date != '2025-10-13'
        AND p1.created_at = (
            SELECT MAX(p2.created_at)
            FROM predictions p2
            WHERE p2.prediction_date = p1.prediction_date
        )
    ''')

    total_result = cursor.fetchone()
    conn.close()

    # Get R² score from recent performance (more realistic than training CV score)
    r2_score = 0.0
    try:
        # Calculate R² based on recent predictions vs actual prices
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get recent predictions with actual prices for R² calculation
        cursor.execute('''
            SELECT predicted_price, actual_price
            FROM predictions p1
            WHERE actual_price IS NOT NULL
            AND prediction_date >= date('now', '-30 days')
            AND prediction_date != '2025-10-13'
            AND p1.created_at = (
                SELECT MAX(p2.created_at)
                FROM predictions p2
                WHERE p2.prediction_date = p1.prediction_date
            )
            ORDER BY prediction_date DESC
            LIMIT 10
        ''')

        recent_data = cursor.fetchall()
        conn.close()

        if len(recent_data) >= 3:  # Need at least 3 data points for R²
            from sklearn.metrics import r2_score
            import numpy as np

            predicted = [row[0] for row in recent_data]
            actual = [row[1] for row in recent_data]

            r2 = r2_score(actual, predicted)
            r2_score = round(r2 * 100, 1)  # Convert to percentage
        else:
            # Fallback to training CV score if not enough recent data
            # GRU model removed
            r2_score = 0.0
    except:
        r2_score = 0.0

    if accuracy_result and accuracy_result[1] > 0:
        return {
            # Now shows average accuracy (higher is better)
            'average_accuracy': round(accuracy_result[0], 2),
            'total_predictions': total_result[0] if total_result else 0,
            'evaluated_predictions': accuracy_result[1],
            'r2_score': r2_score
        }
    return {
        'average_accuracy': 0,
        'total_predictions': total_result[0] if total_result else 0,
        'evaluated_predictions': 0,
        'r2_score': r2_score
    }


def update_actual_prices_realtime():
    """Update actual prices for past predictions using real-time data for continuous accuracy updates"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get predictions that need updating (including recent ones for real-time accuracy)
    # Only include predictions for dates that have market data available
    cursor.execute('''
        SELECT id, prediction_date, predicted_price, actual_price, accuracy_percentage
        FROM predictions
        WHERE prediction_date < date('now')
        AND actual_price IS NULL
        ORDER BY prediction_date
    ''')

    predictions = cursor.fetchall()

    # Get real-time market data
    try:
        gold = yf.Ticker("GC=F")
        # Get recent data with higher frequency for real-time updates
        # 2 days with 1-minute intervals
        hist = gold.history(period="2d", interval="1m")

        if hist.empty:
            # Fallback to daily data
            hist = gold.history(period="1mo", interval="1d")

        # Create a mapping of available dates
        available_dates = set()
        for date in hist.index:
            available_dates.add(date.strftime('%Y-%m-%d'))

        logger.info(
            f"Available market data dates: {sorted(list(available_dates))[-10:]}")
        logger.info(
            f"Latest market data date: {max(available_dates) if available_dates else 'None'}")
        logger.info(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        conn.close()
        return

    updated_count = 0
    skipped_dates = set()  # Track which dates we've already logged as skipped
    logger.info(f"Found {len(predictions)} predictions to evaluate")
    for pred_id, pred_date, pred_price, current_actual, current_accuracy in predictions:
        try:
            # Process all predictions including October 11th

            # Only try to fetch data if we know the date has market data
            if pred_date in available_dates:
                # For today's predictions, use real-time price; for past dates, use historical close
                if pred_date == datetime.now().strftime('%Y-%m-%d'):
                    # Use real-time price for today's predictions
                    actual_price = float(hist['Close'].iloc[-1])
                    logger.info(
                        f"Using real-time price for {pred_date}: ${actual_price:.2f}")
                else:
                    # Use historical close price for past dates
                    target_date = datetime.strptime(
                        pred_date, '%Y-%m-%d').date()
                    matching_rows = hist[hist.index.date == target_date]
                    if not matching_rows.empty:
                        actual_price = float(matching_rows['Close'].iloc[0])
                    else:
                        logger.warning(f"Could not find data for {pred_date}")
                        continue

                # Calculate accuracy
                error_percentage = abs(
                    pred_price - actual_price) / actual_price * 100
                accuracy = max(0, 100 - error_percentage)

                # Update prediction with new accuracy
                cursor.execute('''
                    UPDATE predictions
                    SET actual_price = ?, accuracy_percentage = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (actual_price, accuracy, pred_id))

                updated_count += 1
                logger.info(
                    f"Updated {pred_date}: Predicted {pred_price:.2f}, Actual {actual_price:.2f}, Accuracy {accuracy:.2f}%")
            else:
                # Silently skip dates without market data - no logging needed
                pass

        except Exception as e:
            logger.error(f"Error updating actual price for {pred_date}: {e}")

    conn.commit()
    conn.close()
    logger.info(
        f"Updated {updated_count} predictions with real-time market data")


def update_actual_prices():
    """Update actual prices for past predictions with correct accuracy calculation (legacy function)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get predictions without actual prices
    cursor.execute('''
        SELECT id, prediction_date, predicted_price
        FROM predictions
        WHERE actual_price IS NULL
        AND prediction_date <= date('now')
        ORDER BY prediction_date
    ''')

    predictions = cursor.fetchall()

    # First, get available market data to avoid repeated API calls
    try:
        gold = yf.Ticker("GC=F")
        # Get 30 days of data to check what dates are available
        # Note: Using 1mo instead of 30d for better reliability with GC=F
        hist = gold.history(period="1mo", interval="1d")

        # Create a mapping of available dates
        available_dates = set()
        for date in hist.index:
            available_dates.add(date.strftime('%Y-%m-%d'))

        logger.info(
            f"Available market data dates: {sorted(list(available_dates))[-10:]}")

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        conn.close()
        return

    updated_count = 0
    logger.info(f"Found {len(predictions)} predictions to evaluate")
    for pred_id, pred_date, pred_price in predictions:
        try:
            # Get actual price for that specific date using a small range
            from datetime import datetime, timedelta
            pred_datetime = datetime.strptime(pred_date, '%Y-%m-%d')
            start_date = (pred_datetime - timedelta(days=1)
                          ).strftime('%Y-%m-%d')
            end_date = (pred_datetime + timedelta(days=1)
                        ).strftime('%Y-%m-%d')
            hist_specific = gold.history(
                start=start_date, end=end_date, interval="1d")

            if not hist_specific.empty:
                # Find the row for the specific prediction date
                target_date = datetime.strptime(
                    pred_date, '%Y-%m-%d').date()
                matching_rows = hist_specific[hist_specific.index.date == target_date]
                if not matching_rows.empty:
                    actual_price = float(matching_rows['Close'].iloc[0])
                else:
                    logger.warning(
                        f"Could not find data for {pred_date} in range")
                    continue
                # Correct accuracy calculation: accuracy = 100 - error_percentage
                error_percentage = abs(
                    pred_price - actual_price) / actual_price * 100
                # Accuracy - higher is better
                accuracy = max(0, 100 - error_percentage)

                cursor.execute('''
                    UPDATE predictions
                    SET actual_price = ?, accuracy_percentage = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (actual_price, accuracy, pred_id))

                updated_count += 1
                logger.info(
                    f"Updated {pred_date}: Predicted {pred_price:.2f}, Actual {actual_price:.2f}, Accuracy {accuracy:.2f}%")
            else:
                logger.warning(f"No market data available for {pred_date}")
        except Exception as e:
            logger.error(f"Error updating actual price for {pred_date}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Updated {updated_count} predictions with actual market data")


def prediction_exists_for_date(prediction_date):
    """Check if a prediction already exists for the given date"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) FROM predictions 
        WHERE prediction_date = ?
    ''', (prediction_date,))

    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def get_prediction_for_date(prediction_date):
    """Get the most recent prediction for the given date"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT predicted_price FROM predictions 
        WHERE prediction_date = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    ''', (prediction_date,))

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None


def update_same_day_predictions():
    """Update predictions for today's date when market data becomes available"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().strftime('%Y-%m-%d')

    # Get today's predictions that don't have actual prices yet
    cursor.execute('''
        SELECT id, prediction_date, predicted_price, actual_price, accuracy_percentage
        FROM predictions
        WHERE prediction_date = ?
        AND actual_price IS NULL
        ORDER BY created_at DESC
    ''', (today,))

    predictions = cursor.fetchall()

    if not predictions:
        logger.info(f"No same-day predictions found for {today}")
        conn.close()
        return

    # Get today's market data
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="1d", interval="1d")

        if hist.empty:
            logger.info(f"No market data available for {today} yet")
            conn.close()
            return

        # Check if we have data for today
        today_data = hist[hist.index.date == datetime.now().date()]
        if today_data.empty:
            logger.info(
                f"Market data for {today} not yet available (market may not have closed)")
            conn.close()
            return

        actual_price = float(today_data['Close'].iloc[0])
        logger.info(f"Found market data for {today}: ${actual_price:.2f}")

        updated_count = 0
        for pred_id, pred_date, pred_price, current_actual, current_accuracy in predictions:
            # Calculate accuracy
            error_percentage = abs(
                pred_price - actual_price) / actual_price * 100
            accuracy = max(0, 100 - error_percentage)

            # Update prediction with new accuracy
            cursor.execute('''
                UPDATE predictions
                SET actual_price = ?, accuracy_percentage = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (actual_price, accuracy, pred_id))

            updated_count += 1
            logger.info(
                f"Updated {pred_date}: Predicted {pred_price:.2f}, Actual {actual_price:.2f}, Accuracy {accuracy:.2f}%")

        conn.commit()
        logger.info(
            f"Updated {updated_count} same-day predictions for {today}")

    except Exception as e:
        logger.error(f"Error updating same-day predictions: {e}")
    finally:
        conn.close()


def cleanup_invalid_predictions():
    """Remove predictions for dates that don't have market data available"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get available market data dates
        gold = yf.Ticker("GC=F")
        # Note: Using 1mo instead of 30d for better reliability with GC=F
        hist = gold.history(period="1mo", interval="1d")

        available_dates = set()
        for date in hist.index:
            available_dates.add(date.strftime('%Y-%m-%d'))

        # Find predictions for dates that don't have market data
        cursor.execute('''
            SELECT id, prediction_date
            FROM predictions
            WHERE prediction_date < date('now')
            AND prediction_date NOT IN ({})
        '''.format(','.join([f"'{date}'" for date in available_dates])))

        invalid_predictions = cursor.fetchall()

        if invalid_predictions:
            logger.info(
                f"Found {len(invalid_predictions)} predictions for dates without market data")

            # Delete invalid predictions
            cursor.execute('''
                DELETE FROM predictions
                WHERE prediction_date < date('now')
                AND prediction_date NOT IN ({})
            '''.format(','.join([f"'{date}'" for date in available_dates])))

            conn.commit()
            logger.info(
                f"Cleaned up {len(invalid_predictions)} invalid predictions")
        else:
            logger.info("No invalid predictions found")

    except Exception as e:
        logger.error(f"Error cleaning up invalid predictions: {e}")
    finally:
        conn.close()


# Initialize databases on startup
init_database()
init_backup_database()

app = FastAPI(
    title="XAU/USD Real-time Data API", 
    version="1.0.0",
    description=f"Gold price prediction API running in {ENVIRONMENT} environment"
)

# Add CORS middleware to allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for frontend connectivity verification"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "XAU/USD Real-time Data API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "log_level": LOG_LEVEL,
        "cache_duration": CACHE_DURATION,
        "api_cooldown": API_COOLDOWN
    }


@app.post("/backup")
async def create_backup():
    """Create backup of all predictions"""
    success = backup_predictions()
    return {
        "status": "success" if success else "error",
        "message": "Backup created successfully" if success else "Failed to create backup",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/restore")
async def restore_backup():
    """Restore predictions from backup"""
    success = restore_from_backup()
    return {
        "status": "success" if success else "error",
        "message": "Data restored from backup successfully" if success else "Failed to restore from backup",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/backup/status")
async def backup_status():
    """Get backup database status"""
    try:
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)
        backup_cursor = backup_conn.cursor()

        # Get backup statistics
        backup_cursor.execute('SELECT COUNT(*) FROM predictions')
        backup_count = backup_cursor.fetchone()[0]

        backup_cursor.execute('SELECT MAX(backup_created_at) FROM predictions')
        last_backup = backup_cursor.fetchone()[0]

        backup_conn.close()

        # Get main database count for comparison
        main_conn = sqlite3.connect(DB_PATH)
        main_cursor = main_conn.cursor()
        main_cursor.execute('SELECT COUNT(*) FROM predictions')
        main_count = main_cursor.fetchone()[0]
        main_conn.close()

        return {
            "status": "success",
            "main_database_count": main_count,
            "backup_database_count": backup_count,
            "last_backup": last_backup,
            "backup_synced": main_count == backup_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking backup status: {e}",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/debug/realtime")
async def debug_realtime():
    """Debug endpoint to test real-time price data"""
    try:
        realtime_data = get_realtime_price_data()
        if realtime_data:
            return {
                "status": "success",
                "realtime_data": realtime_data,
                "message": "Real-time data fetched successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Real-time data fetch failed",
                "realtime_data": None
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in debug realtime: {e}",
            "realtime_data": None
        }


@app.get("/debug/symbols")
async def debug_symbols():
    """Debug endpoint to test all gold symbols and their prices"""
    symbols_to_test = ["GC=F", "GOLD", "GLD"]
    results = {}
    
    for symbol in symbols_to_test:
        try:
            gold = yf.Ticker(symbol)
            hist = gold.history(period="1d", interval="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                results[symbol] = {
                    "price": current_price,
                    "type": "ETF" if symbol == "GLD" else "Spot Gold",
                    "status": "success"
                }
            else:
                results[symbol] = {
                    "status": "error",
                    "message": "No data available"
                }
        except Exception as e:
            results[symbol] = {
                "status": "error",
                "message": str(e)
            }
    
    return {
        "status": "success",
        "symbols_tested": symbols_to_test,
        "results": results,
        "recommendation": "Use GC=F or GOLD for spot gold prices, avoid GLD (ETF)",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/debug/clear-cache")
async def clear_cache():
    """Clear all caches to force fresh data fetch"""
    global _market_data_cache, _cache_timestamp, _realtime_cache, _realtime_cache_timestamp
    
    # Clear market data cache
    _market_data_cache = {}
    _cache_timestamp = None
    
    # Clear real-time cache
    _realtime_cache = {}
    _realtime_cache_timestamp = None
    
    return {
        "status": "success",
        "message": "All caches cleared successfully",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/performance")
async def get_performance_stats():
    """Performance monitoring endpoint - New"""
    try:
        return {
            "status": "success",
            "cache_info": {
                "market_data_cached": bool(_market_data_cache),
                "cache_age_seconds": (datetime.now() - _cache_timestamp).total_seconds() if _cache_timestamp else None,
                "realtime_cached": bool(_realtime_cache),
                "realtime_cache_age_seconds": (datetime.now() - _realtime_cache_timestamp).total_seconds() if _realtime_cache_timestamp else None
            },
            "websocket_connections": len(manager.active_connections),
            "cache_duration": CACHE_DURATION,
            "realtime_cache_duration": REALTIME_CACHE_DURATION,
            "api_cooldown": API_COOLDOWN,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting performance stats: {e}",
            "timestamp": datetime.now().isoformat()
        }


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")


manager = ConnectionManager()


def predict_next_day_price_ml():
    """Predict next day price using Lasso Regression model"""
    try:
        # Try Lasso Regression model first
        if lasso_predictor.model is not None:
            # Get fresh market data
            market_data = lasso_predictor.fetch_market_data()
            if not market_data:
                logger.error(
                    "Failed to fetch market data for Lasso prediction")
                return None

            # Create features
            features_df = lasso_predictor.create_fundamental_features(
                market_data)

            # Make prediction using Lasso Regression
            predicted_price = lasso_predictor.predict_next_price(features_df)

            logger.info(f"Lasso Regression prediction: ${predicted_price:.2f}")
            return round(predicted_price, 2)

        # Fallback to legacy ML model - REMOVED
        elif False:  # GRU model removed
            return None
        else:
            logger.error("No trained models available")
            return None

    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return None


def generate_prediction_explanation():
    """Get simplified prediction information without detailed market analysis"""
    try:
        # Get fresh market data
        market_data = lasso_predictor.fetch_market_data()
        if not market_data:
            return {"error": "Failed to fetch market data"}

        # Get current market data
        gold_data = market_data['gold']
        current_price = float(gold_data['Close'].iloc[-1])

        return {
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error generating prediction explanation: {e}")
        return {"error": f"Failed to generate explanation: {str(e)}"}


def predict_next_day_price_smc(hist_data):
    """Predict next day price using enhanced SMC (Smart Money Concepts) approach - DEPRECATED"""
    # This function is kept for backward compatibility but ML is now preferred
    logger.warning("SMC prediction is deprecated, using ML instead")
    return predict_next_day_price_ml()


def get_xauusd_daily_data():
    """Fetch XAU/USD daily data using yfinance with continuous predictions"""
    try:
        # Use cached market data to reduce API calls
        hist, symbol_used = get_cached_market_data()

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

        if not hist.empty:
            # Check if we already made a prediction for today
            today = datetime.now().strftime("%Y-%m-%d")
            next_day = (hist.index[-1] + timedelta(days=1)
                        ).strftime("%Y-%m-%d")

            # Make predictions if none exists for the next day
            should_make_prediction = not prediction_exists_for_date(next_day)

            predicted_price = None
            if should_make_prediction:
                # Predict next day price using ML
                predicted_price = predict_next_day_price_ml()

                # Save prediction to database
                if predicted_price:
                    save_prediction(next_day, predicted_price)
                    logger.info(
                        f"New ML prediction made for {next_day}: ${predicted_price:.2f}")
            else:
                # Get existing prediction for display
                existing_prediction = get_prediction_for_date(next_day)
                if existing_prediction:
                    predicted_price = existing_prediction
                    logger.info(
                        f"Using existing prediction for {next_day}: ${predicted_price:.2f}")

            # Update actual prices for past predictions with real-time data
            update_actual_prices_realtime()

            # Update same-day predictions if market data is available
            update_same_day_predictions()

            # Get historical predictions for ghost line
            historical_predictions = get_historical_predictions(30)

            # Get accuracy statistics
            accuracy_stats = get_accuracy_stats()

            # Convert to list of daily data points
            daily_data = []
            for date, row in hist.iterrows():
                daily_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })

            # Always include current price
            current_price = round(float(hist['Close'].iloc[-1]), 2)

            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": daily_data,
                "historical_predictions": historical_predictions,
                "accuracy_stats": accuracy_stats,
                "current_price": current_price,
                "prediction": {
                    "next_day": next_day,
                    "predicted_price": predicted_price,
                    "current_price": current_price,
                    "prediction_method": get_ml_model_display_name()
                } if predicted_price else None,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        else:
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": [],
                "current_price": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": "No data available"
            }
    except Exception as e:
        logger.error(f"Error fetching XAU/USD daily data: {e}")
        return {
            "symbol": "XAUUSD",
            "timeframe": "daily",
            "data": [],
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": str(e)
        }


@app.get("/")
async def root():
    return {"message": "XAU/USD Real-time Data API with News Sentiment Analysis", "status": "running"}


@app.get("/xauusd/news-sentiment")
async def get_news_sentiment():
    """Get current news sentiment analysis for gold"""
    try:
        # Fetch and analyze news
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(days_back=7)
        
        if sentiment_features.empty:
            return {
                "status": "error",
                "message": "No news sentiment data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get latest sentiment data
        latest_sentiment = sentiment_features.iloc[-1]
        
        return {
            "status": "success",
            "sentiment_data": {
                "date": latest_sentiment.get('date', ''),
                "combined_sentiment": round(float(latest_sentiment.get('combined_sentiment_mean', 0)), 4),
                "news_volume": int(latest_sentiment.get('news_count', 0)),
                "sentiment_trend": round(float(latest_sentiment.get('sentiment_trend', 0)), 4),
                "sentiment_volatility": round(float(latest_sentiment.get('sentiment_volatility', 0)), 4),
                "polarity_mean": round(float(latest_sentiment.get('polarity_mean', 0)), 4),
                "gold_sentiment": round(float(latest_sentiment.get('gold_sentiment_mean', 0)), 4)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching news sentiment: {e}")
        return {
            "status": "error",
            "message": f"Error fetching news sentiment: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd/enhanced-prediction")
async def get_enhanced_prediction():
    """Get gold price prediction using news-enhanced Lasso regression"""
    try:
        # Get market data
        market_data = lasso_predictor.fetch_market_data()
        if not market_data:
            return {
                "status": "error",
                "message": "Failed to fetch market data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Fetch news sentiment
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(days_back=7)
        
        # Create enhanced features
        enhanced_features = news_enhanced_predictor.create_enhanced_features(market_data, sentiment_features)
        
        if enhanced_features.empty:
            return {
                "status": "error",
                "message": "Failed to create enhanced features",
                "timestamp": datetime.now().isoformat()
            }
        
        # Make prediction
        prediction = news_enhanced_predictor.predict_with_news(enhanced_features)
        current_price = enhanced_features['gold_close'].iloc[-1]
        change = prediction - current_price
        change_pct = (change / current_price) * 100
        
        # Get feature importance
        feature_importance = news_enhanced_predictor.get_feature_importance()
        top_features = feature_importance.head(10).to_dict('records') if not feature_importance.empty else []
        
        # Get sentiment summary
        sentiment_summary = {}
        if not sentiment_features.empty:
            latest_sentiment = sentiment_features.iloc[-1]
            sentiment_summary = {
                "combined_sentiment": round(float(latest_sentiment.get('combined_sentiment_mean', 0)), 4),
                "news_volume": int(latest_sentiment.get('news_count', 0)),
                "sentiment_trend": round(float(latest_sentiment.get('sentiment_trend', 0)), 4)
            }
        
        return {
            "status": "success",
            "prediction": {
                "current_price": round(current_price, 2),
                "predicted_price": round(prediction, 2),
                "predicted_change": round(change, 2),
                "predicted_change_percentage": round(change_pct, 2),
                "model_type": "News-Enhanced Lasso Regression",
                "model_accuracy": round(news_enhanced_predictor.best_score, 4) if news_enhanced_predictor.best_score > -np.inf else None
            },
            "sentiment_analysis": sentiment_summary,
            "feature_importance": top_features,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making enhanced prediction: {e}")
        return {
            "status": "error",
            "message": f"Error making enhanced prediction: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd/compare-models")
async def compare_models():
    """Compare predictions from different models including news-enhanced"""
    try:
        # Get market data
        market_data = lasso_predictor.fetch_market_data()
        if not market_data:
            return {
                "status": "error",
                "message": "Failed to fetch market data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get current price
        current_price = market_data['gold']['Close'].iloc[-1]
        
        # Lasso prediction
        lasso_features = lasso_predictor.create_fundamental_features(market_data)
        lasso_prediction = lasso_predictor.predict_next_price(lasso_features)
        lasso_change = lasso_prediction - current_price
        lasso_change_pct = (lasso_change / current_price) * 100
        
        # News-enhanced prediction
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(days_back=7)
        enhanced_features = news_enhanced_predictor.create_enhanced_features(market_data, sentiment_features)
        enhanced_prediction = news_enhanced_predictor.predict_with_news(enhanced_features)
        enhanced_change = enhanced_prediction - current_price
        enhanced_change_pct = (enhanced_change / current_price) * 100
        
        # Legacy ML prediction - REMOVED (GRU model)
        
        return {
            "status": "success",
            "current_price": round(current_price, 2),
            "predictions": {
                "lasso_regression": {
                    "predicted_price": round(lasso_prediction, 2),
                    "predicted_change": round(lasso_change, 2),
                    "predicted_change_percentage": round(lasso_change_pct, 2),
                    "model_accuracy": round(lasso_predictor.best_score, 4) if lasso_predictor.best_score > -np.inf else None
                },
                "news_enhanced_lasso": {
                    "predicted_price": round(enhanced_prediction, 2),
                    "predicted_change": round(enhanced_change, 2),
                    "predicted_change_percentage": round(enhanced_change_pct, 2),
                    "model_accuracy": round(news_enhanced_predictor.best_score, 4) if news_enhanced_predictor.best_score > -np.inf else None
                },
                # Legacy ML model removed
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return {
            "status": "error",
            "message": f"Error comparing models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/xauusd")
async def get_daily_data():
    """REST endpoint to get XAU/USD daily data"""
    return get_xauusd_daily_data()


@app.get("/xauusd/realtime")
async def get_realtime_price():
    """REST endpoint to get real-time XAU/USD data with predictions"""
    # Get real-time price data
    realtime_data = get_realtime_price_data()

    if realtime_data:
        # Get the full daily data for predictions and historical data
        daily_data = get_xauusd_daily_data()

        # Merge real-time price data with daily data
        if daily_data.get('status') == 'success':
            daily_data.update({
                'current_price': realtime_data['current_price'],
                'price_change': realtime_data['price_change'],
                'change_percentage': realtime_data['change_percentage'],
                'last_updated': realtime_data['last_updated'],
                'realtime_symbol': realtime_data['symbol']
            })

        return daily_data
    else:
        # Fallback to regular daily data if real-time fails
        return get_xauusd_daily_data()


@app.get("/xauusd/explanation")
async def get_prediction_explanation():
    """REST endpoint to get prediction explanation"""
    return generate_prediction_explanation()


@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def get_exchange_rate(from_currency: str, to_currency: str):
    """REST endpoint to get exchange rate between currencies"""
    try:
        # For now, we'll use a simple approach with yfinance for major currency pairs
        # This is a simplified implementation - in production, you'd want to use a proper forex API

        if from_currency.upper() == "USD" and to_currency.upper() == "LKR":
            # For USD/LKR, we'll use a fallback rate since yfinance doesn't have LKR
            # In production, you'd integrate with a proper forex API like Fixer.io, Alpha Vantage, etc.
            fallback_rate = 300.0  # Approximate USD/LKR rate

            # Try to get a more accurate rate from a free API (this is just an example)
            # You might want to use a proper forex API service
            try:
                import requests
                # Example using a free API (replace with actual forex API)
                # response = requests.get(f"https://api.exchangerate-api.com/v4/latest/USD")
                # if response.status_code == 200:
                #     data = response.json()
                #     lkr_rate = data.get('rates', {}).get('LKR', fallback_rate)
                # else:
                #     lkr_rate = fallback_rate
                lkr_rate = fallback_rate
            except:
                lkr_rate = fallback_rate

            return {
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "exchange_rate": lkr_rate,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        else:
            # For other currency pairs, try using yfinance
            try:
                ticker = yf.Ticker(f"{from_currency}{to_currency}=X")
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    rate = round(float(hist['Close'].iloc[-1]), 4)
                    return {
                        "from_currency": from_currency.upper(),
                        "to_currency": to_currency.upper(),
                        "exchange_rate": rate,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }
                else:
                    return {
                        "from_currency": from_currency.upper(),
                        "to_currency": to_currency.upper(),
                        "exchange_rate": 1.0,
                        "timestamp": datetime.now().isoformat(),
                        "status": "error",
                        "message": f"No exchange rate data available for {from_currency}/{to_currency}"
                    }
            except Exception as e:
                return {
                    "from_currency": from_currency.upper(),
                    "to_currency": to_currency.upper(),
                    "exchange_rate": 1.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "message": f"Error fetching exchange rate: {str(e)}"
                }

    except Exception as e:
        logger.error(f"Error fetching exchange rate: {e}")
        return {
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "exchange_rate": 1.0,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": str(e)
        }


@app.websocket("/ws/xauusd")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for XAU/USD real-time data updates - Optimized"""
    await manager.connect(websocket)

    try:
        last_sent_data = None
        while True:
            # Get real-time price data
            realtime_data = get_realtime_price_data()

            if realtime_data:
                # Get the full daily data for predictions and historical data
                daily_data = get_xauusd_daily_data()

                # Merge real-time price data with daily data
                if daily_data.get('status') == 'success':
                    daily_data.update({
                        'current_price': realtime_data['current_price'],
                        'price_change': realtime_data['price_change'],
                        'change_percentage': realtime_data['change_percentage'],
                        'last_updated': realtime_data['last_updated'],
                        'realtime_symbol': realtime_data['symbol']
                    })
            else:
                # Fallback to regular daily data if real-time fails
                daily_data = get_xauusd_daily_data()

            # Only send if data has changed to reduce unnecessary updates
            if daily_data != last_sent_data:
                await manager.send_personal_message(json.dumps(daily_data), websocket)
                last_sent_data = daily_data

            # Increased interval for better performance - 10 seconds instead of 5
            await asyncio.sleep(10)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_daily_data():
    """Background task to broadcast daily data to all connected clients - Optimized"""
    last_broadcast_data = None
    while True:
        if manager.active_connections:
            daily_data = get_xauusd_daily_data()
            # Only broadcast if data has changed to reduce unnecessary updates
            if daily_data != last_broadcast_data:
                await manager.broadcast(json.dumps(daily_data))
                last_broadcast_data = daily_data
        # Increased interval for better performance - 5 seconds instead of 2
        await asyncio.sleep(5)


async def continuous_accuracy_updates():
    """Background task to continuously update accuracy - Optimized"""
    while True:
        try:
            # Reduced logging for performance
            update_actual_prices_realtime()
            update_same_day_predictions()
        except Exception as e:
            logger.error(f"Error in continuous accuracy update: {e}")

        # Increased interval for better performance - 15 minutes instead of 10
        await asyncio.sleep(900)  # 15 minutes


@app.on_event("startup")
async def startup_event():
    """Start background tasks for broadcasting daily data and continuous accuracy updates"""
    asyncio.create_task(broadcast_daily_data())
    asyncio.create_task(continuous_accuracy_updates())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
