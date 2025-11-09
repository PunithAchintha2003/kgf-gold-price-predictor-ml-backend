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

# PostgreSQL support
try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    psycopg2 = None
from models.lasso_model import LassoGoldPredictor
from models.news_prediction import NewsEnhancedLassoPredictor, NewsSentimentAnalyzer
import requests

# Configure yfinance to avoid Yahoo Finance blocking
# Apply patches to prevent bot detection
try:
    # Try to set a custom user agent
    import yfinance.const as yf_const
    if hasattr(yf_const, 'USER_AGENT'):
        yf_const.USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
except:
    pass


def create_yf_ticker(symbol, session=None):
    """Create a yfinance ticker - let yfinance handle anti-blocking with curl_cffi"""
    # yfinance >= 0.2.40 handles anti-blocking internally with curl_cffi
    # Don't pass session parameter - let yfinance create its own optimized session
    return yf.Ticker(symbol)


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
logger.info(
    f"Starting app in {ENVIRONMENT} environment with log level {LOG_LEVEL}")


# Paths relative to backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
# Ensure data directory exists (important on fresh/ephemeral deployments)
DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration - PostgreSQL (default) or SQLite (fallback)
# Default to PostgreSQL for production deployments
USE_POSTGRESQL = os.getenv("USE_POSTGRESQL", "true").lower() == "true"
POSTGRESQL_HOST = os.getenv("POSTGRESQL_HOST", "localhost")
POSTGRESQL_PORT = os.getenv("POSTGRESQL_PORT", "5432")
POSTGRESQL_DATABASE = os.getenv("POSTGRESQL_DATABASE", "gold_predictor")
POSTGRESQL_USER = os.getenv("POSTGRESQL_USER", "postgres")
# Password should be set via environment variable - no default for security
POSTGRESQL_PASSWORD = os.getenv("POSTGRESQL_PASSWORD")

# SQLite paths (fallback)
DB_PATH = str(BACKEND_DIR / "data/gold_predictions.db")
BACKUP_DB_PATH = str(BACKEND_DIR / "data/gold_predictions_backup.db")

# PostgreSQL connection pool
_postgresql_pool = None

# Database connection context manager for better performance


def get_db_type():
    """Get current database type"""
    use_postgresql = USE_POSTGRESQL and POSTGRESQL_AVAILABLE and _postgresql_pool is not None
    return "postgresql" if use_postgresql else "sqlite"


def get_date_function(days_offset=0):
    """Get database-appropriate date function"""
    db_type = get_db_type()
    if db_type == "postgresql":
        if days_offset == 0:
            return "CURRENT_DATE"
        elif days_offset < 0:
            return f"CURRENT_DATE - INTERVAL '{abs(days_offset)} days'"
        else:
            return f"CURRENT_DATE + INTERVAL '{days_offset} days'"
    else:
        # SQLite
        if days_offset == 0:
            return "date('now')"
        else:
            return f"date('now', '{days_offset:+d} days')"


def init_postgresql_pool():
    """Initialize PostgreSQL connection pool"""
    global _postgresql_pool
    if not POSTGRESQL_AVAILABLE:
        logger.warning("PostgreSQL library not available, falling back to SQLite")
        return False
    
    try:
        _postgresql_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min and max connections
            host=POSTGRESQL_HOST,
            port=POSTGRESQL_PORT,
            database=POSTGRESQL_DATABASE,
            user=POSTGRESQL_USER,
            password=POSTGRESQL_PASSWORD
        )
        if _postgresql_pool:
            logger.info(f"PostgreSQL connection pool initialized: {POSTGRESQL_DATABASE}")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL pool: {e}")
        logger.warning("Falling back to SQLite")
        return False
    return False


@contextmanager
def get_db_connection(db_path=None):
    """Context manager for database connections - Supports PostgreSQL and SQLite"""
    conn = None
    use_postgresql = USE_POSTGRESQL and POSTGRESQL_AVAILABLE and _postgresql_pool is not None
    
    try:
        if use_postgresql:
            # Use PostgreSQL
            conn = _postgresql_pool.getconn()
            if conn:
                yield conn
        else:
            # Fallback to SQLite
            if db_path is None:
                db_path = DB_PATH
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Better performance
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            yield conn
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        raise e
    finally:
        if conn:
            if use_postgresql:
                try:
                    # Return connection to pool (don't close it)
                    _postgresql_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    # If pool is full or connection is bad, close it
                    try:
                        conn.close()
                    except:
                        pass
            else:
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
# Track the last date to invalidate cache on date change
_last_cache_date = None


def get_cached_market_data(period="3mo"):
    """Get cached market data or fetch new data if cache is expired - Optimized
    
    Args:
        period: Period for historical data (e.g., "1mo", "3mo", "90d"). Default: "3mo"
    """
    global _market_data_cache, _cache_timestamp, _last_api_call, _last_cache_date

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # Clear cache if date changed (new day started)
    if _last_cache_date is not None and _last_cache_date != today:
        logger.info(
            f"Date changed from {_last_cache_date} to {today}, clearing cache")
        _market_data_cache = {}
        _cache_timestamp = None
        _realtime_cache = {}
        _realtime_cache_timestamp = None
        _last_cache_date = today

    if (_cache_timestamp is None or
        (now - _cache_timestamp).total_seconds() > CACHE_DURATION or
            not _market_data_cache):

        # Rate limiting: wait if we made a call recently
        current_time = time.time()
        if current_time - _last_api_call < API_COOLDOWN:
            time.sleep(API_COOLDOWN - (current_time - _last_api_call))

        # Try multiple symbols for better reliability - Prioritize XAU/USD spot price
        symbols_to_try = ["GC=F", "GOLD", "XAUUSD=X",
                          "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
        hist = None

        for symbol in symbols_to_try:
            try:
                _last_api_call = time.time()
                gold = create_yf_ticker(symbol)
                hist = gold.history(period=period, interval="1d")

                if hist.empty:
                    logger.warning(f"Empty history data for {symbol}")

                if not hist.empty:
                    # Validate that we're getting a reasonable gold price (not ETF price)
                    current_price = float(hist['Close'].iloc[-1])

                    # Skip ETFs if they're giving prices too low for spot gold
                    etf_symbols = ["GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
                    if symbol in etf_symbols and current_price < 1000:
                        logger.warning(
                            f"Skipping {symbol} ETF price: ${current_price:.2f} - too low for spot gold")
                        continue

                    # Prefer spot gold symbols
                    if symbol in ["GC=F", "GOLD", "XAUUSD=X"] and current_price > 1000:
                        logger.info(
                            f"Using spot gold price from {symbol}: ${current_price:.2f}")
                        _market_data_cache = {'hist': hist, 'symbol': symbol}
                        _cache_timestamp = now
                        _last_cache_date = today
                        break
                    elif symbol in ["GLD", "IAU", "SGOL", "OUNZ", "AAAU"] and current_price > 1000:
                        # Only use ETF symbols if spot symbols fail and ETF gives reasonable price
                        logger.info(
                            f"Using {symbol} price: ${current_price:.2f}")
                        _market_data_cache = {'hist': hist, 'symbol': symbol}
                        _cache_timestamp = now
                        _last_cache_date = today
                        break
            except Exception as e:
                # Log detailed error for each symbol
                logger.error(
                    f"Error fetching {symbol}: {type(e).__name__}: {e}")
                if symbol == symbols_to_try[-1]:  # Summary on last attempt
                    logger.error(
                        f"All {len(symbols_to_try)} gold data sources failed")
                continue

        if hist is None or hist.empty:
            logger.error(
                "No market data available from any symbol - returning None")
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

            # Try multiple symbols for better reliability - Prioritize XAU/USD spot price
            symbols_to_try = ["GC=F", "GOLD", "XAUUSD=X",
                              "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]

            for symbol in symbols_to_try:
                try:
                    _last_api_call = time.time()
                    gold = create_yf_ticker(symbol)
                    # Get very recent data with 1-minute intervals for real-time feel
                    # GC=F doesn't support 1m intervals, so use daily for futures
                    if symbol == "GC=F":
                        hist = gold.history(period="5d", interval="1d")
                    else:
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

# Try multiple possible paths for the enhanced model
def find_enhanced_model_path():
    """Find the enhanced model file in various possible locations"""
    possible_paths = [
        BACKEND_DIR / 'models/enhanced_lasso_gold_model.pkl',
        Path(__file__).resolve().parent.parent / 'models/enhanced_lasso_gold_model.pkl',
        Path.cwd() / 'backend/models/enhanced_lasso_gold_model.pkl',
        Path.cwd() / 'models/enhanced_lasso_gold_model.pkl',
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return str(path)
    return None

enhanced_model_path = find_enhanced_model_path()

if enhanced_model_path:
    try:
        logger.info(f"Loading news-enhanced model from: {enhanced_model_path}")
        news_enhanced_predictor.load_enhanced_model(enhanced_model_path)
        logger.info("News-enhanced Lasso model loaded successfully")
    except Exception as e:
        logger.warning(
            f"News-enhanced Lasso model found but failed to load: {e} - using regular Lasso model")
        logger.debug(
            f"Exception details: {type(e).__name__}: {str(e)}", exc_info=True)
else:
    # Model file doesn't exist - this is expected if not trained yet
    logger.info(
        "News-enhanced Lasso model not found - using regular Lasso model. "
        "Train the model offline with: python -m models.news_prediction to generate the model file.")


def init_database():
    """Initialize database (PostgreSQL or SQLite) for storing predictions"""
    use_postgresql = USE_POSTGRESQL and POSTGRESQL_AVAILABLE and _postgresql_pool is not None
    
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if use_postgresql:
            # PostgreSQL syntax
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    accuracy_percentage REAL,
                    prediction_method TEXT DEFAULT 'Lasso Regression',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prediction_date ON predictions(prediction_date)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_actual_price ON predictions(actual_price)
            ''')
        else:
            # SQLite syntax
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

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes
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
        db_type = "PostgreSQL" if use_postgresql else "SQLite"
        logger.info(f"Database initialized successfully: {db_type}")


def init_backup_database():
    """Initialize backup database (SQLite only for now) - Optimized"""
    # Backup database is always SQLite (separate file)
    # Skip if using PostgreSQL (backup can be handled differently)
    if USE_POSTGRESQL and POSTGRESQL_AVAILABLE and _postgresql_pool is not None:
        logger.info("PostgreSQL enabled - skipping SQLite backup database initialization")
        return
    
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

        # Ensure legacy backups have the new column (migration)
        try:
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [row[1] for row in cursor.fetchall()]
            if "backup_created_at" not in columns:
                cursor.execute(
                    "ALTER TABLE predictions ADD COLUMN backup_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error migrating backup DB schema: {e}")


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
            try:
                backup_cursor.execute('''
                    INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at, backup_created_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', pred)
            except sqlite3.OperationalError as e:
                # Handle legacy schema missing backup_created_at column by migrating then retrying once
                if "no column named backup_created_at" in str(e):
                    try:
                        backup_cursor.execute(
                            "ALTER TABLE predictions ADD COLUMN backup_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                        )
                        backup_cursor.execute('''
                            INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at, backup_created_at)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', pred)
                    except Exception as e2:
                        logger.error(
                            f"Backup insert failed after migration: {e2}")
                        raise
                else:
                    raise

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


def save_prediction(prediction_date, predicted_price, actual_price=None, prediction_method=None):
    """Save prediction to database with correct accuracy calculation - Optimized"""
    try:
        # Get prediction method if not provided
        if prediction_method is None:
            prediction_method = get_ml_model_display_name()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Calculate accuracy if actual price is available (accuracy = 100 - error_percentage)
            accuracy = None
            if actual_price:
                error_percentage = abs(
                    predicted_price - actual_price) / actual_price * 100
                # Accuracy - higher is better
                accuracy = max(0, 100 - error_percentage)

            # Insert prediction (database-agnostic)
            db_type = get_db_type()
            if db_type == "postgresql":
                # PostgreSQL always has prediction_method column
                cursor.execute('''
                    INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method))
            else:
                # SQLite - check if prediction_method column exists
                try:
                    cursor.execute("PRAGMA table_info(predictions)")
                    columns = [row[1] for row in cursor.fetchall()]
                    has_prediction_method = 'prediction_method' in columns
                except:
                    has_prediction_method = False

                if has_prediction_method:
                    cursor.execute('''
                        INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method))
                else:
                    cursor.execute('''
                        INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage)
                        VALUES (?, ?, ?, ?)
                    ''', (prediction_date, predicted_price, actual_price, accuracy))

            conn.commit()
            logger.info(
                f"Successfully saved prediction for {prediction_date}: ${predicted_price:.2f} using {prediction_method}")

        # Automatically backup after saving
        try:
            backup_predictions()
        except Exception as backup_error:
            logger.warning(
                f"Prediction saved but backup failed: {backup_error}")

        return True
    except Exception as e:
        logger.error(
            f"Error saving prediction for {prediction_date}: {e}", exc_info=True)
        return False


def get_historical_predictions(days=90):
    """Get historical predictions for ghost line - all predictions including future ones - Optimized"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        date_func = get_date_function(-days)
        
        # Get the most recent prediction for each date (handle duplicates)
        # Database-agnostic query
        db_type = get_db_type()
        if db_type == "postgresql":
            cursor.execute('''
                SELECT prediction_date, predicted_price, actual_price
                FROM predictions p1
                WHERE prediction_date >= CURRENT_DATE - INTERVAL %s
                AND p1.created_at = (
                    SELECT MAX(p2.created_at)
                    FROM predictions p2
                    WHERE p2.prediction_date = p1.prediction_date
                )
                ORDER BY prediction_date ASC
            ''', (f'{days} days',))
        else:
            cursor.execute(f'''
                SELECT prediction_date, predicted_price, actual_price
                FROM predictions p1
                WHERE prediction_date >= {date_func}
                AND p1.created_at = (
                    SELECT MAX(p2.created_at)
                    FROM predictions p2
                    WHERE p2.prediction_date = p1.prediction_date
                )
                ORDER BY prediction_date ASC
            ''')

        results = cursor.fetchall()

    # Convert to list of dictionaries with proper formatting
    predictions = []
    for row in results:
        # Convert date to string format (YYYY-MM-DD) to match data format
        date_value = row[0]
        if hasattr(date_value, 'strftime'):
            date_str = date_value.strftime('%Y-%m-%d')
        elif isinstance(date_value, str):
            date_str = date_value
        else:
            date_str = str(date_value)
        
        predictions.append({
            'date': date_str,
            'predicted_price': round(float(row[1]), 2) if row[1] is not None else None,
            'actual_price': round(float(row[2]), 2) if row[2] is not None else None
        })

    return predictions


def get_ml_model_display_name():
    """Get the display name for the current ML model"""
    if news_enhanced_predictor.model is not None:
        return "News-Enhanced Lasso Regression"
    elif lasso_predictor.model is not None:
        return "Lasso Regression (Fallback)"
    elif False:  # GRU model removed
        return "Legacy MLP Neural Network"
    else:
        return "No Model Available"


def get_accuracy_stats():
    """Get accuracy statistics for SMC method with real-time updates - only unique dates"""
    date_func_30 = get_date_function(-30)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get accuracy for unique predictions with actual prices (matching chart display)
        cursor.execute(f'''
            SELECT AVG(accuracy_percentage), COUNT(DISTINCT prediction_date)
            FROM predictions p1
            WHERE accuracy_percentage IS NOT NULL
            AND prediction_date >= {date_func_30}
            AND prediction_date != '2025-10-13'
            AND p1.created_at = (
                SELECT MAX(p2.created_at)
                FROM predictions p2
                WHERE p2.prediction_date = p1.prediction_date
            )
        ''')

        accuracy_result = cursor.fetchone()

        # Get total unique prediction dates (matching chart display)
        cursor.execute(f'''
            SELECT COUNT(DISTINCT prediction_date)
            FROM predictions p1
            WHERE prediction_date >= {date_func_30}
            AND prediction_date != '2025-10-13'
            AND p1.created_at = (
                SELECT MAX(p2.created_at)
                FROM predictions p2
                WHERE p2.prediction_date = p1.prediction_date
            )
        ''')

        total_result = cursor.fetchone()

    # Get R² score from recent performance (more realistic than training CV score)
    r2_score = 0.0
    try:
        # Calculate R² based on recent predictions vs actual prices
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get recent predictions with actual prices for R² calculation
            cursor.execute(f'''
                SELECT predicted_price, actual_price
                FROM predictions p1
                WHERE actual_price IS NOT NULL
                AND prediction_date >= {date_func_30}
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
    db_type = get_db_type()
    date_func_now = get_date_function(0)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get predictions that need updating (including recent ones for real-time accuracy)
        # Only include predictions for dates that have market data available
        if db_type == "postgresql":
            cursor.execute(f'''
                SELECT id, prediction_date, predicted_price, actual_price, accuracy_percentage
                FROM predictions
                WHERE prediction_date < {date_func_now}
                    AND actual_price IS NULL
                ORDER BY prediction_date
            ''')
        else:
            cursor.execute(f'''
                SELECT id, prediction_date, predicted_price, actual_price, accuracy_percentage
                FROM predictions
                WHERE prediction_date < {date_func_now}
                    AND actual_price IS NULL
                ORDER BY prediction_date
            ''')

        predictions = cursor.fetchall()

        # Get real-time market data
        try:
            # Try multiple symbols for better reliability - Prioritize actual gold price (GC=F)
            symbols_to_try = ["GC=F", "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
            hist = None

            for symbol in symbols_to_try:
                try:
                    gold = create_yf_ticker(symbol)
                    # Get recent data with higher frequency for real-time updates
                    # 2 days with 1-minute intervals
                    hist = gold.history(period="2d", interval="1m")
                    if not hist.empty:
                        break
                except Exception as e:
                    if symbol == symbols_to_try[-1]:  # Last attempt
                        logger.error(f"All symbols failed for real-time data: {e}")
                    continue

            if hist is None or hist.empty:
                # Fallback to daily data with same symbol selection
                for symbol in symbols_to_try:
                    try:
                        gold = create_yf_ticker(symbol)
                        # Use a longer period to ensure we capture all needed dates
                        hist = gold.history(period="2mo", interval="1d")
                        if not hist.empty:
                            break
                    except Exception as e:
                        if symbol == symbols_to_try[-1]:  # Last attempt
                            logger.error(f"All symbols failed for daily data: {e}")
                        continue

            if hist is None or hist.empty:
                logger.error("Failed to fetch market data for price updates")
                return

            # Create a mapping of available dates
            available_dates = set()
            for date in hist.index:
                available_dates.add(date.strftime('%Y-%m-%d'))

            logger.debug(
                f"Available market data dates: {sorted(list(available_dates))[-15:]}")
            logger.debug(
                f"Latest market data date: {max(available_dates) if available_dates else 'None'}")
            logger.debug(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

            # Log which dates need updating
            dates_needing_update = [pred[1] for pred in predictions]
            missing_dates = [
                d for d in dates_needing_update if d not in available_dates]
            if missing_dates:
                logger.warning(
                    f"Market data not available for dates: {missing_dates}")

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return

        updated_count = 0
        skipped_dates = set()  # Track which dates we've already logged as skipped

        if len(predictions) == 0:
            logger.debug(
                "No past predictions need evaluation (all already evaluated)")
        else:
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
                            # Try to find the closest date (within 3 days) if exact match fails
                            logger.debug(
                                f"Exact date not found for {pred_date}, searching nearby dates")
                            target_datetime = datetime.strptime(
                                pred_date, '%Y-%m-%d')
                            for days_offset in range(-3, 4):  # Check ±3 days
                                search_date = (target_datetime +
                                               timedelta(days=days_offset)).date()
                                matching_rows = hist[hist.index.date ==
                                                     search_date]
                                if not matching_rows.empty:
                                    actual_price = float(
                                        matching_rows['Close'].iloc[0])
                                    logger.debug(
                                        f"Using data from {search_date} for {pred_date}")
                                    break
                            else:
                                logger.warning(
                                    f"Could not find data for {pred_date} (searched ±3 days)")
                                continue

                    # Calculate accuracy
                    error_percentage = abs(
                        pred_price - actual_price) / actual_price * 100
                    accuracy = max(0, 100 - error_percentage)

                    # Update prediction with new accuracy (database-agnostic)
                    db_type = get_db_type()
                    if db_type == "postgresql":
                        cursor.execute('''
                            UPDATE predictions
                            SET actual_price = %s, accuracy_percentage = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        ''', (actual_price, accuracy, pred_id))
                    else:
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
        # Only log if there were actual updates to avoid spam
        if updated_count > 0:
            logger.info(
                f"Updated {updated_count} predictions with real-time market data")
        else:
            logger.debug(
                f"No predictions needed updating (all already have actual prices)")


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
        # Try multiple symbols for better reliability - Prioritize actual gold price (GC=F)
        symbols_to_try = ["GC=F", "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
        hist = None

        for symbol in symbols_to_try:
            try:
                gold = create_yf_ticker(symbol)
                # Get 30 days of data to check what dates are available
                hist = gold.history(period="1mo", interval="1d")
                if not hist.empty:
                    break
            except Exception as e:
                if symbol == symbols_to_try[-1]:  # Last attempt
                    logger.error(f"All symbols failed for market data: {e}")
                continue

        # Create a mapping of available dates
        available_dates = set()
        for date in hist.index:
            available_dates.add(date.strftime('%Y-%m-%d'))

        # Only log at debug level to avoid spam - dates are already converted to strings
        logger.debug(
            f"Available market data dates: {sorted(list(available_dates))[-10:]}")

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        conn.close()
        return

    updated_count = 0

    if len(predictions) == 0:
        logger.debug(
            "No predictions need evaluation (all already have actual prices)")
    else:
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
    db_type = get_db_type()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if db_type == "postgresql":
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE prediction_date = %s
            ''', (prediction_date,))
        else:
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE prediction_date = ?
            ''', (prediction_date,))

        count = cursor.fetchone()[0]

    return count > 0


def get_prediction_for_date(prediction_date):
    """Get the most recent prediction for the given date"""
    db_type = get_db_type()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if db_type == "postgresql":
            cursor.execute('''
                SELECT predicted_price FROM predictions 
                WHERE prediction_date = %s
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (prediction_date,))
        else:
            cursor.execute('''
                SELECT predicted_price FROM predictions 
                WHERE prediction_date = ?
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (prediction_date,))

        result = cursor.fetchone()

    return result[0] if result else None


def predict_historical_date_ml(target_date_str):
    """Make ML prediction for a specific historical date using only data available up to that date
    
    Uses News-Enhanced Lasso (primary) or Lasso Regression (fallback) - ML methods only
    
    Args:
        target_date_str: Date string in 'YYYY-MM-DD' format
        
    Returns:
        Tuple of (predicted_price: float, method: str) or (None, None) if prediction fails
    """
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        
        # Fetch market data up to the target date (use period that includes target date)
        # We need data BEFORE the target date to make a prediction FOR the target date
        market_data = lasso_predictor.fetch_market_data(symbol='GC=F', period='2y')
        if not market_data or market_data['gold'].empty:
            logger.warning(f"No market data available for historical prediction on {target_date_str}")
            return None, None
        
        # Filter market data to only include dates before the target date
        gold_data = market_data['gold']
        gold_data_filtered = gold_data[gold_data.index.date < target_date]
        
        if gold_data_filtered.empty:
            logger.warning(f"No historical market data before {target_date_str} for prediction")
            return None, None
        
        # Filter all market data components to dates before target
        filtered_market_data = {}
        for key, df in market_data.items():
            if df is not None and not df.empty:
                filtered_df = df[df.index.date < target_date]
                if not filtered_df.empty:
                    filtered_market_data[key] = filtered_df
        
        if 'gold' not in filtered_market_data:
            logger.warning(f"Cannot create prediction for {target_date_str}: no gold data")
            return None, None
        
        # Try News-Enhanced Lasso first (if model available)
        if news_enhanced_predictor.model is not None:
            try:
                # For historical dates, we can't fetch historical news easily
                # So we'll skip news sentiment and use only technical features
                # Create enhanced features without news sentiment
                enhanced_features = news_enhanced_predictor.create_enhanced_features(
                    filtered_market_data, sentiment_df=None)
                
                if not enhanced_features.empty:
                    predicted_price = news_enhanced_predictor.predict_with_news(enhanced_features)
                    method = "News-Enhanced Lasso (Historical)"
                    logger.info(f"News-Enhanced Lasso historical prediction for {target_date_str}: ${predicted_price:.2f}")
                    return round(predicted_price, 2), method
            except Exception as e:
                logger.debug(f"News-Enhanced model failed for {target_date_str}: {e}, using Lasso Regression")
        
        # Fallback to Lasso Regression (primary method for historical predictions)
        if lasso_predictor.model is not None:
            # Create features from filtered market data
            features_df = lasso_predictor.create_fundamental_features(filtered_market_data)
            
            if not features_df.empty:
                predicted_price = lasso_predictor.predict_next_price(features_df)
                method = "Lasso Regression"
                logger.info(f"Lasso Regression historical prediction for {target_date_str}: ${predicted_price:.2f}")
                return round(predicted_price, 2), method
        
        logger.warning(f"No ML model available for historical prediction on {target_date_str}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error making historical ML prediction for {target_date_str}: {e}")
        return None, None


def backfill_missing_predictions(days=90):
    """Backfill missing predictions for past dates using ML models only
    
    Uses News-Enhanced Lasso (primary) or Lasso Regression (fallback) - NO simple methods
    
    Args:
        days: Number of days to check for missing predictions (default: 90)
    """
    try:
        # Get market data to see what dates are available (fetch enough for the range)
        period = f"{max(days, 90)}d" if isinstance(days, int) else "3mo"
        hist, symbol_used = get_cached_market_data(period=period)
        if hist is None or hist.empty:
            logger.warning(
                "Cannot backfill predictions: no market data available")
            return

        # Get all available market dates
        market_dates = set()
        for date in hist.index:
            market_dates.add(date.strftime('%Y-%m-%d'))

        # Get existing predictions
        with get_db_connection() as conn:
            cursor = conn.cursor()
            date_func = get_date_function(-days)
            cursor.execute(f'''
                SELECT DISTINCT prediction_date 
                FROM predictions 
                WHERE prediction_date >= {date_func}
            ''')
            existing_dates = set(row[0] for row in cursor.fetchall())

        # Find missing dates (dates with market data but no prediction)
        missing_dates = []
        today = datetime.now().date()
        for date_str in sorted(market_dates):
            if date_str not in existing_dates:
                # Check if it's a past date or today (not future)
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if date_obj <= today:  # Backfill past dates and today
                        missing_dates.append(date_str)
                except:
                    continue

        if not missing_dates:
            logger.debug("No missing predictions to backfill")
            return

        logger.info(f"Backfilling {len(missing_dates)} missing predictions using ML models only")

        # For each missing date, create a prediction using ML models
        backfilled_count = 0
        # Limit to avoid rate limiting but allow more historical data
        for missing_date in missing_dates[:50]:
            try:
                # Use ML prediction for historical date (returns price and method)
                predicted_price, prediction_method = predict_historical_date_ml(missing_date)
                
                if predicted_price and prediction_method:
                    # Save the prediction with the specific ML method used
                    success = save_prediction(
                        missing_date, predicted_price, prediction_method=prediction_method)
                    if success:
                        backfilled_count += 1
                        logger.info(
                            f"Backfilled ML prediction for {missing_date}: ${predicted_price:.2f} using {prediction_method}")
                else:
                    logger.warning(
                        f"Could not generate ML prediction for {missing_date}")
            except Exception as e:
                logger.warning(
                    f"Failed to backfill ML prediction for {missing_date}: {e}")
                continue

        if backfilled_count > 0:
            logger.info(
                f"Successfully backfilled {backfilled_count} predictions using ML models")
    except Exception as e:
        logger.error(
            f"Error backfilling missing predictions: {e}", exc_info=True)


def update_same_day_predictions():
    """Update predictions for today's date when market data becomes available"""
    db_type = get_db_type()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        today = datetime.now().strftime('%Y-%m-%d')

        # Get today's predictions that don't have actual prices yet
        if db_type == "postgresql":
            cursor.execute('''
                SELECT id, prediction_date, predicted_price, actual_price, accuracy_percentage
                FROM predictions
                WHERE prediction_date = %s
                    AND actual_price IS NULL
                ORDER BY created_at DESC
            ''', (today,))
        else:
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
            return

        # Get today's market data
        try:
            gold = create_yf_ticker("GC=F")
            hist = gold.history(period="1d", interval="1d")

            if hist.empty:
                logger.info(f"No market data available for {today} yet")
                return

            # Check if we have data for today
            today_data = hist[hist.index.date == datetime.now().date()]
            if today_data.empty:
                logger.info(
                    f"Market data for {today} not yet available (market may not have closed)")
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
                if db_type == "postgresql":
                    cursor.execute('''
                        UPDATE predictions
                        SET actual_price = %s, accuracy_percentage = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    ''', (actual_price, accuracy, pred_id))
                else:
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


def cleanup_invalid_predictions():
    """Remove predictions for dates that don't have market data available"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get available market data dates - Prioritize actual gold price (GC=F)
        symbols_to_try = ["GC=F", "GLD", "IAU", "SGOL", "OUNZ", "AAAU"]
        hist = None

        for symbol in symbols_to_try:
            try:
                gold = create_yf_ticker(symbol)
                hist = gold.history(period="1mo", interval="1d")
                if not hist.empty:
                    break
            except Exception as e:
                if symbol == symbols_to_try[-1]:  # Last attempt
                    logger.error(f"All symbols failed for cleanup: {e}")
                continue

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


# Initialize PostgreSQL pool if enabled (MUST be before init_database)
if USE_POSTGRESQL:
    if init_postgresql_pool():
        logger.info("✅ PostgreSQL enabled - using PostgreSQL database")
    else:
        logger.warning("⚠️  PostgreSQL initialization failed - falling back to SQLite")
        USE_POSTGRESQL = False  # Disable PostgreSQL if pool init failed

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


@app.get("/debug/db")
async def debug_db():
    """Diagnostics for database: counts and latest records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT MAX(prediction_date) FROM predictions')
        max_pred_date = cursor.fetchone()[0]

        cursor.execute('SELECT MAX(created_at) FROM predictions')
        max_created_at = cursor.fetchone()[0]

        cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 5
        ''')
        latest = cursor.fetchall()

        conn.close()

        latest_rows = [
            {
                "prediction_date": row[0],
                "predicted_price": row[1],
                "actual_price": row[2],
                "accuracy_percentage": row[3],
                "created_at": row[4]
            }
            for row in latest
        ]

        return {
            "status": "success",
            "db_path": DB_PATH,
            "total_predictions": total,
            "max_prediction_date": max_pred_date,
            "max_created_at": max_created_at,
            "latest": latest_rows,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"DB debug error: {e}",
            "db_path": DB_PATH,
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
    symbols_to_test = ["GC=F", "GOLD", "XAUUSD=X", "GLD"]
    results = {}

    for symbol in symbols_to_test:
        try:
            gold = create_yf_ticker(symbol)
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


@app.post("/debug/force-prediction")
async def force_prediction():
    """Force create a new prediction for testing (deletes existing if any)"""
    try:
        from datetime import timedelta
        next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Delete existing prediction if any
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM predictions WHERE prediction_date = ?', (next_day,))
            conn.commit()

        # Generate new prediction
        predicted_price = predict_next_day_price_ml()

        if predicted_price:
            prediction_method = get_ml_model_display_name()
            success = save_prediction(
                next_day, predicted_price, prediction_method=prediction_method)

            if success:
                return {
                    "status": "success",
                    "message": f"New prediction created for {next_day}",
                    "prediction": {
                        "date": next_day,
                        "predicted_price": predicted_price,
                        "method": prediction_method
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save prediction to database",
                    "timestamp": datetime.now().isoformat()
                }
        else:
            return {
                "status": "error",
                "message": "Failed to generate prediction",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error forcing prediction: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error forcing prediction: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/debug/xauusd-direct")
async def debug_xauusd_direct():
    """Direct XAU/USD price fetch bypassing cache with detailed diagnostics"""
    symbols_to_try = ["GC=F", "GOLD", "XAUUSD=X", "GLD"]
    results = []

    for symbol in symbols_to_try:
        try:
            gold = create_yf_ticker(symbol)
            hist = gold.history(period="1d", interval="1d")

            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])

                results.append({
                    "symbol": symbol,
                    "status": "success",
                    "price": current_price,
                    "data_points": len(hist),
                    "latest_date": hist.index[-1].strftime('%Y-%m-%d') if len(hist) > 0 else None,
                    "type": "Spot Gold" if current_price > 1000 else "ETF or Invalid"
                })

                # Only return if price is reasonable for spot gold
                if current_price > 1000 and symbol in ["GC=F", "GOLD", "XAUUSD=X"]:
                    return {
                        "status": "success",
                        "symbol": symbol,
                        "price": current_price,
                        "type": "Spot Gold",
                        "message": f"Direct fetch successful from {symbol}",
                        "timestamp": datetime.now().isoformat(),
                        "all_tested": results
                    }
            else:
                results.append({
                    "symbol": symbol,
                    "status": "empty",
                    "message": "No data returned"
                })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            logger.error(f"Error testing {symbol}: {e}")

    return {
        "status": "error",
        "message": "All XAU/USD symbols failed",
        "timestamp": datetime.now().isoformat(),
        "results": results
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
    """Predict next day price using News-Enhanced Lasso (primary) with Lasso Regression (fallback)"""
    try:
        # Try News-Enhanced Lasso model first (PRIMARY)
        if news_enhanced_predictor.model is not None:
            try:
                # Get fresh market data
                market_data = lasso_predictor.fetch_market_data()
                if not market_data:
                    logger.error("Failed to fetch market data")
                    raise ValueError("No market data")

                # Fetch news sentiment (if available)
                sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(
                    days_back=7)

                # Create enhanced features
                enhanced_features = news_enhanced_predictor.create_enhanced_features(
                    market_data, sentiment_features)

                if enhanced_features.empty:
                    raise ValueError("No enhanced features created")

                # Make prediction using News-Enhanced model
                predicted_price = news_enhanced_predictor.predict_with_news(
                    enhanced_features)

                logger.info(
                    f"News-Enhanced Lasso prediction: ${predicted_price:.2f}")
                return round(predicted_price, 2)

            except Exception as e:
                logger.warning(
                    f"News-Enhanced model failed: {e}, falling back to Lasso Regression")
                # Fallback to base Lasso model

        # Fallback to Lasso Regression model
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

            logger.info(
                f"Lasso Regression (fallback) prediction: ${predicted_price:.2f}")
            return round(predicted_price, 2)

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


def get_xauusd_daily_data(days=90):
    """Fetch XAU/USD daily data using yfinance with continuous predictions
    
    Args:
        days: Number of days of historical data to fetch (default: 90)
    """
    try:
        # Use cached market data to reduce API calls (fetch more days for 90-day range)
        period = f"{max(days, 90)}d" if isinstance(days, int) else "3mo"
        hist, symbol_used = get_cached_market_data(period=period)

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
            # Check if we already made a prediction for the next day
            today = datetime.now().strftime("%Y-%m-%d")
            # Use today as the base for calculating next day, not historical data
            # This ensures predictions are for tomorrow, not yesterday's next day
            next_day = (datetime.now() + timedelta(days=1)
                        ).strftime("%Y-%m-%d")

            # Make predictions if none exists for the next day
            should_make_prediction = not prediction_exists_for_date(next_day)

            predicted_price = None
            prediction_method = None
            if should_make_prediction:
                # Predict next day price using ML
                predicted_price = predict_next_day_price_ml()

                # Save prediction to database
                if predicted_price:
                    prediction_method = get_ml_model_display_name()
                    success = save_prediction(
                        next_day, predicted_price, prediction_method=prediction_method)
                    if success:
                        logger.info(
                            f"New ML prediction made for {next_day}: ${predicted_price:.2f} using {prediction_method}")
                    else:
                        logger.error(
                            f"Failed to save prediction for {next_day}: ${predicted_price:.2f}")
                else:
                    logger.warning(
                        f"Prediction generation returned None for {next_day}")
            else:
                # Get existing prediction for display
                existing_prediction = get_prediction_for_date(next_day)
                if existing_prediction:
                    predicted_price = existing_prediction
                    logger.info(
                        f"Using existing prediction for {next_day}: ${predicted_price:.2f}")

            # Backfill missing predictions for past dates (increased to match days parameter)
            # TEMPORARILY DISABLED - Only keeping Nov 10 prediction
            # try:
            #     backfill_missing_predictions(days)
            # except Exception as e:
            #     logger.warning(f"Backfill failed: {e}")

            # Update actual prices for past predictions with real-time data
            update_actual_prices_realtime()

            # Update same-day predictions if market data is available
            update_same_day_predictions()

            # Get historical predictions for ghost line (increased to 90 days for frontend)
            all_historical_predictions = get_historical_predictions(days)
            logger.info(f"Retrieved {len(all_historical_predictions)} total predictions from database")
            if all_historical_predictions:
                logger.info(f"Prediction date range: {all_historical_predictions[0]['date']} to {all_historical_predictions[-1]['date']}")
                before_oct6 = [p for p in all_historical_predictions if p['date'] < '2025-10-06']
                if before_oct6:
                    logger.info(f"Found {len(before_oct6)} predictions before Oct 6: {[p['date'] for p in before_oct6]}")

            # Get accuracy statistics
            accuracy_stats = get_accuracy_stats()

            # Convert to list of daily data points
            daily_data = []
            data_dates = set()  # Track dates in main data
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

            # Filter historical predictions to include:
            # 1. Dates that exist in main data (for accuracy comparison)
            # 2. Dates before market data starts (for historical predictions before Oct 6)
            # 3. Future dates (for showing predicted prices beyond current market data)
            # 4. Dates between min and max that don't have market data (weekends/holidays)
            if data_dates:
                # Get the min and max dates from market data
                min_data_date = min(data_dates) if data_dates else None
                max_data_date = max(data_dates) if data_dates else None
                
                # Include ALL predictions (don't filter out predictions without market data)
                # This ensures predictions for weekends/holidays are included
                historical_predictions = all_historical_predictions.copy()
                
                # Sort by date to ensure proper ordering
                historical_predictions.sort(key=lambda x: x['date'])
                
                # Debug logging
                before_count = len([p for p in historical_predictions if min_data_date and p['date'] < min_data_date])
                after_count = len([p for p in historical_predictions if max_data_date and p['date'] > max_data_date])
                matched_count = len([p for p in historical_predictions if p['date'] in data_dates])
                missing_count = len([p for p in historical_predictions if p['date'] not in data_dates and 
                                     (not min_data_date or p['date'] >= min_data_date) and 
                                     (not max_data_date or p['date'] <= max_data_date)])
                logger.info(f"Predictions: {matched_count} with market data, {before_count} before ({min_data_date}), {after_count} after ({max_data_date}), {missing_count} missing market data (weekends/holidays), total: {len(historical_predictions)}")
                
                # CRITICAL FIX: Ensure predictions before Oct 6 are included in data array
                # Add predictions that are in historical_predictions but not yet in daily_data
                # This ensures the accuracy line shows predicted prices for dates before Oct 6
                logger.info(f"Checking for predictions to add to data array. min_data_date={min_data_date}, total predictions={len(all_historical_predictions)}")
                
                # Get all dates currently in daily_data
                existing_data_dates = set(d['date'] for d in daily_data)
                
                # Find predictions that should be in data array but aren't yet
                # These are predictions that don't have corresponding market data (weekends, holidays)
                # EXCLUDE future dates - they should not appear in the gold price line
                predictions_to_add = []
                today = datetime.now().date()
                # Use ALL predictions, not just filtered ones, to ensure we don't miss any
                for pred in all_historical_predictions:
                    pred_date = pred['date']
                    try:
                        pred_date_obj = datetime.strptime(pred_date, '%Y-%m-%d').date()
                        # Only add predictions for past dates or today (not future dates)
                        # Future dates should only appear in the prediction line, not the gold price line
                        if pred_date not in existing_data_dates and pred_date_obj <= today:
                            predictions_to_add.append(pred)
                    except:
                        # Skip invalid dates
                        continue
                
                if predictions_to_add:
                    logger.info(f"✅ Adding {len(predictions_to_add)} predictions to data array for chart display")
                    logger.info(f"Predictions to add: {[p['date'] for p in predictions_to_add[:10]]}")
                    
                    # Add predictions to daily_data
                    for pred in predictions_to_add:
                        pred_date = pred['date']
                        # Check if this date has market data (shouldn't if we're adding it here)
                        if pred_date not in existing_data_dates:
                            daily_data.append({
                                "date": pred_date,
                                "open": None,  # No market data available
                                "high": None,
                                "low": None,
                                "close": None,  # No actual price available
                                "volume": 0,
                                "predicted_price": pred.get('predicted_price'),
                                "actual_price": pred.get('actual_price')
                            })
                    
                    # Re-sort daily_data by date to ensure proper chronological ordering
                    daily_data.sort(key=lambda x: x['date'])
                    logger.info(f"✅ Updated daily_data to include {len(daily_data)} total points (added {len(predictions_to_add)} predictions)")
                else:
                    logger.info(f"All predictions already in data array. Total data points: {len(daily_data)}")
            else:
                # If no data dates, return all predictions (shouldn't happen)
                historical_predictions = all_historical_predictions
                historical_predictions.sort(key=lambda x: x['date'])
            
            # Debug logging if filtering removed all predictions
            if len(all_historical_predictions) > 0 and len(historical_predictions) == 0:
                logger.warning(f"Filtered {len(all_historical_predictions)} predictions to 0. Sample data dates: {sorted(list(data_dates))[:3]}, Sample pred dates: {sorted([p['date'] for p in all_historical_predictions])[:3]}")

            # Always include current price
            current_price = round(float(hist['Close'].iloc[-1]), 2)

            # Merge predictions into data array for easier frontend consumption
            # Use ALL historical predictions (not just filtered ones) to ensure we don't miss any
            all_predictions_by_date = {p['date']: p for p in all_historical_predictions}
            
            # Add predicted_price and actual_price to data points where they exist
            enhanced_data = []
            matched_count = 0
            for data_point in daily_data:
                date = data_point['date']
                enhanced_point = {**data_point}
                
                # Check if this date has a prediction (use all predictions, not just filtered)
                if date in all_predictions_by_date:
                    pred = all_predictions_by_date[date]
                    # Only set if not already set (preserve existing values)
                    if 'predicted_price' not in enhanced_point or enhanced_point.get('predicted_price') is None:
                        enhanced_point['predicted_price'] = pred.get('predicted_price')
                    if 'actual_price' not in enhanced_point or enhanced_point.get('actual_price') is None:
                        enhanced_point['actual_price'] = pred.get('actual_price')
                    matched_count += 1
                # If predicted_price is already set (from earlier addition), keep it
                elif enhanced_point.get('predicted_price') is not None:
                    matched_count += 1  # Count as matched since it already has prediction
                
                enhanced_data.append(enhanced_point)
            
            logger.info(f"Merged {matched_count} predictions into {len(enhanced_data)} data points (using all {len(all_historical_predictions)} predictions)")
            
            # Build prediction object - ensure it's clear this is for FUTURE date only
            # IMPORTANT: Frontend should use prediction.next_day for the date, NOT the last data point date
            prediction_obj = None
            if predicted_price:
                prediction_obj = {
                    "next_day": next_day,
                    "predicted_price": predicted_price,
                    "current_price": current_price,
                    "prediction_method": get_ml_model_display_name(),
                    "warning": "This prediction is for a FUTURE date. Use 'next_day' as the date, NOT the last date in data array."
                }
            
            return {
                "symbol": "XAUUSD",
                "timeframe": "daily",
                "data": enhanced_data,  # Data with predictions merged in - use predicted_price field only
                "historical_predictions": historical_predictions,  # Keep separate array for backward compatibility
                "accuracy_stats": accuracy_stats,
                "current_price": current_price,
                "prediction": prediction_obj,  # Future prediction only - use next_day date, not last data date
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


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/")
async def root():
    return {"message": "XAU/USD Real-time Data API with News Sentiment Analysis", "status": "running"}


@app.get("/xauusd/news-sentiment")
async def get_news_sentiment():
    """Get current news sentiment analysis for gold"""
    try:
        # Fetch and analyze news
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(
            days_back=7)

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
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(
            days_back=7)

        # Create enhanced features
        enhanced_features = news_enhanced_predictor.create_enhanced_features(
            market_data, sentiment_features)

        if enhanced_features.empty:
            return {
                "status": "error",
                "message": "Failed to create enhanced features",
                "timestamp": datetime.now().isoformat()
            }

        # Make prediction
        prediction = news_enhanced_predictor.predict_with_news(
            enhanced_features)
        current_price = enhanced_features['gold_close'].iloc[-1]
        change = prediction - current_price
        change_pct = (change / current_price) * 100

        # Get feature importance
        feature_importance = news_enhanced_predictor.get_feature_importance()
        top_features = feature_importance.head(10).to_dict(
            'records') if not feature_importance.empty else []

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
        lasso_features = lasso_predictor.create_fundamental_features(
            market_data)
        lasso_prediction = lasso_predictor.predict_next_price(lasso_features)
        lasso_change = lasso_prediction - current_price
        lasso_change_pct = (lasso_change / current_price) * 100

        # News-enhanced prediction
        sentiment_features = news_enhanced_predictor.fetch_and_analyze_news(
            days_back=7)
        enhanced_features = news_enhanced_predictor.create_enhanced_features(
            market_data, sentiment_features)
        enhanced_prediction = news_enhanced_predictor.predict_with_news(
            enhanced_features)
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
async def get_daily_data(days: int = 90):
    """REST endpoint to get XAU/USD daily data
    
    Args:
        days: Number of days of historical data to return (default: 90)
    """
    return get_xauusd_daily_data(days=days)


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
                ticker = create_yf_ticker(f"{from_currency}{to_currency}=X")
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
