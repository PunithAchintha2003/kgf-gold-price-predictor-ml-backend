"""Optimized prediction repository with query result caching and combined queries"""
import logging
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
from functools import lru_cache
import time

from ..core.database import get_db_connection, get_db_type, get_date_function

logger = logging.getLogger(__name__)

# Query result cache with TTL
_query_cache: Dict[str, tuple[any, float]] = {}
_CACHE_TTL = 30  # 30 seconds for query results


def _get_cached_query(key: str) -> Optional[any]:
    """Get cached query result if not expired"""
    if key in _query_cache:
        result, expiry = _query_cache[key]
        if time.time() < expiry:
            return result
        else:
            del _query_cache[key]
    return None


def _cache_query_result(key: str, result: any, ttl: int = _CACHE_TTL):
    """Cache query result with TTL"""
    expiry = time.time() + ttl
    _query_cache[key] = (result, expiry)
    
    # Cleanup old entries if cache gets too large
    if len(_query_cache) > 500:
        current_time = time.time()
        expired_keys = [k for k, (_, exp) in _query_cache.items() if current_time >= exp]
        for k in expired_keys:
            del _query_cache[k]


class PredictionRepository:
    """Repository for prediction database operations with performance optimizations"""

    @staticmethod
    def save_prediction(
        prediction_date: str,
        predicted_price: float,
        actual_price: Optional[float] = None,
        prediction_method: Optional[str] = None,
        prediction_reasons: Optional[str] = None
    ) -> bool:
        """Save prediction to database with accuracy calculation"""
        try:
            # Invalidate cache when saving new data
            _query_cache.clear()
            
            predicted_price = float(predicted_price) if predicted_price is not None else None
            actual_price = float(actual_price) if actual_price is not None else None

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Calculate accuracy if actual price is available
                accuracy = None
                if actual_price and predicted_price:
                    error_percentage = abs(predicted_price - actual_price) / actual_price * 100
                    accuracy = float(max(0, 100 - error_percentage))

                db_type = get_db_type()
                if db_type == "postgresql":
                    # Single UPSERT query
                    cursor.execute('''
                        INSERT INTO predictions 
                        (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (prediction_date) 
                        DO UPDATE SET
                            predicted_price = EXCLUDED.predicted_price,
                            actual_price = EXCLUDED.actual_price,
                            accuracy_percentage = EXCLUDED.accuracy_percentage,
                            prediction_method = EXCLUDED.prediction_method,
                            prediction_reasons = EXCLUDED.prediction_reasons,
                            updated_at = CURRENT_TIMESTAMP
                    ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method, prediction_reasons))
                else:
                    # SQLite - Single INSERT OR REPLACE
                    cursor.execute('''
                        INSERT OR REPLACE INTO predictions 
                        (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method, prediction_reasons))

                conn.commit()
                logger.debug(f"Saved prediction for {prediction_date}: ${predicted_price:.2f}")
                return True
        except Exception as e:
            logger.error(f"Error saving prediction for {prediction_date}: {e}", exc_info=True)
            return False

    @staticmethod
    def prediction_exists_for_date(prediction_date: str) -> bool:
        """Check if a prediction exists for the given date"""
        cache_key = f"exists_{prediction_date}"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute('SELECT EXISTS(SELECT 1 FROM predictions WHERE prediction_date = %s)', (prediction_date,))
            else:
                cursor.execute('SELECT EXISTS(SELECT 1 FROM predictions WHERE prediction_date = ?)', (prediction_date,))
            result = cursor.fetchone()[0]
            _cache_query_result(cache_key, bool(result), ttl=60)
            return bool(result)

    @staticmethod
    def get_prediction_for_date(prediction_date: str) -> Optional[float]:
        """Get predicted price for a specific date"""
        cache_key = f"pred_{prediction_date}"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute('''
                    SELECT predicted_price FROM predictions 
                    WHERE prediction_date = %s 
                    ORDER BY created_at DESC LIMIT 1
                ''', (prediction_date,))
            else:
                cursor.execute('''
                    SELECT predicted_price FROM predictions 
                    WHERE prediction_date = ? 
                    ORDER BY created_at DESC LIMIT 1
                ''', (prediction_date,))
            row = cursor.fetchone()
            result = float(row[0]) if row and row[0] is not None else None
            if result:
                _cache_query_result(cache_key, result, ttl=60)
            return result

    @staticmethod
    def get_prediction_details_for_date(prediction_date: str) -> Optional[Dict]:
        """Get full prediction details for a specific date"""
        cache_key = f"pred_details_{prediction_date}"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute('''
                    SELECT predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons
                    FROM predictions 
                    WHERE prediction_date = %s 
                    ORDER BY created_at DESC LIMIT 1
                ''', (prediction_date,))
            else:
                cursor.execute('''
                    SELECT predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons
                    FROM predictions 
                    WHERE prediction_date = ? 
                    ORDER BY created_at DESC LIMIT 1
                ''', (prediction_date,))
            row = cursor.fetchone()
            if row:
                result = {
                    'predicted_price': float(row[0]) if row[0] is not None else None,
                    'actual_price': float(row[1]) if row[1] is not None else None,
                    'accuracy_percentage': float(row[2]) if row[2] is not None else None,
                    'method': row[3],
                    'prediction_reasons': row[4]
                }
                _cache_query_result(cache_key, result, ttl=60)
                return result
            return None

    @staticmethod
    def get_historical_predictions(days: int = 90) -> List[Dict]:
        """Get historical predictions for the specified number of days - OPTIMIZED"""
        cache_key = f"hist_preds_{days}"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        date_func = get_date_function(-days)
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            if db_type == "postgresql":
                cursor.execute('''
                    SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method
                    FROM (
                        SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE prediction_date >= CURRENT_DATE - INTERVAL %s
                    ) ranked
                    WHERE rn = 1
                    ORDER BY prediction_date ASC
                ''', (f'{days} days',))
            else:
                cursor.execute(f'''
                    SELECT p1.prediction_date, p1.predicted_price, p1.actual_price, p1.accuracy_percentage, p1.prediction_method
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE prediction_date >= {date_func}
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                    ORDER BY p1.prediction_date ASC
                ''')

            results = cursor.fetchall()

        predictions = []
        for row in results:
            date_value = row[0]
            if isinstance(date_value, str):
                date_str = date_value
            else:
                date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, 'strftime') else str(date_value)
            
            predictions.append({
                'date': date_str,
                'predicted_price': float(row[1]) if row[1] is not None else None,
                'actual_price': float(row[2]) if row[2] is not None else None,
                'accuracy_percentage': float(row[3]) if row[3] is not None else None,
                'method': row[4] if row[4] else 'Lasso Regression'
            })
        
        _cache_query_result(cache_key, predictions, ttl=60)
        return predictions

    @staticmethod
    def get_accuracy_stats() -> Dict:
        """Get accuracy statistics - OPTIMIZED with caching"""
        cache_key = "accuracy_stats"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        
        # Check if table exists first (graceful handling for startup)
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                if db_type == "postgresql":
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'predictions'
                        )
                    """)
                else:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='predictions'
                    """)
                table_exists = cursor.fetchone()
                if not table_exists or (db_type == "postgresql" and not table_exists[0]):
                    # Table doesn't exist yet - return default stats
                    default_result = {
                        'average_accuracy': 0.0,
                        'r2_score': None,
                        'total_predictions': 0,
                        'evaluated_predictions': 0
                    }
                    _cache_query_result(cache_key, default_result, ttl=30)
                    return default_result
        except Exception as e:
            # If we can't check, assume table doesn't exist
            logger.debug(f"Could not check if predictions table exists: {e}")
            default_result = {
                'average_accuracy': 0.0,
                'r2_score': None,
                'total_predictions': 0,
                'evaluated_predictions': 0
            }
            _cache_query_result(cache_key, default_result, ttl=30)
            return default_result
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Single optimized query for PostgreSQL
                if db_type == "postgresql":
                    cursor.execute('''
                        WITH ranked_predictions AS (
                            SELECT 
                                prediction_date,
                                accuracy_percentage,
                                predicted_price,
                                actual_price,
                                prediction_method,
                                ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                            FROM predictions
                            WHERE accuracy_percentage IS NOT NULL
                        ),
                        weekday_predictions AS (
                            SELECT 
                                accuracy_percentage,
                                predicted_price,
                                actual_price,
                                prediction_method
                            FROM ranked_predictions
                            WHERE rn = 1
                            AND EXTRACT(DOW FROM prediction_date) BETWEEN 1 AND 5
                        )
                        SELECT 
                            COUNT(*) as total_evaluated,
                            AVG(accuracy_percentage) as avg_accuracy,
                            COUNT(CASE WHEN prediction_method IS NULL OR prediction_method != 'Manual Entry' THEN 1 END) as model_predictions
                        FROM weekday_predictions
                    ''')
                    row = cursor.fetchone()
                    evaluated_count = row[0] if row[0] else 0
                    avg_accuracy = float(row[1]) if row[1] is not None else 0.0
                    
                    # Get R² score in same query
                    cursor.execute('''
                        WITH ranked_predictions AS (
                            SELECT 
                                predicted_price,
                                actual_price,
                                ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                            FROM predictions
                            WHERE actual_price IS NOT NULL
                            AND predicted_price IS NOT NULL
                            AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                            AND ABS(predicted_price - actual_price) > 0.01
                        )
                        SELECT predicted_price, actual_price
                        FROM ranked_predictions
                        WHERE rn = 1
                    ''')
                else:
                    # SQLite - optimized single query
                    cursor.execute('''
                        SELECT 
                            COUNT(DISTINCT p1.prediction_date) as total_evaluated,
                            AVG(p1.accuracy_percentage) as avg_accuracy
                        FROM predictions p1
                        INNER JOIN (
                            SELECT prediction_date, MAX(created_at) as max_created_at
                            FROM predictions
                            WHERE accuracy_percentage IS NOT NULL
                            GROUP BY prediction_date
                        ) p2 ON p1.prediction_date = p2.prediction_date 
                            AND p1.created_at = p2.max_created_at
                        WHERE CAST(strftime('%w', p1.prediction_date) AS INTEGER) BETWEEN 1 AND 5
                    ''')
                    row = cursor.fetchone()
                    evaluated_count = row[0] if row[0] else 0
                    avg_accuracy = float(row[1]) if row[1] is not None else 0.0
                    
                    cursor.execute('''
                        SELECT p1.predicted_price, p1.actual_price
                        FROM predictions p1
                        INNER JOIN (
                            SELECT prediction_date, MAX(created_at) as max_created_at
                            FROM predictions
                            WHERE actual_price IS NOT NULL
                            AND predicted_price IS NOT NULL
                            AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                            AND ABS(predicted_price - actual_price) > 0.01
                            GROUP BY prediction_date
                        ) p2 ON p1.prediction_date = p2.prediction_date 
                            AND p1.created_at = p2.max_created_at
                    ''')

                price_data = cursor.fetchall()
                
                # Calculate R² score
                r2_score = None
                if len(price_data) >= 2:
                    try:
                        predicted = np.array([float(row[0]) for row in price_data])
                        actual = np.array([float(row[1]) for row in price_data])
                        
                        ss_res = np.sum((actual - predicted) ** 2)
                        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                        
                        if ss_tot > 0:
                            r2_score = float(1 - (ss_res / ss_tot))
                    except Exception as e:
                        logger.debug(f"Error calculating R²: {e}")

                result = {
                    'average_accuracy': round(avg_accuracy, 2),
                    'r2_score': round(r2_score, 4) if r2_score is not None else None,
                    'total_predictions': 0,  # Will be filled by comprehensive stats
                    'evaluated_predictions': evaluated_count
                }
                
                _cache_query_result(cache_key, result, ttl=60)
                return result
        except Exception as e:
            # Handle any database errors gracefully (e.g., table doesn't exist)
            error_msg = str(e).lower()
            if "no such table" in error_msg or "does not exist" in error_msg or "relation" not in error_msg:
                logger.debug(f"Predictions table not available: {e}")
            else:
                logger.warning(f"Error getting accuracy stats: {e}")
            
            # Return default values on error
            default_result = {
                'average_accuracy': 0.0,
                'r2_score': None,
                'total_predictions': 0,
                'evaluated_predictions': 0
            }
            _cache_query_result(cache_key, default_result, ttl=30)
            return default_result

    @staticmethod
    def get_comprehensive_stats() -> Dict:
        """Get comprehensive prediction statistics - OPTIMIZED single query"""
        cache_key = "comprehensive_stats"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Single comprehensive query for PostgreSQL
            if db_type == "postgresql":
                cursor.execute('''
                    WITH ranked_predictions AS (
                        SELECT 
                            prediction_date,
                            predicted_price,
                            actual_price,
                            accuracy_percentage,
                            prediction_method,
                            ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                    ),
                    weekday_stats AS (
                        SELECT 
                            prediction_date,
                            predicted_price,
                            actual_price,
                            accuracy_percentage,
                            prediction_method,
                            CASE WHEN actual_price IS NULL THEN 1 ELSE 0 END as is_pending
                        FROM ranked_predictions
                        WHERE rn = 1
                        AND EXTRACT(DOW FROM prediction_date) BETWEEN 1 AND 5
                    )
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN actual_price IS NOT NULL THEN 1 END) as evaluated_predictions,
                        COUNT(CASE WHEN actual_price IS NULL THEN 1 END) as pending_predictions,
                        AVG(accuracy_percentage) as avg_accuracy,
                        AVG(CASE WHEN prediction_method IS NULL OR prediction_method != 'Manual Entry' 
                            AND actual_price IS NOT NULL AND predicted_price IS NOT NULL
                            AND ABS(predicted_price - actual_price) > 0.01
                            THEN accuracy_percentage END) as model_avg_accuracy
                    FROM weekday_stats
                ''')
                row = cursor.fetchone()
                total_count = row[0] if row[0] else 0
                evaluated_count = row[1] if row[1] else 0
                pending_count = row[2] if row[2] else 0
                avg_accuracy = float(row[3]) if row[3] is not None else 0.0
                
                # Get R² in separate optimized query
                cursor.execute('''
                    WITH ranked_predictions AS (
                        SELECT 
                            predicted_price,
                            actual_price,
                            ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        AND predicted_price IS NOT NULL
                        AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                        AND ABS(predicted_price - actual_price) > 0.01
                    )
                    SELECT predicted_price, actual_price
                    FROM ranked_predictions
                    WHERE rn = 1
                ''')
            else:
                # SQLite optimized
                cursor.execute('''
                    SELECT 
                        COUNT(DISTINCT p1.prediction_date) as total_predictions,
                        COUNT(DISTINCT CASE WHEN p1.actual_price IS NOT NULL THEN p1.prediction_date END) as evaluated_predictions,
                        COUNT(DISTINCT CASE WHEN p1.actual_price IS NULL THEN p1.prediction_date END) as pending_predictions,
                        AVG(p1.accuracy_percentage) as avg_accuracy
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                    WHERE CAST(strftime('%w', p1.prediction_date) AS INTEGER) BETWEEN 1 AND 5
                ''')
                row = cursor.fetchone()
                total_count = row[0] if row[0] else 0
                evaluated_count = row[1] if row[1] else 0
                pending_count = row[2] if row[2] else 0
                avg_accuracy = float(row[3]) if row[3] is not None else 0.0
                
                cursor.execute('''
                    SELECT p1.predicted_price, p1.actual_price
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        AND predicted_price IS NOT NULL
                        AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                        AND ABS(predicted_price - actual_price) > 0.01
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                ''')

            price_data = cursor.fetchall()
            
            # Calculate R² score
            r2_score = None
            if len(price_data) >= 2:
                try:
                    predicted = np.array([float(row[0]) for row in price_data])
                    actual = np.array([float(row[1]) for row in price_data])
                    
                    ss_res = np.sum((actual - predicted) ** 2)
                    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                    
                    if ss_tot > 0:
                        r2_score = float(1 - (ss_res / ss_tot))
                except Exception as e:
                    logger.debug(f"Error calculating R²: {e}")

            evaluation_rate = (evaluated_count / total_count * 100) if total_count > 0 else 0.0

            result = {
                'total_predictions': total_count,
                'evaluated_predictions': evaluated_count,
                'pending_predictions': pending_count,
                'average_accuracy': round(avg_accuracy, 2),
                'r2_score': round(r2_score, 4) if r2_score is not None else None,
                'evaluation_rate': round(evaluation_rate, 2)
            }
            
            _cache_query_result(cache_key, result, ttl=60)
            return result

    @staticmethod
    def get_pending_predictions() -> List[Dict]:
        """Get all pending predictions - OPTIMIZED"""
        cache_key = "pending_predictions"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        db_type = get_db_type()
        with get_db_connection() as conn:
            cursor = conn.cursor()

            if db_type == "postgresql":
                cursor.execute('''
                    SELECT prediction_date, predicted_price, prediction_method
                    FROM (
                        SELECT prediction_date, predicted_price, prediction_method,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NULL
                    ) ranked
                    WHERE rn = 1
                    ORDER BY prediction_date ASC
                ''')
            else:
                cursor.execute('''
                    SELECT p1.prediction_date, p1.predicted_price, p1.prediction_method
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE actual_price IS NULL
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                    ORDER BY p1.prediction_date ASC
                ''')

            results = cursor.fetchall()

        pending = []
        for row in results:
            date_value = row[0]
            if isinstance(date_value, str):
                date_str = date_value
            else:
                date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, 'strftime') else str(date_value)
            
            # Skip weekends
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if date_obj.weekday() >= 5:  # Saturday or Sunday
                    continue
            except:
                pass
            
            pending.append({
                'date': date_str,
                'predicted_price': float(row[1]) if row[1] is not None else None,
                'method': row[2] if row[2] else 'Lasso Regression'
            })
        
        _cache_query_result(cache_key, pending, ttl=30)
        return pending

    @staticmethod
    def update_prediction_with_actual_price(
        prediction_date: str,
        actual_price: float
    ) -> bool:
        """Update an existing prediction with the actual price and recalculate accuracy"""
        try:
            actual_price = float(actual_price) if actual_price is not None else None
            if actual_price is None:
                logger.warning(f"Cannot update prediction for {prediction_date}: actual_price is None")
                return False

            # Get the existing prediction to preserve other fields
            existing_pred = PredictionRepository.get_prediction_details_for_date(prediction_date)
            if not existing_pred:
                logger.warning(f"No prediction found for {prediction_date} to update")
                return False
            
            # Use save_prediction to update (it handles upserts)
            # This preserves predicted_price, prediction_method, and prediction_reasons
            return PredictionRepository.save_prediction(
                prediction_date=prediction_date,
                predicted_price=existing_pred.get('predicted_price'),
                actual_price=actual_price,
                prediction_method=existing_pred.get('method'),
                prediction_reasons=existing_pred.get('prediction_reasons')
            )
        except Exception as e:
            logger.error(f"Error updating prediction with actual price for {prediction_date}: {e}", exc_info=True)
            return False

    @staticmethod
    def get_accuracy_visualization_data(days: int = 90) -> Dict:
        """Get detailed accuracy data for visualization - OPTIMIZED"""
        cache_key = f"accuracy_viz_{days}"
        cached = _get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        date_func = get_date_function(-days)
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            if db_type == "postgresql":
                cursor.execute('''
                    SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method
                    FROM (
                        SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        AND predicted_price IS NOT NULL
                        AND prediction_date >= CURRENT_DATE - INTERVAL %s
                        AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                        AND ABS(predicted_price - actual_price) > 0.01
                    ) ranked
                    WHERE rn = 1
                    ORDER BY prediction_date ASC
                ''', (f'{days} days',))
            else:
                cursor.execute(f'''
                    SELECT p1.prediction_date, p1.predicted_price, p1.actual_price, 
                           p1.accuracy_percentage, p1.prediction_method
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        AND predicted_price IS NOT NULL
                        AND prediction_date >= {date_func}
                        AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                        AND ABS(predicted_price - actual_price) > 0.01
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                    ORDER BY p1.prediction_date ASC
                ''')

            results = cursor.fetchall()

        data_points = []
        accuracy_values = []
        error_values = []

        for row in results:
            date_value = row[0]
            if isinstance(date_value, str):
                date_str = date_value
            else:
                date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, 'strftime') else str(date_value)
            
            predicted = float(row[1])
            actual = float(row[2])
            accuracy = float(row[3]) if row[3] is not None else None
            
            error_absolute = abs(predicted - actual)
            error_percentage = (error_absolute / actual * 100) if actual > 0 else 0
            
            if accuracy is None:
                accuracy = max(0, 100 - error_percentage)
            
            data_points.append({
                'date': date_str,
                'predicted_price': round(predicted, 2),
                'actual_price': round(actual, 2),
                'accuracy_percentage': round(accuracy, 2),
                'error_absolute': round(error_absolute, 2),
                'error_percentage': round(error_percentage, 2),
                'method': row[4] if row[4] else 'Lasso Regression'
            })
            
            accuracy_values.append(accuracy)
            error_values.append(error_absolute)

        statistics = {
            'average_accuracy': round(sum(accuracy_values) / len(accuracy_values), 2) if accuracy_values else 0.0,
            'min_accuracy': round(min(accuracy_values), 2) if accuracy_values else 0.0,
            'max_accuracy': round(max(accuracy_values), 2) if accuracy_values else 0.0,
            'average_error': round(sum(error_values) / len(error_values), 2) if error_values else 0.0,
            'total_predictions': len(data_points)
        }

        result = {
            'data': data_points,
            'statistics': statistics
        }
        
        _cache_query_result(cache_key, result, ttl=60)
        return result

