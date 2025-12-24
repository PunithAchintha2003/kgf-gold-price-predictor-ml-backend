"""Repository for prediction database operations"""
import logging
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np

from ..core.database import get_db_connection, get_db_type, get_date_function

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repository for prediction database operations"""

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
            # Convert numpy types to Python native types
            predicted_price = float(
                predicted_price) if predicted_price is not None else None
            actual_price = float(
                actual_price) if actual_price is not None else None

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Calculate accuracy if actual price is available
                accuracy = None
                if actual_price and predicted_price:
                    error_percentage = abs(
                        predicted_price - actual_price) / actual_price * 100
                    accuracy = float(max(0, 100 - error_percentage))

                # Insert or update prediction
                db_type = get_db_type()
                if db_type == "postgresql":
                    # Check if prediction exists first
                    cursor.execute('''
                        SELECT id FROM predictions WHERE prediction_date = %s
                    ''', (prediction_date,))
                    existing = cursor.fetchone()

                    # Check if prediction_reasons column exists
                    cursor.execute('''
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = 'predictions' AND column_name = 'prediction_reasons'
                    ''')
                    has_prediction_reasons = cursor.fetchone() is not None
                    
                    if existing:
                        # Update existing prediction
                        if has_prediction_reasons:
                            cursor.execute('''
                                UPDATE predictions
                                SET predicted_price = %s,
                                    actual_price = %s,
                                    accuracy_percentage = %s,
                                    prediction_method = %s,
                                    prediction_reasons = %s,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE prediction_date = %s
                            ''', (predicted_price, actual_price, accuracy, prediction_method, prediction_reasons, prediction_date))
                        else:
                            cursor.execute('''
                                UPDATE predictions
                                SET predicted_price = %s,
                                    actual_price = %s,
                                    accuracy_percentage = %s,
                                    prediction_method = %s,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE prediction_date = %s
                            ''', (predicted_price, actual_price, accuracy, prediction_method, prediction_date))
                    else:
                        # Insert new prediction
                        if has_prediction_reasons:
                            cursor.execute('''
                                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method, prediction_reasons))
                        else:
                            cursor.execute('''
                                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method)
                                VALUES (%s, %s, %s, %s, %s)
                            ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method))
                else:
                    # SQLite - check which columns exist
                    try:
                        cursor.execute("PRAGMA table_info(predictions)")
                        columns = [row[1] for row in cursor.fetchall()]
                        has_prediction_method = 'prediction_method' in columns
                        has_prediction_reasons = 'prediction_reasons' in columns
                    except Exception as e:
                        logger.debug(f"Error checking for columns: {e}")
                        has_prediction_method = False
                        has_prediction_reasons = False

                    if has_prediction_method and has_prediction_reasons:
                        cursor.execute('''
                            INSERT OR REPLACE INTO predictions 
                            (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, prediction_reasons, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method, prediction_reasons))
                    elif has_prediction_method:
                        cursor.execute('''
                            INSERT OR REPLACE INTO predictions 
                            (prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (prediction_date, predicted_price, actual_price, accuracy, prediction_method))
                    else:
                        cursor.execute('''
                            INSERT OR REPLACE INTO predictions 
                            (prediction_date, predicted_price, actual_price, accuracy_percentage, updated_at)
                            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (prediction_date, predicted_price, actual_price, accuracy))

                conn.commit()
                logger.info(
                    f"Saved prediction for {prediction_date}: ${predicted_price:.2f}")
                return True
        except Exception as e:
            logger.error(
                f"Error saving prediction for {prediction_date}: {e}", exc_info=True)
            return False

    @staticmethod
    def prediction_exists_for_date(prediction_date: str) -> bool:
        """Check if a prediction exists for the given date"""
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

    @staticmethod
    def get_prediction_for_date(prediction_date: str) -> Optional[float]:
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

        return float(result[0]) if result and result[0] is not None else None

    @staticmethod
    def get_prediction_details_for_date(prediction_date: str) -> Optional[Dict]:
        """Get full prediction details (price, method, reasons, etc.) for the given date"""
        db_type = get_db_type()
        has_prediction_method = False
        has_prediction_reasons = False

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check which columns exist
            try:
                if db_type == "postgresql":
                    cursor.execute("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = 'predictions' 
                        AND column_name IN ('prediction_method', 'prediction_reasons')
                    """)
                    existing_columns = [row[0] for row in cursor.fetchall()]
                    has_prediction_method = 'prediction_method' in existing_columns
                    has_prediction_reasons = 'prediction_reasons' in existing_columns
                else:
                    cursor.execute("PRAGMA table_info(predictions)")
                    columns = [row[1] for row in cursor.fetchall()]
                    has_prediction_method = 'prediction_method' in columns
                    has_prediction_reasons = 'prediction_reasons' in columns
            except Exception as e:
                logger.debug(f"Error checking for columns: {e}")
                has_prediction_method = False
                has_prediction_reasons = False

            # Build query based on available columns
            try:
                select_fields = ["predicted_price"]
                if has_prediction_method:
                    select_fields.append("prediction_method")
                if has_prediction_reasons:
                    select_fields.append("prediction_reasons")
                
                select_clause = ", ".join(select_fields)
                
                if db_type == "postgresql":
                    cursor.execute(f'''
                        SELECT {select_clause} FROM predictions 
                        WHERE prediction_date = %s
                        ORDER BY created_at DESC 
                        LIMIT 1
                    ''', (prediction_date,))
                else:
                    cursor.execute(f'''
                        SELECT {select_clause} FROM predictions 
                        WHERE prediction_date = ?
                        ORDER BY created_at DESC 
                        LIMIT 1
                    ''', (prediction_date,))
            except Exception as e:
                logger.error(f"Error querying prediction details: {e}", exc_info=True)
                return None

            result = cursor.fetchone()
            
            if not result:
                return None
            
            # Build result dictionary
            details = {
                "predicted_price": float(result[0]) if result[0] is not None else None,
            }
            
            idx = 1
            if has_prediction_method and len(result) > idx:
                details["method"] = result[idx] if result[idx] else None
                idx += 1
            else:
                details["method"] = None
                
            if has_prediction_reasons and len(result) > idx:
                details["prediction_reasons"] = result[idx] if result[idx] else None
            else:
                details["prediction_reasons"] = None
            
            return details

    @staticmethod
    def get_historical_predictions(days: int = 90) -> List[Dict]:
        """Get historical predictions for the specified number of days"""
        date_func = get_date_function(-days)
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            if db_type == "postgresql":
                # Optimized query using window function instead of correlated subquery
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
                # SQLite optimized using JOIN
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

        # Convert to list of dictionaries, filtering out weekends
        predictions = []
        for row in results:
            date_value = row[0]
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
                # Check if it's a weekend
                weekday = date_value.weekday()
            elif isinstance(date_value, str):
                date_str = date_value
                from datetime import datetime
                try:
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                    weekday = date_obj.weekday()
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing date '{date_value}': {e}")
                    weekday = None
            else:
                date_str = str(date_value)
                weekday = None

            # Skip weekends (Saturday=5, Sunday=6)
            if weekday is not None and weekday >= 5:
                continue

            predicted_price = round(
                float(row[1]), 2) if row[1] is not None else None
            actual_price = round(
                float(row[2]), 2) if row[2] is not None else None
            accuracy_percentage = round(float(row[3]), 2) if len(
                row) > 3 and row[3] is not None else None
            method = row[4] if len(row) > 4 and row[4] else 'Lasso Regression'

            # Calculate accuracy if not stored but we have both prices
            if accuracy_percentage is None and predicted_price is not None and actual_price is not None:
                accuracy_percentage = round(
                    100 - abs((predicted_price - actual_price) / actual_price * 100), 2)

            predictions.append({
                'date': date_str,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'accuracy_percentage': accuracy_percentage,
                'method': method
            })

        return predictions

    @staticmethod
    def get_accuracy_stats() -> Dict:
        """Get accuracy statistics"""
        try:
            db_type = get_db_type()

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Check if predictions table exists
                try:
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
                        logger.debug("Predictions table does not exist yet, returning default stats")
                        return {
                            'average_accuracy': 0.0,
                            'r2_score': None,
                            'total_predictions': 0,
                            'evaluated_predictions': 0
                        }
                except Exception as e:
                    logger.debug(f"Error checking if predictions table exists: {e}")
                    # If we can't check, assume table doesn't exist
                    return {
                        'average_accuracy': 0.0,
                        'r2_score': None,
                        'total_predictions': 0,
                        'evaluated_predictions': 0
                    }

                # Get accuracy for unique predictions with actual prices (excluding weekends)
                if db_type == "postgresql":
                    # Optimized query using window function
                    cursor.execute('''
                        SELECT prediction_date, accuracy_percentage
                        FROM (
                            SELECT prediction_date, accuracy_percentage,
                                   ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                            FROM predictions
                            WHERE accuracy_percentage IS NOT NULL
                        ) ranked
                        WHERE rn = 1
                    ''')
                else:
                    # SQLite optimized using JOIN
                    cursor.execute('''
                        SELECT p1.prediction_date, p1.accuracy_percentage
                        FROM predictions p1
                        INNER JOIN (
                            SELECT prediction_date, MAX(created_at) as max_created_at
                            FROM predictions
                            WHERE accuracy_percentage IS NOT NULL
                            GROUP BY prediction_date
                        ) p2 ON p1.prediction_date = p2.prediction_date 
                            AND p1.created_at = p2.max_created_at
                    ''')

                accuracy_results = cursor.fetchall()
                # Filter out weekends and calculate average
                evaluated_count = 0
                accuracy_values = []
                for row in accuracy_results:
                    try:
                        date_value = row[0]
                        if hasattr(date_value, 'weekday'):
                            weekday = date_value.weekday()
                        elif isinstance(date_value, str):
                            date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                            weekday = date_obj.weekday()
                        else:
                            continue

                        # Only count weekdays (Monday=0 to Friday=4)
                        if weekday < 5:
                            evaluated_count += 1
                            accuracy_values.append(float(row[1]))
                    except Exception as e:
                        logger.warning(f"Error processing accuracy row: {e}")
                        continue

                avg_accuracy = sum(accuracy_values) / \
                    len(accuracy_values) if accuracy_values else 0.0

                # Get total unique prediction dates (excluding weekends)
                if db_type == "postgresql":
                    # Optimized query using window function
                    cursor.execute('''
                        SELECT DISTINCT prediction_date
                        FROM (
                            SELECT prediction_date,
                                   ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                            FROM predictions
                        ) ranked
                        WHERE rn = 1
                    ''')
                else:
                    # SQLite optimized using JOIN
                    cursor.execute('''
                        SELECT DISTINCT p1.prediction_date
                        FROM predictions p1
                        INNER JOIN (
                            SELECT prediction_date, MAX(created_at) as max_created_at
                            FROM predictions
                            GROUP BY prediction_date
                        ) p2 ON p1.prediction_date = p2.prediction_date 
                            AND p1.created_at = p2.max_created_at
                    ''')

                total_dates = cursor.fetchall()
                # Filter out weekends - only count weekdays
                total_count = 0
                for row in total_dates:
                    try:
                        date_value = row[0]
                        if hasattr(date_value, 'weekday'):
                            weekday = date_value.weekday()
                        elif isinstance(date_value, str):
                            date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                            weekday = date_obj.weekday()
                        else:
                            continue

                        # Only count weekdays (Monday=0 to Friday=4)
                        if weekday < 5:
                            total_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing date row: {e}")
                        continue

                # Get predicted and actual prices for R² calculation
                # Exclude manual entries (where prediction_method = 'Manual Entry' or predicted_price = actual_price)
                if db_type == "postgresql":
                    # Optimized query using window function
                    cursor.execute('''
                        SELECT predicted_price, actual_price, prediction_method
                        FROM (
                            SELECT predicted_price, actual_price, prediction_method,
                                   ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                            FROM predictions
                            WHERE actual_price IS NOT NULL
                            AND predicted_price IS NOT NULL
                            AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                            AND ABS(predicted_price - actual_price) > 0.01
                        ) ranked
                        WHERE rn = 1
                    ''')
                else:
                    # SQLite optimized using JOIN
                    cursor.execute('''
                        SELECT p1.predicted_price, p1.actual_price, p1.prediction_method
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

                # Debug logging
                logger.debug(
                    f"R² calculation: Found {len(price_data) if price_data else 0} model predictions (excluding manual entries)")

            # avg_accuracy and evaluated_count are already calculated above (excluding weekends)

            # Calculate R² score (only for model predictions, excluding manual entries)
            r2_score = None
            if price_data and len(price_data) > 1:
                try:
                    predicted_prices = np.array(
                        [float(row[0]) for row in price_data])
                    actual_prices = np.array([float(row[1]) for row in price_data])

                    # Calculate R² = 1 - (SS_res / SS_tot)
                    ss_res = np.sum((actual_prices - predicted_prices) ** 2)
                    ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)

                    if ss_tot > 0:
                        r2_score = 1 - (ss_res / ss_tot)
                        r2_score = round(float(r2_score), 4)
                    else:
                        r2_score = None
                except Exception as e:
                    logger.warning(f"Error calculating R² score: {e}", exc_info=True)
                    r2_score = None

            return {
                'average_accuracy': round(avg_accuracy, 2),
                'r2_score': r2_score if r2_score is not None else None,
                'total_predictions': total_count,
                'evaluated_predictions': evaluated_count
            }
        except Exception as e:
            # Check if it's a "table doesn't exist" error
            error_msg = str(e).lower()
            if "no such table" in error_msg or "does not exist" in error_msg or "relation" in error_msg:
                logger.debug(f"Predictions table does not exist yet: {e}")
            else:
                logger.warning(f"Error getting accuracy stats: {e}", exc_info=True)
            # Return default values on error
            return {
                'average_accuracy': 0.0,
                'r2_score': None,
                'total_predictions': 0,
                'evaluated_predictions': 0
            }

    @staticmethod
    def get_accuracy_visualization_data(days: int = 90) -> Dict:
        """Get detailed accuracy data for visualization"""
        date_func = get_date_function(-days)
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get all predictions with actual prices (excluding weekends and manual entries)
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

        # Convert to list of accuracy data points, filtering out weekends
        data_points = []
        accuracy_values = []
        error_values = []

        for row in results:
            date_value = row[0]
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
                weekday = date_value.weekday()
            elif isinstance(date_value, str):
                date_str = date_value
                try:
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                    weekday = date_obj.weekday()
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing date '{date_value}': {e}")
                    weekday = None
            else:
                date_str = str(date_value)
                weekday = None

            # Skip weekends (Saturday=5, Sunday=6)
            if weekday is not None and weekday >= 5:
                continue

            predicted_price = float(row[1]) if row[1] is not None else None
            actual_price = float(row[2]) if row[2] is not None else None
            accuracy_percentage = float(row[3]) if row[3] is not None else None
            method = row[4] if len(row) > 4 and row[4] else 'Lasso Regression'

            if predicted_price is None or actual_price is None:
                continue

            # Calculate accuracy if not present
            if accuracy_percentage is None:
                accuracy_percentage = 100 - \
                    abs((predicted_price - actual_price) / actual_price * 100)

            # Calculate errors
            error_absolute = abs(predicted_price - actual_price)
            error_percentage = abs(
                (predicted_price - actual_price) / actual_price * 100)

            data_points.append({
                'date': date_str,
                'predicted_price': round(predicted_price, 2),
                'actual_price': round(actual_price, 2),
                'accuracy_percentage': round(accuracy_percentage, 2),
                'error_absolute': round(error_absolute, 2),
                'error_percentage': round(error_percentage, 2),
                'method': method
            })

            accuracy_values.append(accuracy_percentage)
            error_values.append(error_absolute)

        # Calculate statistics
        if accuracy_values:
            statistics = {
                'average_accuracy': round(sum(accuracy_values) / len(accuracy_values), 2),
                'min_accuracy': round(min(accuracy_values), 2),
                'max_accuracy': round(max(accuracy_values), 2),
                'average_error': round(sum(error_values) / len(error_values), 2),
                'total_predictions': len(data_points)
            }
        else:
            statistics = {
                'average_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'average_error': 0.0,
                'total_predictions': 0
            }

        return {
            'data': data_points,
            'statistics': statistics
        }

    @staticmethod
    def get_comprehensive_stats() -> Dict:
        """Get comprehensive prediction statistics (all time)"""
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get total unique predictions (all time) - need to filter weekends
            if db_type == "postgresql":
                # Optimized using window function
                cursor.execute('''
                    SELECT DISTINCT prediction_date
                    FROM (
                        SELECT prediction_date,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                    ) ranked
                    WHERE rn = 1
                ''')
            else:
                # SQLite optimized using JOIN
                cursor.execute('''
                    SELECT DISTINCT p1.prediction_date
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                ''')

            total_dates = cursor.fetchall()
            # Filter out weekends - only count weekdays
            total_count = 0
            for row in total_dates:
                date_value = row[0]
                if hasattr(date_value, 'weekday'):
                    weekday = date_value.weekday()
                elif isinstance(date_value, str):
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                    weekday = date_obj.weekday()
                else:
                    continue

                # Only count weekdays (Monday=0 to Friday=4)
                if weekday < 5:
                    total_count += 1

            # Get evaluated predictions (those with actual_price) - need to filter weekends
            if db_type == "postgresql":
                # Optimized using window function
                cursor.execute('''
                    SELECT DISTINCT prediction_date, accuracy_percentage
                    FROM (
                        SELECT prediction_date, accuracy_percentage,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                    ) ranked
                    WHERE rn = 1
                ''')
            else:
                # SQLite optimized using JOIN
                cursor.execute('''
                    SELECT DISTINCT p1.prediction_date, p1.accuracy_percentage
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                ''')

            evaluated_results = cursor.fetchall()
            # Filter out weekends and calculate average accuracy
            evaluated_count = 0
            accuracy_values = []
            for row in evaluated_results:
                date_value = row[0]
                if hasattr(date_value, 'weekday'):
                    weekday = date_value.weekday()
                elif isinstance(date_value, str):
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                    weekday = date_obj.weekday()
                else:
                    continue

                # Only count weekdays (Monday=0 to Friday=4)
                if weekday < 5:
                    evaluated_count += 1
                    if row[1] is not None:
                        accuracy_values.append(float(row[1]))

            avg_accuracy = sum(accuracy_values) / \
                len(accuracy_values) if accuracy_values else 0.0

            # Get predicted and actual prices for R² calculation (all time)
            # Exclude manual entries to get accurate model performance
            if db_type == "postgresql":
                # Optimized using window function
                cursor.execute('''
                    SELECT predicted_price, actual_price
                    FROM (
                        SELECT predicted_price, actual_price,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NOT NULL
                        AND predicted_price IS NOT NULL
                        AND (prediction_method IS NULL OR prediction_method != 'Manual Entry')
                        AND ABS(predicted_price - actual_price) > 0.01
                    ) ranked
                    WHERE rn = 1
                ''')
            else:
                # SQLite optimized using JOIN
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

            # Pending predictions are those without actual_price, excluding weekends
            # Get all predictions without actual_price and filter out weekends
            if db_type == "postgresql":
                # Optimized using window function
                cursor.execute('''
                    SELECT DISTINCT prediction_date
                    FROM (
                        SELECT prediction_date,
                               ROW_NUMBER() OVER (PARTITION BY prediction_date ORDER BY created_at DESC) as rn
                        FROM predictions
                        WHERE actual_price IS NULL
                    ) ranked
                    WHERE rn = 1
                ''')
            else:
                # SQLite optimized using JOIN
                cursor.execute('''
                    SELECT DISTINCT p1.prediction_date
                    FROM predictions p1
                    INNER JOIN (
                        SELECT prediction_date, MAX(created_at) as max_created_at
                        FROM predictions
                        WHERE actual_price IS NULL
                        GROUP BY prediction_date
                    ) p2 ON p1.prediction_date = p2.prediction_date 
                        AND p1.created_at = p2.max_created_at
                ''')

            pending_dates = cursor.fetchall()
            # Filter out weekends
            pending_count = 0
            for row in pending_dates:
                date_value = row[0]
                if hasattr(date_value, 'weekday'):
                    weekday = date_value.weekday()
                elif isinstance(date_value, str):
                    date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                    weekday = date_obj.weekday()
                else:
                    continue

                # Only count weekdays (Monday=0 to Friday=4)
                if weekday < 5:
                    pending_count += 1

        # Calculate R² score
        r2_score = None
        if price_data and len(price_data) > 1:
            try:
                predicted_prices = np.array(
                    [float(row[0]) for row in price_data])
                actual_prices = np.array([float(row[1]) for row in price_data])

                # Calculate R² = 1 - (SS_res / SS_tot)
                ss_res = np.sum((actual_prices - predicted_prices) ** 2)
                ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)

                if ss_tot > 0:
                    r2_score = 1 - (ss_res / ss_tot)
                    r2_score = round(float(r2_score), 4)
                else:
                    r2_score = None
            except Exception as e:
                logger.warning(f"Error calculating R² score: {e}")
                r2_score = None

        return {
            'total_predictions': total_count,
            'evaluated_predictions': evaluated_count,
            'pending_predictions': pending_count,
            'average_accuracy': round(avg_accuracy, 2) if evaluated_count > 0 else None,
            'r2_score': r2_score,
            'evaluation_rate': round((evaluated_count / total_count * 100), 2) if total_count > 0 else 0.0
        }

    @staticmethod
    def get_pending_predictions() -> List[Dict]:
        """Get all pending predictions (those without actual_price)"""
        db_type = get_db_type()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            if db_type == "postgresql":
                # Optimized using window function
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
                # SQLite optimized using JOIN
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

        # Convert to list of dictionaries and filter out weekends
        pending = []
        for row in results:
            date_value = row[0]
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
                date_obj = date_value
            elif isinstance(date_value, str):
                date_str = date_value
                from datetime import datetime
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date_str = str(date_value)
                from datetime import datetime
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing date '{date_str}': {e}")
                    continue

            # Skip weekends - don't include Saturday/Sunday in pending predictions
            if date_obj.weekday() >= 5:  # Saturday (5) or Sunday (6)
                continue

            pending.append({
                'date': date_str,
                'predicted_price': round(float(row[1]), 2) if row[1] is not None else None,
                'method': row[2] if len(row) > 2 and row[2] else 'Lasso Regression'
            })

        return pending

    @staticmethod
    def update_prediction_with_actual_price(prediction_date: str, actual_price: float) -> bool:
        """Update a prediction with actual market price"""
        try:
            # Get the existing prediction to preserve predicted_price and method
            db_type = get_db_type()

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get existing prediction data
                if db_type == "postgresql":
                    cursor.execute('''
                        SELECT predicted_price, prediction_method
                        FROM predictions
                        WHERE prediction_date = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (prediction_date,))
                else:
                    cursor.execute('''
                        SELECT predicted_price, prediction_method
                        FROM predictions
                        WHERE prediction_date = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (prediction_date,))

                result = cursor.fetchone()
                if not result:
                    logger.warning(
                        f"No prediction found for date {prediction_date}")
                    return False

                predicted_price = float(
                    result[0]) if result[0] is not None else None
                prediction_method = result[1] if len(
                    result) > 1 and result[1] else None

                # Update with actual price using save_prediction (which calculates accuracy)
                return PredictionRepository.save_prediction(
                    prediction_date=prediction_date,
                    predicted_price=predicted_price,
                    actual_price=actual_price,
                    prediction_method=prediction_method
                )
        except Exception as e:
            logger.error(
                f"Error updating prediction for {prediction_date}: {e}", exc_info=True)
            return False
