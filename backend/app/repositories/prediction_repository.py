"""Repository for prediction database operations"""
import logging
from typing import Optional, List, Dict
from datetime import datetime

from ..core.database import get_db_connection, get_db_type, get_date_function

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repository for prediction database operations"""
    
    @staticmethod
    def save_prediction(
        prediction_date: str,
        predicted_price: float,
        actual_price: Optional[float] = None,
        prediction_method: Optional[str] = None
    ) -> bool:
        """Save prediction to database with accuracy calculation"""
        try:
            # Convert numpy types to Python native types
            predicted_price = float(predicted_price) if predicted_price is not None else None
            actual_price = float(actual_price) if actual_price is not None else None
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate accuracy if actual price is available
                accuracy = None
                if actual_price and predicted_price:
                    error_percentage = abs(predicted_price - actual_price) / actual_price * 100
                    accuracy = float(max(0, 100 - error_percentage))
                
                # Insert or update prediction
                db_type = get_db_type()
                if db_type == "postgresql":
                    # Check if prediction exists first
                    cursor.execute('''
                        SELECT id FROM predictions WHERE prediction_date = %s
                    ''', (prediction_date,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing prediction
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
                logger.info(f"Saved prediction for {prediction_date}: ${predicted_price:.2f}")
                return True
        except Exception as e:
            logger.error(f"Error saving prediction for {prediction_date}: {e}", exc_info=True)
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
    def get_historical_predictions(days: int = 90) -> List[Dict]:
        """Get historical predictions for the specified number of days"""
        date_func = get_date_function(-days)
        db_type = get_db_type()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if db_type == "postgresql":
                cursor.execute('''
                    SELECT prediction_date, predicted_price, actual_price, prediction_method
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
                    SELECT prediction_date, predicted_price, actual_price, prediction_method
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
        
        # Convert to list of dictionaries
        predictions = []
        for row in results:
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
                'actual_price': round(float(row[2]), 2) if row[2] is not None else None,
                'method': row[3] if len(row) > 3 and row[3] else 'Lasso Regression'
            })
        
        return predictions
    
    @staticmethod
    def get_accuracy_stats() -> Dict:
        """Get accuracy statistics"""
        date_func_30 = get_date_function(-30)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get accuracy for unique predictions with actual prices
            cursor.execute(f'''
                SELECT AVG(accuracy_percentage), COUNT(DISTINCT prediction_date)
                FROM predictions p1
                WHERE accuracy_percentage IS NOT NULL
                AND prediction_date >= {date_func_30}
                AND p1.created_at = (
                    SELECT MAX(p2.created_at)
                    FROM predictions p2
                    WHERE p2.prediction_date = p1.prediction_date
                )
            ''')
            
            accuracy_result = cursor.fetchone()
            
            # Get total unique prediction dates
            cursor.execute(f'''
                SELECT COUNT(DISTINCT prediction_date)
                FROM predictions p1
                WHERE prediction_date >= {date_func_30}
                AND p1.created_at = (
                    SELECT MAX(p2.created_at)
                    FROM predictions p2
                    WHERE p2.prediction_date = p1.prediction_date
                )
            ''')
            
            total_result = cursor.fetchone()
        
        avg_accuracy = float(accuracy_result[0]) if accuracy_result[0] is not None else 0.0
        evaluated_count = int(accuracy_result[1]) if accuracy_result[1] is not None else 0
        total_count = int(total_result[0]) if total_result[0] is not None else 0
        
        return {
            'average_accuracy': round(avg_accuracy, 2),
            'r2_score': 0.0,  # Can be calculated separately if needed
            'total_predictions': total_count,
            'evaluated_predictions': evaluated_count
        }


