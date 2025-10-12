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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import os
from models.ml_model import GoldPriceMLPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = "data/gold_predictions.db"

# Initialize ML predictor
ml_predictor = GoldPriceMLPredictor()
try:
    ml_predictor.load_model('models/gold_ml_model.pkl')
    logger.info("ML model loaded successfully")
except:
    logger.warning("ML model not found, will train new model")
    # Train new model
    market_data = ml_predictor.fetch_market_data()
    if market_data:
        features_df = ml_predictor.create_fundamental_features(market_data)
        X, y = ml_predictor.prepare_training_data(features_df)
        if not X.empty:
            ml_predictor.train_models(X, y)
            ml_predictor.save_model('models/gold_ml_model.pkl')
            logger.info("New ML model trained and saved")
        else:
            logger.error("Failed to prepare training data for ML model")
    else:
        logger.error("Failed to fetch market data for ML model")


def init_database():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create predictions table
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

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_prediction(prediction_date, predicted_price, actual_price=None):
    """Save prediction to database with correct accuracy calculation"""
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()


def get_historical_predictions(days=30):
    """Get historical predictions for ghost line - all predictions including future ones"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT prediction_date, predicted_price, actual_price
        FROM predictions p1
        WHERE prediction_date >= date('now', '-{} days')
        AND prediction_date != '2025-10-11'
        AND p1.created_at = (
            SELECT MAX(p2.created_at)
            FROM predictions p2
            WHERE p2.prediction_date = p1.prediction_date
        )
        ORDER BY prediction_date
    '''.format(days))

    results = cursor.fetchall()
    conn.close()

    return [{
        'date': row[0],
        'predicted_price': row[1],
        'actual_price': row[2]
    } for row in results]


def get_ml_model_display_name():
    """Get the display name for the current ML model"""
    return "Lasso Regression"


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
        AND prediction_date != '2025-10-11'
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
        AND prediction_date != '2025-10-11'
        AND p1.created_at = (
            SELECT MAX(p2.created_at)
            FROM predictions p2
            WHERE p2.prediction_date = p1.prediction_date
        )
    ''')

    total_result = cursor.fetchone()
    conn.close()

    if accuracy_result and accuracy_result[1] > 0:
        return {
            # Now shows average accuracy (higher is better)
            'average_accuracy': round(accuracy_result[0], 2),
            'total_predictions': total_result[0] if total_result else 0,
            'evaluated_predictions': accuracy_result[1]
        }
    return {
        'average_accuracy': 0,
        'total_predictions': total_result[0] if total_result else 0,
        'evaluated_predictions': 0
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
    logger.info(f"Found {len(predictions)} predictions to evaluate")
    for pred_id, pred_date, pred_price, current_actual, current_accuracy in predictions:
        try:
            # Skip accuracy calculation for October 11th as requested
            if pred_date == "2025-10-11":
                logger.info(
                    f"Skipping accuracy calculation for {pred_date} as requested")
                continue

            logger.info(
                f"Processing prediction for {pred_date}, available_dates contains: {pred_date in available_dates}")
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
                logger.info(
                    f"Skipping {pred_date} - no market data available (market may not have closed yet for today's date)")

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


# Initialize database on startup
init_database()

app = FastAPI(title="XAU/USD Real-time Data API", version="1.0.0")

# Add CORS middleware to allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Predict next day price using Machine Learning approach"""
    try:
        # Get fresh market data
        market_data = ml_predictor.fetch_market_data()
        if not market_data:
            logger.error("Failed to fetch market data for ML prediction")
            return None

        # Create features
        features_df = ml_predictor.create_fundamental_features(market_data)

        # Make prediction
        predicted_price = ml_predictor.predict_next_price(features_df)

        logger.info(f"ML prediction: ${predicted_price:.2f}")
        return round(predicted_price, 2)

    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return None


def generate_prediction_explanation():
    """Get user-friendly explanation for the current prediction - simplified to 5 main reasons"""
    try:
        # Get fresh market data
        market_data = ml_predictor.fetch_market_data()
        if not market_data:
            return {"error": "Failed to fetch market data"}

        # Create features
        features_df = ml_predictor.create_fundamental_features(market_data)

        # Get current market data
        gold_data = market_data['gold']
        current_price = float(gold_data['Close'].iloc[-1])

        # Get latest features for analysis
        latest_features = features_df.iloc[-1]

        # Analyze market conditions - focus on 5 most important factors
        explanations = []

        # 1. US Dollar Strength (Most Important)
        if 'dxy_close' in latest_features and not pd.isna(latest_features['dxy_close']):
            dxy_close = latest_features['dxy_close']
            dxy_returns = latest_features.get('dxy_returns', 0)

            if dxy_returns > 0.01:  # 1% increase
                explanations.append({
                    "factor": "💵 US Dollar Strength",
                    "value": f"${dxy_close:.2f} (+{dxy_returns*100:.1f}%)",
                    "interpretation": "Strong dollar makes gold more expensive for international buyers",
                    "impact": "Bearish",
                    "confidence": "High"
                })
            elif dxy_returns < -0.01:  # 1% decrease
                explanations.append({
                    "factor": "💵 US Dollar Strength",
                    "value": f"${dxy_close:.2f} ({dxy_returns*100:.1f}%)",
                    "interpretation": "Weak dollar makes gold cheaper for international buyers",
                    "impact": "Bullish",
                    "confidence": "High"
                })
            else:
                explanations.append({
                    "factor": "💵 US Dollar Strength",
                    "value": f"${dxy_close:.2f}",
                    "interpretation": "Stable dollar has neutral effect on gold prices",
                    "impact": "Neutral",
                    "confidence": "Medium"
                })

        # 2. Interest Rates (Treasury Yields)
        if 'treasury_close' in latest_features and not pd.isna(latest_features['treasury_close']):
            treasury_close = latest_features['treasury_close']
            treasury_returns = latest_features.get('treasury_returns', 0)

            if treasury_returns > 0.02:  # 2% increase
                explanations.append({
                    "factor": "📈 Interest Rates",
                    "value": f"{treasury_close:.2f}% (+{treasury_returns*100:.1f}%)",
                    "interpretation": "Higher rates make bonds more attractive than gold",
                    "impact": "Bearish",
                    "confidence": "High"
                })
            elif treasury_returns < -0.02:  # 2% decrease
                explanations.append({
                    "factor": "📈 Interest Rates",
                    "value": f"{treasury_close:.2f}% ({treasury_returns*100:.1f}%)",
                    "interpretation": "Lower rates make gold more attractive than bonds",
                    "impact": "Bullish",
                    "confidence": "High"
                })
            else:
                explanations.append({
                    "factor": "📈 Interest Rates",
                    "value": f"{treasury_close:.2f}%",
                    "interpretation": "Stable rates have neutral effect on gold",
                    "impact": "Neutral",
                    "confidence": "Medium"
                })

        # 3. Market Fear (VIX)
        if 'vix_close' in latest_features and not pd.isna(latest_features['vix_close']):
            vix_close = latest_features['vix_close']

            if vix_close > 30:
                explanations.append({
                    "factor": "😰 Market Fear",
                    "value": f"{vix_close:.1f}",
                    "interpretation": "High fear drives investors to safe haven assets like gold",
                    "impact": "Bullish",
                    "confidence": "High"
                })
            elif vix_close < 15:
                explanations.append({
                    "factor": "😰 Market Fear",
                    "value": f"{vix_close:.1f}",
                    "interpretation": "Low fear means investors prefer riskier assets over gold",
                    "impact": "Bearish",
                    "confidence": "High"
                })
            else:
                explanations.append({
                    "factor": "😰 Market Fear",
                    "value": f"{vix_close:.1f}",
                    "interpretation": "Normal fear levels have neutral effect on gold",
                    "impact": "Neutral",
                    "confidence": "Medium"
                })

        # 4. Gold Technical Analysis (RSI)
        gold_rsi = latest_features.get('gold_rsi', 50)
        if gold_rsi > 70:
            explanations.append({
                "factor": "📊 Gold Technicals",
                "value": f"RSI: {gold_rsi:.1f}",
                "interpretation": "Gold is overbought and may be overvalued",
                "impact": "Bearish",
                "confidence": "High" if gold_rsi > 80 else "Medium"
            })
        elif gold_rsi < 30:
            explanations.append({
                "factor": "📊 Gold Technicals",
                "value": f"RSI: {gold_rsi:.1f}",
                "interpretation": "Gold is oversold and may be undervalued",
                "impact": "Bullish",
                "confidence": "High" if gold_rsi < 20 else "Medium"
            })
        else:
            explanations.append({
                "factor": "📊 Gold Technicals",
                "value": f"RSI: {gold_rsi:.1f}",
                "interpretation": "Gold is in normal trading range",
                "impact": "Neutral",
                "confidence": "Medium"
            })

        # 5. Inflation Pressure (Oil Prices)
        if 'oil_close' in latest_features and not pd.isna(latest_features['oil_close']):
            oil_close = latest_features['oil_close']
            oil_returns = latest_features.get('oil_returns', 0)

            if oil_returns > 0.02:  # 2% increase
                explanations.append({
                    "factor": "🛢️ Inflation Pressure",
                    "value": f"${oil_close:.2f} (+{oil_returns*100:.1f}%)",
                    "interpretation": "Rising oil prices increase inflation, supporting gold as hedge",
                    "impact": "Bullish",
                    "confidence": "Medium"
                })
            elif oil_returns < -0.02:  # 2% decrease
                explanations.append({
                    "factor": "🛢️ Inflation Pressure",
                    "value": f"${oil_close:.2f} ({oil_returns*100:.1f}%)",
                    "interpretation": "Falling oil prices reduce inflation pressure on gold",
                    "impact": "Bearish",
                    "confidence": "Medium"
                })
            else:
                explanations.append({
                    "factor": "🛢️ Inflation Pressure",
                    "value": f"${oil_close:.2f}",
                    "interpretation": "Stable oil prices have neutral effect on gold",
                    "impact": "Neutral",
                    "confidence": "Low"
                })

        # Calculate overall sentiment
        bullish_count = sum(
            1 for exp in explanations if exp['impact'] == 'Bullish')
        bearish_count = sum(
            1 for exp in explanations if exp['impact'] == 'Bearish')
        neutral_count = sum(
            1 for exp in explanations if exp['impact'] == 'Neutral')

        if bullish_count > bearish_count + neutral_count:
            overall_sentiment = "Bullish"
            sentiment_explanation = "Most factors suggest gold prices may rise"
        elif bearish_count > bullish_count + neutral_count:
            overall_sentiment = "Bearish"
            sentiment_explanation = "Most factors suggest gold prices may fall"
        else:
            overall_sentiment = "Neutral"
            sentiment_explanation = "Mixed signals - gold prices may remain stable"

        return {
            "current_price": current_price,
            "overall_sentiment": overall_sentiment,
            "sentiment_explanation": sentiment_explanation,
            "factors": explanations[:5],  # Ensure only 5 factors
            "summary": {
                "bullish_factors": bullish_count,
                "bearish_factors": bearish_count,
                "neutral_factors": neutral_count,
                "total_factors": min(len(explanations), 5)
            },
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
        # XAU/USD is represented as GC=F (Gold Futures) in Yahoo Finance
        gold = yf.Ticker("GC=F")

        # Get daily data for the last 30 days
        # Note: Using 1mo instead of 30d for better reliability with GC=F
        hist = gold.history(period="1mo", interval="1d")

        if not hist.empty:
            # Check if we already made a prediction for today
            today = datetime.now().strftime("%Y-%m-%d")
            next_day = (hist.index[-1] + timedelta(days=1)
                        ).strftime("%Y-%m-%d")

            # Make predictions more frequently (every 2 hours for continuous updates)
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute
            should_make_prediction = (
                not prediction_exists_for_date(next_day) or
                # Every 2 hours, within first 5 minutes
                (current_hour % 2 == 0 and current_minute < 5)
            )

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
    return {"message": "XAU/USD Real-time Data API", "status": "running"}


@app.get("/xauusd")
async def get_daily_data():
    """REST endpoint to get XAU/USD daily data"""
    return get_xauusd_daily_data()


@app.get("/xauusd/realtime")
async def get_realtime_price():
    """REST endpoint to get real-time XAU/USD current price"""
    try:
        # Get real-time price using 2-day period with 1-minute interval for more current data
        # Note: period="1d" doesn't work with GC=F, so we use "2d" and take the latest data
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="2d", interval="1m")

        if not hist.empty:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            timestamp = hist.index[-1].isoformat()

            return {
                "symbol": "XAUUSD",
                "current_price": current_price,
                "timestamp": timestamp,
                "status": "success"
            }
        else:
            return {
                "symbol": "XAUUSD",
                "current_price": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": "No real-time data available"
            }
    except Exception as e:
        logger.error(f"Error fetching real-time price: {e}")
        return {
            "symbol": "XAUUSD",
            "current_price": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": str(e)
        }


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
    """WebSocket endpoint for XAU/USD real-time data updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Get daily data
            daily_data = get_xauusd_daily_data()

            # Send to client
            await manager.send_personal_message(json.dumps(daily_data), websocket)

            # Wait 2 seconds before next update for real-time feel
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_daily_data():
    """Background task to broadcast daily data to all connected clients"""
    while True:
        if manager.active_connections:
            daily_data = get_xauusd_daily_data()
            await manager.broadcast(json.dumps(daily_data))
        # Update data every 2 seconds for real-time feel
        await asyncio.sleep(2)


async def continuous_accuracy_updates():
    """Background task to continuously update accuracy every 10 minutes"""
    while True:
        try:
            logger.info("Running continuous accuracy update...")
            update_actual_prices_realtime()
            update_same_day_predictions()
            logger.info("Continuous accuracy update completed")
        except Exception as e:
            logger.error(f"Error in continuous accuracy update: {e}")

        # Update accuracy every 10 minutes
        await asyncio.sleep(600)  # 10 minutes


@app.on_event("startup")
async def startup_event():
    """Start background tasks for broadcasting daily data and continuous accuracy updates"""
    asyncio.create_task(broadcast_daily_data())
    asyncio.create_task(continuous_accuracy_updates())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
