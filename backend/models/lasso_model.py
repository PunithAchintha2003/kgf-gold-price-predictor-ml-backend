"""
Lasso Regression Model for XAU/USD Future Price Prediction
Simple, stable, and effective machine learning approach
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
import joblib
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LassoGoldPredictor:
    """
    Lasso Regression Gold Price Predictor
    Uses Lasso regression with feature selection for gold price prediction
    """

    def __init__(self, alpha=0.01, max_iter=2000, random_state=42):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_columns = []
        self.selected_features = []
        self.best_score = -np.inf
        self.training_history = None

    def fetch_market_data(self, symbol='GC=F', period='2y'):
        """Fetch comprehensive market data for gold and related assets"""
        try:
            # Gold futures data (required - must succeed)
            gold = yf.Ticker(symbol)
            gold_data = gold.history(period=period, interval='1d')
            
            if gold_data is None or gold_data.empty:
                logger.error(f"Failed to fetch gold data for {symbol}")
                return None

            # Dollar Index (DXY) - optional
            dxy_data = pd.DataFrame()
            try:
                dxy = yf.Ticker('DX-Y.NYB')
                dxy_data = dxy.history(period=period, interval='1d')
                if dxy_data is None or dxy_data.empty:
                    logger.debug("DXY data unavailable or empty")
                    dxy_data = pd.DataFrame()
            except Exception as e:
                logger.debug(f"DXY data fetch failed (non-critical): {e}")

            # 10-Year Treasury Yield - optional
            treasury_data = pd.DataFrame()
            try:
                treasury = yf.Ticker('^TNX')
                treasury_data = treasury.history(period=period, interval='1d')
                if treasury_data is None or treasury_data.empty:
                    logger.debug("Treasury data unavailable or empty")
                    treasury_data = pd.DataFrame()
            except Exception as e:
                logger.debug(f"Treasury data fetch failed (non-critical): {e}")

            # VIX (Volatility Index) - optional
            vix_data = pd.DataFrame()
            try:
                vix = yf.Ticker('^VIX')
                vix_data = vix.history(period=period, interval='1d')
                if vix_data is None or vix_data.empty:
                    logger.debug("VIX data unavailable or empty")
                    vix_data = pd.DataFrame()
            except Exception as e:
                logger.debug(f"VIX data fetch failed (non-critical): {e}")

            # Oil prices (WTI) - optional
            oil_data = pd.DataFrame()
            try:
                oil = yf.Ticker('CL=F')
                oil_data = oil.history(period=period, interval='1d')
                if oil_data is None or oil_data.empty:
                    logger.debug("Oil data unavailable or empty")
                    oil_data = pd.DataFrame()
            except Exception as e:
                logger.debug(f"Oil data fetch failed (non-critical): {e}")

            return {
                'gold': gold_data,
                'dxy': dxy_data,
                'treasury': treasury_data,
                'vix': vix_data,
                'oil': oil_data
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}", exc_info=True)
            return None

    def create_technical_features(self, df):
        """Create comprehensive technical indicators"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None - cannot create technical features")
        
        if 'Close' not in df.columns:
            raise ValueError("DataFrame missing 'Close' column required for technical features")
        
        df = df.copy()

        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = (
                df['Close'] - df[f'sma_{window}']) / df[f'sma_{window}']

        # Exponential moving averages
        for span in [12, 26]:
            df[f'ema_{span}'] = df['Close'].ewm(span=span).mean()
            df[f'price_vs_ema_{span}'] = (
                df['Close'] - df[f'ema_{span}']) / df[f'ema_{span}']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / \
            (df['bb_upper'] - df['bb_lower'])

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)

        # Volatility indicators
        df['volatility_5'] = df['returns'].rolling(
            window=5).std() * np.sqrt(252)
        df['volatility_20'] = df['returns'].rolling(
            window=20).std() * np.sqrt(252)

        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'roc_{period}'] = df['Close'].pct_change(period) * 100

        return df

    def create_fundamental_features(self, market_data):
        """Create fundamental features from multiple market data sources"""
        # Validate gold data exists and is not empty
        if market_data is None or 'gold' not in market_data:
            raise ValueError("Market data is missing or gold data is not available")
        
        gold_data = market_data['gold']
        if gold_data is None or gold_data.empty:
            raise ValueError("Gold market data is empty or unavailable")
        
        # Start with gold data as base
        gold_df = self.create_technical_features(gold_data)
        
        # Validate required columns exist
        required_cols = ['Close', 'returns', 'rsi', 'macd', 'bb_position', 'volatility_20']
        missing_cols = [col for col in required_cols if col not in gold_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in gold data: {missing_cols}")
        
        features_df = gold_df[required_cols].copy()
        features_df.columns = ['gold_close', 'gold_returns', 'gold_rsi',
                               'gold_macd', 'gold_bb_position', 'gold_volatility']

        # DXY features
        if not market_data['dxy'].empty:
            dxy_df = self.create_technical_features(market_data['dxy'])
            dxy_aligned = dxy_df.reindex(features_df.index, method='ffill')
            features_df['dxy_close'] = dxy_aligned['Close']
            features_df['dxy_returns'] = dxy_aligned['returns']
            features_df['dxy_rsi'] = dxy_aligned['rsi']
            features_df['dxy_macd'] = dxy_aligned['macd']

        # Treasury yield features
        if not market_data['treasury'].empty:
            treasury_df = self.create_technical_features(
                market_data['treasury'])
            treasury_aligned = treasury_df.reindex(
                features_df.index, method='ffill')
            features_df['treasury_close'] = treasury_aligned['Close']
            features_df['treasury_returns'] = treasury_aligned['returns']

        # VIX features
        if not market_data['vix'].empty:
            vix_df = self.create_technical_features(market_data['vix'])
            vix_aligned = vix_df.reindex(features_df.index, method='ffill')
            features_df['vix_close'] = vix_aligned['Close']
            features_df['vix_returns'] = vix_aligned['returns']

        # Oil features
        if not market_data['oil'].empty:
            oil_df = self.create_technical_features(market_data['oil'])
            oil_aligned = oil_df.reindex(features_df.index, method='ffill')
            features_df['oil_close'] = oil_aligned['Close']
            features_df['oil_returns'] = oil_aligned['returns']

        # Cross-asset correlations
        if 'dxy_returns' in features_df.columns and 'gold_returns' in features_df.columns:
            features_df['gold_dxy_corr'] = features_df['gold_returns'].rolling(
                window=20).corr(features_df['dxy_returns'])

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features_df[f'gold_close_lag_{lag}'] = features_df['gold_close'].shift(
                lag)
            features_df[f'gold_returns_lag_{lag}'] = features_df['gold_returns'].shift(
                lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'gold_close_ma_{window}'] = features_df['gold_close'].rolling(
                window).mean()
            features_df[f'gold_close_std_{window}'] = features_df['gold_close'].rolling(
                window).std()
            features_df[f'gold_returns_ma_{window}'] = features_df['gold_returns'].rolling(
                window).mean()

        return features_df

    def prepare_training_data(self, features_df, target_col='gold_close', prediction_horizon=1):
        """Prepare training data for Lasso regression"""
        logger.info(f"Preparing Lasso regression training data")

        # Create target variable (future price)
        data = features_df.copy()
        data['target'] = data[target_col].shift(-prediction_horizon)

        # Remove rows with NaN values
        data_clean = data.dropna()

        if data_clean.empty:
            logger.error("No valid data after cleaning NaN values")
            return pd.DataFrame(), np.array([])

        # Separate features and target
        feature_cols = [
            col for col in data_clean.columns if col not in ['target', target_col]]
        X = data_clean[feature_cols]
        y = data_clean['target']

        self.feature_columns = feature_cols

        logger.info(
            f"Training data prepared - X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")

        return X, y

    def train_model(self, X, y, test_size=0.2):
        """Train the Lasso regression model with feature selection"""
        logger.info(
            f"Training Lasso Regression with {int((1-test_size)*100)}% training data")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False)

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use LassoCV to find optimal alpha
        lasso_cv = LassoCV(
            alphas=np.logspace(-4, 1, 50),
            cv=5,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Fit LassoCV to find best alpha
        logger.info("Finding optimal alpha using cross-validation...")
        lasso_cv.fit(X_train_scaled, y_train)
        best_alpha = lasso_cv.alpha_

        logger.info(f"Best alpha found: {best_alpha:.6f}")

        # Create final Lasso model with best alpha
        self.model = Lasso(
            alpha=best_alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Train the model
        logger.info("Training Lasso Regression...")
        self.model.fit(X_train_scaled, y_train)

        # Feature selection
        self.feature_selector = SelectFromModel(self.model, prefit=True)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [self.feature_columns[i] for i in range(
            len(self.feature_columns)) if selected_mask[i]]

        logger.info(
            f"Selected {len(self.selected_features)} features out of {len(self.feature_columns)}")
        # Show first 10
        logger.info(f"Selected features: {self.selected_features[:10]}...")

        # Create a new Lasso model for the selected features
        self.model = Lasso(
            alpha=best_alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Retrain with selected features
        self.model.fit(X_train_selected, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_selected)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.best_score = r2

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_selected, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        logger.info(
            f"Lasso Regression - R² = {r2:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")
        logger.info(
            f"Cross-validation R² = {cv_mean:.4f} (+/- {cv_std*2:.4f})")

        return {
            'lasso_model': {
                'model': self.model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_r2': cv_mean,
                'cv_std': cv_std,
                'best_alpha': best_alpha,
                'selected_features': self.selected_features
            }
        }

    def predict_next_price(self, features_df):
        """Predict next day's gold price using Lasso regression"""
        if self.model is None:
            raise ValueError(
                "Model not trained yet. Call train_model() first.")

        # Get the last row of features
        if len(features_df) < 1:
            raise ValueError("Not enough data for prediction")

        # Prepare features
        X_pred = features_df[self.feature_columns].iloc[-1:].values

        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)

        # Select features
        X_pred_selected = self.feature_selector.transform(X_pred_scaled)

        # Make prediction
        prediction = self.model.predict(X_pred_selected)[0]

        return prediction

    def get_model_summary(self):
        """Get model summary information"""
        if self.model is None:
            return None

        return {
            'model_type': 'Lasso Regression',
            'alpha': self.model.alpha,
            'feature_count': len(self.feature_columns),
            'selected_features': len(self.selected_features),
            'test_r2_score': self.best_score,
            'max_iter': self.max_iter
        }

    def get_feature_importance(self):
        """Get feature importance from Lasso coefficients"""
        if self.model is None or self.feature_selector is None:
            return None

        # Get coefficients for selected features
        coefficients = self.model.coef_
        feature_names = self.selected_features

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df

    def save_model(self, filepath='lasso_gold_model.pkl'):
        """Save the trained Lasso model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'best_score': self.best_score,
            'alpha': self.alpha,
            'max_iter': self.max_iter
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Lasso Regression model saved to {filepath}")

    def load_model(self, filepath='lasso_gold_model.pkl'):
        """Load a trained Lasso model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_columns = model_data['feature_columns']
        self.selected_features = model_data['selected_features']
        self.best_score = model_data['best_score']
        self.alpha = model_data.get('alpha', 0.01)
        self.max_iter = model_data.get('max_iter', 2000)

        logger.info(f"Lasso Regression model loaded from {filepath}")


def main():
    """Main function to train and test the Lasso regression model"""
    predictor = LassoGoldPredictor(
        alpha=0.01,
        max_iter=2000,
        random_state=42
    )

    # Fetch market data
    logger.info("Fetching market data...")
    market_data = predictor.fetch_market_data()

    if market_data is None:
        logger.error("Failed to fetch market data")
        return

    # Create features
    logger.info("Creating features...")
    features_df = predictor.create_fundamental_features(market_data)

    # Prepare training data
    logger.info("Preparing training data...")
    X, y = predictor.prepare_training_data(features_df)

    if X.empty:
        logger.error("No training data available")
        return

    logger.info(f"Training data shape: {X.shape}")

    # Train Lasso model
    logger.info("Training Lasso Regression model...")
    training_results = predictor.train_model(X, y, test_size=0.2)

    # Get model summary
    model_summary = predictor.get_model_summary()
    if model_summary:
        logger.info("Model Summary:")
        for key, value in model_summary.items():
            logger.info(f"{key}: {value}")

    # Get feature importance
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        logger.info("Top 10 Most Important Features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['coefficient']:.6f}")

    # Save model
    predictor.save_model()

    # Test prediction
    logger.info("Testing prediction...")
    try:
        next_price = predictor.predict_next_price(features_df)
        current_price = features_df['gold_close'].iloc[-1]
        change = next_price - current_price
        change_pct = (change / current_price) * 100

        logger.info(f"Current price: ${current_price:.2f}")
        logger.info(f"Predicted next price: ${next_price:.2f}")
        logger.info(f"Predicted change: ${change:.2f} ({change_pct:.2f}%)")

    except Exception as e:
        logger.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
