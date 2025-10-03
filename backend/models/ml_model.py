import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldPriceMLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.best_model = None
        self.best_score = -np.inf

    def fetch_market_data(self, symbol='GC=F', period='2y'):
        """Fetch comprehensive market data for gold and related assets"""
        try:
            # Gold futures data
            gold = yf.Ticker(symbol)
            gold_data = gold.history(period=period, interval='1d')

            # Dollar Index (DXY)
            dxy = yf.Ticker('DX-Y.NYB')
            dxy_data = dxy.history(period=period, interval='1d')

            # 10-Year Treasury Yield
            treasury = yf.Ticker('^TNX')
            treasury_data = treasury.history(period=period, interval='1d')

            # VIX (Volatility Index)
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period=period, interval='1d')

            # Oil prices (WTI)
            oil = yf.Ticker('CL=F')
            oil_data = oil.history(period=period, interval='1d')

            return {
                'gold': gold_data,
                'dxy': dxy_data,
                'treasury': treasury_data,
                'vix': vix_data,
                'oil': oil_data
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def create_technical_features(self, df):
        """Create comprehensive technical indicators"""
        df = df.copy()

        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']

        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'price_vs_sma_{window}'] = (
                df['Close'] - df[f'sma_{window}']) / df[f'sma_{window}']

        # Exponential moving averages
        for span in [12, 26, 50]:
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

        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['Close']

        # Volume indicators (if available)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            df['price_volume'] = df['Close'] * df['Volume']

        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'roc_{period}'] = df['Close'].pct_change(period) * 100

        # Volatility indicators
        df['volatility_5'] = df['returns'].rolling(
            window=5).std() * np.sqrt(252)
        df['volatility_20'] = df['returns'].rolling(
            window=20).std() * np.sqrt(252)

        # Price patterns
        df['doji'] = (abs(df['Close'] - df['Open']) /
                      (df['High'] - df['Low'])) < 0.1
        df['hammer'] = ((df['Low'] < df['Open']) & (df['Low'] < df['Close']) &
                        (df['Close'] - df['Low']) > 2 * (df['Open'] - df['Low']))
        df['shooting_star'] = ((df['High'] > df['Open']) & (df['High'] > df['Close']) &
                               (df['High'] - df['Close']) > 2 * (df['Close'] - df['Open']))

        return df

    def create_fundamental_features(self, market_data):
        """Create fundamental features from multiple market data sources"""
        # Start with gold data as base
        gold_df = self.create_technical_features(market_data['gold'])
        features_df = gold_df[['Close', 'returns', 'rsi',
                               'macd', 'bb_position', 'volatility_20']].copy()
        features_df.columns = ['gold_close', 'gold_returns', 'gold_rsi',
                               'gold_macd', 'gold_bb_position', 'gold_volatility']

        # DXY features
        if not market_data['dxy'].empty:
            dxy_df = self.create_technical_features(market_data['dxy'])
            # Align dates with gold data
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
        for lag in [1, 2, 3, 5]:
            features_df[f'gold_close_lag_{lag}'] = features_df['gold_close'].shift(
                lag)
            features_df[f'gold_returns_lag_{lag}'] = features_df['gold_returns'].shift(
                lag)

        return features_df

    def prepare_training_data(self, features_df, target_col='gold_close', prediction_horizon=1):
        """Prepare training data for ML models"""
        logger.info(f"Original features shape: {features_df.shape}")
        logger.info(
            f"NaN count before cleaning: {features_df.isnull().sum().sum()}")

        # Create target variable (future price)
        features_df['target'] = features_df[target_col].shift(
            -prediction_horizon)

        # Remove rows with NaN values
        features_df_clean = features_df.dropna()
        logger.info(
            f"Features shape after cleaning: {features_df_clean.shape}")
        logger.info(
            f"NaN count after cleaning: {features_df_clean.isnull().sum().sum()}")

        if features_df_clean.empty:
            logger.error("No valid data after cleaning NaN values")
            return pd.DataFrame(), pd.Series()

        # Separate features and target
        feature_cols = [
            col for col in features_df_clean.columns if col not in ['target', target_col]]
        X = features_df_clean[feature_cols]
        y = features_df_clean['target']

        self.feature_columns = feature_cols

        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def train_models(self, X, y, test_size=0.2):
        """Train multiple ML models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        # Define models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }

        # Train and evaluate models
        model_scores = {}

        for name, model in models.items():
            try:
                # Train model
                if name in ['ridge', 'lasso', 'svr']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Cross-validation score
                if name in ['ridge', 'lasso', 'svr']:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=5, scoring='r2')

                model_scores[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }

                logger.info(
                    f"{name}: R² = {r2:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        # Select best model based on cross-validation score
        best_model_name = max(model_scores.keys(),
                              key=lambda k: model_scores[k]['cv_mean'])
        self.best_model = model_scores[best_model_name]['model']
        self.best_score = model_scores[best_model_name]['cv_mean']

        logger.info(
            f"Best model: {best_model_name} with CV score: {self.best_score:.4f}")

        # Store all models
        self.models = {name: data['model']
                       for name, data in model_scores.items()}

        return model_scores

    def predict_next_price(self, features_df):
        """Predict next day's gold price"""
        if self.best_model is None:
            raise ValueError(
                "Model not trained yet. Call train_models() first.")

        # Get latest features
        latest_features = features_df[self.feature_columns].iloc[-1:].values

        # Scale features if needed
        if hasattr(self.best_model, 'feature_importances_') or isinstance(self.best_model, (Ridge, Lasso, SVR)):
            latest_features = self.scalers['standard'].transform(
                latest_features)

        # Make prediction
        prediction = self.best_model.predict(latest_features)[0]

        return prediction

    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            return None

        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            logger.warning("Best model doesn't support feature importance")
            return None

    def save_model(self, filepath='gold_ml_model.pkl'):
        """Save the trained model and scaler"""
        model_data = {
            'best_model': self.best_model,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'best_score': self.best_score
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath='gold_ml_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.best_score = model_data['best_score']
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main function to train and test the ML model"""
    predictor = GoldPriceMLPredictor()

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

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")

    # Train models
    logger.info("Training models...")
    model_scores = predictor.train_models(X, y)

    # Get feature importance
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        logger.info("Top 10 most important features:")
        logger.info(importance_df.head(10).to_string())

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
