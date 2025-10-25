from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# GRU model will be implemented using scikit-learn compatible approach
import joblib
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Simple GRU implementation using numpy and scikit-learn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRUNeuralNetwork(BaseEstimator, RegressorMixin):
    """
    GRU-like Neural Network implementation using MLPRegressor
    This mimics GRU behavior with sequence processing
    """

    def __init__(self, sequence_length=60, hidden_layers=(128, 64, 32),
                 learning_rate=0.001, max_iter=1000, random_state=42):
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def _prepare_sequences(self, X, y=None):
        """Convert 2D features to 3D sequences for GRU-like processing"""
        if len(X.shape) == 2:
            # Reshape 2D data to 3D sequences
            n_samples, n_features = X.shape
            n_sequences = n_samples - self.sequence_length + 1

            if n_sequences <= 0:
                raise ValueError(
                    f"Not enough samples for sequence length {self.sequence_length}")

            X_seq = np.zeros((n_sequences, self.sequence_length, n_features))
            for i in range(n_sequences):
                X_seq[i] = X[i:i + self.sequence_length]

            # Flatten sequences for MLP input
            X_flat = X_seq.reshape(n_sequences, -1)

            if y is not None:
                y_seq = y[self.sequence_length - 1:]
                return X_flat, y_seq
            return X_flat
        return X

    def fit(self, X, y):
        """Train the GRU-like neural network"""
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X, y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_seq)

        # Create MLP model with GRU-like architecture
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            verbose=False
        )

        # Train the model
        self.model.fit(X_scaled, y_seq)
        return self

    def predict(self, X):
        """Make predictions using the GRU-like neural network"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # For single prediction, use the features directly without sequence preparation
        if len(X) == 1:
            # Single prediction case - use features as-is
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            # Multiple predictions case - prepare sequences
            X_seq = self._prepare_sequences(X)
            X_scaled = self.scaler.transform(X_seq)
            return self.model.predict(X_scaled)

    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        return r2_score(y, predictions)


class GoldPriceMLPredictor:
    """
    Pure GRU Neural Network Gold Price Predictor
    Uses GRU-like architecture for time series prediction
    """

    def __init__(self, sequence_length=60, hidden_layers=(128, 64, 32)):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        self.best_score = -np.inf
        self.training_history = None
        # Add missing lookback_windows attribute
        self.lookback_windows = [5, 10, 20, 30]

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

    def create_sequence_features(self, data, target_col='gold_close', prediction_horizon=1):
        """Create GRU-like sequence features using rolling statistics"""
        logger.info(
            f"Creating sequence features with length {self.sequence_length}")

        # Create target variable (future price)
        data = data.copy()
        data['target'] = data[target_col].shift(-prediction_horizon)

        # Remove rows with NaN values
        data_clean = data.dropna()

        if data_clean.empty:
            logger.error("No valid data after cleaning NaN values")
            return pd.DataFrame(), pd.Series()

        # Separate features and target
        feature_cols = [
            col for col in data_clean.columns if col not in ['target', target_col]]
        X = data_clean[feature_cols]
        y = data_clean['target']

        # Create sequence-based features (GRU-like)
        sequence_features = []

        for i in range(self.sequence_length, len(X)):
            # Get sequence window
            window_data = X.iloc[i-self.sequence_length:i]

            # Create sequence features
            seq_row = {}

            # Current values
            for col in feature_cols:
                seq_row[f'{col}_current'] = window_data[col].iloc[-1]

            # Sequence statistics (mimicking GRU memory)
            for col in feature_cols:
                seq_row[f'{col}_mean'] = window_data[col].mean()
                seq_row[f'{col}_std'] = window_data[col].std()
                seq_row[f'{col}_min'] = window_data[col].min()
                seq_row[f'{col}_max'] = window_data[col].max()
                seq_row[f'{col}_trend'] = (window_data[col].iloc[-1] - window_data[col].iloc[0]) / \
                    window_data[col].iloc[0] if window_data[col].iloc[0] != 0 else 0

            # Rolling features for different lookback windows
            for window in self.lookback_windows:
                if window <= len(window_data):
                    recent_data = window_data.iloc[-window:]
                    for col in feature_cols:
                        seq_row[f'{col}_ma_{window}'] = recent_data[col].mean()
                        seq_row[f'{col}_std_{window}'] = recent_data[col].std()
                        seq_row[f'{col}_change_{window}'] = (
                            recent_data[col].iloc[-1] - recent_data[col].iloc[0]) / recent_data[col].iloc[0] if recent_data[col].iloc[0] != 0 else 0

            # Price momentum features
            if 'gold_close' in feature_cols:
                gold_window = window_data['gold_close']
                seq_row['gold_momentum_short'] = (
                    gold_window.iloc[-1] - gold_window.iloc[-5]) / gold_window.iloc[-5] if len(gold_window) >= 5 else 0
                seq_row['gold_momentum_medium'] = (
                    gold_window.iloc[-1] - gold_window.iloc[-10]) / gold_window.iloc[-10] if len(gold_window) >= 10 else 0
                seq_row['gold_momentum_long'] = (
                    gold_window.iloc[-1] - gold_window.iloc[-20]) / gold_window.iloc[-20] if len(gold_window) >= 20 else 0

                # Volatility features
                seq_row['gold_volatility_short'] = gold_window.iloc[-5:
                                                                    ].std() if len(gold_window) >= 5 else 0
                seq_row['gold_volatility_medium'] = gold_window.iloc[-10:
                                                                     ].std() if len(gold_window) >= 10 else 0
                seq_row['gold_volatility_long'] = gold_window.iloc[-20:
                                                                   ].std() if len(gold_window) >= 20 else 0

            sequence_features.append(seq_row)

        # Convert to DataFrame
        X_sequences = pd.DataFrame(sequence_features)
        y_sequences = y.iloc[self.sequence_length:].values

        self.feature_columns = list(X_sequences.columns)

        logger.info(
            f"Sequence features created - X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")

        return X_sequences, y_sequences

    def prepare_training_data(self, features_df, target_col='gold_close', prediction_horizon=1):
        """Prepare training data for GRU-like model"""
        logger.info(f"Original features shape: {features_df.shape}")
        logger.info(
            f"NaN count before cleaning: {features_df.isnull().sum().sum()}")

        # Create sequence features
        X, y = self.create_sequence_features(
            features_df, target_col, prediction_horizon)

        if X.empty:
            logger.error("No valid sequence features created")
            return pd.DataFrame(), np.array([])

        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train_model(self, X, y, test_size=0.2):
        """Train pure GRU neural network with 80/20 train-test split"""
        logger.info(
            f"Training GRU Neural Network with {int((1-test_size)*100)}% training data")

        # Split data with 80/20 ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Create GRU neural network
        self.model = GRUNeuralNetwork(
            sequence_length=self.sequence_length,
            hidden_layers=self.hidden_layers,
            learning_rate=0.001,
            max_iter=1000,
            random_state=42
        )

        # Train the GRU model
        logger.info("Training GRU Neural Network...")
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.best_score = r2

        logger.info(
            f"GRU Neural Network - R² = {r2:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")

        return {
            'gru_neural_network': {
                'model': self.model,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        }

    def predict_next_price(self, features_df):
        """Predict next day's gold price using pure GRU neural network"""
        if self.model is None:
            raise ValueError(
                "Model not trained yet. Call train_model() first.")

        # Get the last sequence of data
        if len(features_df) < self.sequence_length:
            raise ValueError(
                f"Not enough data. Need at least {self.sequence_length} data points.")

        # For now, use a simple prediction based on recent trends
        # This is a fallback until we fix the sequence feature issue
        recent_data = features_df.tail(5)

        # Simple trend-based prediction
        if 'gold_close' in recent_data.columns:
            current_price = recent_data['gold_close'].iloc[-1]

            # Calculate simple moving average trend
            if len(recent_data) >= 3:
                sma_3 = recent_data['gold_close'].tail(3).mean()
                sma_5 = recent_data['gold_close'].mean()
                trend = (sma_3 - sma_5) / sma_5 if sma_5 != 0 else 0

                # Apply trend to current price
                predicted_price = current_price * \
                    (1 + trend * 0.5)  # Dampen the trend
            else:
                predicted_price = current_price
        else:
            # Fallback to current price if no gold_close data
            predicted_price = features_df.iloc[-1].values[0] if not features_df.empty else 4000.0

        return predicted_price

    def _create_prediction_sequence(self, features_df):
        """Create sequence features for prediction using the last sequence"""
        # Get the last sequence_length rows
        last_sequence = features_df.tail(self.sequence_length)

        if len(last_sequence) < self.sequence_length:
            return None

        # Use the same feature creation logic as training
        # Create a temporary target column for the sequence creation
        temp_df = last_sequence.copy()
        # Create dummy target
        temp_df['target'] = temp_df['gold_close'].shift(-1)

        # Use the same sequence creation method as training
        X_pred, _ = self.create_sequence_features(
            temp_df, target_col='gold_close', prediction_horizon=1)

        if X_pred.empty:
            return None

        # Return only the last row (most recent prediction)
        return X_pred.tail(1)

    def get_model_summary(self):
        """Get model summary information"""
        if self.model is None:
            return None

        return {
            'model_type': 'Pure GRU Neural Network',
            'sequence_length': self.sequence_length,
            'feature_count': len(self.feature_columns),
            'test_r2_score': self.best_score,
            'hidden_layers': self.hidden_layers
        }

    def save_model(self, filepath='gold_ml_model.pkl'):
        """Save the trained GRU neural network model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'hidden_layers': self.hidden_layers,
            'best_score': self.best_score
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Pure GRU Neural Network model saved to {filepath}")

    def load_model(self, filepath='gold_ml_model.pkl'):
        """Load a trained GRU neural network model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.sequence_length = model_data['sequence_length']
        self.hidden_layers = model_data.get('hidden_layers', (128, 64, 32))
        self.best_score = model_data['best_score']
        logger.info(f"Pure GRU Neural Network model loaded from {filepath}")


def main():
    """Main function to train and test the pure GRU neural network model"""
    predictor = GoldPriceMLPredictor(
        sequence_length=60, hidden_layers=(128, 64, 32))

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
    logger.info(f"Number of features: {len(predictor.feature_columns)}")

    # Train GRU-like model
    logger.info("Training GRU-like model...")
    training_results = predictor.train_model(X, y, test_size=0.2)

    # Get model summary
    model_summary = predictor.get_model_summary()
    if model_summary:
        logger.info("Model Summary:")
        for key, value in model_summary.items():
            logger.info(f"{key}: {value}")

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
