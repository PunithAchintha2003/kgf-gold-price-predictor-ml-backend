import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """PyTorch GRU model for gold price prediction"""

    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.2):
        super(GRUModel, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # GRU layers
        self.gru_layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.gru_layers.append(
                nn.GRU(input_dim, hidden_size, batch_first=True,
                       dropout=dropout if i < len(hidden_sizes)-1 else 0)
            )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through GRU layers
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Take the last output
        x = x[:, -1, :]

        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class GoldPriceGRUPredictor:
    def __init__(self, sequence_length=60, hidden_sizes=[128, 64, 32], dropout=0.2):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.sequence_length = sequence_length
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.best_score = -np.inf
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

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

    def create_sequences(self, data, target_col='gold_close', prediction_horizon=1):
        """Create sequences for time series data"""
        logger.info(f"Creating sequences with length {self.sequence_length}")

        # Create target variable (future price)
        data = data.copy()
        data['target'] = data[target_col].shift(-prediction_horizon)

        # Remove rows with NaN values
        data_clean = data.dropna()

        if data_clean.empty:
            logger.error("No valid data after cleaning NaN values")
            return np.array([]), np.array([])

        # Separate features and target
        feature_cols = [
            col for col in data_clean.columns if col not in ['target', target_col]]
        X = data_clean[feature_cols].values
        y = data_clean['target'].values

        self.feature_columns = feature_cols

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        logger.info(
            f"Sequences created - X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")

        return X_sequences, y_sequences

    def prepare_training_data(self, features_df, target_col='gold_close', prediction_horizon=1):
        """Prepare training data for GRU model"""
        logger.info(f"Original features shape: {features_df.shape}")
        logger.info(
            f"NaN count before cleaning: {features_df.isnull().sum().sum()}")

        # Create sequences
        X, y = self.create_sequences(
            features_df, target_col, prediction_horizon)

        if X.size == 0:
            logger.error("No valid sequences created")
            return np.array([]), np.array([])

        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train_model(self, X, y, test_size=0.2, epochs=100, batch_size=32, learning_rate=0.001):
        """Train GRU model with 80/20 train-test split"""
        logger.info(
            f"Training GRU model with {int((1-test_size)*100)}% training data")

        # Split data with 80/20 ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # Create model
        self.model = GRUModel(
            input_size=X_train.shape[2],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        logger.info(
            f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor).squeeze()
                val_loss = criterion(val_outputs, y_test_tensor).item()

            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(X_train_tensor).squeeze().cpu().numpy()
            y_test_pred = self.model(X_test_tensor).squeeze().cpu().numpy()

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        self.best_score = test_r2

        logger.info("Training Results:")
        logger.info(
            f"Train - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(
            f"Test - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

        # Clean up
        if os.path.exists('best_model.pth'):
            os.remove('best_model.pth')

        return {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'training_history': self.training_history
        }

    def predict_next_price(self, features_df):
        """Predict next day's gold price using GRU model"""
        if self.model is None:
            raise ValueError(
                "Model not trained yet. Call train_model() first.")

        # Get the last sequence of data
        if len(features_df) < self.sequence_length:
            raise ValueError(
                f"Not enough data. Need at least {self.sequence_length} data points.")

        # Get latest sequence
        latest_sequence = features_df[self.feature_columns].iloc[-self.sequence_length:].values

        # Scale the sequence
        latest_sequence_scaled = self.scaler.transform(latest_sequence)

        # Convert to tensor and add batch dimension
        latest_sequence_tensor = torch.FloatTensor(
            latest_sequence_scaled).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(
                latest_sequence_tensor).squeeze().cpu().numpy()

        return prediction

    def get_model_summary(self):
        """Get model summary information"""
        if self.model is None:
            return None

        return {
            'model_type': 'PyTorch GRU',
            'sequence_length': self.sequence_length,
            'feature_count': len(self.feature_columns),
            'test_r2_score': self.best_score,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'hidden_sizes': self.hidden_sizes,
            'device': str(self.device)
        }

    def save_model(self, filepath='gold_gru_model.pkl'):
        """Save the trained GRU model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': len(self.feature_columns),
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout
            },
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'best_score': self.best_score
        }
        joblib.dump(model_data, filepath)
        logger.info(f"GRU model saved to {filepath}")

    def load_model(self, filepath='gold_gru_model.pkl'):
        """Load a trained GRU model"""
        model_data = joblib.load(filepath)

        # Recreate model
        self.model = GRUModel(
            input_size=model_data['model_config']['input_size'],
            hidden_sizes=model_data['model_config']['hidden_sizes'],
            dropout=model_data['model_config']['dropout']
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(model_data['model_state_dict'])

        # Load other attributes
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.sequence_length = model_data['sequence_length']
        self.best_score = model_data['best_score']

        logger.info(f"GRU model loaded from {filepath}")


def main():
    """Main function to train and test the GRU model"""
    predictor = GoldPriceGRUPredictor(sequence_length=60)

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

    if X.size == 0:
        logger.error("No training data available")
        return

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Number of features: {len(predictor.feature_columns)}")

    # Train GRU model
    logger.info("Training GRU model...")
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
