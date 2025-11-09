"""
News-Based Gold Price Prediction Enhancement
Integrates news sentiment analysis with Lasso regression for improved accuracy
"""

import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
from textblob import TextBlob
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
from config.news_config import NEWS_API_KEY, ALPHA_VANTAGE_KEY, RSS_FEEDS, GOLD_KEYWORDS

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    News sentiment analysis for gold price prediction
    Fetches news from multiple sources and analyzes sentiment
    """

    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words='english')
        self.sentiment_scaler = StandardScaler()

        # Gold-related keywords for filtering
        self.gold_keywords = GOLD_KEYWORDS

    def fetch_news_data(self, days_back: int = 30) -> List[Dict]:
        """
        Fetch news data from multiple sources
        """
        news_data = []

        # Fetch from NewsAPI (if key available)
        if self.news_api_key:
            news_data.extend(self._fetch_newsapi_data(days_back))

        # Fetch from Alpha Vantage (if key available)
        if self.alpha_vantage_key:
            news_data.extend(self._fetch_alpha_vantage_data(days_back))

        # Fetch from Yahoo Finance news
        news_data.extend(self._fetch_yahoo_news_data(days_back))

        # Fetch from RSS feeds
        news_data.extend(self._fetch_rss_news_data(days_back))

        return news_data

    def _fetch_newsapi_data(self, days_back: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'gold OR precious metals OR inflation OR federal reserve',
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 100
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            news_items = []

            for article in data.get('articles', []):
                news_items.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', '')
                })

            logger.info(f"Fetched {len(news_items)} articles from NewsAPI")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
            return []

    def _fetch_alpha_vantage_data(self, days_back: int) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'GOLD',
                'apikey': self.alpha_vantage_key,
                'limit': 1000
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            news_items = []

            for item in data.get('feed', []):
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published_at': item.get('time_published', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'sentiment_score': item.get('overall_sentiment_score', 0),
                    'sentiment_label': item.get('overall_sentiment_label', 'neutral')
                })

            logger.info(
                f"Fetched {len(news_items)} articles from Alpha Vantage")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return []

    def _fetch_yahoo_news_data(self, days_back: int) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        try:
            # Use yfinance to get news
            gold_ticker = yf.Ticker("GC=F")
            news = gold_ticker.news

            news_items = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for item in news:
                try:
                    pub_date = datetime.fromtimestamp(
                        item.get('providerPublishTime', 0))
                    if pub_date >= cutoff_date:
                        news_items.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'published_at': pub_date.isoformat(),
                            'source': 'Yahoo Finance',
                            'url': item.get('link', '')
                        })
                except:
                    continue

            logger.info(
                f"Fetched {len(news_items)} articles from Yahoo Finance")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return []

    def _fetch_rss_news_data(self, days_back: int) -> List[Dict]:
        """Fetch news from RSS feeds"""
        try:
            try:
                import feedparser
            except ImportError as e:
                logger.warning(f"feedparser not available or has compatibility issues: {e}")
                return []
            
            # Handle Python 3.13+ compatibility issue with feedparser
            try:
                import html
            except ImportError:
                # Python 3.13+ removed html.parser fallback, feedparser may have issues
                pass

            rss_feeds = RSS_FEEDS

            news_items = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)

                    for entry in feed.entries:
                        try:
                            # Handle published_parsed safely
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                                if pub_date >= cutoff_date:
                                    # Check if content is gold-related
                                    content = f"{entry.title} {entry.get('summary', '')}"
                                    if self._is_gold_related(content):
                                        news_items.append({
                                            'title': entry.title,
                                            'summary': entry.get('summary', ''),
                                            'published_at': pub_date.isoformat(),
                                            'source': 'RSS',
                                            'url': entry.link
                                        })
                        except Exception as entry_error:
                            logger.debug(f"Error processing RSS entry: {entry_error}")
                            continue

                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue

            logger.info(f"Fetched {len(news_items)} articles from RSS feeds")
            return news_items

        except Exception as e:
            logger.warning(f"Error fetching RSS data: {e} - continuing without RSS feeds")
            return []

    def _is_gold_related(self, text: str) -> bool:
        """Check if text is related to gold/precious metals"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.gold_keywords)

    def analyze_sentiment(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze sentiment of news data
        """
        if not news_data:
            return pd.DataFrame()

        sentiment_data = []

        for item in news_data:
            # Combine title, description, and content
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('summary', '')} {item.get('content', '')}"
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters

            if not text.strip():
                continue

            # TextBlob sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Custom sentiment scoring based on gold-specific keywords
            gold_sentiment = self._calculate_gold_sentiment(text)

            # Combine different sentiment measures
            combined_sentiment = (polarity + gold_sentiment) / 2

            sentiment_data.append({
                'date': item.get('published_at', ''),
                'title': item.get('title', ''),
                'source': item.get('source', ''),
                'text': text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'gold_sentiment': gold_sentiment,
                'combined_sentiment': combined_sentiment,
                'sentiment_score': item.get('sentiment_score', combined_sentiment),
                'sentiment_label': item.get('sentiment_label', self._get_sentiment_label(combined_sentiment))
            })

        return pd.DataFrame(sentiment_data)

    def _calculate_gold_sentiment(self, text: str) -> float:
        """Calculate gold-specific sentiment score"""
        text_lower = text.lower()

        # Positive keywords for gold
        positive_keywords = [
            'bullish', 'rise', 'increase', 'up', 'gain', 'strong', 'positive',
            'inflation hedge', 'safe haven', 'uncertainty', 'crisis', 'recession',
            'dollar weakness', 'fed cut', 'rate cut', 'stimulus', 'qe'
        ]

        # Negative keywords for gold
        negative_keywords = [
            'bearish', 'fall', 'decrease', 'down', 'loss', 'weak', 'negative',
            'dollar strength', 'fed hike', 'rate hike', 'tapering', 'recovery',
            'risk on', 'equity rally', 'bond rally'
        ]

        positive_count = sum(
            1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(
            1 for keyword in negative_keywords if keyword in text_lower)

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def create_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment-based features for ML model
        """
        if sentiment_df.empty:
            return pd.DataFrame()

        # Convert date column
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        sentiment_df = sentiment_df.sort_values('date')

        # Daily sentiment aggregation
        daily_sentiment = sentiment_df.groupby('date').agg({
            'polarity': ['mean', 'std', 'count'],
            'subjectivity': ['mean', 'std'],
            'gold_sentiment': ['mean', 'std'],
            'combined_sentiment': ['mean', 'std'],
            'sentiment_score': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        daily_sentiment.columns = ['date', 'polarity_mean', 'polarity_std', 'news_count',
                                   'subjectivity_mean', 'subjectivity_std',
                                   'gold_sentiment_mean', 'gold_sentiment_std',
                                   'combined_sentiment_mean', 'combined_sentiment_std',
                                   'sentiment_score_mean', 'sentiment_score_std']

        # Fill NaN values
        daily_sentiment = daily_sentiment.fillna(0)

        # Create additional features
        daily_sentiment['sentiment_volatility'] = daily_sentiment['polarity_std']
        daily_sentiment['news_volume'] = daily_sentiment['news_count']
        daily_sentiment['sentiment_trend'] = daily_sentiment['combined_sentiment_mean'].rolling(
            5).mean()
        daily_sentiment['sentiment_momentum'] = daily_sentiment['combined_sentiment_mean'].diff()

        # Create lagged features
        for lag in [1, 2, 3, 5, 10]:
            daily_sentiment[f'sentiment_lag_{lag}'] = daily_sentiment['combined_sentiment_mean'].shift(
                lag)
            daily_sentiment[f'news_volume_lag_{lag}'] = daily_sentiment['news_count'].shift(
                lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            daily_sentiment[f'sentiment_ma_{window}'] = daily_sentiment['combined_sentiment_mean'].rolling(
                window).mean()
            daily_sentiment[f'sentiment_std_{window}'] = daily_sentiment['combined_sentiment_mean'].rolling(
                window).std()
            daily_sentiment[f'news_volume_ma_{window}'] = daily_sentiment['news_count'].rolling(
                window).mean()

        return daily_sentiment


class NewsEnhancedLassoPredictor:
    """
    Enhanced Lasso Regression with News Sentiment Features
    Combines traditional technical/fundamental features with news sentiment
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

        self.news_analyzer = NewsSentimentAnalyzer()
        self.sentiment_features = None

    def fetch_and_analyze_news(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch and analyze news data for sentiment features
        """
        logger.info(
            f"Fetching and analyzing news data for the last {days_back} days")

        # Fetch news data
        news_data = self.news_analyzer.fetch_news_data(days_back)

        if not news_data:
            logger.warning("No news data fetched")
            return pd.DataFrame()

        # Analyze sentiment
        sentiment_df = self.news_analyzer.analyze_sentiment(news_data)

        if sentiment_df.empty:
            logger.warning("No sentiment data generated")
            return pd.DataFrame()

        # Create sentiment features
        self.sentiment_features = self.news_analyzer.create_sentiment_features(
            sentiment_df)

        logger.info(
            f"Created sentiment features with shape: {self.sentiment_features.shape}")
        return self.sentiment_features

    def create_enhanced_features(self, market_data: Dict, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create enhanced features combining market data with news sentiment
        """
        # Start with basic market features (similar to original lasso model)
        from .lasso_model import LassoGoldPredictor

        base_predictor = LassoGoldPredictor()
        base_features = base_predictor.create_fundamental_features(market_data)

        if sentiment_df is not None and not sentiment_df.empty:
            # Align sentiment features with market data
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
            base_features['date'] = base_features.index.date

            # Merge sentiment features
            enhanced_features = base_features.merge(
                sentiment_df,
                on='date',
                how='left'
            )

            # Fill missing sentiment values with neutral scores
            sentiment_cols = [
                col for col in enhanced_features.columns if 'sentiment' in col or 'news' in col]
            enhanced_features[sentiment_cols] = enhanced_features[sentiment_cols].fillna(
                0)

            # Remove date column
            enhanced_features = enhanced_features.drop('date', axis=1)

            logger.info(
                f"Enhanced features created with shape: {enhanced_features.shape}")
            logger.info(f"Added {len(sentiment_cols)} sentiment features")

            return enhanced_features
        else:
            logger.warning(
                "No sentiment data available, using base features only")
            return base_features

    def train_enhanced_model(self, enhanced_features: pd.DataFrame, target_col='gold_close', test_size=0.2):
        """
        Train enhanced Lasso model with news sentiment features
        """
        logger.info(
            "Training enhanced Lasso model with news sentiment features")

        # Prepare training data
        data = enhanced_features.copy()
        data['target'] = data[target_col].shift(-1)
        data_clean = data.dropna()

        if data_clean.empty:
            logger.error("No valid data after cleaning")
            return None

        # Separate features and target
        feature_cols = [
            col for col in data_clean.columns if col not in ['target', target_col]]
        X = data_clean[feature_cols]
        y = data_clean['target']

        self.feature_columns = feature_cols

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use LassoCV for optimal alpha
        from sklearn.linear_model import LassoCV
        lasso_cv = LassoCV(
            alphas=np.logspace(-4, 1, 50),
            cv=5,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )

        lasso_cv.fit(X_train_scaled, y_train)
        best_alpha = lasso_cv.alpha_

        # Create final model and fit it first
        self.model = Lasso(
            alpha=best_alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Fit the model first
        self.model.fit(X_train_scaled, y_train)

        # Feature selection - manually select features with non-zero coefficients
        # Get important features based on coefficient magnitude
        coef_threshold = np.abs(self.model.coef_).mean() * 0.1
        important_features_mask = np.abs(self.model.coef_) > coef_threshold

        # Store the feature selector info
        self.feature_selector = {
            'mask': important_features_mask,
            'n_features': important_features_mask.sum()
        }

        # Transform with the selector
        X_train_selected = X_train_scaled[:, important_features_mask]
        X_test_selected = X_test_scaled[:, important_features_mask]

        # Get selected features
        selected_indices = np.where(important_features_mask)[0]
        self.selected_features = [self.feature_columns[i]
                                  for i in selected_indices]

        logger.info(
            f"Feature selection - Input: {X_train_scaled.shape[1]}, Selected: {len(self.selected_features)}")

        # Retrain with selected features
        self.model.fit(X_train_selected, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.best_score = r2

        logger.info(f"Enhanced Lasso Model - R² = {r2:.4f}, MSE = {mse:.4f}")
        logger.info(
            f"Selected {len(self.selected_features)} features out of {len(self.feature_columns)}")

        return {
            'enhanced_lasso_model': {
                'model': self.model,
                'mse': mse,
                'r2': r2,
                'best_alpha': best_alpha,
                'selected_features': self.selected_features,
                'feature_importance': self.get_feature_importance()
            }
        }

    def predict_with_news(self, enhanced_features: pd.DataFrame) -> float:
        """
        Make prediction using enhanced model with news sentiment
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get latest features - only use the features that were used in training
        available_features = [
            col for col in self.feature_columns if col in enhanced_features.columns]

        if not available_features:
            logger.error(
                f"None of the required features found in enhanced_features. Available: {list(enhanced_features.columns)}")
            raise ValueError(
                "Feature mismatch between training and prediction")

        X_pred = enhanced_features[available_features].iloc[-1:].values

        # Handle missing features by filling with 0
        if len(available_features) < len(self.feature_columns):
            missing_features = [
                col for col in self.feature_columns if col not in available_features]
            logger.warning(
                f"Missing {len(missing_features)} features. Filling with zeros.")
            # Create a full feature vector with zeros for missing features
            full_X_pred = np.zeros((1, len(self.feature_columns)))
            for i, col in enumerate(self.feature_columns):
                if col in available_features:
                    col_idx = available_features.index(col)
                    full_X_pred[0, i] = X_pred[0, col_idx]
            X_pred = full_X_pred

        try:
            X_pred_scaled = self.scaler.transform(X_pred)

            # Check if feature selector expects different number of features
            if X_pred_scaled.shape[1] != len(self.feature_columns):
                logger.error(
                    f"Scaled feature size ({X_pred_scaled.shape[1]}) doesn't match feature columns ({len(self.feature_columns)})")
                raise ValueError("Feature size mismatch after scaling")

            # Transform features using custom selector
            if isinstance(self.feature_selector, dict):
                # Custom selector - use mask
                mask = self.feature_selector['mask']
                X_pred_selected = X_pred_scaled[:, mask]
            else:
                # Sklearn selector - use transform method
                X_pred_selected = self.feature_selector.transform(
                    X_pred_scaled)

            prediction = self.model.predict(X_pred_selected)[0]
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(f"X_pred shape: {X_pred.shape}")
            logger.error(f"Feature columns count: {len(self.feature_columns)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from enhanced model"""
        if self.model is None or self.feature_selector is None:
            return pd.DataFrame()

        coefficients = self.model.coef_
        feature_names = self.selected_features

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df

    def save_enhanced_model(self, filepath='enhanced_lasso_gold_model.pkl'):
        """Save the enhanced model"""
        if self.model is None:
            raise ValueError("No model to save")

        # If relative path, save to models directory
        from pathlib import Path
        model_path = Path(filepath)
        if not model_path.is_absolute():
            # Get the models directory (where this file is located)
            models_dir = Path(__file__).resolve().parent
            filepath = str(models_dir / filepath)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'best_score': self.best_score,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'news_analyzer': self.news_analyzer
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Enhanced Lasso model saved to {filepath}")

    def load_enhanced_model(self, filepath='enhanced_lasso_gold_model.pkl'):
        """Load the enhanced model"""
        from pathlib import Path
        
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"Enhanced model file not found: {filepath}")
        
        if not model_path.is_file():
            raise ValueError(f"Path exists but is not a file: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_columns = model_data['feature_columns']
        self.selected_features = model_data['selected_features']
        self.best_score = model_data['best_score']
        self.alpha = model_data.get('alpha', 0.01)
        self.max_iter = model_data.get('max_iter', 2000)
        self.news_analyzer = model_data.get(
            'news_analyzer', NewsSentimentAnalyzer())

        logger.info(f"Enhanced Lasso model loaded from {filepath}")


def main():
    """Main function to train enhanced model with news sentiment"""
    logger.info("Starting enhanced gold price prediction with news sentiment")

    # Initialize enhanced predictor
    predictor = NewsEnhancedLassoPredictor()

    # Fetch market data
    logger.info("Fetching market data...")
    from .lasso_model import LassoGoldPredictor
    base_predictor = LassoGoldPredictor()
    market_data = base_predictor.fetch_market_data()

    if market_data is None:
        logger.error("Failed to fetch market data")
        return

    # Fetch and analyze news
    logger.info("Fetching and analyzing news...")
    sentiment_features = predictor.fetch_and_analyze_news(days_back=30)

    # Create enhanced features
    logger.info("Creating enhanced features...")
    enhanced_features = predictor.create_enhanced_features(
        market_data, sentiment_features)

    if enhanced_features.empty:
        logger.error("No enhanced features created")
        return

    # Train enhanced model
    logger.info("Training enhanced model...")
    training_results = predictor.train_enhanced_model(enhanced_features)

    if training_results:
        # Save model
        predictor.save_enhanced_model()

        # Test prediction
        logger.info("Testing enhanced prediction...")
        try:
            prediction = predictor.predict_with_news(enhanced_features)
            current_price = enhanced_features['gold_close'].iloc[-1]
            change = prediction - current_price
            change_pct = (change / current_price) * 100

            logger.info(f"Current price: ${current_price:.2f}")
            logger.info(f"Enhanced prediction: ${prediction:.2f}")
            logger.info(f"Predicted change: ${change:.2f} ({change_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
