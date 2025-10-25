"""
News API Configuration
Set your API keys here for enhanced news sentiment analysis
"""

import os

# News API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', None)  # Get from NewsAPI.org
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', None)  # Get from Alpha Vantage

# News sources configuration
RSS_FEEDS = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US',
    'https://feeds.marketwatch.com/marketwatch/marketpulse/',
    'https://feeds.bloomberg.com/markets/news.rss',
    'https://feeds.reuters.com/news/wealth',
    'https://feeds.cnn.com/rss/money_latest.rss'
]

# Gold-related keywords for filtering news
GOLD_KEYWORDS = [
    'gold', 'precious metals', 'bullion', 'xau', 'xauusd', 'gold futures',
    'gold etf', 'gld', 'gold mining', 'gold stocks', 'inflation', 'dollar',
    'fed', 'federal reserve', 'interest rates', 'treasury', 'bonds',
    'safe haven', 'economic uncertainty', 'recession', 'crisis',
    'quantitative easing', 'monetary policy', 'central bank'
]

# Sentiment analysis configuration
SENTIMENT_WEIGHTS = {
    'polarity': 0.4,
    'gold_sentiment': 0.6
}

# News fetching configuration
DEFAULT_DAYS_BACK = 30
MAX_NEWS_ARTICLES = 1000
NEWS_CACHE_DURATION = 3600  # 1 hour in seconds
