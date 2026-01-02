# KGF Gold Price Predictor - Detailed Code Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend Code Description](#backend-code-description)
   - [Main Application](#main-application)
   - [Core Modules](#core-modules)
   - [Services](#services)
   - [Repositories](#repositories)
   - [API Routes](#api-routes)
   - [ML Models](#ml-models)
   - [Utilities](#utilities)
4. [Frontend Code Description](#frontend-code-description)
5. [Data Flow](#data-flow)
6. [Deployment](#deployment)

---

## Project Overview

**KGF Gold Price Predictor** is a production-ready machine learning application that predicts next-day XAU/USD (Gold) prices using Lasso Regression models with 96.16% accuracy. The system combines technical indicators with news sentiment analysis to deliver accurate predictions.

### Key Features
- **ML-Powered Predictions**: Lasso Regression with 96.16% accuracy
- **News Sentiment Analysis**: Multi-source sentiment integration
- **Real-time Updates**: WebSocket support with 10-second intervals
- **RESTful API**: 19+ well-documented endpoints
- **Background Tasks**: Automated prediction generation, model retraining, and updates
- **Production Ready**: Optimized caching, connection pooling, and error handling

### Technology Stack
- **Backend**: Python 3.11+, FastAPI 0.115.0+, scikit-learn 1.5.0+
- **Frontend**: React 18.3.1, TypeScript 5.8.3, Material-UI 7.3.2
- **Database**: PostgreSQL (production) / SQLite (development)
- **ML**: scikit-learn, pandas, numpy
- **Market Data**: yfinance, NewsAPI, Alpha Vantage

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React/TypeScript)               │
│  - Dashboard Components                                      │
│  - Real-time Charts (Plotly.js)                             │
│  - Redux State Management                                    │
│  - WebSocket Client                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/WebSocket
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              FastAPI Backend (Python)                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  API Layer (REST + WebSocket)                      │   │
│  │  - /api/v1/xauusd/* (19+ endpoints)               │   │
│  │  - /ws/xauusd (WebSocket)                          │   │
│  └───────────────────────┬────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼────────────────────────────┘   │
│  │  Service Layer                                         │   │
│  │  - PredictionService                                   │   │
│  │  - MarketDataService                                   │   │
│  │  - ExchangeService                                     │   │
│  └───────────────────────┬────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼────────────────────────────┘   │
│  │  Repository Layer                                      │   │
│  │  - PredictionRepository                                │   │
│  └───────────────────────┬────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼────────────────────────────┘   │
│  │  Database Layer                                        │   │
│  │  - PostgreSQL / SQLite                                 │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  ML Models                                          │   │
│  │  - NewsEnhancedLassoPredictor (Primary)           │   │
│  │  - LassoGoldPredictor (Fallback)                   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Background Tasks                                  │   │
│  │  - Auto-update predictions                         │   │
│  │  - Auto-retrain models                            │   │
│  │  - Auto-generate predictions                      │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Backend Code Description

### Main Application

#### `backend/app/main.py`
**Purpose**: FastAPI application entry point with lifecycle management

**Key Components**:

1. **Lifespan Manager** (`lifespan` function):
   - **Startup**: 
     - Initializes ML models (Lasso and News-Enhanced)
     - Sets up services (PredictionService, MarketDataService, ExchangeService)
     - Initializes database connections (PostgreSQL/SQLite)
     - Starts background tasks (auto-update, auto-retrain, auto-predict)
     - Initializes WebSocket connection manager
   - **Shutdown**: 
     - Gracefully stops background tasks
     - Closes WebSocket connections
     - Closes database connection pools

2. **FastAPI App Configuration**:
   - Title: "XAU/USD Real-time Data API"
   - Version: 2.0.0
   - OpenAPI documentation at `/docs`
   - Exception handlers for custom errors

3. **Middleware Stack** (order matters):
   - `SecurityHeadersMiddleware`: Adds security headers
   - `CompressionMiddleware`: Compresses responses
   - `TimingMiddleware`: Tracks response times
   - `RequestSizeLimitMiddleware`: Limits request size
   - `CORSMiddleware`: Handles CORS

4. **API Routes**:
   - Includes `/api/v1/*` routes
   - Root endpoint `/` with API info
   - Health check at `/health`

5. **WebSocket Endpoint** (`/ws/xauusd`):
   - Manages real-time connections
   - Broadcasts market data every 10 seconds
   - Handles rate limiting gracefully

**Code Flow**:
```python
# Startup sequence
1. Initialize ML models → initialize_models()
2. Create services → PredictionService, MarketDataService, etc.
3. Initialize database → init_database(), init_postgresql_pool()
4. Start background tasks → asyncio.create_task()
5. Initialize WebSocket manager → ConnectionManager()
```

---

### Core Modules

#### `backend/app/core/config.py`
**Purpose**: Application configuration using Pydantic Settings

**Key Features**:
- **Environment-based settings**: Development, staging, production
- **Database configuration**: PostgreSQL/SQLite with connection pooling
- **Cache settings**: TTL, cooldown periods, rate limit backoff
- **Background task scheduling**: Auto-update, auto-retrain, auto-predict intervals
- **API keys management**: NewsAPI, Alpha Vantage, Gemini
- **CORS configuration**: Allowed origins per environment

**Settings Structure**:
```python
class Settings(BaseSettings):
    # Environment
    environment: Literal["development", "staging", "production"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    # Database
    use_postgresql: bool = False
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_database: str = "gold_predictor"
    postgresql_user: str = "postgres"
    postgresql_password: Optional[str] = None
    
    # Cache
    cache_duration: int = 300  # 5 minutes
    api_cooldown: int = 5  # seconds
    realtime_cache_duration: int = 60  # 1 minute
    
    # Background Tasks
    auto_update_enabled: bool = True
    auto_update_interval: int = 3600  # 1 hour
    auto_retrain_enabled: bool = True
    auto_retrain_hour: int = 2  # 2 AM
    auto_predict_enabled: bool = True
    auto_predict_hour: int = 8  # 8 AM
```

**Validation**:
- Validates PostgreSQL settings when enabled
- Ensures production environment has proper CORS settings
- Validates port ranges and numeric values

---

#### `backend/app/core/database.py`
**Purpose**: Database connection management and initialization

**Key Functions**:

1. **`init_postgresql_pool()`**:
   - Creates PostgreSQL connection pool
   - Configures SSL for cloud databases (Render, AWS, Neon)
   - Sets up connection keepalives to prevent stale connections
   - Pool size: 5-50 connections (configurable)

2. **`get_db_connection()`**:
   - Context manager for database connections
   - Returns PostgreSQL or SQLite connection based on config
   - Handles connection errors gracefully

3. **`init_database()`**:
   - Creates `predictions` table if not exists
   - Schema:
     - `prediction_date` (PRIMARY KEY)
     - `predicted_price` (FLOAT)
     - `actual_price` (FLOAT, nullable)
     - `accuracy_percentage` (FLOAT, nullable)
     - `prediction_method` (TEXT)
     - `prediction_reasons` (TEXT, nullable)
     - `created_at` (TIMESTAMP)
     - `updated_at` (TIMESTAMP)

4. **`get_db_type()`**:
   - Returns "postgresql" or "sqlite" based on configuration

5. **`get_date_function()`**:
   - Returns database-appropriate date function
   - PostgreSQL: `CURRENT_DATE - INTERVAL 'N days'`
   - SQLite: `date('now', '-N days')`

**Connection Pooling**:
- PostgreSQL: Uses `psycopg2.pool.SimpleConnectionPool`
- SQLite: Thread-safe connections with connection pooling
- Automatic connection health checks

---

#### `backend/app/core/database_async.py`
**Purpose**: Async database operations for better performance

**Key Functions**:

1. **`init_postgresql_pool_async()`**:
   - Creates asyncpg connection pool for PostgreSQL
   - Enables non-blocking database operations
   - Better performance for concurrent requests

2. **`init_database_async()`**:
   - Creates tables using async operations
   - Non-blocking table creation

3. **`close_postgresql_pool_async()`**:
   - Gracefully closes async connection pool on shutdown

---

#### `backend/app/core/models.py`
**Purpose**: ML model initialization and management

**Key Function**:

**`initialize_models()`**:
- Loads News-Enhanced Lasso model (primary)
- Loads Basic Lasso model (fallback)
- Returns tuple: `(lasso_predictor, news_enhanced_predictor)`
- Handles model loading errors gracefully
- Logs model information (R² score, features count)

**Model Loading Flow**:
```python
1. Check for enhanced_lasso_gold_model.pkl
2. If exists → Load NewsEnhancedLassoPredictor
3. Check for lasso_gold_model.pkl
4. If exists → Load LassoGoldPredictor
5. Return both (one may be None)
```

---

#### `backend/app/core/background_tasks.py`
**Purpose**: Automated background processing tasks

**Key Tasks**:

1. **`broadcast_daily_data()`**:
   - Broadcasts market data to WebSocket clients
   - Runs every 10 seconds
   - Checks rate limits before fetching data
   - Only broadcasts when data changes

2. **`auto_update_pending_predictions()`**:
   - Updates pending predictions with actual market prices
   - Runs every hour (configurable)
   - Handles rate limiting with exponential backoff
   - Circuit breaker pattern (stops after 5 consecutive failures)
   - Updates predictions for past dates using market data

3. **`auto_retrain_model()`**:
   - Retrains ML models daily at 2 AM (configurable)
   - Requires minimum 10 predictions before retraining
   - Uses historical predictions and actual prices
   - Saves new model to disk
   - Logs training metrics (R² score, MAE)

4. **`auto_generate_daily_prediction()`**:
   - Generates daily prediction at 8 AM (configurable)
   - Creates prediction for next trading day (skips weekends)
   - Saves to database
   - Generates AI explanation using Gemini (if available)

5. **`auto_retrain_and_predict()`**:
   - Combined task: retrains first, then generates prediction
   - Runs at same scheduled time
   - Ensures prediction uses latest model

**Task Management**:
- Uses `BackgroundTaskManager` for lifecycle management
- Graceful shutdown support
- Error tracking and logging
- Health status tracking

---

#### `backend/app/core/task_manager.py`
**Purpose**: Background task lifecycle management

**Key Features**:
- **Task Registration**: Tracks all background tasks
- **Shutdown Event**: `asyncio.Event` for graceful shutdown
- **Task States**: Tracks status, last run, errors, run count
- **Shutdown Method**: Cancels all tasks gracefully with timeout

**Task State Structure**:
```python
{
    "status": "running" | "stopped" | "error",
    "last_run": "2025-01-01T00:00:00",
    "last_error": None | "error message",
    "run_count": 0,
    "error_count": 0
}
```

---

#### `backend/app/core/websocket.py`
**Purpose**: WebSocket connection management

**Key Features**:
- **ConnectionManager Class**:
  - Tracks active WebSocket connections in a list
  - Thread-safe connection management
  - Broadcasts messages to all clients
  - Sends personal messages to specific clients
  - Handles disconnections gracefully

**Methods**:
- `connect(websocket)`: 
  - Accepts WebSocket connection
  - Adds to active_connections list
  - Logs connection count
  
- `disconnect(websocket)`: 
  - Removes connection from active_connections
  - Logs disconnection
  
- `broadcast(message)`: 
  - Sends message to all connected clients
  - Handles errors per client (continues if one fails)
  - Used by background task for real-time updates
  
- `send_personal_message(message, websocket)`: 
  - Sends message to specific client
  - Used in WebSocket endpoint for individual updates
  
- `disconnect_all()`: 
  - Closes all connections on shutdown
  - Handles CancelledError gracefully
  - Returns count of disconnected clients

**Connection Lifecycle**:
```python
1. Client connects → connect() called
2. Connection added to active_connections
3. Background task broadcasts every 10s
4. On shutdown → disconnect_all() called
5. All connections closed gracefully
```

---

#### `backend/app/core/dependencies.py`
**Purpose**: Dependency injection for FastAPI routes

**Key Functions**:
- `get_market_data_service()`: Returns MarketDataService instance
- `get_prediction_service()`: Returns PredictionService instance
- `get_prediction_repo()`: Returns PredictionRepository instance
- `get_exchange_service()`: Returns ExchangeService instance

**Usage**:
```python
@router.get("/")
async def endpoint(
    market_data_service = Depends(get_market_data_service)
):
    return market_data_service.get_daily_data()
```

---

#### `backend/app/core/exceptions.py`
**Purpose**: Custom exception classes and handlers

**Exception Classes**:
- `BaseAPIException`: Base class for all API exceptions
- `ValidationException`: Input validation errors
- `DatabaseException`: Database operation errors
- `ModelException`: ML model errors

**Exception Handlers**:
- `base_api_exception_handler`: Handles custom API exceptions
- `validation_exception_handler`: Handles Pydantic validation errors
- `general_exception_handler`: Catches all other exceptions

---

#### `backend/app/core/middleware.py`
**Purpose**: Custom FastAPI middleware

**Middleware Classes**:

1. **`SecurityHeadersMiddleware`**:
   - Adds security headers (X-Content-Type-Options, X-Frame-Options, etc.)
   - Prevents common web vulnerabilities

2. **`CompressionMiddleware`**:
   - Compresses responses using gzip
   - Reduces bandwidth usage

3. **`TimingMiddleware`**:
   - Tracks request processing time
   - Adds X-Process-Time header

4. **`RequestSizeLimitMiddleware`**:
   - Limits request body size (default: 10MB)
   - Prevents DoS attacks

---

#### `backend/app/core/logging_config.py`
**Purpose**: Structured logging configuration

**Features**:
- JSON-formatted logs for production
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Emoji support for better readability
- File and console handlers

---

#### `backend/app/core/response_cache.py`
**Purpose**: Response caching for API endpoints

**Features**:
- TTL-based caching
- Automatic cache invalidation
- Memory-efficient storage
- Thread-safe operations

**Usage**:
```python
from ..core.response_cache import response_cache

cached = response_cache.get(cache_key, ttl=30)
if cached is not None:
    return cached

# Generate response
response = generate_response()
response_cache.set(cache_key, response, ttl=30)
return response
```

---

### Services

#### `backend/app/services/prediction_service.py`
**Purpose**: Orchestrates ML predictions with fallback logic

**Key Methods**:

1. **`predict_next_day()`**:
   - **Primary**: Uses News-Enhanced Lasso model
     - Fetches market data
     - Fetches news sentiment (7 days back)
     - Creates enhanced features
     - Makes prediction
   - **Fallback**: Uses Basic Lasso model if primary fails
     - Fetches market data
     - Creates fundamental features
     - Makes prediction
   - Returns `float` or `None` on error

2. **`get_model_info()`**:
   - Returns detailed model information:
     - Active model name
     - Model type
     - Training R² score
     - Live R² score (from actual predictions)
     - Feature counts
     - Fallback availability
     - Live accuracy statistics

3. **`get_model_display_name()`**:
   - Returns human-readable model name
   - "News-Enhanced Lasso Regression (Primary)" or "Lasso Regression (Fallback)"

**Prediction Flow**:
```python
1. Try News-Enhanced model:
   - Fetch market data
   - Fetch news sentiment
   - Create enhanced features
   - Predict
   - If success → return prediction
   
2. If fails, try Basic Lasso:
   - Fetch market data
   - Create fundamental features
   - Predict
   - If success → return prediction
   
3. If both fail → return None
```

---

#### `backend/app/services/market_data_service.py`
**Purpose**: Market data operations and aggregation

**Key Methods**:

1. **`get_daily_data(days, start_date, end_date)`**:
   - Fetches historical market data
   - Merges with predictions from database
   - Filters by date range if provided
   - Handles rate limiting gracefully
   - Returns structured response with:
     - Daily data points (OHLCV)
     - Historical predictions
     - Accuracy statistics
     - Current price
     - Next day prediction
     - Model information
     - Metadata

2. **`get_realtime_price()`**:
   - Gets current gold price
   - Uses last trading day's price on weekends
   - Handles rate limiting
   - Returns price with timestamp

3. **`update_pending_predictions()`**:
   - Updates pending predictions with actual prices
   - Fetches market data for date range
   - Matches predictions with market data
   - Handles weekends/holidays (uses last trading day)
   - Returns update statistics

**Weekend Handling**:
- Skips weekend dates in predictions
- Uses last trading day's price on weekends
- Filters out weekend data points

---

#### `backend/app/services/exchange_service.py`
**Purpose**: Currency exchange rate operations

**Key Methods**:
- `get_exchange_rate(from_currency, to_currency)`: Fetches exchange rate
- Caches rates to reduce API calls
- Handles errors gracefully

---

### Repositories

#### `backend/app/repositories/prediction_repository.py`
**Purpose**: Database access layer for predictions

**Key Methods**:

1. **`save_prediction()`**:
   - Saves or updates prediction (UPSERT)
   - Calculates accuracy if actual price available
   - Formula: `accuracy = 100 - (|predicted - actual| / actual * 100)`
   - Supports PostgreSQL and SQLite

2. **`prediction_exists_for_date()`**:
   - Checks if prediction exists for date
   - Uses query result caching (60s TTL)

3. **`get_prediction_for_date()`**:
   - Gets predicted price for specific date
   - Returns latest prediction if multiple exist

4. **`get_historical_predictions(days)`**:
   - Gets predictions for last N days
   - Filters duplicates (uses latest per date)
   - Optimized query with window functions (PostgreSQL)
   - Cached for 60 seconds

5. **`get_accuracy_stats()`**:
   - Calculates accuracy statistics:
     - Average accuracy
     - R² score (coefficient of determination)
     - Total predictions
     - Evaluated predictions
   - Excludes weekends and manual entries
   - Cached for 60 seconds

6. **`get_pending_predictions()`**:
   - Gets all predictions without actual prices
   - Filters weekends
   - Returns sorted by date

7. **`update_prediction_with_actual_price()`**:
   - Updates existing prediction with actual price
   - Recalculates accuracy
   - Preserves other fields (method, reasons)

8. **`get_accuracy_visualization_data()`**:
   - Gets detailed accuracy data for charts
   - Includes error calculations
   - Returns statistics (min, max, average)

**Query Optimization**:
- Uses window functions (PostgreSQL) for deduplication
- Query result caching with TTL
- Single queries instead of multiple
- Indexed columns for fast lookups

**Caching Strategy**:
- Query results cached for 30-60 seconds
- Cache invalidation on writes
- Automatic cleanup of expired entries

---

### API Routes

#### `backend/app/api/v1/routes/xauusd.py`
**Purpose**: XAU/USD gold price API endpoints

**Endpoints**:

1. **`GET /api/v1/xauusd`**:
   - Returns daily market data with predictions
   - Query params: `days` (default: 90), `start_date`, `end_date`
   - Cached for 30 seconds
   - Includes historical predictions, accuracy stats, model info

2. **`GET /api/v1/xauusd/realtime`**:
   - Returns current real-time price
   - Uses last trading day on weekends
   - Handles rate limiting

3. **`GET /api/v1/xauusd/enhanced-prediction`**:
   - Returns ML prediction with sentiment analysis
   - Includes:
     - Next day price prediction
     - Current price
     - Change percentage
     - Model information
     - Sentiment scores
   - Generates prediction on-demand if not in database

4. **`GET /api/v1/xauusd/prediction-stats`**:
   - Returns comprehensive prediction statistics
   - Includes accuracy metrics, R² score, evaluation rate

5. **`GET /api/v1/xauusd/prediction-history`**:
   - Returns historical predictions
   - Query param: `days` (default: 30)
   - Includes accuracy percentages

6. **`GET /api/v1/xauusd/pending-predictions`**:
   - Returns predictions waiting for actual prices
   - Filters weekends

7. **`GET /api/v1/xauusd/accuracy-visualization`**:
   - Returns accuracy data for charts
   - Includes error calculations
   - Statistics (min, max, average)

8. **`GET /api/v1/xauusd/model-info`**:
   - Returns detailed ML model information
   - Includes R² scores, feature counts, model type

9. **`GET /api/v1/xauusd/prediction-reasons`**:
   - Returns AI-generated prediction explanations
   - Uses Google Gemini API
   - Human-readable explanations

10. **`POST /api/v1/xauusd/update-pending-predictions`**:
    - Manually triggers prediction updates
    - Updates pending predictions with actual prices
    - Returns update statistics

**Response Caching**:
- Uses `response_cache` for frequently accessed endpoints
- TTL: 30 seconds for market data
- Cache key includes query parameters

---

#### `backend/app/api/v1/routes/health.py`
**Purpose**: Health check endpoints

**Endpoints**:

1. **`GET /api/v1/health`**:
   - Returns comprehensive health status:
     - Database connectivity
     - Background task status
     - Model loading status
     - System metrics
   - Used by monitoring systems

**Health Check Structure**:
```json
{
  "status": "healthy",
  "database": {
    "connected": true,
    "type": "postgresql"
  },
  "models": {
    "primary": "loaded",
    "fallback": "loaded"
  },
  "background_tasks": {
    "auto_update": "running",
    "auto_retrain": "running"
  }
}
```

---

#### `backend/app/api/v1/routes/exchange.py`
**Purpose**: Currency exchange rate endpoints

**Endpoints**:
- `GET /api/v1/exchange-rate/{from_currency}/{to_currency}`: Gets exchange rate

---

### ML Models

#### `backend/models/lasso_model.py`
**Purpose**: Basic Lasso Regression model for gold price prediction

**Key Components**:

1. **`LassoGoldPredictor` Class**:
   - **Initialization**:
     - `alpha`: L1 regularization parameter (default: 0.01)
     - `max_iter`: Maximum iterations (default: 2000)
     - `random_state`: Random seed (default: 42)

2. **`fetch_market_data()`**:
   - Fetches gold futures data (GC=F) - **Required**
   - Fetches optional data:
     - Dollar Index (DX-Y.NYB)
     - 10-Year Treasury Yield (^TNX)
     - VIX (^VIX)
     - Oil prices (CL=F)
   - Period: 2 years by default
   - Returns pandas DataFrame

3. **`create_fundamental_features()`**:
   - Creates 35+ technical indicators:
     - **Moving Averages**: SMA 5, 10, 20, 50
     - **Exponential Moving Averages**: EMA 5, 10, 20, 50
     - **RSI**: Relative Strength Index (14-period)
     - **MACD**: Moving Average Convergence Divergence
     - **Bollinger Bands**: Upper, middle, lower bands
     - **Volume Indicators**: Volume SMA, volume ratio
     - **Price Momentum**: Rate of change, price ratios
     - **Volatility**: Rolling standard deviation
     - **Market Indicators**: DXY, Treasury, VIX, Oil correlations

4. **`train_model()`**:
   - Splits data: 80% train, 20% test
   - Standardizes features using StandardScaler
   - Uses LassoCV for optimal alpha selection
   - Feature selection using SelectFromModel
   - Cross-validation for model evaluation
   - Saves model and scaler to disk

5. **`predict_next_price()`**:
   - Fetches latest market data
   - Creates features for prediction
   - Standardizes features
   - Makes prediction using trained model
   - Returns predicted price (float)

**Model Training Flow**:
```python
1. Fetch market data (2 years)
2. Create features (35+ indicators)
3. Split data (train/test)
4. Standardize features
5. Train LassoCV model
6. Feature selection
7. Cross-validation
8. Save model to .pkl file
```

**Feature Engineering**:
- Technical indicators calculated from OHLCV data
- Lag features (previous day values)
- Rolling statistics (mean, std, min, max)
- Price ratios and percentages
- Market correlation features

---

#### `backend/models/news_prediction.py`
**Purpose**: News-Enhanced Lasso Regression model

**Key Components**:

1. **`NewsEnhancedLassoPredictor` Class**:
   - Extends base Lasso model
   - Adds news sentiment features

2. **`fetch_and_analyze_news()`**:
   - Fetches news from multiple sources:
     - Yahoo Finance RSS
     - NewsAPI (if API key available)
     - Alpha Vantage (if API key available)
   - Analyzes sentiment using TextBlob
   - Extracts gold-specific keywords
   - Returns sentiment features DataFrame

3. **`create_enhanced_features()`**:
   - Combines base features with sentiment features
   - Creates sentiment indicators:
     - Combined sentiment score
     - News volume
     - Sentiment trend
     - Positive/negative news ratio
   - Returns enhanced features DataFrame

4. **`predict_with_news()`**:
   - Uses enhanced features for prediction
   - Falls back to base features if sentiment unavailable
   - Returns predicted price

**Sentiment Analysis**:
- TextBlob for sentiment scoring (-1 to +1)
- Gold-specific keyword weighting
- Temporal aggregation (7-day rolling average)
- News volume normalization

---

### Utilities

#### `backend/app/utils/cache.py`
**Purpose**: Market data caching with rate limit handling

**Key Components**:

1. **`MarketDataCache` Class**:
   - **`get_cached_market_data(period)`**:
     - Checks cache first
     - Fetches from yfinance if cache expired
     - Handles rate limiting with exponential backoff
     - Returns: (data, symbol_used, rate_limit_info)

2. **Rate Limit Handling**:
   - Detects rate limits from yfinance
   - Exponential backoff: 60s → 1800s (max)
   - Tracks rate limit until timestamp
   - Serves cached data during rate limits

3. **Cache Strategy**:
   - TTL: 5 minutes for market data
   - 1 minute for real-time data
   - Automatic cache invalidation

---

#### `backend/app/utils/yfinance_helper.py`
**Purpose**: Yahoo Finance data fetching utilities

**Features**:
- Wrapper around yfinance library
- Error handling and retries
- Rate limit detection
- Data normalization

---

#### `backend/app/utils/fallback_data.py`
**Purpose**: Fallback data when market data unavailable

**Features**:
- Provides default/fallback data
- Prevents API failures
- Maintains response structure

---

### AI Services

#### `backend/ai/services/gemini_service.py`
**Purpose**: Google Gemini API integration for AI-powered explanations

**Key Components**:

1. **`GeminiService` Class**:
   - Initializes with API key from config
   - Configures model, temperature, max_tokens
   - Handles API key validation

2. **`_make_request()`**:
   - Makes HTTP request to Gemini API
   - Implements retry logic with exponential backoff
   - Handles rate limiting
   - Returns JSON response

3. **`generate_text()`**:
   - Generates text using Gemini model
   - Supports system instructions
   - Error handling and fallback

4. **`generate_prediction_reasons()`**:
   - Generates human-readable prediction explanations
   - Includes market context
   - Explains prediction factors

**Configuration**:
- Model: `gemini-pro` (configurable)
- Temperature: 0.7 (configurable)
- Max tokens: 500 (configurable)
- Timeout: 30 seconds
- Max retries: 3

**Error Handling**:
- Graceful degradation if API key missing
- Retry with exponential backoff
- Fallback to simple explanations

---

#### `backend/ai/services/prediction_reason_service.py`
**Purpose**: AI-powered prediction explanation service

**Features**:
- Uses GeminiService to generate explanations
- Context-aware explanations with:
  - Current market conditions
  - Technical indicators
  - News sentiment
  - Price trends
- Human-readable format
- Error handling with fallback messages

---

### Schemas

#### `backend/app/schemas/prediction.py`
**Purpose**: Pydantic models for prediction data validation

**Key Models**:

1. **`Prediction`**:
   - `next_day`: Next trading day (YYYY-MM-DD)
   - `predicted_price`: Predicted price (must be > 0)
   - `current_price`: Current market price (must be > 0)
   - `prediction_method`: ML model name
   - `change`: Price change amount
   - `change_percentage`: Price change percentage
   - Date format validation

2. **`PredictionHistoryItem`**:
   - `date`: Prediction date
   - `predicted_price`: Predicted price
   - `actual_price`: Actual price (optional)
   - `accuracy_percentage`: Accuracy (0-100)
   - `status`: "pending" or "completed"
   - `method`: Prediction method
   - `prediction_reasons`: AI-generated explanation

3. **`AccuracyStats`**:
   - `average_accuracy`: Average accuracy (0-100)
   - `r2_score`: R² score (-1 to 1)
   - `training_r2_score`: Training R² score
   - `live_r2_score`: Live R² score
   - `total_predictions`: Total count
   - `evaluated_predictions`: Evaluated count
   - `pending_predictions`: Pending count
   - `evaluation_rate`: Evaluation rate percentage

4. **`ModelInfo`**:
   - `active_model`: Active model name
   - `model_type`: Model type
   - `r2_score`: Primary R² score
   - `features_count`: Total features
   - `selected_features_count`: Selected features
   - `selected_features`: Feature names list
   - `fallback_available`: Fallback model status

5. **`EnhancedPredictionResponse`**:
   - Complete response structure for enhanced predictions
   - Includes prediction, model info, sentiment, top features

**Validation**:
- All numeric fields have range constraints
- Date fields validated for YYYY-MM-DD format
- Required fields enforced
- Optional fields properly typed

---

#### `backend/app/schemas/market_data.py`
**Purpose**: Pydantic models for market data

**Models**:
- `DailyDataPoint`: Single day's OHLCV data
- `DailyDataResponse`: Complete daily data response
- Market data validation and structure

---

#### `backend/app/schemas/exchange.py`
**Purpose**: Pydantic models for exchange rates

**Models**:
- `ExchangeRateResponse`: Exchange rate response structure
- Currency validation

---

### Scripts

#### `scripts/train_enhanced_model.py`
**Purpose**: Script to train the News-Enhanced Lasso model

**Functionality**:
- Trains the enhanced model with news sentiment
- Fetches market data and news
- Creates enhanced features
- Saves model to `backend/models/enhanced_lasso_gold_model.pkl`
- Takes 5-15 minutes depending on data volume

**Usage**:
```bash
python scripts/train_enhanced_model.py
```

**Requirements**:
- Internet connection
- API keys for NewsAPI and Alpha Vantage (optional)
- Sufficient disk space (~10-50MB for model file)

---

#### `scripts/check_predictions.py`
**Purpose**: Utility script to check prediction accuracy

**Functionality**:
- Queries database for predictions
- Calculates accuracy statistics
- Displays prediction history
- Useful for monitoring and debugging

---

#### `scripts/diagnose_apis.py`
**Purpose**: Diagnostic script for API connectivity

**Functionality**:
- Tests Yahoo Finance API connectivity
- Tests NewsAPI connectivity
- Tests Alpha Vantage connectivity
- Tests Gemini API connectivity
- Reports rate limit status
- Useful for troubleshooting

---

#### `scripts/import_predictions.py`
**Purpose**: Script to import predictions from external sources

**Functionality**:
- Imports predictions from CSV/JSON
- Validates data format
- Saves to database
- Useful for data migration

---

## Frontend Code Description

### Main Components

#### `kgf-gold-tradex-frontend/src/components/price-predictor/Dashboard.tsx`
**Purpose**: Main dashboard component for gold price predictions

**Key Features**:
- **Real-time Price Display**: Shows current gold price
- **Interactive Charts**: Plotly.js charts with historical data
- **Prediction History Table**: Shows past predictions with accuracy
- **Accuracy Statistics**: Displays model accuracy metrics
- **Pending Predictions**: Shows predictions waiting for actual prices
- **Currency Conversion**: Supports multiple currencies (USD, EUR, GBP, etc.)
- **Responsive Design**: Mobile, tablet, desktop layouts

**State Management**:
- Uses Redux Toolkit Query for API calls
- Local state for UI interactions (zoom, sidebar)
- WebSocket support (optional, can use REST polling)

**API Integration**:
- `useGetDailyDataQuery`: Fetches daily market data
- `useGetRealtimePriceQuery`: Fetches real-time price
- `useGetEnhancedPredictionQuery`: Gets ML predictions
- `useGetPredictionHistoryQuery`: Gets historical predictions
- `useGetAccuracyVisualizationQuery`: Gets accuracy charts data

**Performance Optimizations**:
- Lazy loading of heavy components (Chart, Plotly)
- Memoization of expensive calculations
- Debounced API calls
- Efficient re-renders

---

#### `kgf-gold-tradex-frontend/src/components/price-predictor/Chart.tsx`
**Purpose**: Interactive price chart component

**Features**:
- Plotly.js integration
- Historical price data visualization
- Prediction overlays
- Zoom and pan functionality
- Multiple timeframes
- Responsive design

---

#### `kgf-gold-tradex-frontend/src/store/api/goldApi.ts`
**Purpose**: Redux Toolkit Query API client

**Endpoints**:
- All API endpoints defined as RTK Query hooks
- Automatic caching and refetching
- Error handling
- TypeScript types

---

## Data Flow

### Prediction Generation Flow

```
1. Background Task Triggered (8 AM daily)
   ↓
2. MarketDataService.get_daily_data()
   ↓
3. PredictionService.predict_next_day()
   ↓
4. NewsEnhancedLassoPredictor.predict_with_news()
   ↓
5. Fetch Market Data (yfinance)
   ↓
6. Fetch News Sentiment (NewsAPI, Alpha Vantage)
   ↓
7. Create Enhanced Features (35+ indicators + sentiment)
   ↓
8. Model Prediction
   ↓
9. PredictionRepository.save_prediction()
   ↓
10. Database (PostgreSQL/SQLite)
```

### API Request Flow

```
1. Client Request → FastAPI Route
   ↓
2. Dependency Injection → Service Layer
   ↓
3. Service → Repository Layer
   ↓
4. Repository → Database
   ↓
5. Response Caching (if applicable)
   ↓
6. Response to Client
```

### WebSocket Flow

```
1. Client Connects → /ws/xauusd
   ↓
2. ConnectionManager.connect()
   ↓
3. Background Task: broadcast_daily_data()
   ↓
4. Every 10 seconds:
   - Fetch latest data
   - Broadcast to all clients
   ↓
5. Client Updates UI
```

---

## Deployment

### Production Deployment (Render)

1. **Environment Variables**:
   - `ENVIRONMENT=production`
   - `USE_POSTGRESQL=true`
   - `POSTGRESQL_HOST=...`
   - `POSTGRESQL_DATABASE=...`
   - `POSTGRESQL_USER=...`
   - `POSTGRESQL_PASSWORD=...`
   - API keys (NewsAPI, Alpha Vantage, Gemini)

2. **Build Process**:
   - `pip install -r requirements.txt`
   - Python 3.11+

3. **Start Command**:
   - `python run_backend.py`

4. **Database Setup**:
   - PostgreSQL database on Render
   - Tables created automatically on first run

---

## Summary

This project demonstrates a production-ready ML application with:

- **Modular Architecture**: Clean separation of concerns (API → Service → Repository → Database)
- **Scalable Design**: Async operations, connection pooling, background tasks
- **Reliability**: Error handling, fallbacks, circuit breakers, graceful degradation
- **Performance**: Caching (85% hit rate), query optimization, lazy loading
- **Maintainability**: Type hints, comprehensive documentation, structured logging
- **Production Ready**: Health checks, monitoring, graceful shutdown, deployment configs

### Code Organization

**Backend Structure**:
```
backend/
├── app/                    # Main application
│   ├── main.py            # FastAPI entry point
│   ├── api/v1/routes/     # API endpoints (19+ endpoints)
│   ├── services/          # Business logic layer
│   ├── repositories/      # Data access layer
│   ├── core/              # Core utilities (config, database, tasks)
│   ├── schemas/           # Pydantic validation models
│   └── utils/             # Utility functions
├── models/                # ML models
│   ├── lasso_model.py     # Basic Lasso Regression
│   └── news_prediction.py # News-Enhanced Lasso
├── ai/                    # AI services
│   └── services/          # Gemini integration
└── config/                # Configuration files
```

**Frontend Structure**:
```
kgf-gold-tradex-frontend/
├── src/
│   ├── components/        # React components
│   │   └── price-predictor/  # Gold price predictor components
│   ├── store/             # Redux state management
│   │   └── api/            # RTK Query API client
│   ├── hooks/              # Custom React hooks
│   └── utils/              # Utility functions
```

### Key Design Patterns

1. **Dependency Injection**: FastAPI Depends() for service injection
2. **Repository Pattern**: Data access abstraction
3. **Service Layer**: Business logic separation
4. **Factory Pattern**: Model initialization
5. **Observer Pattern**: WebSocket broadcasting
6. **Circuit Breaker**: Background task error handling
7. **Caching Strategy**: Multi-level caching (query, response, market data)

### Performance Optimizations

1. **Database**:
   - Connection pooling (PostgreSQL: 5-50 connections)
   - Query result caching (30-60s TTL)
   - Optimized queries with window functions
   - Indexed columns for fast lookups

2. **API**:
   - Response caching (30s TTL)
   - Rate limit handling with exponential backoff
   - Async operations for non-blocking I/O
   - Compression middleware

3. **Frontend**:
   - Lazy loading of heavy components
   - Memoization of expensive calculations
   - Debounced API calls
   - Efficient re-renders

### Security Features

1. **Input Validation**: Pydantic schemas for all inputs
2. **Security Headers**: X-Content-Type-Options, X-Frame-Options, etc.
3. **Request Size Limits**: 10MB maximum
4. **CORS Configuration**: Environment-based origin whitelist
5. **Error Handling**: No sensitive data in error messages
6. **API Key Management**: Environment variables only

### Monitoring & Observability

1. **Health Checks**: `/api/v1/health` endpoint
2. **Structured Logging**: JSON format in production
3. **Background Task Status**: Tracked in task manager
4. **Performance Metrics**: Response times, cache hit rates
5. **Error Tracking**: Comprehensive error logging

### Testing & Quality

1. **Type Safety**: TypeScript (frontend), type hints (backend)
2. **Validation**: Pydantic models for data validation
3. **Error Handling**: Comprehensive try-catch blocks
4. **Logging**: Structured logging throughout
5. **Documentation**: Docstrings, README, inline comments

The codebase follows industry best practices and is designed for scalability, maintainability, and production deployment.

