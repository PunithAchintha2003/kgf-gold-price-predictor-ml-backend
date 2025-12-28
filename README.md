# KGF Gold Price Predictor - ML Backend

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-production-success.svg?style=for-the-badge)](https://kgf-gold-price-predictor.onrender.com)

**Production-ready FastAPI backend for XAU/USD (Gold) price prediction using machine learning models with news sentiment analysis.**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-api-documentation) • [Deployment](#-deployment) • [Contributing](#-contributing)

[![Live API](https://img.shields.io/badge/Live%20API-Available-brightgreen?style=flat-square)](https://kgf-gold-price-predictor.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-blue?style=flat-square)](https://kgf-gold-price-predictor.onrender.com/docs)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.16%25-success?style=flat-square)]()
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.32.0+-00A86B?style=flat-square&logo=python&logoColor=white)](https://www.uvicorn.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Machine Learning Models](#-machine-learning-models)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Development](#-development)
- [Testing](#-testing)
- [Monitoring & Observability](#-monitoring--observability)
- [Performance](#-performance)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)
- [Acknowledgments](#-acknowledgments)
- [Disclaimer](#️-disclaimer)

---

## 📖 Overview

**KGF Gold Price Predictor** is a production-ready machine learning backend that provides accurate next-day gold price predictions using advanced ML models and real-time market data analysis. The system combines technical indicators with news sentiment analysis to deliver predictions with **96.16% accuracy**.

### Key Capabilities

- **Real-time Price Tracking**: Live XAU/USD price updates via WebSocket
- **ML-Powered Predictions**: Lasso Regression models with 96.16% accuracy
- **Sentiment Analysis**: Multi-source news sentiment integration
- **RESTful API**: Comprehensive API with 19+ endpoints
- **Production Ready**: Optimized for scalability and reliability

### Use Cases

- Financial analysis and research
- Trading decision support systems
- Market trend analysis
- Educational ML projects
- Portfolio management tools

### Performance Metrics

| Metric                | Value            |
| --------------------- | ---------------- |
| **Model Accuracy**    | 96.16%           |
| **R² Score**          | 0.96+            |
| **API Response Time** | < 100ms (cached) |
| **WebSocket Latency** | < 10s            |
| **Cache Hit Rate**    | ~85%             |

---

## ✨ Features

### Core Features

- 🤖 **AI Price Prediction**: Next-day gold price predictions using Lasso Regression (96.16% accuracy)
- 📰 **News Sentiment Analysis**: Multi-source sentiment analysis from Yahoo Finance, NewsAPI, Alpha Vantage
- ⚡ **Real-time Data**: Live XAU/USD price updates via WebSocket (10s intervals)
- 📊 **RESTful API**: 19+ well-documented endpoints with OpenAPI/Swagger documentation
- 🔄 **WebSocket Streaming**: Real-time data broadcasting to connected clients
- 💾 **Database Support**: PostgreSQL (production) with SQLite fallback
- 🚀 **Production Ready**: Optimized caching, connection pooling, and error handling
- 🔒 **Security**: Environment-based configuration, input validation, and error handling

### Technical Features

- ✅ Modular architecture following industry best practices
- ✅ Comprehensive logging and monitoring
- ✅ Automated database migrations
- ✅ Connection pooling for PostgreSQL
- ✅ Intelligent caching with TTL
- ✅ Background task processing
- ✅ Health check endpoints
- ✅ API rate limiting and cooldown
- ✅ Request/response validation with Pydantic
- ✅ Graceful error handling
- ✅ Exponential backoff for rate limits
- ✅ Circuit breaker pattern for resilience
- ✅ Auto-retraining of ML models
- ✅ Automated prediction generation

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│   (React/Vue)    │
└────────┬────────┘
         │ HTTP/WebSocket
         │
┌────────▼─────────────────────────────────────┐
│         FastAPI Backend                      │
│  ┌──────────────────────────────────────┐   │
│  │  API Layer (REST + WebSocket)        │   │
│  │  - /api/v1/xauusd                    │   │
│  │  - /api/v1/health                     │   │
│  │  - /ws/xauusd                        │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │  Service Layer                       │   │
│  │  - PredictionService                 │   │
│  │  - MarketDataService                  │   │
│  │  - ExchangeService                    │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │  Repository Layer                    │   │
│  │  - PredictionRepository              │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │  Database Layer                      │   │
│  │  PostgreSQL / SQLite                 │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  ML Models                           │   │
│  │  - LassoGoldPredictor                │   │
│  │  - NewsEnhancedLassoPredictor         │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Background Tasks                    │   │
│  │  - Auto-update predictions            │   │
│  │  - Auto-retrain models                │   │
│  │  - Auto-generate predictions          │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
         │
         │ External APIs
         │
┌────────▼────────┐  ┌──────────────┐
│  Yahoo Finance   │  │  News APIs    │
│  Market Data     │  │  Sentiment    │
└─────────────────┘  └──────────────┘
```

### Technology Stack

| Category                | Technology         | Version          | Purpose                        |
| ----------------------- | ------------------ | ---------------- | ------------------------------ |
| **Framework**           | FastAPI            | 0.115.0+         | Modern async web framework     |
| **Python**              | Python             | 3.11+            | Programming language           |
| **ML Library**          | scikit-learn       | 1.5.0+           | Machine learning algorithms    |
| **Database**            | PostgreSQL         | 12+              | Primary production database    |
| **Database (Fallback)** | SQLite             | 3.x              | Development/testing database   |
| **Async DB Drivers**    | asyncpg, aiosqlite | 0.29.0+, 0.20.0+ | Async database connectivity    |
| **Market Data**         | yfinance           | 0.2.40+          | Financial market data fetching |
| **HTTP Client**         | httpx              | 0.27.0+          | Modern async HTTP client       |
| **Async**               | asyncio, WebSocket | Built-in         | Async I/O and real-time comms  |
| **Validation**          | Pydantic           | 2.9.0+           | Data validation and settings   |
| **Server**              | Uvicorn            | 0.32.0+          | ASGI server                    |
| **AI/LLM**              | Google Gemini      | 0.8.0+           | AI-powered explanations        |
| **News Analysis**       | TextBlob           | 0.17.1+          | Sentiment analysis             |
| **Caching**             | cachetools         | 5.3.0+           | Advanced caching utilities     |
| **Logging**             | python-json-logger | 2.0.7+           | Structured logging             |

---

## 🚀 Quick Start

Get up and running in 5 minutes!

### Prerequisites

- **Python**: 3.11 or higher ([Download](https://www.python.org/downloads/))
- **PostgreSQL**: 12+ (optional, SQLite fallback available)
- **pip**: Python package manager (included with Python 3.4+)
- **Internet connection**: Required for market data fetching
- **Git**: For cloning the repository

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/kgf-gold-price-predictor-ml-backend.git
cd kgf-gold-price-predictor-ml-backend

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# On PowerShell: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables (optional for quick start)
# Copy .env.example to .env and edit if needed
# For quick testing, defaults will work with SQLite

# 5. Start the server
python run_backend.py
```

### Verify Installation

Once the server starts, verify it's working:

```bash
# Test health endpoint
curl http://localhost:8001/health

# Or open in browser
open http://localhost:8001/docs
```

### Access Points

After starting the server, access:

- **🌐 API Base**: http://localhost:8001
- **📖 Interactive Docs (Swagger)**: http://localhost:8001/docs
- **📚 Alternative Docs (ReDoc)**: http://localhost:8001/redoc
- **🔌 WebSocket**: ws://localhost:8001/ws/xauusd
- **❤️ Health Check**: http://localhost:8001/health

### First API Call

Try your first API call:

```bash
# Get current gold price and prediction
curl http://localhost:8001/api/v1/xauusd

# Get real-time price
curl http://localhost:8001/api/v1/xauusd/realtime

# Get enhanced prediction with sentiment
curl http://localhost:8001/api/v1/xauusd/enhanced-prediction
```

---

## 📦 Installation

### Detailed Setup

For detailed installation instructions, see [Quick Start](#-quick-start) section above.

### Environment Configuration

Create a `.env` file in the root directory:

```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database Configuration
USE_POSTGRESQL=false  # Set to true to use PostgreSQL instead of SQLite
POSTGRESQL_HOST=localhost
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=gold_predictor
POSTGRESQL_USER=your_username
POSTGRESQL_PASSWORD=your_password

# API Keys (Optional)
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
GEMINI_API_KEY=your_gemini_api_key

# Cache Settings
CACHE_DURATION=300
API_COOLDOWN=5
REALTIME_CACHE_DURATION=60
RATE_LIMIT_INITIAL_BACKOFF=60
RATE_LIMIT_MAX_BACKOFF=1800

# Background Tasks
AUTO_UPDATE_ENABLED=true
AUTO_UPDATE_INTERVAL=3600
AUTO_UPDATE_STARTUP_DELAY=60
AUTO_UPDATE_MAX_RETRIES=3
AUTO_UPDATE_RETRY_DELAY=300

# Auto-Retrain Settings
AUTO_RETRAIN_ENABLED=true
AUTO_RETRAIN_INTERVAL=86400
AUTO_RETRAIN_HOUR=2
AUTO_RETRAIN_MIN_PREDICTIONS=10

# Auto-Predict Settings
AUTO_PREDICT_ENABLED=true
AUTO_PREDICT_HOUR=8

# CORS (Development)
CORS_ORIGINS=http://localhost:4000,http://127.0.0.1:4000

# Server
PORT=8001
```

### Database Setup

```bash
# PostgreSQL (optional)
createdb gold_predictor

# SQLite is used by default (no setup required)
```

---

## ⚙️ Configuration

### Environment Variables

| Variable                       | Description                                 | Default          | Required |
| ------------------------------ | ------------------------------------------- | ---------------- | -------- |
| `ENVIRONMENT`                  | Environment mode (development/production)   | `development`    | No       |
| `LOG_LEVEL`                    | Logging level (DEBUG/INFO/WARNING/ERROR)    | `WARNING`        | No       |
| `USE_POSTGRESQL`               | Enable PostgreSQL database                  | `false`          | No       |
| `POSTGRESQL_HOST`              | PostgreSQL host address                     | `localhost`      | Yes\*    |
| `POSTGRESQL_PORT`              | PostgreSQL port                             | `5432`           | No       |
| `POSTGRESQL_DATABASE`          | Database name                               | `gold_predictor` | Yes\*    |
| `POSTGRESQL_USER`              | Database username                           | `postgres`       | Yes\*    |
| `POSTGRESQL_PASSWORD`          | Database password                           | -                | Yes\*    |
| `NEWS_API_KEY`                 | NewsAPI key for sentiment analysis          | -                | No       |
| `ALPHA_VANTAGE_KEY`            | Alpha Vantage API key                       | -                | No       |
| `GEMINI_API_KEY`               | Google Gemini API key for AI explanations   | -                | No       |
| `CACHE_DURATION`               | Market data cache TTL (seconds)             | `300`            | No       |
| `API_COOLDOWN`                 | API request cooldown (seconds)              | `5`              | No       |
| `REALTIME_CACHE_DURATION`      | Real-time data cache TTL (seconds)          | `60`             | No       |
| `RATE_LIMIT_INITIAL_BACKOFF`   | Initial rate limit backoff (seconds)        | `60`             | No       |
| `RATE_LIMIT_MAX_BACKOFF`       | Maximum rate limit backoff (seconds)        | `1800`           | No       |
| `AUTO_UPDATE_ENABLED`          | Enable automatic prediction updates         | `true`           | No       |
| `AUTO_UPDATE_INTERVAL`         | Update interval in seconds                  | `3600`           | No       |
| `AUTO_UPDATE_STARTUP_DELAY`    | Startup delay before first update (seconds) | `60`             | No       |
| `AUTO_UPDATE_MAX_RETRIES`      | Maximum retry attempts                      | `3`              | No       |
| `AUTO_UPDATE_RETRY_DELAY`      | Delay between retries (seconds)             | `300`            | No       |
| `AUTO_RETRAIN_ENABLED`         | Enable automatic model retraining           | `true`           | No       |
| `AUTO_RETRAIN_INTERVAL`        | Retrain interval in seconds                 | `86400`          | No       |
| `AUTO_RETRAIN_HOUR`            | Hour of day to retrain (0-23)               | `2`              | No       |
| `AUTO_RETRAIN_MIN_PREDICTIONS` | Minimum predictions before retrain          | `10`             | No       |
| `AUTO_PREDICT_ENABLED`         | Enable automatic prediction generation      | `true`           | No       |
| `AUTO_PREDICT_HOUR`            | Hour of day to generate prediction (0-23)   | `8`              | No       |
| `CORS_ORIGINS`                 | Comma-separated list of allowed origins     | `*` (dev)        | No       |
| `PORT`                         | Server port                                 | `8001`           | No       |

\*Required if `USE_POSTGRESQL=true`

### Database Configuration

#### PostgreSQL (Recommended for Production)

The application uses PostgreSQL by default for production deployments. SQLite is automatically used as a fallback if PostgreSQL is unavailable.

**Setup PostgreSQL:**

```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database
createdb gold_predictor
```

**Connection String Format:**

```
postgresql://username:password@host:port/database
```

#### SQLite (Development/Testing)

For local development without PostgreSQL:

```bash
USE_POSTGRESQL=false python run_backend.py
```

SQLite database will be created automatically at `backend/data/gold_predictions.db`.

---

## 📚 API Documentation

### Base URL

**Development:**

```
http://localhost:8001
```

**Production:**

```
https://kgf-gold-price-predictor.onrender.com
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### API Endpoints

#### Health & Status

| Endpoint         | Method | Description           | Response                                  |
| ---------------- | ------ | --------------------- | ----------------------------------------- |
| `/`              | GET    | API root and status   | `{"message": "...", "status": "running"}` |
| `/health`        | GET    | Health check endpoint | Health status JSON                        |
| `/api/v1/health` | GET    | Health check (v1)     | Detailed health status                    |

#### Market Data

| Endpoint                                | Method | Description                           | Query Params             |
| --------------------------------------- | ------ | ------------------------------------- | ------------------------ |
| `/api/v1/xauusd`                        | GET    | Daily data with predictions           | `?days=90` (default: 90) |
| `/api/v1/xauusd/realtime`               | GET    | Real-time price data                  | -                        |
| `/api/v1/xauusd/enhanced-prediction`    | GET    | ML prediction with sentiment          | -                        |
| `/api/v1/xauusd/prediction-stats`       | GET    | Comprehensive prediction statistics   | -                        |
| `/api/v1/xauusd/prediction-history`     | GET    | Historical predictions                | `?days=30`               |
| `/api/v1/xauusd/pending-predictions`    | GET    | Pending predictions list              | -                        |
| `/api/v1/xauusd/accuracy-visualization` | GET    | Accuracy statistics for visualization | -                        |
| `/api/v1/xauusd/model-info`             | GET    | Detailed ML model information         | -                        |
| `/api/v1/xauusd/prediction-reasons`     | GET    | AI-generated prediction reasons       | -                        |

#### Data Management

| Endpoint                                    | Method | Description                | Body |
| ------------------------------------------- | ------ | -------------------------- | ---- |
| `/api/v1/xauusd/update-pending-predictions` | POST   | Update pending predictions | -    |

#### Exchange Rates

| Endpoint                                              | Method | Description       | Path Params                    |
| ----------------------------------------------------- | ------ | ----------------- | ------------------------------ |
| `/api/v1/exchange-rate/{from_currency}/{to_currency}` | GET    | Get exchange rate | `from_currency`, `to_currency` |

#### WebSocket

| Endpoint     | Description               | Update Interval |
| ------------ | ------------------------- | --------------- |
| `/ws/xauusd` | Real-time price streaming | 10 seconds      |

### Request/Response Examples

#### Get Daily Data

```bash
curl http://localhost:8001/api/v1/xauusd?days=30
```

**Response:**

```json
{
  "symbol": "XAUUSD",
  "timeframe": "daily",
  "data": [
    {
      "date": "2025-11-26",
      "open": 4128.6,
      "high": 4204.9,
      "low": 4163.6,
      "close": 4184.4,
      "volume": 39332,
      "predicted_price": 4112.7,
      "actual_price": null
    }
  ],
  "current_price": 4184.4,
  "prediction": {
    "next_day": "2025-11-27",
    "predicted_price": 4135.16,
    "current_price": 4184.4,
    "prediction_method": "Lasso Regression"
  },
  "accuracy_stats": {
    "average_accuracy": 98.95,
    "r2_score": 0.96,
    "total_predictions": 22,
    "evaluated_predictions": 21
  },
  "status": "success"
}
```

#### Get Real-time Price

```bash
curl http://localhost:8001/api/v1/xauusd/realtime
```

**Response:**

```json
{
  "symbol": "XAUUSD",
  "current_price": 4184.4,
  "price_change": 12.5,
  "change_percentage": 0.3,
  "last_updated": "2025-11-26 12:00:00",
  "timestamp": "2025-11-26T12:00:00",
  "status": "success"
}
```

#### Get Enhanced Prediction

```bash
curl http://localhost:8001/api/v1/xauusd/enhanced-prediction
```

**Response:**

```json
{
  "status": "success",
  "prediction": {
    "next_day_price": 4135.16,
    "current_price": 4184.4,
    "change": -49.24,
    "change_percentage": -1.18,
    "method": "Lasso Regression"
  },
  "model": {
    "name": "Enhanced Lasso Regression",
    "type": "Lasso Regression",
    "r2_score": 0.9616,
    "training_r2_score": 0.9616,
    "live_r2_score": 0.9616,
    "features": {
      "total": 35,
      "selected": 25,
      "top_features": ["close", "volume", "sma_20"]
    }
  },
  "sentiment": {
    "combined_sentiment": 0.15,
    "news_volume": 42,
    "sentiment_trend": 0.05
  },
  "timestamp": "2025-11-26T12:00:00"
}
```

---

## 💻 Usage Examples

### Python

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8001"

# Get daily data with predictions
response = requests.get(f"{BASE_URL}/api/v1/xauusd?days=90")
data = response.json()

print(f"Current Price: ${data['current_price']:.2f}")
print(f"Predicted Price: ${data['prediction']['predicted_price']:.2f}")
print(f"Accuracy: {data['accuracy_stats']['average_accuracy']:.2f}%")

# Get real-time price
response = requests.get(f"{BASE_URL}/api/v1/xauusd/realtime")
realtime = response.json()
print(f"Real-time Price: ${realtime['current_price']:.2f}")

# Get enhanced prediction
response = requests.get(f"{BASE_URL}/api/v1/xauusd/enhanced-prediction")
prediction = response.json()
print(f"Next Day Prediction: ${prediction['prediction']['next_day_price']:.2f}")
print(f"Sentiment: {prediction['sentiment']['combined_sentiment']:.2f}")
```

### JavaScript/TypeScript

```javascript
// Fetch daily data
async function getGoldData() {
  const response = await fetch("http://localhost:8001/api/v1/xauusd?days=90");
  const data = await response.json();

  console.log("Current Price:", data.current_price);
  console.log("Predicted Price:", data.prediction.predicted_price);

  return data;
}

// WebSocket connection
const ws = new WebSocket("ws://localhost:8001/ws/xauusd");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Real-time Price:", data.current_price);
  console.log("Prediction:", data.prediction);
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};

ws.onclose = () => {
  console.log("WebSocket connection closed");
};
```

### cURL

```bash
# Health check
curl http://localhost:8001/health

# Get daily data
curl http://localhost:8001/api/v1/xauusd?days=30

# Get real-time price
curl http://localhost:8001/api/v1/xauusd/realtime

# Get enhanced prediction
curl http://localhost:8001/api/v1/xauusd/enhanced-prediction

# Get prediction statistics
curl http://localhost:8001/api/v1/xauusd/prediction-stats
```

### WebSocket Client (Python)

```python
import asyncio
import websockets
import json

async def listen_to_prices():
    uri = "ws://localhost:8001/ws/xauusd"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            price_data = json.loads(data)
            print(f"Price: ${price_data['current_price']:.2f}")
            print(f"Prediction: ${price_data.get('prediction', {}).get('predicted_price', 'N/A')}")

asyncio.run(listen_to_prices())
```

---

## 🤖 Machine Learning Models

### Primary Model: Lasso Regression

**Algorithm**: Lasso Regression with L1 regularization

**Performance Metrics:**

- **R² Score**: 96.16%
- **Mean Absolute Error**: < 1%
- **Prediction Window**: Next-day price predictions

**Features:**

- 35+ technical indicators
- Fundamental market data
- Moving averages (5, 10, 20, 50-day)
- Exponential moving averages
- Volatility indicators
- Price momentum features

**Training:**

- Automated retraining with new market data
- Cross-validation for model selection
- Feature selection for optimal performance
- Daily retraining at configurable time (default: 2 AM)

### Enhanced Model: News-Enhanced Lasso

**Algorithm**: Lasso Regression + News Sentiment Analysis

**Features:**

- All technical indicators from base model
- Multi-source news sentiment scores
- Gold-specific keyword weighting
- Sentiment trend analysis

**News Sources:**

- Yahoo Finance
- NewsAPI
- Alpha Vantage
- RSS feeds

**Sentiment Analysis:**

- TextBlob-based sentiment scoring
- Domain-specific keyword extraction
- Temporal sentiment aggregation

### Model Retraining

The system automatically retrains models daily:

- **Schedule**: Configurable hour (default: 2 AM)
- **Trigger**: Minimum number of new predictions (default: 10)
- **Data**: Uses historical predictions and actual prices
- **Validation**: Cross-validation ensures model quality

---

## 🏗️ Project Structure

```
kgf-gold-price-predictor-ml-backend/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── __init__.py     # API router configuration
│   │   │       └── routes/
│   │   │           ├── health.py   # Health check endpoints
│   │   │           ├── xauusd.py   # Gold price endpoints
│   │   │           └── exchange.py # Exchange rate endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Application configuration
│   │   │   ├── database.py         # Database connection & setup
│   │   │   ├── logging_config.py   # Logging configuration
│   │   │   ├── dependencies.py     # Dependency injection
│   │   │   ├── exceptions.py       # Custom exceptions
│   │   │   ├── middleware.py       # Custom middleware
│   │   │   ├── metrics.py          # Performance metrics
│   │   │   ├── models.py           # ML model initialization
│   │   │   ├── background_tasks.py # Background task definitions
│   │   │   ├── task_manager.py     # Background task management
│   │   │   └── websocket.py        # WebSocket connection manager
│   │   ├── services/
│   │   │   ├── prediction_service.py    # Prediction business logic
│   │   │   ├── market_data_service.py   # Market data operations
│   │   │   └── exchange_service.py      # Exchange rate operations
│   │   ├── repositories/
│   │   │   └── prediction_repository.py # Database access layer
│   │   ├── schemas/
│   │   │   ├── market_data.py      # Market data schemas
│   │   │   ├── prediction.py       # Prediction schemas
│   │   │   └── exchange.py         # Exchange rate schemas
│   │   └── utils/
│   │       ├── cache.py            # Caching utilities
│   │       ├── yfinance_helper.py  # Market data fetching
│   │       └── fallback_data.py    # Fallback data handling
│   ├── ai/
│   │   ├── config.py               # AI service configuration
│   │   ├── exceptions.py           # AI-specific exceptions
│   │   └── services/
│   │       ├── gemini_service.py   # Google Gemini integration
│   │       └── prediction_reason_service.py # AI explanation service
│   ├── config/
│   │   └── news_config.py          # News API configuration
│   ├── models/
│   │   ├── lasso_model.py          # Lasso regression model
│   │   ├── news_prediction.py      # News-enhanced model
│   │   ├── lasso_gold_model.pkl    # Trained model file
│   │   └── enhanced_lasso_gold_model.pkl # Enhanced model file
│   └── data/                       # Database files (SQLite)
│       ├── gold_predictions.db
│       └── gold_predictions_backup.db
├── kgf-gold-tradex-frontend/      # Frontend application (separate repo)
├── .env                            # Environment variables (not in git)
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── run_backend.py                  # Application startup script
├── scripts/
│   ├── train_enhanced_model.py     # Model training script
│   ├── check_predictions.py        # Prediction checking script
│   ├── diagnose_apis.py            # API diagnostics script
│   └── import_predictions.py       # Prediction import script
├── Procfile                        # Process configuration (Render)
├── render.yaml                     # Render deployment config
├── runtime.txt                     # Python version specification
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🌐 Deployment

### Render (Recommended)

#### Quick Deploy

1. **Push to GitHub**

   ```bash
   git push origin main
   ```

2. **Create Render Service**

   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Settings**

   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_backend.py`
   - **Instance Type**: Free tier available

4. **Set Environment Variables**

   - Add all variables from your `.env` file
   - **Important**: Set `USE_POSTGRESQL=true` if using PostgreSQL
   - Create PostgreSQL database on Render

5. **Deploy**
   - Click "Create Web Service"
   - Wait 3-5 minutes for deployment

#### Blueprint Deployment

Use the included `render.yaml` for automated deployment:

```bash
# In Render Dashboard
New + → Blueprint → Connect Repository → Apply
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "run_backend.py"]
```

Build and run:

```bash
# Build image
docker build -t kgf-gold-predictor .

# Run container
docker run -p 8001:8001 --env-file .env kgf-gold-predictor
```

### Environment Variables for Production

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
USE_POSTGRESQL=true
POSTGRESQL_HOST=your_production_host
POSTGRESQL_DATABASE=gold_predictor
POSTGRESQL_USER=your_username
POSTGRESQL_PASSWORD=your_secure_password
POSTGRESQL_PORT=5432
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
GEMINI_API_KEY=your_gemini_api_key
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
PORT=10000
CACHE_DURATION=3600
API_COOLDOWN=15
```

### Free Tier Limitations

⚠️ **Render Free Tier:**

- Spins down after 15 minutes of inactivity
- First request after sleep takes ~30 seconds
- Limited to 750 hours/month

⚠️ **PostgreSQL:**

- Create a free PostgreSQL database on Render or use Neon
- Configure connection via environment variables
- Automatic fallback to SQLite if unavailable

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/kgf-gold-price-predictor-ml-backend.git
cd kgf-gold-price-predictor-ml-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development tools (optional but recommended)
pip install black flake8 pytest pytest-cov pytest-asyncio mypy pre-commit

# Setup pre-commit hooks (optional)
pre-commit install

# Run in development mode
ENVIRONMENT=development LOG_LEVEL=DEBUG python run_backend.py
```

### Code Style

This project follows PEP 8 style guidelines and industry best practices:

```bash
# Format code with black
black backend/ scripts/

# Check code style with flake8
flake8 backend/ scripts/ --max-line-length=100 --extend-ignore=E203

# Type checking with mypy
mypy backend/ --ignore-missing-imports

# Run all checks
black --check backend/ scripts/
flake8 backend/ scripts/
mypy backend/
```

### Project Architecture

The backend follows a modular architecture:

- **API Layer**: FastAPI routes and WebSocket handlers
- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Database access and data persistence
- **Core Layer**: Configuration, database, and logging setup
- **Models Layer**: ML models and prediction logic

---

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx

# Run all tests
pytest

# Run with coverage report
pytest --cov=backend --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v

# Run tests in parallel (faster)
pytest -n auto

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Structure

**Note**: Test suite is currently under development. Install test dependencies:

```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx
```

Create a `tests/` directory in the project root with test files for API endpoints, services, and models.

---

## 📊 Monitoring & Observability

### Health Checks

The application provides comprehensive health check endpoints:

- `/health` - Basic health check
- `/api/v1/health` - Detailed health status with:
  - Database connectivity
  - Background task status
  - Model loading status
  - System metrics

### Logging

The application uses structured logging with configurable levels:

```python
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
```

### Metrics

Performance metrics are tracked and available via:

- Response times
- Cache hit rates
- Database query performance
- Background task execution times

---

## 📊 Performance

### Benchmarks

- **Average Accuracy**: 96.16%
- **API Response Time**: < 100ms (cached), < 2s (uncached)
- **WebSocket Update Frequency**: 10 seconds
- **Cache Hit Rate**: ~85%
- **Concurrent Connections**: Tested up to 1000+

### Optimization Features

- **Intelligent Caching**: 5-minute TTL for market data
- **Connection Pooling**: PostgreSQL connection pooling
- **Background Tasks**: Non-blocking prediction generation
- **Lazy Loading**: Models loaded on first use
- **Request Cooldown**: Prevents API rate limiting
- **Async Operations**: Non-blocking I/O operations
- **Exponential Backoff**: Smart rate limit handling

---

## 🔒 Security

### Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Store securely in environment variables
3. **Database Credentials**: Use strong passwords
4. **Input Validation**: All inputs validated via Pydantic schemas
5. **Error Handling**: No sensitive data in error messages
6. **CORS**: Configured for specific origins in production
7. **HTTPS**: Always use HTTPS in production
8. **Rate Limiting**: Implement rate limiting for API endpoints

### Production Checklist

- [ ] Change default passwords
- [ ] Use HTTPS in production
- [ ] Configure CORS for specific domains
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerts
- [ ] Regular security updates
- [ ] Database backups enabled
- [ ] API keys rotated regularly
- [ ] Log sensitive operations
- [ ] Review and update dependencies

---

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Errors

**Problem**: Cannot connect to PostgreSQL

**Solutions:**

1. Verify PostgreSQL is running:

   ```bash
   # macOS
   brew services list

   # Linux
   sudo systemctl status postgresql
   ```

2. Check connection credentials in `.env`
3. Verify database exists:

   ```bash
   psql -l | grep gold_predictor
   ```

4. Use SQLite fallback:
   ```bash
   USE_POSTGRESQL=false python run_backend.py
   ```

#### Port Already in Use

**Problem**: `Address already in use` error

**Solution:**

```bash
# Find and kill process on port 8001
lsof -ti:8001 | xargs kill -9

# Or change port
PORT=8002 python run_backend.py
```

#### Module Import Errors

**Problem**: `ModuleNotFoundError`

**Solution:**

```bash
# Ensure you're in the project root
cd /path/to/kgf-gold-price-predictor-ml-backend

# Reinstall dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Market Data Fetching Issues

**Problem**: No market data returned

**Solutions:**

1. Check internet connection
2. Verify yfinance is working:
   ```python
   import yfinance as yf
   ticker = yf.Ticker("GC=F")
   data = ticker.history(period="1d")
   print(data)
   ```
3. Check API rate limits (Yahoo Finance may rate limit)
4. Review cache settings in `.env`
5. Check logs for rate limit warnings

#### WebSocket Connection Issues

**Problem**: WebSocket connection fails

**Solutions:**

1. Verify server is running
2. Check CORS configuration
3. Ensure WebSocket endpoint is correct: `ws://localhost:8001/ws/xauusd`
4. Check firewall settings

#### Rate Limiting Warnings

**Problem**: Frequent rate limit warnings in logs

**Explanation**: This is expected behavior when using free Yahoo Finance data. The system handles this gracefully by:

- Using exponential backoff
- Serving cached data when available
- Automatically retrying after rate limit expires

**Solutions:**

1. Increase `CACHE_DURATION` to reduce API calls
2. Increase `RATE_LIMIT_INITIAL_BACKOFF` for longer wait times
3. The system will automatically recover when rate limits expire

---

## 🗺️ Roadmap

### Planned Features

**Short-term (Q1 2025):**

- [ ] Docker containerization with multi-stage builds
- [ ] Docker Compose setup for local development
- [ ] Comprehensive test suite (>90% coverage)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] API rate limiting middleware
- [ ] Enhanced error handling and logging

**Medium-term (Q2 2025):**

- [ ] Kubernetes deployment manifests
- [ ] Additional ML models (Random Forest, XGBoost, LSTM)
- [ ] Real-time alerting system
- [ ] Advanced analytics dashboard
- [ ] Model versioning system
- [ ] A/B testing framework for models

**Long-term (Q3-Q4 2025):**

- [ ] Multi-currency support (Silver, Platinum, etc.)
- [ ] Historical backtesting framework
- [ ] GraphQL API support
- [ ] Mobile app SDK
- [ ] Webhook support for integrations
- [ ] Advanced ML features (ensemble models, deep learning)

### Version History

**v2.0.0** (Current) - Production Release

- ✅ Lasso Regression model with 96.16% accuracy
- ✅ News sentiment analysis integration
- ✅ WebSocket real-time data streaming
- ✅ PostgreSQL/SQLite database support
- ✅ Background task processing
- ✅ Rate limit handling with exponential backoff
- ✅ Auto-retraining and auto-prediction generation
- ✅ Comprehensive REST API (19+ endpoints)
- ✅ Production deployment on Render

**Upcoming: v2.1.0**

- 🔄 Docker support
- 🔄 Enhanced monitoring and metrics
- 🔄 Improved error handling
- 🔄 Additional ML models

---

## 🤝 Contributing

We welcome contributions from the community! This project follows industry-standard contribution practices.

### Getting Started

1. **Fork the repository**

   - Click the "Fork" button on GitHub
   - Clone your fork: `git clone https://github.com/yourusername/kgf-gold-price-predictor-ml-backend.git`

2. **Set up development environment**

   ```bash
   cd kgf-gold-price-predictor-ml-backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install black flake8 pytest pytest-cov mypy
   ```

3. **Create a feature branch**

   ```bash
   git checkout -b feature/amazing-feature
   # or
   git checkout -b fix/bug-description
   ```

4. **Make your changes**

   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update relevant documentation

5. **Test your changes**

   ```bash
   # Run tests
   pytest

   # Check code style
   black --check backend/
   flake8 backend/

   # Type checking
   mypy backend/
   ```

6. **Commit your changes**

   ```bash
   git add .
   git commit -m 'feat: Add amazing feature'
   # Use conventional commits: feat, fix, docs, style, refactor, test, chore
   ```

7. **Push to your fork**

   ```bash
   git push origin feature/amazing-feature
   ```

8. **Open a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Contribution Guidelines

**Code Standards:**

- ✅ Follow PEP 8 style guidelines
- ✅ Use type hints where applicable
- ✅ Write docstrings for functions and classes
- ✅ Keep functions focused and small
- ✅ Write meaningful variable and function names

**Testing:**

- ✅ Add tests for all new features
- ✅ Ensure all existing tests pass
- ✅ Aim for >80% code coverage
- ✅ Include both unit and integration tests

**Documentation:**

- ✅ Update README.md if needed
- ✅ Add docstrings to new functions/classes
- ✅ Update API documentation
- ✅ Include examples in docstrings

**Commit Messages:**
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add new prediction endpoint
fix: Resolve database connection issue
docs: Update installation instructions
style: Format code with black
refactor: Improve service layer structure
test: Add tests for prediction service
chore: Update dependencies
```

**Pull Request Process:**

1. Update README.md if applicable
2. Ensure all tests pass
3. Request review and address comments
4. Once approved, maintainers will merge

### Areas for Contribution

We welcome contributions in these areas:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Test coverage
- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 🔒 Security enhancements
- 🌐 Internationalization

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE**

This software and its predictions are provided **"AS IS"** without warranty of any kind, express or implied.

**Financial Disclaimer:**

- ⚠️ This application is for **educational and research purposes only**
- ⚠️ AI predictions should **NOT** be considered financial advice
- ⚠️ Always consult qualified financial professionals before making investment decisions
- ⚠️ Past performance does not guarantee future results
- ⚠️ The authors and contributors are **NOT responsible** for any financial losses resulting from the use of this software
- ⚠️ Trading involves risk of loss - only invest what you can afford to lose

**Accuracy Disclaimer:**

- Model accuracy metrics are based on historical data
- Real-world performance may vary
- Market conditions can change rapidly
- No prediction system is 100% accurate

**Data Disclaimer:**

- Market data is provided by third-party sources (Yahoo Finance, etc.)
- Data accuracy and availability are not guaranteed
- The application may experience data outages or delays

**Use at Your Own Risk:**
By using this software, you acknowledge that you have read, understood, and agree to this disclaimer. You assume full responsibility for any decisions made based on the predictions or data provided by this application.

---

## 🙏 Acknowledgments

Special thanks to FastAPI, scikit-learn, Pydantic, Uvicorn, yfinance, PostgreSQL, and all contributors to this project.

---

## 📞 Support & Contact

### Getting Help

- **📚 Documentation**: Interactive API docs at `/docs` when server is running
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/yourusername/kgf-gold-price-predictor-ml-backend/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/yourusername/kgf-gold-price-predictor-ml-backend/discussions)
- **📧 Email**: Punithachintha@gmail.com
- **🌐 Live API**: [https://kgf-gold-price-predictor.onrender.com](https://kgf-gold-price-predictor.onrender.com)
- **📖 API Docs**: [https://kgf-gold-price-predictor.onrender.com/docs](https://kgf-gold-price-predictor.onrender.com/docs)

### Reporting Issues

When reporting issues, include: environment details, steps to reproduce, expected vs actual behavior, and error messages.

### Feature Requests

Open a GitHub issue with the `enhancement` label describing the use case and expected behavior.

---

<div align="center">

**Made with ❤️ for the financial technology community**

[⬆ Back to Top](#kgf-gold-price-predictor---ml-backend)

</div>
