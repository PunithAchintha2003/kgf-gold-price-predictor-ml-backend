# KGF Gold Price Predictor - ML Backend

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Production-ready FastAPI backend for XAU/USD (Gold) price prediction using machine learning models with news sentiment analysis.

## ✨ Features

- 🤖 **AI Price Prediction**: Next-day gold price predictions using Lasso Regression (96.16% accuracy)
- 📰 **News Sentiment Analysis**: Multi-source sentiment analysis from Yahoo Finance, NewsAPI, Alpha Vantage
- ⚡ **Real-time Data**: Live XAU/USD price updates via WebSocket (10s intervals)
- 📊 **RESTful API**: 18+ well-documented endpoints
- 🔄 **WebSocket Streaming**: Real-time data broadcasting to connected clients
- 💾 **Automated Tracking**: SQLite database with automatic backups and accuracy tracking
- 🚀 **Production Ready**: Optimized caching, connection pooling, and error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager
- Internet connection (for market data)

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/kgf-gold-price-predictor-ml-backend.git
cd kgf-gold-price-predictor Victor-ml-backend
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the server

```bash
python run_backend.py
```

4. Access the API

- API: http://localhost:8001
- Interactive Docs: http://localhost:8001/docs
- WebSocket: ws://localhost:8001/ws/xauusd

## 📚 API Documentation

### Core Endpoints

| Endpoint                      | Method | Description                  |
| ----------------------------- | ------ | ---------------------------- |
| `/`                           | GET    | API status                   |
| `/health`                     | GET    | Health check                 |
| `/xauusd`                     | GET    | Daily data + predictions (supports `?days=90` for historical data) |
| `/xauusd/realtime`            | GET    | Real-time price data         |
| `/xauusd/news-sentiment`      | GET    | News sentiment analysis      |
| `/xauusd/enhanced-prediction` | GET    | ML prediction with sentiment |
| `/xauusd/compare-models`      | GET    | Model comparison             |

### Data Management

| Endpoint         | Method | Description         |
| ---------------- | ------ | ------------------- |
| `/backup`        | POST   | Create backup       |
| `/restore`       | POST   | Restore from backup |
| `/backup/status` | GET    | Backup status       |
| `/performance`   | GET    | Performance stats   |

### WebSocket

| Endpoint     | Description         | Interval |
| ------------ | ------------------- | -------- |
| `/ws/xauusd` | Real-time streaming | 10s      |

## 🗄️ Database: PostgreSQL (Default) or SQLite (Fallback)

The backend uses **PostgreSQL by default** for production deployments. SQLite is used as a fallback if PostgreSQL is unavailable or not configured.

**PostgreSQL is the default** - no configuration needed if PostgreSQL connection details are provided via environment variables.

To use SQLite instead (for local development), set `USE_POSTGRESQL=false`.

### PostgreSQL Configuration (Default)

**PostgreSQL is enabled by default.** Provide connection details via environment variables:

```bash
# Required for PostgreSQL (default)
POSTGRESQL_HOST=localhost          # or your PostgreSQL host
POSTGRESQL_DATABASE=gold_predictor
POSTGRESQL_USER=postgres
POSTGRESQL_PASSWORD=your_password
POSTGRESQL_PORT=5432

# Optional: Explicitly enable PostgreSQL (default is true)
USE_POSTGRESQL=true
```

**For Local Development:**
1. **Install PostgreSQL** (if not already installed):
   ```bash
   brew install postgresql@15  # macOS
   # or
   sudo apt-get install postgresql  # Linux
   ```

2. **Start PostgreSQL**:
   ```bash
   brew services start postgresql@15  # macOS
   # or
   sudo systemctl start postgresql  # Linux
   ```

3. **Create Database**:
   ```bash
   createdb gold_predictor
   ```

4. **Run with PostgreSQL** (default):
   ```bash
   POSTGRESQL_HOST=localhost POSTGRESQL_DATABASE=gold_predictor POSTGRESQL_USER=your_username POSTGRESQL_PASSWORD=your_password python3 run_backend.py
   ```

   Or use the provided script:
   ```bash
   ./run_with_postgresql.sh
   ```

### SQLite Fallback (Local Development)

To use SQLite instead of PostgreSQL (for local development without PostgreSQL):

```bash
USE_POSTGRESQL=false python3 run_backend.py
```

**Note:** If PostgreSQL connection fails, the system automatically falls back to SQLite.

## 📊 Frontend Integration: Accuracy Line

The API returns predicted prices for chart display. Use the `/xauusd?days=90` endpoint to get 90 days of historical data including predictions.

### Data Structure

```json
{
  "data": [
    {
      "date": "2025-10-06",
      "close": 3948.5,           // For gold price line
      "predicted_price": 3927.2, // For accuracy line
      "actual_price": 3935.0
    }
  ],
  "historical_predictions": [
    {
      "date": "2025-10-06",
      "predicted_price": 3927.2,
      "actual_price": 3935.0
    }
  ]
}
```

### JavaScript Example (Chart.js)

```javascript
const response = await fetch('http://localhost:8001/xauusd?days=90');
const apiData = await response.json();

// Extract data for chart
const labels = apiData.data.map(d => d.date);
const actualPrices = apiData.data.map(d => d.close);
// IMPORTANT: Use predicted_price from data array (already merged)
const predictedPrices = apiData.data.map(d => d.predicted_price || null);

new Chart(ctx, {
  type: 'line',
  data: {
    labels: labels,
    datasets: [
      {
        label: 'Actual Price (Gold Price Line)',
        data: actualPrices,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      {
        label: 'Predicted Price (Accuracy Line)',
        data: predictedPrices,  // Use predicted_price from data array
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],  // Dashed line for predictions
        tension: 0.1
      }
    ]
  },
  options: {
    scales: {
      y: {
        beginAtZero: false
      }
    }
  }
});
```

### Alternative: Using historical_predictions Array

```javascript
// If you prefer to use the separate historical_predictions array:
const response = await fetch('http://localhost:8001/xauusd?days=90');
const apiData = await response.json();

// Create a map of predictions by date
const predictionsMap = {};
apiData.historical_predictions.forEach(pred => {
  predictionsMap[pred.date] = pred.predicted_price;
});

// Map predictions to data points
const predictedPrices = apiData.data.map(d => 
  predictionsMap[d.date] || null
);
```

**Key Points:**
- ✅ All predicted prices come from PostgreSQL database
- ✅ Use `predicted_price` field from `data[]` array for accuracy line (already merged)
- ✅ Data includes 90 days of history (including dates before Oct 6)
- ✅ Gold price line uses `close` values
- ✅ Accuracy line uses `predicted_price` values
- ✅ Some dates may have `null` predicted_price (dates before predictions started) - handle with `|| null`

## 💻 Usage

### Python Example

```python
import requests

# Get daily data with predictions
response = requests.get('http://localhost:8001/xauusd')
data = response.json()
print(f"Current price: ${data['current_price']}")
print(f"Predicted price: ${data['prediction']['predicted_price']}")

# Get real-time price
response = requests.get('http://localhost:8001/xauusd/realtime')
realtime = response.json()
print(f"Real-time price: ${realtime['current_price']}")
```

### JavaScript WebSocket Example

```javascript
const ws = new WebSocket("ws://localhost:8001/ws/xauusd");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Current price:", data.current_price);
  console.log("Prediction:", data.prediction);
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

### cURL Examples

```bash
# Health check
curl http://localhost:8001/health

# Get gold price data
curl http://localhost:8001/xauusd

# Get real-time price
curl http://localhost:8001/xauusd/realtime

# Get enhanced prediction
curl http://localhost:8001/xauusd/enhanced-prediction
```

## 🤖 Machine Learning Models

### Primary Model: Lasso Regression

- **Algorithm**: Lasso Regression with L1 regularization
- **Accuracy**: 96.16% R² score
- **Features**: 35+ technical and fundamental indicators
- **Training**: Automated with new market data
- **Prediction Window**: Next-day price predictions

### Enhanced Model: News-Enhanced Lasso

- **Algorithm**: Lasso Regression + News Sentiment Analysis
- **Features**: Technical indicators + multi-source sentiment features
- **News Sources**: Yahoo Finance, NewsAPI, Alpha Vantage, RSS feeds
- **Sentiment Analysis**: TextBlob with gold-specific keyword weighting

## 🛠️ Technology Stack

| Category               | Technology         |
| ---------------------- | ------------------ |
| **Framework**          | FastAPI            |
| **ML Library**         | scikit-learn       |
| **Data Processing**    | pandas, numpy      |
| **Market Data**        | yfinance           |
| **Sentiment Analysis** | TextBlob           |
| **Database**           | SQLite (WAL mode)  |
| **WebSocket**          | FastAPI native     |
| **API Docs**           | Swagger UI / ReDoc |

## 🏗️ Project Structure

```
kgf-gold-price-predictor-ml-backend/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── config/                  # Configuration files
│   ├── data/                    # SQLite databases
│   └── models/                  # ML models
├── requirements.txt             # Dependencies
├── run_backend.py               # Startup script
├── render.yaml                  # Render deployment config
├── Procfile                     # Process configuration
└── README.md                    # This file
```

## 📦 Dependencies

Core dependencies required for the application:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
yfinance>=0.2.40
textblob==0.17.1
feedparser>=6.0.11
```

See [requirements.txt](requirements.txt) for complete list.

## 🌐 Deployment

### Database Deployment

The application uses **PostgreSQL by default** for production deployments. SQLite is available as a fallback for local development. 

📖 **See [DEPLOYMENT_SQLITE.md](DEPLOYMENT_SQLITE.md) for detailed SQLite deployment guide.**

### Quick Deploy to Render (Free Tier)

1. **Push to GitHub**

   ```bash
   git push origin main
   ```

2. **Go to [Render Dashboard](https://dashboard.render.com)**

   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure Settings**

   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_backend.py`
   - **Instance Type**: FREE ($0/month)

4. **Deploy**

   Wait 3-5 minutes for build to complete.

#### Alternative: Blueprint Deployment

- Click "New +" → "Blueprint"
- Connect repository
- Render auto-detects `render.yaml`
- Click "Apply"

**Note**: The `render.yaml` is configured to use PostgreSQL (`USE_POSTGRESQL=true`). Make sure to create a PostgreSQL database on Render and set the connection environment variables.

### Environment Variables (Optional)

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
USE_POSTGRESQL=true  # Use PostgreSQL (default)
POSTGRESQL_HOST=your_postgres_host
POSTGRESQL_DATABASE=gold_predictor
POSTGRESQL_USER=your_username
POSTGRESQL_PASSWORD=your_password
POSTGRESQL_PORT=5432
NEWS_API_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here
```

### Free Tier Limitations

⚠️ **Note**: Render's free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

⚠️ **PostgreSQL Required**: The application uses PostgreSQL by default. Create a free PostgreSQL database on Render and configure the connection environment variables. If PostgreSQL is unavailable, the system will automatically fall back to SQLite (ephemeral on free tier).

## 📊 Performance

- **Average Accuracy**: 96.16%
- **Update Frequency**: 10 seconds (WebSocket)
- **Cache Duration**: 5 minutes (market data), 1 minute (real-time)
- **Response Time**: <100ms (cached data)
- **Concurrent Connections**: Unlimited

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. AI predictions should not be considered financial advice. Always consult qualified financial professionals before making investment decisions.

## 👥 Authors

- **KGF Team** - _Initial work_

## 🙏 Acknowledgments

- Yahoo Finance for market data
- FastAPI community for excellent documentation
- scikit-learn contributors

## 📞 Support

- **Documentation**: Available at `/docs` when server is running
- **Issues**: Open an issue on GitHub
- **Email**: [Your Email]

## 📈 Status

![Status](https://img.shields.io/badge/status-active-success.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

Made with ❤️ by KGF Team
