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
| `/xauusd`                     | GET    | Daily data + predictions     |
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

### Deploy to Render (Free Tier)

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

### Environment Variables (Optional)

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
NEWS_API_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here
```

### Free Tier Limitations

⚠️ **Note**: Render's free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

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

## 🙏 Acknowledgments

- Yahoo Finance for market data
- FastAPI community for excellent documentation
- scikit-learn contributors

## 📞 Support

- **Documentation**: Available at `/docs` when server is running
- **Issues**: Open an issue on GitHub
- **Email**: Punithachintha@gmail.com

## 📈 Status

![Status](https://img.shields.io/badge/status-active-success.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

---
