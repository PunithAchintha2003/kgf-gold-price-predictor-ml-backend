# KGF Gold Price Predictor - ML Backend

A production-ready FastAPI backend service for XAU/USD (Gold) price prediction using advanced machine learning models. Features real-time data streaming, ML predictions, and comprehensive market analysis.

## 🎯 Project Status

**✅ Backend Ready** - The FastAPI backend is fully functional and ready to serve data to any frontend application.

- **Backend**: FastAPI with WebSocket support and ML prediction engine ✅ **ACTIVE**
- **API Endpoints**: Complete REST API with real-time data endpoints ✅ **ACTIVE**
- **WebSocket**: Real-time data streaming every 2 seconds ✅ **ACTIVE**
- **ML Predictions**: Lasso Regression model with 99.17% accuracy ✅ **ACTIVE**
- **Database**: SQLite storage for predictions and historical data ✅ **ACTIVE**
- **Documentation**: Interactive API docs at /docs endpoint ✅ **ACTIVE**

**Current Status**: Backend server running on port 8001, ready for frontend integration.

## ⚡ Quick Start

```bash
# 1. Navigate to backend directory
cd backend

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Start the server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 4. Access the API
# - API: http://localhost:8001
# - Docs: http://localhost:8001/docs
# - WebSocket: ws://localhost:8001/ws/xauusd
```

## 🚀 Features

### Core ML & Prediction Engine

- **AI Price Prediction**: Next-day gold price predictions using Lasso Regression ML model
- **High Accuracy**: 99.17% prediction accuracy based on recent evaluations
- **Real-time Data**: Live XAU/USD price updates every 2 seconds via WebSocket
- **Market Analysis**: Comprehensive prediction explanations with 5 key market factors
- **Historical Tracking**: SQLite database storing all predictions and accuracy metrics

### API & Data Services

- **RESTful API**: Complete REST API with 6+ endpoints for all data needs
- **WebSocket Streaming**: Real-time data streaming with 2-second update frequency
- **Market Data**: Historical daily XAU/USD data with OHLCV information
- **Exchange Rates**: USD/LKR and other currency conversion support
- **Interactive Docs**: Auto-generated Swagger UI documentation

### Technical Features

- **FastAPI Backend**: High-performance async Python web framework
- **ML Pipeline**: Automated model training and prediction pipeline
- **Database Integration**: SQLite for prediction storage and historical tracking
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Ready for frontend integration from any domain
- **Production Ready**: Optimized for deployment and scaling

## 📋 Requirements

### System Requirements

- **Python**: 3.8 or higher
- **Node.js**: 18+ (for React frontend)
- **Internet Connection**: Required for fetching live gold prices
- **Operating System**: Windows, macOS, or Linux

### Python Dependencies

```txt
fastapi==0.104.1          # Web framework
uvicorn==0.24.0           # ASGI server
streamlit==1.28.1         # Legacy frontend (optional)
pandas==2.1.3             # Data processing
plotly==5.17.0            # Interactive charts
requests==2.31.0          # HTTP requests
yfinance==0.2.28          # Market data
websockets==12.0          # WebSocket support
asyncio-mqtt==0.16.1      # MQTT support
python-multipart==0.0.6   # File uploads
scikit-learn==1.3.2       # Machine learning
numpy==1.24.3             # Numerical computing
```

### React Frontend Dependencies

```json
{
  "react": "^19.1.1", // React framework
  "typescript": "~5.8.3", // Type safety
  "@mui/material": "^7.3.2", // Material UI components
  "tailwindcss": "^4.1.13", // Utility-first CSS
  "@reduxjs/toolkit": "^2.9.0", // State management
  "react-plotly.js": "^2.6.0", // Interactive charts
  "redux-persist": "^6.0.0", // State persistence
  "react-router-dom": "^7.9.2", // Client-side routing
  "vite": "^7.1.7" // Build tool
}
```

## 🛠️ Installation

### Quick Setup (Recommended)

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd kgf-gold-ai-price-prediction-frontend
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install React frontend dependencies**:

   ```bash
   cd react-frontend
   npm install
   cd ..
   ```

4. **Start the complete application**:
   ```bash
   python3 run_full_app.py
   ```

### Manual Setup

If you prefer to run components separately:

1. **Backend only**:

   ```bash
   pip install -r requirements.txt
   python3 run_backend.py
   ```

2. **React frontend only**:
   ```bash
   cd react-frontend
   npm install
   python3 ../run_react_frontend.py
   ```

### Development Setup

For development with hot reload:

1. **Start backend in development mode**:

   ```bash
   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start React development server** (in another terminal):
   ```bash
   cd react-frontend
   npm run dev
   ```

## 🏃‍♂️ Running the Application

### 🚀 Quick Start (Recommended)

**Start the Backend Server:**

```bash
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Access Points:**

- 🔧 **FastAPI Backend**: http://localhost:8001
- 📚 **API Documentation**: http://localhost:8001/docs
- 📡 **WebSocket**: ws://localhost:8001/ws/xauusd

### 🔧 Individual Components

#### Backend Only

```bash
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
# Available at: http://localhost:8001
```

#### React Frontend Only

```bash
python3 run_react_frontend.py
# Available at: http://localhost:5173
# Note: Requires backend running on port 8000
```

### 🛠️ Development Mode

#### Backend Development

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

#### React Frontend Development

```bash
cd react-frontend
npm run dev
```

#### Production Build

```bash
cd react-frontend
npm run build
npm run preview
```

### 📱 Available Interfaces

| Interface   | URL                           | Description                   | Status       |
| ----------- | ----------------------------- | ----------------------------- | ------------ |
| API Backend | http://localhost:8001         | FastAPI REST + WebSocket      | ✅ Active    |
| API Docs    | http://localhost:8001/docs    | Interactive API documentation | ✅ Available |
| WebSocket   | ws://localhost:8001/ws/xauusd | Real-time data streaming      | ✅ Available |

## 📊 API Endpoints

### REST API Endpoints

| Method | Endpoint                     | Description                                | Response                                                         |
| ------ | ---------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| `GET`  | `/`                          | Health check endpoint                      | `{"message": "XAU/USD Real-time Data API", "status": "running"}` |
| `GET`  | `/xauusd`                    | Daily XAU/USD data with AI prediction      | Historical data + prediction                                     |
| `GET`  | `/xauusd/realtime`           | Real-time current price (2s updates)       | Current price data                                               |
| `GET`  | `/xauusd/explanation`        | Prediction explanation with market factors | Detailed analysis                                                |
| `GET`  | `/exchange-rate/{from}/{to}` | Currency exchange rates                    | Exchange rate data                                               |
| `GET`  | `/docs`                      | Interactive API documentation              | Swagger UI                                                       |

### WebSocket Endpoints

| Endpoint     | Description                    | Update Frequency |
| ------------ | ------------------------------ | ---------------- |
| `/ws/xauusd` | Real-time daily data streaming | 2 seconds        |

### API Response Examples

#### Daily Data (`/xauusd`)

```json
{
  "symbol": "XAUUSD",
  "timeframe": "daily",
  "data": [
    {
      "date": "2025-10-03",
      "open": 3880.4,
      "high": 3916.8,
      "low": 3861.1,
      "close": 3912.1,
      "volume": 209911
    }
  ],
  "historical_predictions": [
    {
      "date": "2025-10-03",
      "predicted_price": 3872.73,
      "actual_price": 3866.8
    }
  ],
  "accuracy_stats": {
    "average_accuracy": 99.17,
    "total_predictions": 5,
    "evaluated_predictions": 4
  },
  "current_price": 3912.1,
  "prediction": {
    "next_day": "2025-10-04",
    "predicted_price": 3875.68,
    "current_price": 3912.1,
    "prediction_method": "Lasso Regression"
  },
  "timestamp": "2025-10-04T03:21:51.669527",
  "status": "success"
}
```

#### Real-time Price (`/xauusd/realtime`)

```json
{
  "symbol": "XAUUSD",
  "current_price": 3912.1,
  "timestamp": "2025-10-03T16:59:00-04:00",
  "status": "success"
}
```

#### Prediction Explanation (`/xauusd/explanation`)

```json
{
  "current_price": 3912.1,
  "overall_sentiment": "Bullish",
  "sentiment_explanation": "Most factors suggest gold prices may rise",
  "factors": [
    {
      "factor": "💵 US Dollar Strength",
      "value": "$104.25 (-0.8%)",
      "interpretation": "Weak dollar makes gold cheaper for international buyers",
      "impact": "Bullish",
      "confidence": "High"
    }
  ],
  "summary": {
    "bullish_factors": 3,
    "bearish_factors": 1,
    "neutral_factors": 1,
    "total_factors": 5
  },
  "timestamp": "2025-10-04T03:21:51.669527",
  "status": "success"
}
```

## 🧠 Machine Learning Model

### Lasso Regression Model

- **Algorithm**: Lasso Regression (L1 regularization)
- **Features**: 5 key market factors including USD strength, interest rates, VIX, technical indicators, and oil prices
- **Accuracy**: 99.17% based on recent predictions
- **Training**: Automated retraining with new market data
- **Prediction Window**: Next-day price predictions

### Market Analysis Factors

1. **💵 US Dollar Strength (DXY)**: Most important factor affecting gold prices
2. **📈 Interest Rates (Treasury Yields)**: Bond yields vs gold attractiveness
3. **😰 Market Fear (VIX)**: Volatility index indicating market sentiment
4. **📊 Gold Technicals (RSI)**: Technical analysis indicators
5. **🛢️ Inflation Pressure (Oil Prices)**: Commodity price impact on gold

## 🔧 Technical Details

### Backend Architecture

- **Framework**: FastAPI with async/await support for high performance
- **Data Source**: Yahoo Finance Gold Futures (GC=F) as XAU/USD proxy via `yfinance`
- **ML Engine**: Lasso Regression model with automated training pipeline
- **Database**: SQLite for prediction storage and historical tracking
- **WebSocket**: Real-time data streaming every 2 seconds
- **CORS**: Enabled for cross-origin requests from any frontend
- **Logging**: Comprehensive error handling and request logging
- **Auto-reload**: Development mode with automatic server restart on code changes

### Data Flow

1. **Market Data Collection**: Yahoo Finance API → FastAPI Backend
2. **ML Processing**: Historical data → Lasso Regression model → Predictions
3. **Database Storage**: Predictions → SQLite Database
4. **Real-time Updates**: WebSocket → Connected clients
5. **API Responses**: REST endpoints → Frontend applications

## 📊 Current Performance Metrics

### Real-time Data

- **Current Gold Price**: $3,912.10 (as of latest update)
- **Next Day Prediction**: $3,875.68
- **Prediction Method**: Lasso Regression
- **Update Frequency**: Every 2 seconds via WebSocket

### Model Performance

- **Average Accuracy**: 99.17%
- **Total Predictions**: 5 recent predictions
- **Evaluated Predictions**: 4 with actual results
- **Model Status**: Active and continuously learning

## 🧠 Machine Learning Approach

The prediction model uses Lasso Regression with 5 key market factors to predict next-day gold prices:

### Key Features Analyzed:

1. **💵 US Dollar Strength (DXY)**: Most influential factor - weak dollar supports gold prices
2. **📈 Interest Rates (Treasury Yields)**: Higher rates make bonds more attractive than gold
3. **😰 Market Fear (VIX)**: High volatility drives investors to safe haven assets like gold
4. **📊 Gold Technicals (RSI)**: Technical indicators showing overbought/oversold conditions
5. **🛢️ Inflation Pressure (Oil Prices)**: Rising oil prices increase inflation, supporting gold

### Prediction Logic:

- **Factor Weighting**: DXY and interest rates have highest impact on predictions
- **Sentiment Analysis**: Combines all factors to determine overall market sentiment
- **Confidence Levels**: Each factor includes confidence rating (High/Medium/Low)
- **Real-time Updates**: Model continuously updates with latest market data

## 🎯 Current Features

### API Endpoints

- **🔴 Real-time Price**: `/xauusd/realtime` - Live XAU/USD price every 2 seconds
- **📊 Daily Data**: `/xauusd` - Historical data with ML predictions
- **🧠 Prediction Analysis**: `/xauusd/explanation` - Detailed market factor analysis
- **💱 Exchange Rates**: `/exchange-rate/{from}/{to}` - Currency conversion
- **📚 Documentation**: `/docs` - Interactive Swagger UI
- **🔌 WebSocket**: `/ws/xauusd` - Real-time data streaming

### ML Prediction Engine

- **Next Day Prediction**: AI prediction for tomorrow's gold price
- **Model Method**: Lasso Regression with 99.17% accuracy
- **Market Analysis**: 5-factor sentiment analysis with confidence levels
- **Historical Tracking**: SQLite database storing all predictions and results
- **Auto-retraining**: Model updates with new market data

### Real-time Data

- **Live Price Updates**: Every 2 seconds via WebSocket
- **Market Data**: Yahoo Finance Gold Futures (GC=F) as XAU/USD proxy
- **Historical Data**: 30 days of OHLCV data for analysis
- **Prediction Storage**: All predictions saved with accuracy tracking

## 📁 Project Structure

```
KGF-gold-price-predictor-ml-backend/
├── backend/                    # Main backend application
│   ├── app/                   # FastAPI application
│   │   └── main.py           # Main application entry point
│   ├── models/               # Machine learning models
│   │   ├── ml_model.py       # ML prediction engine
│   │   └── gold_ml_model.pkl # Trained Lasso Regression model
│   ├── data/                 # Data storage
│   │   └── gold_predictions.db # SQLite database
│   ├── config/               # Configuration files
│   ├── utils/                # Utility functions
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Backend documentation
├── run_backend.py            # Backend startup script
└── README.md                 # This file
```

## 🚨 Important Notes

- **Data Source**: Yahoo Finance Gold Futures (GC=F) as XAU/USD proxy
- **Update Frequency**: Live price updates every 2 seconds via WebSocket
- **Data Retention**: 30 days of historical data for ML model training
- **Market Hours**: Data availability depends on market trading hours
- **Prediction Accuracy**: 99.17% based on recent evaluations
- **Educational Use**: Predictions are for educational purposes, not financial advice
- **Model Training**: Lasso Regression model with automated retraining

## 🔍 Troubleshooting

### 🚨 Common Issues & Solutions

#### Backend Issues

| Problem                         | Solution                        | Check                                                                               |
| ------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| **Backend not starting**        | Check if port 8001 is available | `lsof -i :8001`                                                                     |
| **API not responding**          | Ensure backend is running       | Visit http://localhost:8001                                                         |
| **No data updates**             | Check internet connection       | Test Yahoo Finance access                                                           |
| **WebSocket connection failed** | Verify WebSocket endpoint       | Check `/ws/xauusd` endpoint                                                         |
| **Module not found error**      | Run from backend directory      | `cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload` |

#### React Frontend Issues

| Problem                           | Solution                     | Check                              |
| --------------------------------- | ---------------------------- | ---------------------------------- |
| **Frontend not loading**          | Install dependencies         | `cd react-frontend && npm install` |
| **Chart not displaying**          | Verify backend connection    | Check API endpoints accessibility  |
| **Theme not persisting**          | Clear browser storage        | Clear localStorage in DevTools     |
| **Real-time updates not working** | Check connection status      | Look for WebSocket indicator       |
| **Build errors**                  | Check TypeScript compilation | `npm run build` for errors         |
| **Development server issues**     | Check console for errors     | `npm run dev` and check output     |

#### General Issues

| Problem                | Solution                 | Check                                 |
| ---------------------- | ------------------------ | ------------------------------------- |
| **Port conflicts**     | Use different ports      | Check 8000 (backend) and 5173 (React) |
| **CORS errors**        | Backend has CORS enabled | Verify frontend URL configuration     |
| **Data not updating**  | Check connection status  | Refresh page and check indicators     |
| **Performance issues** | Check browser console    | Monitor network tab for API calls     |

### 🔧 Debug Commands

#### Check System Status

```bash
# Check if ports are available
lsof -i :8001  # Backend port

# Check Python dependencies
pip list | grep -E "(fastapi|uvicorn|pandas|yfinance)"

# Check if backend directory exists
ls -la backend/
```

#### Backend Debug

```bash
# Test backend directly
curl http://localhost:8001/
curl http://localhost:8001/xauusd

# Start backend with proper command
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

#### React Frontend Debug

```bash
# Check React build
cd react-frontend
npm run build

# Check for TypeScript errors
npx tsc --noEmit

# Check for linting errors
npm run lint
```

### 🆘 Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Both backend and frontend provide detailed error messages
2. **Verify dependencies**: Ensure all Python and Node.js packages are installed
3. **Check network**: Verify internet connection for live data
4. **Browser compatibility**: Use modern browsers (Chrome, Firefox, Safari, Edge)
5. **Clear cache**: Clear browser cache and localStorage if experiencing UI issues

## 📈 Usage

1. Start the backend server using the correct command:

   ```bash
   cd backend
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

2. Access the API at http://localhost:8001
3. View API documentation at http://localhost:8001/docs
4. Connect to WebSocket at ws://localhost:8001/ws/xauusd for real-time updates
5. The API provides:
   - Real-time gold price data
   - ML predictions for next-day prices
   - Historical price data with OHLCV information
   - Market analysis and prediction explanations
   - Exchange rate information
6. Live price updates every 2 seconds via WebSocket
7. Historical predictions with accuracy tracking
8. ML model using Lasso Regression for price predictions

## 🛡️ Limitations

- **Data Source**: Uses free Yahoo Finance data (may have delays)
- **Proxy Data**: Limited to Gold Futures (GC=F) data as XAU/USD proxy
- **Prediction Scope**: Only provides next-day predictions, not intraday forecasts
- **Model Simplicity**: Uses Linear Regression; more complex models could improve accuracy
- **Market Dependency**: Predictions based on historical patterns may not account for unexpected market events
- **Educational Use**: Not intended for actual trading decisions

## 📈 Recent Updates & Project Status

### ✅ Latest Updates (Current Version)

- **React Frontend**: Modern TypeScript interface with Material UI and Tailwind CSS
- **Real-time Updates**: Live price updates every 2 seconds with synchronized chart display
- **State Management**: Redux Toolkit with RTK Query for efficient API calls and caching
- **Theme Support**: Light/Dark mode toggle with persistent user preferences
- **WebSocket Integration**: Real-time data streaming with automatic reconnection
- **Responsive Design**: Mobile-first design that works on all screen sizes
- **Production Ready**: Full build system with TypeScript compilation and optimization

### 🔄 Migration Status

| Component                | Status             | Notes                         |
| ------------------------ | ------------------ | ----------------------------- |
| **React Frontend**       | ✅ **Primary**     | Modern TypeScript interface   |
| **Streamlit Frontend**   | 🔄 **Alternative** | Legacy Python interface       |
| **FastAPI Backend**      | ✅ **Active**      | REST + WebSocket API          |
| **ML Prediction Engine** | ✅ **Active**      | Smart Money Concepts model    |
| **Database**             | ✅ **Active**      | SQLite for prediction storage |

### 🚀 Future Improvements

- **Advanced ML Models**: Implement LSTM, Random Forest, or XGBoost for better prediction accuracy
- **Multiple Timeframes**: Add hourly and weekly predictions alongside daily forecasts
- **Technical Indicators**: Integrate RSI, MACD, Bollinger Bands, and other technical analysis tools
- **Sentiment Analysis**: Incorporate news sentiment and social media analysis
- **Backtesting**: Add historical prediction accuracy testing and performance metrics
- **Risk Management**: Include volatility predictions and confidence intervals
- **Multi-Asset Support**: Extend to other precious metals (Silver, Platinum, Palladium)
- **Real-time Alerts**: Add price alert notifications
- **Mobile App**: Native mobile application for iOS and Android

## 🏗️ Project Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend Apps │    │  FastAPI Backend │    │  External APIs  │
│   (Any Client)  │◄──►│   (Python)      │◄──►│  (Yahoo Finance)│
│   Port: Various │    │   Port: 8001    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  SQLite Database│
         │              │  (Predictions)  │
         │              └─────────────────┘
         │
         ▼
┌─────────────────┐
│   WebSocket     │
│   Real-time     │
│   Updates       │
└─────────────────┘
```

### Technology Stack

#### Frontend (React)

- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite 7.1.7
- **UI Library**: Material UI 7.3.2
- **Styling**: Tailwind CSS 4.1.13
- **State Management**: Redux Toolkit 2.9.0
- **Charts**: Plotly.js 3.1.0
- **Routing**: React Router DOM 7.9.2

#### Backend (Python)

- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0
- **ML Library**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.3, numpy 1.24.3
- **Market Data**: yfinance 0.2.28
- **WebSocket**: websockets 12.0
- **Database**: SQLite (built-in)

#### Alternative Frontend (Streamlit)

- **Framework**: Streamlit 1.28.1
- **Charts**: Plotly 5.17.0
- **Data**: pandas, numpy

### Data Flow

1. **Data Collection**: Yahoo Finance API → FastAPI Backend
2. **ML Processing**: Historical data → SMC model → Predictions
3. **Storage**: Predictions → SQLite Database
4. **Real-time Updates**: WebSocket → React Frontend
5. **Visualization**: Plotly.js → Interactive Charts
6. **State Management**: Redux → Component Updates

### Key Components

#### Backend Components

- `backend.py` - Main FastAPI application
- `run_backend.py` - Backend startup script
- `gold_predictions.db` - SQLite database
- ML prediction engine with SMC analysis

#### React Frontend Components

- `src/components/` - React components
- `src/store/` - Redux store and API slices
- `src/hooks/` - Custom React hooks
- `src/theme/` - Material UI theme configuration

#### Scripts

- `run_full_app.py` - Start both backend and frontend
- `run_react_frontend.py` - Start React frontend only
- `run_backend.py` - Start backend only

### Security & Performance

- **CORS**: Enabled for cross-origin requests
- **Error Handling**: Comprehensive error handling and logging
- **Caching**: RTK Query automatic caching
- **Real-time**: WebSocket with fallback to REST API
- **Type Safety**: Full TypeScript integration
- **Production Ready**: Optimized build system

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements. Some areas where contributions would be particularly valuable:

- Improving the ML model accuracy
- Adding new SMC features
- Enhancing the UI/UX
- Adding backtesting capabilities
- Implementing additional technical indicators
- Improving real-time data sources

## ⚠️ Disclaimer

This application is for educational and research purposes only. The AI predictions should not be considered as financial advice or used for actual trading decisions. Gold price movements are influenced by numerous factors including economic indicators, geopolitical events, and market sentiment that may not be captured by historical data analysis. Always consult with qualified financial professionals before making investment decisions.

## 📄 License

This project is open source and available under the MIT License.
