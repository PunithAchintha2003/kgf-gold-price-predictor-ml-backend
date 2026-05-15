<a id="top"></a>

# 🥇 KGF Gold Price Predictor — ML Backend

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

**Production-ready FastAPI backend for XAU/USD (gold) price prediction, real-time market data, news sentiment, and spot trading.**

**🌐 Live (Render):** [https://kgf-gold-price-predictor.onrender.com](https://kgf-gold-price-predictor.onrender.com) · [Swagger /docs](https://kgf-gold-price-predictor.onrender.com/docs) · [ReDoc](https://kgf-gold-price-predictor.onrender.com/redoc)

[Features](#features) · [Quick Start](#quick-start) · [API](#api-documentation) · [Deployment](#deployment) · [Contributing](#contributing)

[![Live API](https://img.shields.io/badge/Live%20API-kgf--gold--price--predictor.onrender.com-brightgreen?style=flat-square)](https://kgf-gold-price-predictor.onrender.com)
[![Swagger](https://img.shields.io/badge/OpenAPI-Swagger-85EA2D?style=flat-square&logo=swagger&logoColor=white)](https://kgf-gold-price-predictor.onrender.com/docs)
[![ReDoc](https://img.shields.io/badge/Docs-ReDoc-8CA1AF?style=flat-square)](https://kgf-gold-price-predictor.onrender.com/redoc)
[![Model R²](https://img.shields.io/badge/Model%20R%C2%B2-0.96+-success?style=flat-square)]()
[![WebSocket](https://img.shields.io/badge/WebSocket-%2Fws%2Fxauusd-blue?style=flat-square)](https://kgf-gold-price-predictor.onrender.com/docs)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.32+-00A86B?style=flat-square&logo=uvicorn&logoColor=white)](https://www.uvicorn.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

---

<a id="table-of-contents"></a>

## 📋 Table of Contents

- [Live deployment (Render)](#live-deployment-render)
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Development](#development)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Support](#support)

---

<a id="live-deployment-render"></a>

## 🌐 Live deployment (Render)

| Service | URL |
| --- | --- |
| **API (production)** | https://kgf-gold-price-predictor.onrender.com |
| **Swagger UI** | https://kgf-gold-price-predictor.onrender.com/docs |
| **ReDoc** | https://kgf-gold-price-predictor.onrender.com/redoc |
| **Health check** | https://kgf-gold-price-predictor.onrender.com/api/v1/health |
| **WebSocket** | `wss://kgf-gold-price-predictor.onrender.com/ws/xauusd` |

```bash
curl https://kgf-gold-price-predictor.onrender.com/health
curl https://kgf-gold-price-predictor.onrender.com/api/v1/xauusd/realtime
```

---

<a id="overview"></a>

## 📖 Overview

**KGF Gold Price Predictor** is a FastAPI backend that serves next-day XAU/USD predictions, live market data, news sentiment, exchange rates, and a JWT-protected spot-trading wallet API. It is designed for production use on [Render](https://render.com) with PostgreSQL, with SQLite as a local fallback.

| Capability | Description |
| --- | --- |
| **Predictions** | Lasso regression with optional news-sentiment features |
| **Market data** | Yahoo Finance via `yfinance`, with caching and rate-limit backoff |
| **Real-time** | WebSocket stream at `/ws/xauusd` (~10s updates) |
| **Spot trading** | Buy/sell, deposits (Stripe), withdrawals, admin flows |
| **AI explanations** | Optional Google Gemini summaries for predictions |

> **Note:** Reported model metrics (e.g. ~96% R²) are based on historical training/evaluation data. Live market performance may differ. See [Disclaimer](#disclaimer).

---

<a id="features"></a>

## ✨ Features

### Core

- 🤖 **ML predictions** — Lasso regression for next-day XAU/USD prices
- 📰 **Sentiment** — Multi-source news analysis (Yahoo Finance, NewsAPI, Alpha Vantage, RSS)
- ⚡ **Real-time data** — REST + WebSocket with intelligent caching
- 📊 **REST API** — OpenAPI/Swagger at `/docs`, 30+ endpoints
- 💾 **Databases** — PostgreSQL (production) or SQLite (local dev)
- 🔄 **Background jobs** — Auto-update predictions, retrain, and daily forecast generation
- 💳 **Spot trading** — Wallet, Stripe checkout, JWT auth shared with the Node frontend
- 🔒 **Hardening** — Pydantic validation, security headers, request size limits, structured logging

### Technical

- Layered architecture: routes → services → repositories
- Async PostgreSQL pool (`asyncpg`) + sync fallback (`psycopg2` / `sqlite3`)
- Connection pooling, TTL caches, exponential backoff for external APIs
- Graceful shutdown of background tasks and WebSocket clients

---

<a id="architecture"></a>

## 🏗️ Architecture

```
┌──────────────┐     HTTP / WebSocket      ┌─────────────────────────────────────┐
│   Clients    │ ────────────────────────► │           FastAPI (v2.0)            │
│ (Web / App)  │                           │  /api/v1/xauusd  /api/v1/spot-trade   │
└──────────────┘                           │  /api/v1/health  /ws/xauusd           │
                                           └──────────┬────────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────┐
                    ▼                                 ▼                         ▼
           ┌────────────────┐              ┌─────────────────┐       ┌─────────────────┐
           │ Service layer  │              │  ML models      │       │  PostgreSQL /   │
           │ Prediction     │              │  Lasso + News   │       │  SQLite         │
           │ Market / Trade │              │  (.pkl)         │       └─────────────────┘
           └────────┬───────┘              └─────────────────┘
                    │
                    ▼
           ┌────────────────┐       ┌─────────────────┐
           │ Yahoo Finance  │       │ News + Gemini   │
           │ (yfinance)     │       │ APIs (optional) │
           └────────────────┘       └─────────────────┘
```

---

<a id="tech-stack"></a>

## 🧰 Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| API | FastAPI 0.115+, Uvicorn | Async HTTP + WebSocket |
| Language | Python 3.12 | See `runtime.txt` |
| ML | scikit-learn, pandas, numpy, joblib | Training & inference |
| Data | yfinance, httpx, TextBlob | Market & news data |
| Database | PostgreSQL 12+, SQLite, asyncpg, aiosqlite | Persistence |
| AI | google-generativeai (Gemini) | Prediction explanations |
| Payments | Stripe, PyJWT | Deposits & auth |
| Deploy | Render, GitHub Actions | Hosting & deploy hook |

---

<a id="quick-start"></a>

## 🚀 Quick Start

### Prerequisites

- Python **3.12** (matches `runtime.txt`)
- `pip` and `git`
- Internet access (market data)
- PostgreSQL 12+ (optional; SQLite works out of the box)

### Install & run

```bash
git clone https://github.com/PunithAchintha2003/kgf-gold-price-predictor-ml-backend.git
cd kgf-gold-price-predictor-ml-backend

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: create .env (see Configuration)
python run_backend.py
```

### Verify

```bash
curl http://localhost:8001/health
curl http://localhost:8001/api/v1/xauusd/realtime
```

### Local URLs

| Resource | URL |
| --- | --- |
| API root | http://localhost:8001 |
| Swagger UI | http://localhost:8001/docs |
| ReDoc | http://localhost:8001/redoc |
| Health | http://localhost:8001/api/v1/health |
| WebSocket | `ws://localhost:8001/ws/xauusd` |

Default port is **8001** (`PORT` env overrides; Render uses **10000**).

---

<a id="configuration"></a>

## ⚙️ Configuration

Create a `.env` file in the project root. Settings are loaded via **Pydantic Settings** (`backend/app/core/config.py`).

### Common variables

```bash
# Core
ENVIRONMENT=development          # development | staging | production
LOG_LEVEL=INFO                   # DEBUG | INFO | WARNING | ERROR | CRITICAL
PORT=8001

# Database (SQLite by default)
USE_POSTGRESQL=false
POSTGRESQL_HOST=localhost
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=gold_predictor
POSTGRESQL_USER=postgres
POSTGRESQL_PASSWORD=your_password

# External APIs (optional)
NEWS_API_KEY=
ALPHA_VANTAGE_KEY=
GEMINI_API_KEY=

# Auth & payments (required for spot-trade)
JWT_SECRET=change-me-min-32-chars
JWT_ALGORITHM=HS256
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
FRONTEND_BASE_URL=http://localhost:4000

# CORS (comma-separated; do not use * in production)
CORS_ORIGINS=http://localhost:4000,http://127.0.0.1:4000

# Caching & rate limits
CACHE_DURATION=300
API_COOLDOWN=5
REALTIME_CACHE_DURATION=60

# Background tasks
AUTO_UPDATE_ENABLED=true
AUTO_RETRAIN_ENABLED=true
AUTO_PREDICT_ENABLED=true
AUTO_RETRAIN_HOUR=2
AUTO_PREDICT_HOUR=8
```

### Environment reference

| Variable | Default | Description |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Runtime mode |
| `LOG_LEVEL` | `WARNING` | Log verbosity |
| `USE_POSTGRESQL` | `false` | Use PostgreSQL instead of SQLite |
| `POSTGRESQL_*` | see config | Required when `USE_POSTGRESQL=true` |
| `GEMINI_API_KEY` | — | AI prediction reasons |
| `NEWS_API_KEY` | — | NewsAPI sentiment |
| `ALPHA_VANTAGE_KEY` | — | Alpha Vantage news |
| `JWT_SECRET` | (dev default) | Must match Node backend |
| `STRIPE_SECRET_KEY` | — | Stripe API key |
| `STRIPE_WEBHOOK_SECRET` | — | Webhook signature verification |
| `CORS_ORIGINS` | `*` (dev) | Allowed browser origins |
| `CACHE_DURATION` | `300` | Market data cache TTL (seconds) |
| `AUTO_UPDATE_INTERVAL` | `3600` | Pending-prediction update interval |

SQLite files are created automatically under `backend/data/`.

---

<a id="api-documentation"></a>

## 📚 API Documentation

### Base URLs

| Environment | URL |
| --- | --- |
| Local | `http://localhost:8001` |
| Production | `https://kgf-gold-price-predictor.onrender.com` |

Interactive docs: **[/docs](https://kgf-gold-price-predictor.onrender.com/docs)** · **[/redoc](https://kgf-gold-price-predictor.onrender.com/redoc)**

All versioned routes are prefixed with **`/api/v1`**.

### Health & root

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API metadata |
| `GET` | `/health` | Legacy health check |
| `GET` | `/api/v1/health` | Health + background task status |

### XAU/USD — market & predictions

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/xauusd` | Daily OHLCV + predictions (`?days=90`) |
| `GET` | `/api/v1/xauusd/realtime` | Latest spot price |
| `GET` | `/api/v1/xauusd/enhanced-prediction` | Prediction + sentiment + model info |
| `GET` | `/api/v1/xauusd/prediction-stats` | Aggregate accuracy stats |
| `GET` | `/api/v1/xauusd/prediction-history` | Historical predictions (`?days=30`) |
| `GET` | `/api/v1/xauusd/pending-predictions` | Unevaluated predictions |
| `POST` | `/api/v1/xauusd/update-pending-predictions` | Resolve pending rows |
| `GET` | `/api/v1/xauusd/accuracy-visualization` | Chart-friendly accuracy data |
| `GET` | `/api/v1/xauusd/model-info` | Active model metadata |
| `GET` | `/api/v1/xauusd/prediction-reasons` | Gemini-generated explanations |

### Exchange rates

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/exchange-rate/{from}/{to}` | FX conversion |

### Spot trading (JWT required)

Prefix: **`/api/v1/spot-trade`** · Header: `Authorization: Bearer <token>`

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/price` | — | Gold price in LKR with spread |
| `POST` | `/buy` | User | Market buy |
| `POST` | `/sell` | User | Market sell |
| `GET` | `/balance` | User | Wallet balance |
| `POST` | `/deposit` | User | Stripe checkout session |
| `POST` | `/deposit/confirm` | User | Confirm deposit |
| `POST` | `/stripe/webhook` | Stripe | Payment webhooks |
| `POST` | `/withdraw` | User | Withdrawal request |
| `GET` | `/history` | User | Trade history |
| `GET` | `/orders` | User | Open orders |
| `GET` | `/wallet-transactions` | User | Wallet ledger |
| `GET` | `/admin/*` | Admin | Admin wallet & withdrawal tools |

### WebSocket

| Endpoint | Description |
| --- | --- |
| `WS /ws/xauusd` | Streams daily market payload (~10s; respects rate limits) |

### Example response

```bash
curl "http://localhost:8001/api/v1/xauusd?days=7"
```

```json
{
  "symbol": "XAUUSD",
  "current_price": 4184.4,
  "prediction": {
    "next_day": "2025-11-27",
    "predicted_price": 4135.16,
    "prediction_method": "Lasso Regression"
  },
  "accuracy_stats": {
    "average_accuracy": 98.95,
    "r2_score": 0.96
  },
  "status": "success"
}
```

---

<a id="usage-examples"></a>

## 💻 Usage Examples

### cURL

```bash
curl http://localhost:8001/api/v1/xauusd/realtime
curl http://localhost:8001/api/v1/xauusd/enhanced-prediction
curl -H "Authorization: Bearer $TOKEN" http://localhost:8001/api/v1/spot-trade/balance
```

### Python

```python
import requests

BASE = "http://localhost:8001"
data = requests.get(f"{BASE}/api/v1/xauusd", params={"days": 30}, timeout=30).json()
print(data["current_price"], data["prediction"]["predicted_price"])
```

### WebSocket (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8001/ws/xauusd");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

<a id="machine-learning-models"></a>

## 🤖 Machine Learning Models

### Lasso regression (primary)

- **Algorithm:** Lasso (L1) regression via scikit-learn
- **Artifact:** `backend/models/lasso_gold_model.pkl`
- **Features:** Technical indicators (moving averages, momentum, volatility, volume, etc.)
- **Output:** Next-day XAU/USD price estimate

### News-enhanced Lasso

- **Module:** `backend/models/news_prediction.py`
- **Adds:** Aggregated sentiment from configured news sources
- **Training script:** `scripts/train_enhanced_model.py`

### Automation

| Task | Default schedule | Setting |
| --- | --- | --- |
| Update pending predictions | Every 1h | `AUTO_UPDATE_*` |
| Retrain + predict | Daily (retrain then predict) | `AUTO_RETRAIN_HOUR`, `AUTO_PREDICT_HOUR` |
| WebSocket broadcast | Continuous | Background task |

Retraining uses historical predictions vs actuals once `AUTO_RETRAIN_MIN_PREDICTIONS` (default 10) are available.

---

<a id="project-structure"></a>

## 📁 Project Structure

```
kgf-gold-price-predictor-ml-backend/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app & lifespan
│   │   ├── api/v1/routes/          # health, xauusd, exchange
│   │   ├── core/                   # config, db, tasks, websocket
│   │   ├── services/               # prediction, market, exchange
│   │   ├── repositories/           # prediction persistence
│   │   ├── schemas/                # Pydantic models
│   │   └── utils/                  # cache, yfinance, validation
│   ├── ai/                         # Gemini & prediction reasons
│   ├── config/                     # news API config
│   ├── models/                     # ML code + .pkl weights
│   ├── spot_trade/                 # trading routes, models, migrations
│   └── data/                       # SQLite databases (gitignored)
├── scripts/                        # train, diagnose, import utilities
├── .github/workflows/deploy.yml    # Render deploy hook on push
├── requirements.txt
├── run_backend.py                  # Local & production entrypoint
├── render.yaml                     # Render blueprint
├── Procfile
├── runtime.txt                     # python-3.12.0
└── LICENSE
```

---

<a id="deployment"></a>

## 🌐 Deployment

### Render (recommended)

1. Push to `main` on GitHub.
2. Create a **Web Service** on [Render](https://dashboard.render.com) and connect the repo.
3. Use the settings from `render.yaml`:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `python -W ignore::SyntaxWarning run_backend.py`
4. Add a **PostgreSQL** instance and link it, or set `POSTGRESQL_*` manually.
5. Set secrets in the dashboard: `GEMINI_API_KEY`, `NEWS_API_KEY`, `STRIPE_*`, `JWT_SECRET`, etc.

**Blueprint:** Render Dashboard → New → Blueprint → select repo → apply `render.yaml`.

### CI/CD

`.github/workflows/deploy.yml` triggers a Render deploy hook on push to `main`/`master` (requires `RENDER_DEPLOY_HOOK` secret).

### Production checklist

- [ ] `ENVIRONMENT=production`
- [ ] `USE_POSTGRESQL=true` with valid credentials
- [ ] `CORS_ORIGINS` set to real frontend URLs (not `*`)
- [ ] `LOG_LEVEL=INFO` or `WARNING`
- [ ] Rotate `JWT_SECRET` and API keys
- [ ] Configure Stripe webhook URL → `/api/v1/spot-trade/stripe/webhook`

### Free-tier notes

- Render free web services sleep after inactivity (~30s cold start).
- Yahoo Finance may rate-limit; the app caches and backs off automatically.

---

<a id="development"></a>

## 🛠️ Development

```bash
ENVIRONMENT=development LOG_LEVEL=DEBUG python run_backend.py
```

Uvicorn **auto-reload** is enabled when `ENVIRONMENT=development`.

### Code quality (optional)

```bash
pip install black flake8 mypy
black backend/ scripts/
flake8 backend/ scripts/ --max-line-length=100
mypy backend/ --ignore-missing-imports
```

### Utility scripts

| Script | Purpose |
| --- | --- |
| `scripts/train_enhanced_model.py` | Train news-enhanced model |
| `scripts/check_predictions.py` | Inspect prediction DB rows |
| `scripts/diagnose_apis.py` | Test external API connectivity |
| `scripts/import_predictions.py` | Import historical predictions |

### Tests

Automated tests are not yet in the repository. Contributions adding `pytest` coverage are welcome.

---

<a id="security"></a>

## 🔒 Security

- Secrets only via environment variables — never commit `.env`
- Pydantic request/response validation on all inputs
- Security headers, timing, compression, and request size middleware
- JWT verification for spot-trade routes (`JWT_SECRET` shared with Node backend)
- Stripe webhook signature verification
- Production validator rejects `CORS_ORIGINS=*` and `LOG_LEVEL=DEBUG`

---

<a id="troubleshooting"></a>

## 🔧 Troubleshooting

| Issue | Fix |
| --- | --- |
| PostgreSQL connection fails | Verify `POSTGRESQL_*`; or run with `USE_POSTGRESQL=false` |
| Port in use | `lsof -ti:8001 \| xargs kill -9` or set `PORT=8002` |
| `ModuleNotFoundError` | Run from repo root; `pip install -r requirements.txt` |
| No market data | Check network; test `yfinance`; increase `CACHE_DURATION` |
| Rate-limit warnings | Expected with Yahoo Finance; app uses cache + backoff |
| WebSocket disconnects | Confirm `ws://host:port/ws/xauusd`; check CORS/proxy settings |
| Spot-trade 401 | Send `Authorization: Bearer <valid JWT>` matching `JWT_SECRET` |

---

<a id="roadmap"></a>

## 🗺️ Roadmap

- [ ] Docker / Docker Compose for local dev
- [ ] `pytest` suite with CI test job
- [ ] API rate-limiting middleware
- [ ] Additional models (ensemble / time-series)
- [ ] `.env.example` template in repo

**Current release:** v2.0.0 — Lasso + sentiment, WebSocket, PostgreSQL/SQLite, spot trading, Render deployment.

---

<a id="contributing"></a>

## 🤝 Contributing

1. Fork [the repository](https://github.com/PunithAchintha2003/kgf-gold-price-predictor-ml-backend).
2. Create a branch: `git checkout -b feat/your-feature`.
3. Make changes; follow PEP 8 and add tests when applicable.
4. Open a pull request with a clear description.

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.).

---

<a id="license"></a>

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

---

<a id="disclaimer"></a>

## ⚠️ Disclaimer

This software is provided **"AS IS"** without warranty.

- For **education and research** only — not financial advice.
- Model metrics reflect historical evaluation; live results may differ.
- Market data comes from third parties; availability is not guaranteed.
- Trading involves risk. Authors are not liable for financial losses.

By using this API you accept these terms.

---

<a id="support"></a>

## 📞 Support

| Channel | Link |
| --- | --- |
| Live API | https://kgf-gold-price-predictor.onrender.com |
| API docs | https://kgf-gold-price-predictor.onrender.com/docs |
| Issues | https://github.com/PunithAchintha2003/kgf-gold-price-predictor-ml-backend/issues |
| Email | Punithachintha@gmail.com |

When filing issues, include OS, Python version, `.env` flags (redact secrets), steps to reproduce, and logs.

---

<div align="center">

**Made with ❤️ for the financial technology community**

[⬆ Back to Top](#top)

</div>
