# Quick Start: Deploy to Render

## 🚀 What's Been Set Up

Your project is now ready to deploy on Render! Here's what I've prepared:

### ✅ Files Created

1. **`render.yaml`** - Render configuration (declarative deployment)
2. **`runtime.txt`** - Python 3.11 specification
3. **`Procfile`** - Process file for Render
4. **`DEPLOYMENT.md`** - Full deployment guide
5. **Updated `run_backend.py`** - Now uses dynamic PORT from Render

## 📋 Deployment Steps

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Go to Render Dashboard

1. Visit [dashboard.render.com](https://dashboard.render.com)
2. Sign up/login if needed

### Step 3: Deploy

**Option A: Using Blueprint (Easiest)**

- Click "New +" → "Blueprint"
- Connect your GitHub repository
- Render will auto-detect `render.yaml`
- Click "Apply"
- Done! 🎉

**Option B: Manual Setup**

- Click "New +" → "Web Service"
- Connect your GitHub repository
- Settings:
  - **Name**: `kgf-gold-price-predictor`
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `python run_backend.py`
  - **Plan**: Free or Starter ($7/month)
- Add Environment Variables (optional):
  ```
  ENVIRONMENT=production
  LOG_LEVEL=INFO
  ```
- Click "Create Web Service"
- Wait 2-5 minutes for build

### Step 4: Access Your API

Once deployed, your API will be available at:

- **Base URL**: `https://your-service-name.onrender.com`
- **API Docs**: `https://your-service-name.onrender.com/docs`
- **Health Check**: `https://your-service-name.onrender.com/health`

## ⚠️ Important Notes

### Database (SQLite)

- SQLite databases are **ephemeral** on Render
- Data will be lost on each restart/deployment
- For production, consider using Render PostgreSQL

### Model Files (.pkl)

- Models will retrain on first startup (2-5 minutes)
- Trained models are saved in memory
- Will retrain on each deployment

### First Startup

- Takes 2-5 minutes to train models
- This is normal and expected
- Models cache after first training

## 🧪 Test Your Deployment

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-service-name.onrender.com/health

# Get gold data
curl https://your-service-name.onrender.com/xauusd

# Real-time price
curl https://your-service-name.onrender.com/xauusd/realtime

# ML prediction
curl https://your-service-name.onrender.com/xauusd/enhanced-prediction
```

## 📊 What Happens on Startup

1. **Dependency Installation** (1-2 min)

   - Installs Python packages from `requirements.txt`

2. **Model Training** (2-5 min - first time only)

   - Trains Lasso Regression model
   - Trains News-Enhanced Lasso model
   - Saves models (in memory/ephemeral storage)

3. **Database Initialization**

   - Creates SQLite database tables
   - Sets up indexes for performance

4. **Service Ready** (✅)
   - API is now live and accepting requests
   - WebSocket endpoints available
   - Background tasks running

## 🔧 Environment Variables (Optional)

Add these in Render dashboard for enhanced features:

```
NEWS_API_KEY=your_key_here          # For news sentiment analysis
ALPHA_VANTAGE_KEY=your_key_here     # For alternative news source
```

## 💰 Cost Estimates

- **Free Tier**: $0/month (spins down after inactivity)
- **Starter**: $7/month (always-on)
- **Professional**: $25/month (high performance)

## 📝 Next Steps

1. ✅ Deploy to Render (follow steps above)
2. 🧪 Test the API endpoints
3. 🔗 Connect your frontend (if you have one)
4. 📊 Monitor logs in Render dashboard
5. 🎯 Set up custom domain (optional)

## 🆘 Troubleshooting

**Build Fails?**

- Check `requirements.txt` has all dependencies
- Verify Python version in `runtime.txt`

**Service Crashes?**

- Check logs in Render dashboard
- Verify all model files are present

**No Data?**

- Models will train on first startup
- Wait 2-5 minutes for training to complete
- Check logs for errors

## 📚 Full Documentation

For detailed information, see:

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Complete deployment guide
- **[README.md](./README.md)** - Project documentation

## 🎉 That's It!

Your FastAPI backend is now ready for Render deployment. Follow the steps above to get it live!

**Questions?** Check the logs in Render dashboard for detailed error messages.
