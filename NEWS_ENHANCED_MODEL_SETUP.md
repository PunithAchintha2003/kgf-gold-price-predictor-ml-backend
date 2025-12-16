# News-Enhanced Lasso Model Setup

## Problem

The news-enhanced Lasso model was not working on production because:

1. **Model file was not in git** - `.gitignore` was blocking all `.pkl` files
2. **Model was never trained** - The `enhanced_lasso_gold_model.pkl` file didn't exist
3. **Silent failure** - Logs were at DEBUG level, so failures weren't visible

## What Was Fixed

### 1. Trained the News-Enhanced Model ✅

```bash
cd backend
python3 -m models.news_prediction
```

**Model Performance:**

- **R² Score:** 76.23% (compared to basic Lasso's 96.16%)
- **Selected Features:** 8 out of 69 features
- **News Sources:** RSS feeds only (25 articles)

⚠️ **Note:** The enhanced model has lower accuracy due to limited news data. To improve:

- Add `NEWS_API_KEY` from [NewsAPI.org](https://newsapi.org)
- Add `ALPHA_VANTAGE_KEY` from [Alpha Vantage](https://www.alphavantage.co)

### 2. Updated `.gitignore` ✅

Added exception to allow model files in `backend/models/`:

```gitignore
*.pkl
!backend/models/*.pkl  # Allow model files
```

### 3. Enhanced Logging ✅

Updated `backend/app/core/models.py` to show INFO-level messages:

- Model loading success/failure
- Model accuracy metrics
- Selected features count

### 4. Updated `render.yaml` ✅

Added placeholders for news API keys:

```yaml
# - key: NEWS_API_KEY
#   sync: false
# - key: ALPHA_VANTAGE_KEY
#   sync: false
```

## How It Works Now

The system automatically tries models in this order:

1. **News-Enhanced Lasso** (if available)

   - Uses market data + news sentiment
   - Fetches and analyzes news articles
   - R² = 76.23% (with current limited data)

2. **Basic Lasso** (fallback)
   - Uses market data only
   - R² = 96.16%
   - Used when enhanced model fails

## Deployment Steps

### Step 1: Push Changes to GitHub

```bash
git push origin 012-model
```

### Step 2: Deploy to Render

Render will automatically detect the changes and redeploy.

### Step 3: Verify Deployment

Check logs for:

```
✅ News-Enhanced Lasso model loaded successfully
   Model accuracy (R²): 0.7623
   Selected features: 8
```

If you see this instead:

```
ℹ️  News-Enhanced model not found - using Lasso Regression only
```

Then the model file wasn't deployed correctly.

## Optional: Add News API Keys (Recommended)

To improve the news-enhanced model's accuracy:

1. **Get API Keys:**

   - NewsAPI: https://newsapi.org (free tier: 100 requests/day)
   - Alpha Vantage: https://www.alphavantage.co (free tier: 500 requests/day)

2. **Add to Render Dashboard:**

   - Go to your service → Environment
   - Add variables:
     - `NEWS_API_KEY=your_newsapi_key`
     - `ALPHA_VANTAGE_KEY=your_alphavantage_key`

3. **Retrain the Model:**
   ```bash
   cd backend
   python3 -m models.news_prediction
   ```
4. **Commit and Deploy:**
   ```bash
   git add backend/models/enhanced_lasso_gold_model.pkl
   git commit -m "Update news-enhanced model with better data"
   git push origin 012-model
   ```

## Monitoring

### Check Which Model is Being Used

Look for these log messages:

```
🤖 Using News-Enhanced Lasso model for prediction
```

or

```
🤖 Using Lasso Regression model for prediction
```

### Check Model Performance

The backend will show in startup logs:

```
🤖 ML Model: News-Enhanced Lasso Regression  (or "Lasso Regression")
📊 Model Accuracy (R²): X.XXXX
🔧 Features: X/XX selected
```

## Troubleshooting

### Enhanced Model Not Loading on Production

**Check 1:** Model file exists in git

```bash
git ls-files backend/models/*.pkl
```

Should show:

```
backend/models/enhanced_lasso_gold_model.pkl
backend/models/lasso_gold_model.pkl
```

**Check 2:** Render logs show model loading

```
✅ News-Enhanced Lasso model loaded successfully
```

**Check 3:** File exists on production
The model file should be at:

```
/opt/render/project/src/backend/models/enhanced_lasso_gold_model.pkl
```

### Enhanced Model Has Low Accuracy

This is normal with limited news data. To improve:

1. Add news API keys (see above)
2. Retrain with more historical data (increase `days_back` parameter)
3. Wait for more news data to accumulate over time

### Prediction Using Wrong Model

The enhanced model may fail and fallback to basic Lasso if:

- News fetching fails (rate limited)
- Feature creation fails (missing data)
- Prediction error (feature mismatch)

Check logs for warnings:

```
WARNING News-Enhanced prediction failed: <error message>
```

## Current Status

- ✅ News-enhanced model trained and saved locally
- ✅ Git configuration updated to include model files
- ✅ Enhanced logging implemented
- ✅ Changes committed to git
- ⏳ **PENDING:** Push to GitHub
- ⏳ **PENDING:** Automatic Render deployment
- ⏳ **PENDING:** Production verification

## Files Changed

1. `.gitignore` - Allow `.pkl` files in `backend/models/`
2. `backend/models/enhanced_lasso_gold_model.pkl` - Trained news-enhanced model
3. `backend/app/core/models.py` - Enhanced logging
4. `backend/app/services/prediction_service.py` - Better prediction logs
5. `render.yaml` - Added news API key placeholders

## Next Steps

1. **Push to production:**

   ```bash
   git push origin 012-model
   ```

2. **Wait for Render deployment** (automatic)

3. **Check logs** to verify news-enhanced model is loaded

4. **Optional:** Add news API keys to improve model accuracy

5. **Optional:** Retrain model with more data after adding API keys
