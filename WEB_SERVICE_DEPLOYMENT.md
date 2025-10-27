# Deploy as Web Service on Render (Step-by-Step)

## 🎯 What is Web Service?

Web Service is Render's dynamic web app deployment option, perfect for:

- ✅ API servers (like yours!)
- ✅ Full-stack apps
- ✅ Mobile backends
- ✅ FastAPI applications

This is the **correct deployment method** for your project!

---

## 🚀 Complete Deployment Guide

### Step 1: Push Code to GitHub

First, make sure your code is on GitHub:

```bash
# Check current status
git status

# Add all files
git add .

# Commit changes
git commit -m "Ready for Render deployment"

# Push to GitHub
git push origin main
```

**Verify**: Go to your GitHub repository and make sure all files are there.

---

### Step 2: Go to Render Dashboard

1. Visit [render.com](https://render.com)
2. Click **"Get Started"** or **"Sign Up"**
3. Sign up with GitHub (recommended) or email
4. After login, go to [dashboard.render.com](https://dashboard.render.com)

---

### Step 3: Create Web Service

1. Click the **"+ New +"** button (top right)
2. Select **"Web Service"**
3. You'll be asked to connect a repository
4. Click **"Connect GitHub"** (or **"Public Git repository"**)

---

### Step 4: Connect Your Repository

1. If using GitHub:

   - Authorize Render (if first time)
   - Select your repository: `KGF-gold-price-predictor-ml-backend`
   - Click **"Connect"**

2. If using Public Git:
   - Enter your Git repository URL
   - Click **"Connect"**

---

### Step 5: Configure Web Service Settings

This is the most important step! Fill in these settings:

#### Basic Settings

```
┌────────────────────────────────────────────┐
│ Service Name: kgf-gold-price-predictor    │
│                         [Enter your name]  │
├────────────────────────────────────────────┤
│ Region: (Choose closest to you)           │
│              ↓                              │
│        [Oregon (US West)]                  │
│        [Frankfurt (EU)]                    │
│        [Singapore (Asia Pacific)]          │
└────────────────────────────────────────────┘
```

#### Runtime Settings

```
┌────────────────────────────────────────────┐
│ Branch: main                               │
│                  (Your default branch)     │
├────────────────────────────────────────────┤
│ Root Directory: (leave EMPTY)              │
│                  [                    ]    │
├────────────────────────────────────────────┤
│ Runtime: Python 3                          │
│        [Python 3 ▼]                        │
└────────────────────────────────────────────┘
```

#### Build Settings

```
┌────────────────────────────────────────────┐
│ Build Command: pip install -r requirements.txt
│              [Enter exactly this]          │
├────────────────────────────────────────────┤
│ Start Command: python run_backend.py      │
│              [Enter exactly this]           │
└────────────────────────────────────────────┘
```

---

### Step 6: Select FREE Plan (CRITICAL!)

**This is where you ensure $0 cost:**

```
┌────────────────────────────────────────────┐
│ Instance Type:                             │
│              ↓                              │
│    Free ($0/month)        ← SELECT THIS   │
│    Starter ($7/month)     ← Don't select  │
│    Professional ($25/month) ← Don't select│
└────────────────────────────────────────────┘
```

**Important**: You MUST manually select "Free" or it will default to $7/month!

---

### Step 7: Add Environment Variables (Optional)

Scroll down to find **"Environment"** section, click **"Add Environment Variable"**:

```bash
KEY:   ENVIRONMENT
VALUE: production

KEY:   LOG_LEVEL
VALUE: INFO
```

(These are optional - the app works without them using defaults)

---

### Step 8: Advanced Settings

Leave these as defaults:

- **Auto-Deploy**: Enabled (deploys on git push)
- **Health Check Path**: (leave empty)
- **Override Connection Details**: Not needed

---

### Step 9: Create and Deploy

1. Click **"Create Web Service"** button (bottom)
2. Render will start building your app
3. Wait for deployment (3-5 minutes)
4. Watch the logs for progress

---

### Step 10: Monitor Deployment

You'll see logs like:

```
Checking for Python
Python found: 3.11.0
Installing dependencies...
Installing from requirements.txt...
✅ Dependencies installed
Starting application...
✅ Application started
```

First deployment takes **2-5 minutes** because:

- Installing Python packages
- Training ML models (first time only)
- Initializing database

---

### Step 11: Access Your API

Once deployment is complete, you'll see:

```
✅ Live
https://your-service-name.onrender.com
```

Test your API:

- **Health Check**: `https://your-service-name.onrender.com/health`
- **API Docs**: `https://your-service-name.onrender.com/docs`
- **Gold Data**: `https://your-service-name.onrender.com/xauusd`

---

## 📋 Summary of Settings

When creating Web Service, use these EXACT settings:

| Setting            | Value                             |
| ------------------ | --------------------------------- |
| **Service Type**   | Web Service                       |
| **Name**           | kgf-gold-price-predictor          |
| **Region**         | Choose closest to you             |
| **Branch**         | main                              |
| **Root Directory** | (leave empty)                     |
| **Runtime**        | Python 3                          |
| **Build Command**  | `pip install -r requirements.txt` |
| **Start Command**  | `python run_backend.py`           |
| **Instance Type**  | **FREE** ← This is critical!      |
| **Auto-Deploy**    | Enabled                           |

---

## ✅ Post-Deployment

### 1. Verify It's Free

- Go to your service dashboard
- Check "Plan" in sidebar - should say "Free"
- Billing should show $0/month

### 2. Test Your API

```bash
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Test gold data
curl https://your-service-name.onrender.com/xauusd

# View API docs in browser
open https://your-service-name.onrender.com/docs
```

### 3. Monitor Logs

- Click "Logs" tab in Render dashboard
- Check for any errors
- Watch model training progress

---

## 🎯 What Happens on First Deploy?

```
Step 1: Installing Dependencies (1-2 min)
  ✅ Installing FastAPI
  ✅ Installing pandas, numpy
  ✅ Installing scikit-learn
  ✅ Installing yfinance
  ✅ All packages installed

Step 2: Starting Application (30 sec)
  ✅ Loading main.py
  ✅ Initializing FastAPI app
  ✅ Setting up database

Step 3: Training ML Models (2-5 min - first time only)
  ⏳ Training Lasso model...
  ✅ Lasso model trained
  ⏳ Training News-Enhanced model...
  ✅ News-Enhanced model trained

Step 4: Service Ready! ✅
  ✅ API is live
  ✅ Models loaded
  ✅ Database initialized
  ✅ WebSocket ready
```

After first deploy, subsequent deployments are faster (1-2 minutes).

---

## 🆘 Troubleshooting

### Issue: Build Fails

**Error**: "Module not found"
**Solution**: Make sure requirements.txt is in project root

**Error**: "Python version not supported"
**Solution**: Check runtime.txt has `python-3.11.0`

### Issue: Service Crashes

**Check logs for**:

- Port binding errors (should use $PORT env var)
- Database path errors (should use backend/data/)
- Model loading errors (will retrain automatically)

### Issue: Shows $7/month

**Solution**: You selected wrong plan

- Go to Settings → Instance Type
- Change to "Free"
- Save changes

### Issue: Slow Response

**This is normal on first request**:

- Free tier service sleeps after 15 min
- Wakes up in ~30 seconds
- Subsequent requests are fast

---

## 📱 Mobile/API Usage

Your Web Service is perfect for:

- ✅ Mobile app backends
- ✅ React/Vue frontends
- ✅ API consumers
- ✅ WebSocket clients

Example usage:

```javascript
// In your frontend app
const API_URL = "https://your-service.onrender.com";

fetch(`${API_URL}/xauusd`)
  .then((res) => res.json())
  .then((data) => console.log(data));
```

---

## 🎊 You're Done!

Your FastAPI backend is now deployed as a **Web Service** on Render!

- **URL**: `https://your-service-name.onrender.com`
- **Docs**: `https://your-service-name.onrender.com/docs`
- **Cost**: **$0/month** (FREE tier)
- **Type**: Web Service (Dynamic web app)

**No monthly fees!** Happy coding! 🚀
