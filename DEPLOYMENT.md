# Deployment Guide - KGF Gold Price Predictor to Render

This guide will help you deploy the KGF Gold Price Predictor backend to Render.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **Environment Variables**: Prepare API keys (optional, for enhanced features)

## Quick Deploy (Recommended)

### Option 1: Using render.yaml (Declarative Deployment)

This is the easiest method using the provided configuration file.

1. **Push your code to GitHub** (if not already done):

   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Go to Render Dashboard**:

   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" and select "Blueprint"

3. **Connect your repository**:

   - Select your GitHub repository
   - Render will automatically detect the `render.yaml` file
   - Click "Apply"

4. **Review and deploy**:
   - Render will read the configuration from `render.yaml`
   - Review the settings and click "Save Changes"
   - Wait for the build to complete

### Option 2: Manual Web Service Setup

If you prefer to set it up manually:

1. **Go to Render Dashboard**:

   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" and select "Web Service"

2. **Connect your repository**:

   - Select your GitHub repository
   - Click "Connect"

3. **Configure the service**:

   - **Name**: `kgf-gold-price-predictor` (or your preferred name)
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (project root)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_backend.py`
   - **Instance Type**: Choose based on your needs:
     - **Free**: Suitable for testing (spins down after inactivity)
     - **Starter ($7/month)**: Better for production use

4. **Environment Variables** (Optional):
   Add the following in the "Environment" section:

   ```
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   CACHE_DURATION=300
   API_COOLDOWN=2
   REALTIME_CACHE_DURATION=60
   ```

   Optional (for enhanced features):

   ```
   NEWS_API_KEY=your_news_api_key
   ALPHA_VANTAGE_KEY=your_alpha_vantage_key
   ```

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for the build to complete (usually 2-5 minutes)
   - Your service will be live at `https://your-service-name.onrender.com`

## Important Notes

### Database Storage

⚠️ **Important**: Render's filesystem is ephemeral, meaning files are lost when the service restarts. SQLite databases in `backend/data/` will be reset on each deployment.

**Solutions**:

1. **Use Render PostgreSQL** (Recommended for production):

   - Add a PostgreSQL database in Render
   - Update the database connection in the code
   - Modify `backend/app/main.py` to use PostgreSQL instead of SQLite

2. **External Cloud Storage**:

   - Use AWS S3, Google Cloud Storage, or similar
   - Store databases and model files externally
   - Modify the code to sync databases on startup

3. **Accept Ephemeral Storage**:
   - For development/testing: Accept that data is temporary
   - Models will retrain automatically on each restart
   - Predictions history will be reset

### Model Files

The ML models (`.pkl` files) will be trained on first startup if they don't exist. The training process:

- Takes 2-5 minutes on first startup
- Happens automatically
- Models are saved in memory during runtime
- Will be retrained on each deployment

To avoid retraining on every deploy:

1. Store trained models in external storage (S3, Google Cloud Storage)
2. Download and load models on startup
3. Only retrain if models are outdated

### File Structure for Render

```
kgf-gold-price-predictor-ml-backend/
├── Procfile                      # Process file for Render
├── render.yaml                   # Render configuration (Option 1)
├── runtime.txt                   # Python version specification
├── requirements.txt              # Python dependencies
├── run_backend.py                # Main startup script
├── backend/
│   ├── app/
│   │   └── main.py               # FastAPI application
│   ├── config/                   # Configuration files
│   ├── data/                     # SQLite databases (ephemeral on Render)
│   ├── models/                   # ML model files (ephemeral on Render)
│   └── requirements.txt          # Backend dependencies
└── DEPLOYMENT.md                 # This file
```

## Post-Deployment

### Verify Deployment

1. **Health Check**:

   ```bash
   curl https://your-service-name.onrender.com/health
   ```

2. **API Documentation**:
   Visit: `https://your-service-name.onrender.com/docs`

3. **Test Endpoints**:

   ```bash
   # Get gold price data
   curl https://your-service-name.onrender.com/xauusd

   # Get real-time price
   curl https://your-service-name.onrender.com/xauusd/realtime

   # Get predictions
   curl https://your-service-name.onrender.com/xauusd/enhanced-prediction
   ```

### Monitor Logs

1. Go to your service dashboard in Render
2. Click on "Logs" tab
3. Monitor for any errors or issues
4. Check for model training messages

### Troubleshooting

#### Common Issues

1. **Build Fails**:

   - Check that all dependencies are in `requirements.txt`
   - Verify Python version in `runtime.txt`
   - Check build logs for specific errors

2. **Service Crashes on Startup**:

   - Check logs for Python errors
   - Verify database path permissions
   - Ensure all required files are present

3. **Model Training Takes Too Long**:

   - First startup takes 2-5 minutes for model training
   - This is normal and expected
   - Subsequent requests are fast

4. **No Data Available**:

   - Check internet connectivity in Render (should work automatically)
   - Verify Yahoo Finance API access
   - Check logs for API errors

5. **Port Binding Errors**:
   - Ensure `run_backend.py` uses `PORT` environment variable
   - Verify the startup command uses the correct port

### Performance Optimization

For better performance on Render:

1. **Instance Type**:

   - Free tier: Suitable for testing
   - Starter ($7/month): Better for production
   - Professional: For high traffic

2. **Enable Auto-Deploy**:

   - Automatic deployments on git push
   - Enable in "Settings" tab

3. **Environment Variables**:

   - Set `LOG_LEVEL=WARNING` to reduce log volume
   - Adjust cache durations based on usage

4. **Monitoring**:
   - Set up alerts for service downtime
   - Monitor memory usage
   - Track response times

## Environment Variables Reference

| Variable                  | Description                               | Default     | Required |
| ------------------------- | ----------------------------------------- | ----------- | -------- |
| `PORT`                    | Server port (set automatically by Render) | 8001        | No       |
| `ENVIRONMENT`             | Running environment                       | development | No       |
| `LOG_LEVEL`               | Logging level                             | WARNING     | No       |
| `CACHE_DURATION`          | Market data cache duration (seconds)      | 300         | No       |
| `API_COOLDOWN`            | API rate limiting (seconds)               | 2           | No       |
| `REALTIME_CACHE_DURATION` | Real-time cache duration (seconds)        | 60          | No       |
| `NEWS_API_KEY`            | NewsAPI.org API key (optional)            | None        | No       |
| `ALPHA_VANTAGE_KEY`       | Alpha Vantage API key (optional)          | None        | No       |

## Database Migration (Optional)

To use PostgreSQL instead of SQLite:

1. **Add PostgreSQL Service in Render**:

   - Create a PostgreSQL database in Render
   - Note the connection string

2. **Update Connection**:

   - Modify database connection in `backend/app/main.py`
   - Replace SQLite with PostgreSQL connection

3. **Install PostgreSQL Driver**:
   - Add `psycopg2-binary` to `requirements.txt`
   - Update connection management code

## Security Considerations

1. **API Keys**:

   - Use Render's environment variable encryption
   - Never commit API keys to git
   - Rotate keys regularly

2. **CORS**:

   - Update CORS origins in `backend/app/main.py`
   - Add your frontend domain to allowed origins

3. **Rate Limiting**:
   - Current rate limiting is basic
   - Consider adding proper rate limiting middleware

## Cost Estimates

| Service Type | Cost      | Best For                   |
| ------------ | --------- | -------------------------- |
| Free         | $0/month  | Testing, personal projects |
| Starter      | $7/month  | Small production apps      |
| Professional | $25/month | High traffic, production   |

## Next Steps

1. **Deploy Frontend** (if applicable):

   - Deploy your frontend separately
   - Update API endpoints to point to Render URL

2. **Set Up Custom Domain** (optional):

   - Add custom domain in Render settings
   - Configure DNS records

3. **Monitoring**:

   - Set up uptime monitoring
   - Configure alerts
   - Monitor API usage

4. **Backup Strategy**:
   - Implement database backups (for PostgreSQL)
   - Store trained models externally
   - Backup configuration

## Support

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **Render Community**: [community.render.com](https://community.render.com)
- **Project Issues**: Check project repository for known issues

## Summary

Your FastAPI backend is now configured for deployment on Render. Key changes made:

✅ **Created `render.yaml`** - Declarative deployment configuration  
✅ **Created `runtime.txt`** - Python version specification  
✅ **Created `Procfile`** - Process file for Render  
✅ **Updated `run_backend.py`** - Dynamic port handling for Render  
✅ **Created `DEPLOYMENT.md`** - This comprehensive guide

**Ready to deploy?** Follow the "Quick Deploy" instructions above!
