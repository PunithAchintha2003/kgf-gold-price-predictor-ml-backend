# Production Deployment Checklist for Render.com

## ✅ Pre-Deployment Checklist

### 1. Verify Service Status in Render Dashboard
- [ ] Go to [Render Dashboard](https://dashboard.render.com)
- [ ] Navigate to your service: `kgf-gold-price-predictor`
- [ ] Check service status:
  - ✅ **Running** (Green) - Service is active
  - ⚠️ **Building** - Deployment in progress
  - ❌ **Failed** - Check logs for errors
  - ⏸️ **Suspended** - Service paused (free tier)

### 2. Check Build Logs
- [ ] Click on your service → **Logs** tab
- [ ] Verify build completed successfully:
  - ✅ `pip install -r requirements.txt` completed
  - ✅ No import errors
  - ✅ Database connection successful
- [ ] Check for any error messages

### 3. Verify Environment Variables

Go to **Environment** tab in Render Dashboard and verify these variables are set:

#### Required Variables:
- [ ] `PORT` = `10000` (Render auto-sets this, but verify)
- [ ] `ENVIRONMENT` = `production`
- [ ] `LOG_LEVEL` = `INFO`
- [ ] `USE_POSTGRESQL` = `true`

#### CORS Configuration (CRITICAL):
- [ ] `CORS_ORIGINS` = `http://localhost:4000,https://kgf-gold-price-predictor.onrender.com`
  - Add your production frontend URL if different
  - Format: comma-separated, no spaces after commas

#### PostgreSQL Database Variables:
- [ ] `POSTGRESQL_HOST` - Auto-set if database is linked
- [ ] `POSTGRESQL_DATABASE` - Auto-set if database is linked
- [ ] `POSTGRESQL_USER` - Auto-set if database is linked
- [ ] `POSTGRESQL_PASSWORD` - Auto-set if database is linked
- [ ] `POSTGRESQL_PORT` - Usually `5432` (auto-set if linked)

#### Optional Variables:
- [ ] `CACHE_DURATION` = `300` (5 minutes)
- [ ] `API_COOLDOWN` = `2` (seconds)
- [ ] `REALTIME_CACHE_DURATION` = `60` (1 minute)

### 4. Verify PostgreSQL Database

#### Database Status:
- [ ] PostgreSQL database service is **Running**
- [ ] Database is **linked** to your web service (if using Render's managed PostgreSQL)
- [ ] Or manually configured with connection details

#### Database Connection Test:
- [ ] Check logs for: `✅ PostgreSQL enabled - using PostgreSQL database`
- [ ] If you see: `⚠️ PostgreSQL initialization failed - falling back to SQLite`
  - Verify all PostgreSQL environment variables are set correctly
  - Check database is accessible from Render's network

### 5. Test API Endpoints

After deployment, test these endpoints:

#### Health Check:
```bash
curl https://kgf-gold-price-predictor.onrender.com/health
```
Expected: `{"status":"healthy",...}`

#### Main Endpoint:
```bash
curl https://kgf-gold-price-predictor.onrender.com/xauusd?days=90
```
Expected: JSON response with market data

#### Exchange Rate:
```bash
curl https://kgf-gold-price-predictor.onrender.com/exchange-rate/USD/LKR
```
Expected: JSON response with exchange rate

### 6. Check CORS Headers

Test CORS headers are being sent:
```bash
curl -I -H "Origin: http://localhost:4000" \
  https://kgf-gold-price-predictor.onrender.com/health
```

Look for:
- `Access-Control-Allow-Origin: http://localhost:4000`
- `Access-Control-Allow-Methods: *`
- `Access-Control-Allow-Headers: *`

### 7. Common Issues & Solutions

#### Issue: 502 Bad Gateway
**Possible Causes:**
- Service crashed on startup
- Database connection failed
- Missing environment variables
- Port mismatch

**Solutions:**
1. Check **Logs** tab for error messages
2. Verify all environment variables are set
3. Check PostgreSQL connection
4. Verify `PORT` environment variable matches Render's port (usually 10000)

#### Issue: CORS Error
**Possible Causes:**
- `CORS_ORIGINS` not set or incorrect
- Frontend origin not in allowed list

**Solutions:**
1. Set `CORS_ORIGINS` environment variable:
   ```
   http://localhost:4000,https://your-frontend-domain.com
   ```
2. Redeploy service after adding environment variable
3. Verify no typos in origin URLs

#### Issue: Database Connection Failed
**Possible Causes:**
- PostgreSQL not running
- Wrong credentials
- Database not linked to service

**Solutions:**
1. Verify PostgreSQL service is running
2. Check all `POSTGRESQL_*` environment variables
3. If using Render's managed PostgreSQL, ensure it's linked
4. Check network connectivity in logs

#### Issue: Service Keeps Restarting
**Possible Causes:**
- Application crash on startup
- Import errors
- Missing dependencies

**Solutions:**
1. Check **Logs** for error messages
2. Verify `requirements.txt` includes all dependencies
3. Check Python version matches `runtime.txt` (3.12.0)
4. Verify all file paths are correct

### 8. Manual Environment Variable Setup

If `render.yaml` doesn't apply automatically, set these manually in Render Dashboard:

1. Go to your service → **Environment** tab
2. Click **Add Environment Variable**
3. Add each variable:

```
CORS_ORIGINS = http://localhost:4000,https://kgf-gold-price-predictor.onrender.com
```

**Important:** 
- No quotes around the value
- Comma-separated list for CORS_ORIGINS
- Case-sensitive variable names

### 9. Redeploy After Changes

After updating environment variables:
- [ ] Click **Manual Deploy** → **Deploy latest commit**
- [ ] Or push a new commit to trigger auto-deploy
- [ ] Wait for build to complete
- [ ] Verify service status is **Running**

### 10. Monitor Service Health

- [ ] Check **Metrics** tab for:
  - CPU usage
  - Memory usage
  - Request rate
- [ ] Monitor **Logs** for:
  - Error messages
  - Database connection status
  - API request logs

## 🔧 Quick Fix Commands

### Test Health Endpoint:
```bash
curl https://kgf-gold-price-predictor.onrender.com/health
```

### Test CORS:
```bash
curl -H "Origin: http://localhost:4000" \
  -H "Access-Control-Request-Method: GET" \
  -X OPTIONS \
  https://kgf-gold-price-predictor.onrender.com/health
```

### View Service Logs:
- Go to Render Dashboard → Your Service → **Logs** tab
- Or use Render CLI: `render logs <service-name>`

## 📝 Notes

- Render free tier services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds (cold start)
- Consider upgrading to paid tier for always-on service
- Database backups are recommended for production

## ✅ Final Verification

Before marking as complete:
- [ ] Service status: **Running** (Green)
- [ ] Health endpoint returns: `{"status":"healthy"}`
- [ ] CORS headers present in response
- [ ] Database connection successful
- [ ] All API endpoints responding
- [ ] Frontend can connect without CORS errors

---

**Last Updated:** 2025-11-27
**Service URL:** https://kgf-gold-price-predictor.onrender.com

