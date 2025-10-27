# Free Tier Deployment Notes

## ✅ Free Tier Configuration

Your project is now configured for **Render's FREE tier** ($0/month).

### Changes Made:

- Updated `render.yaml` plan from `starter` to `free`
- All configuration remains the same
- Works perfectly on free tier!

---

## 🎯 What You Get on Free Tier

✅ **Everything Works**:

- Full FastAPI backend
- ML predictions
- News sentiment analysis (optional)
- WebSocket support
- API documentation
- All endpoints functional

---

## ⚠️ Free Tier Limitations

### 1. **Spins Down After Inactivity**

- Service goes to sleep after 15 minutes of inactivity
- Takes ~30 seconds to wake up when accessed
- First request after sleep is slow

### 2. **Forced Sleep Times**

- Free tier services sleep when not in use
- This is normal and expected
- Service wakes automatically when accessed

### 3. **Performance Considerations**

- Slower startup time (2-5 min for model training)
- Wake-up time (~30 seconds)
- Limited resources

### 4. **Data Persistence**

- SQLite databases are **ephemeral**
- Data resets on each restart/deploy
- Models retrain on each wake

---

## 🚀 How to Deploy (Same Process)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Configure for Render free tier"
git push
```

### Step 2: Deploy on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Click "Apply"
5. Wait for deployment (3-5 minutes)

### Step 3: Access Your API

- **Your URL**: `https://your-service-name.onrender.com`
- **API Docs**: `https://your-service-name.onrender.com/docs`

---

## 💡 Tips for Free Tier Usage

### 1. **Test Regularly**

- First request after sleep takes ~30 seconds
- Subsequent requests are fast
- Keep service warm with periodic requests

### 2. **Use Webhooks**

If you have a frontend, set up a health check ping:

```javascript
// Ping service every 10 minutes to keep it warm
setInterval(() => {
  fetch("https://your-service-name.onrender.com/health");
}, 600000);
```

### 3. **Accept Model Retraining**

- Models train on each wake (takes 2-5 min)
- This is normal on free tier
- First request after wake is slow

### 4. **Monitor Logs**

- Check Render dashboard for errors
- Logs are available in dashboard
- First startup takes longer due to training

---

## 📊 Free Tier vs Paid Tiers

| Feature            | Free                | Starter ($7) | Professional ($25) |
| ------------------ | ------------------- | ------------ | ------------------ |
| **Cost**           | $0/month            | $7/month     | $25/month          |
| **Sleep Behavior** | Sleeps after 15 min | Always on    | Always on          |
| **Wake-up Time**   | ~30 seconds         | Instant      | Instant            |
| **Resources**      | Limited             | Moderate     | High               |
| **Better for**     | Testing, demos      | Production   | High traffic       |

---

## 🎯 Recommended Usage

### Use Free Tier If:

- ✅ Testing/demo purposes
- ✅ Personal projects
- ✅ Low traffic application
- ✅ Budget-conscious

### Upgrade to Starter If:

- ❌ Need instant responses (can't wait 30s)
- ❌ Production application
- ❌ Need reliable uptime
- ❌ Want to keep service always warm

---

## 🔧 Configuration

Your `render.yaml` is configured for free tier:

```yaml
services:
  - type: web
    name: kgf-gold-price-predictor
    env: python
    plan: free # ← Free tier
    buildCommand: pip install -r requirements.txt
    startCommand: python run_backend.py
```

---

## ✅ Summary

**You're all set for FREE deployment!**

- ✅ No cost
- ✅ Full functionality
- ⚠️ Sleeps after inactivity
- ⚠️ Takes time to wake up
- ⚠️ Models retrain on wake

**Ready to deploy?** Just follow the deployment steps above!

---

## 🆘 Common Issues on Free Tier

### Service Seems Slow?

- First request after sleep takes 30s
- Model training on wake takes 2-5 min
- This is normal, not a bug

### Service Not Responding?

- Service is asleep
- Send a request to wake it up
- Wait 30 seconds for response

### Build Fails?

- Check logs in Render dashboard
- Verify requirements.txt
- Check Python version

---

## 💰 Can I Upgrade Later?

Yes! You can upgrade anytime:

1. Go to your service in Render
2. Click "Settings"
3. Change plan from Free to Starter
4. Pay $7/month

**No need to redeploy** - it just wakes up and stays on.
