# How to Deploy for FREE on Render

## ⚠️ Important: You MUST Select Free Tier Manually

Even though `render.yaml` is configured for free tier, you need to manually ensure you select the **FREE** plan when deploying.

---

## 🚀 Deployment Steps for FREE Tier

### Method 1: Using Blueprint (Automated)

1. **Push your code to GitHub**:

   ```bash
   git add .
   git commit -m "Configure for free tier"
   git push origin main
   ```

2. **Go to Render Dashboard**:

   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Sign up/Login (it's free to sign up)

3. **Create Blueprint**:

   - Click "New +" button
   - Select "Blueprint"
   - Click "Connect GitHub" or "Public Git repository"
   - Connect your repository

4. **IMPORTANT - Select FREE Plan**:

   - Look for the services section
   - You'll see a service created from render.yaml
   - **Click on the service to edit it**
   - Look for "Plan" section
   - **Change from "Starter" to "Free"**
   - Click "Save Changes"

5. **Apply Blueprint**:
   - Click "Apply" or "Save changes"
   - Wait for deployment (3-5 minutes)

### Method 2: Manual Web Service Setup (Recommended for Free)

1. **Push your code to GitHub**:

   ```bash
   git add .
   git commit -m "Configure for free tier"
   git push origin main
   ```

2. **Go to Render Dashboard**:

   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +"
   - Select "Web Service"

3. **Connect Repository**:

   - Select your GitHub repository
   - Click "Connect"

4. **Configure Settings** (VERY IMPORTANT):

   - **Name**: `kgf-gold-price-predictor`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_backend.py`

   **CRITICAL - Plan Selection**:

   - Look for "Instance Type" or "Plan" dropdown
   - **Select "Free" from the dropdown**
   - This ensures $0/month cost

   - **Auto-Deploy**: Leave enabled (auto-deploys on git push)

5. **Environment Variables** (Optional - add these):
   Click "Advanced" or scroll down to Environment section:

   ```
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

6. **Create Web Service**:
   - Click "Create Web Service"
   - Wait for build to complete (3-5 minutes)
   - Your service will be live!

---

## ✅ How to Verify You're on Free Tier

1. **Check Pricing**:

   - Go to your service in Render dashboard
   - Click on the service name
   - Look at "Plan" in the sidebar - should say "Free"

2. **Verify $0 Cost**:

   - In the service overview
   - Should show $0/month or "Free" tier

3. **Test Your Service**:
   - Your URL: `https://your-service-name.onrender.com`
   - Health check: `https://your-service-name.onrender.com/health`

---

## ⚠️ If You're Seeing $7/month

If Render shows "Starter $7/month":

1. **Don't save/deploy yet!**
2. **Find the "Plan" or "Instance Type" dropdown**
3. **Change it to "Free"**
4. **Then save/deploy**

The reason you might see $7 is because:

- Render default is "Starter" plan
- You must manually select "Free"
- This is intentional (they want you to upgrade)

---

## 🎯 Free Tier Features

✅ **What You Get for FREE**:

- Full FastAPI backend
- ML predictions
- All API endpoints
- WebSocket support
- 100% functionality

⚠️ **Free Tier Limitations**:

- Sleeps after 15 min inactivity
- Takes ~30 seconds to wake up
- Models retrain on wake (takes 2-5 min)
- Data resets on restart

---

## 📋 Step-by-Step Visual Guide

When creating web service, you'll see:

```
┌─────────────────────────────────────┐
│ Name: kgf-gold-price-predictor     │
├─────────────────────────────────────┤
│ Instance Type: [Free ▼]   ← IMPORTANT! │
│              ↓                      │
│            [Free]     ← SELECT THIS │
│            [Starter]  ← Don't select│
│            [Pro]      ← Don't select│
└─────────────────────────────────────┘
```

**Make sure "Free" is selected before clicking "Create Web Service"**

---

## 🆘 Troubleshooting

### Issue: I already deployed with Starter ($7)

**Solution**:

1. Go to your service in Render
2. Click "Settings"
3. Scroll to "Plan" or "Instance Type"
4. Change to "Free"
5. Click "Save Changes"
6. You'll be refunded (prorated) and switched to free

### Issue: Can't find "Free" option

**Solution**:

- "Free" might be called "Free Instance" or just "Free"
- It's usually in a dropdown under "Instance Type"
- If you can't find it, Render may have changed their UI
- Check: https://render.com/docs/free-tier

### Issue: Free option is disabled

**Possible reasons**:

- You already have too many free services
- Free tier has limits (usually 1-3 free services)
- Contact Render support

---

## ✅ Summary

**To Get Free Tier ($0/month)**:

1. ✅ Push code to GitHub
2. ✅ Go to Render Dashboard
3. ✅ Create Web Service (or Blueprint)
4. ⚠️ **MANUALLY SELECT "FREE" in Plan dropdown**
5. ✅ Configure build/start commands
6. ✅ Deploy!

**Remember**: Render won't automatically select free tier - YOU must select it manually!

---

## 💰 Cost Verification

After deployment, verify your cost:

- Go to Render dashboard → Your service
- Check "Plan" in sidebar
- Should show "Free" not "Starter"
- Billing should show $0/month

---

## 🎉 You're Done!

Once deployed, your API will be at:

- `https://your-service-name.onrender.com`
- API docs: `https://your-service-name.onrender.com/docs`

**No monthly fees!** 🎊
