# 🚀 Streamlit Cloud Deployment - Quick Start

## ✅ Code Successfully Pushed to GitHub
Your code is now live at: https://github.com/Ashbinbiju/s.genie

## 📝 Next Steps: Deploy to Streamlit Cloud

### Step 1: Go to Streamlit Cloud
1. Visit https://share.streamlit.io/
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub account

### Step 2: Deploy Your App
1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository:** `Ashbinbiju/s.genie`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** `trading-genie` or leave blank for auto-generated

### Step 3: Configure Secrets (CRITICAL!)
1. Before clicking "Deploy", scroll down to **"Advanced settings"**
2. Click on **"Secrets"**
3. Copy the content from `secrets.toml.template` and paste it
4. **Replace the placeholder values** with your actual Angel One credentials:

```toml
[angel_one]
CLIENT_ID = "YOUR_ACTUAL_CLIENT_ID"
PASSWORD = "YOUR_ACTUAL_PASSWORD"
TOTP_SECRET = "YOUR_ACTUAL_TOTP_SECRET"
API_KEY = "YOUR_ACTUAL_API_KEY"
CORRELATION_ID = "YOUR_ACTUAL_CORRELATION_ID"
SOURCE_ID = "YOUR_ACTUAL_SOURCE_ID"
```

### Step 4: Deploy!
1. Click **"Deploy!"** button
2. Wait 2-3 minutes for deployment
3. Your app will be live at: `https://ashbinbiju-s-genie-app-xxxxx.streamlit.app`

---

## 🔒 Security Checklist

✅ **Credentials NOT in GitHub** - Protected by `.gitignore`  
✅ **Secrets configured in Streamlit Cloud** - Using secure secrets management  
✅ **config_secure.py** - Reads from `st.secrets` in production  
✅ **Local development** - Falls back to `config.py` for testing

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Check that `requirements.txt` includes all packages:
```
streamlit==1.28.0
SmartApi-Python==1.3.0
pandas==2.0.3
ta==0.11.0
plotly==5.17.0
pyotp==2.9.0
numpy==1.24.3
```

### Issue: "Login Failed" or "Invalid Credentials"
**Solution:** 
1. Go to Streamlit Cloud dashboard
2. Click on your app → "Settings" → "Secrets"
3. Verify all credentials are correctly entered
4. Click "Save" and wait for app to restart

### Issue: "No Data Available"
**Solution:**
- Check market hours (9:20 AM - 3:30 PM IST)
- Try different timeframe (15min instead of 30min)
- Check if Angel One API is working (visit their website)

### Issue: App keeps restarting or crashing
**Solution:**
1. Check logs in Streamlit Cloud: Click on "Manage app" → "Logs"
2. Look for error messages
3. Common fix: Restart the app (hamburger menu → "Reboot app")

---

## 📊 Post-Deployment Testing

Once deployed, test these features:

1. **Login** - Should show "✅ Successfully logged in to Angel One"
2. **Data Loading** - Select a stock, data should load within 5-10 seconds
3. **Indicators** - Check RSI, MACD, VWAP calculations
4. **Charts** - Interactive Plotly chart should render with buy/sell signals
5. **Refresh** - Click refresh button to update data
6. **Debug Mode** - Enable in sidebar to see technical details

---

## 🎉 Success!

Your trading dashboard is now live and accessible from anywhere! 

**Share your app URL with:**
- Bookmark it for daily trading
- Access from mobile/tablet
- Share with trading team (if applicable)

---

## 📞 Need Help?

If you encounter issues:
1. Check `DEPLOYMENT_GUIDE.md` for detailed instructions
2. Check `DATA_SYNC_GUIDE.md` for TradingView sync issues
3. Review Streamlit Cloud logs for error messages
4. Verify Angel One API credentials are correct

---

**Remember:** Never share your Streamlit Cloud app URL publicly if it contains sensitive trading data!
