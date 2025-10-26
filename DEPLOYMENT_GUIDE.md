# 🚀 Deployment Guide - GitHub & Streamlit Cloud

## Step-by-Step Deployment

### Prerequisites ✅
- GitHub account
- Git installed on your computer
- Angel One SmartAPI credentials

---

## Part 1: Push to GitHub

### 1. Initialize Git Repository

Open PowerShell in the webtest directory:

```powershell
cd e:\abcd\webtest
git init
```

### 2. Configure Git (First time only)

```powershell
git config --global user.name "Ashbinbiju"
git config --global user.email "your-email@example.com"
```

### 3. Add Files to Git

```powershell
git add .
git status  # Check what will be committed
```

### 4. Create Initial Commit

```powershell
git commit -m "Initial commit: Advanced Intraday Trading System"
```

### 5. Add GitHub Remote

```powershell
git remote add origin https://github.com/Ashbinbiju/s.genie.git
git branch -M main
```

### 6. Push to GitHub

```powershell
git push -u origin main
```

If you get an authentication error, you may need to:
- Create a Personal Access Token on GitHub
- Use: `git remote set-url origin https://YOUR_TOKEN@github.com/Ashbinbiju/s.genie.git`

---

## Part 2: Deploy on Streamlit Cloud

### 1. Go to Streamlit Cloud

Visit: https://share.streamlit.io

### 2. Sign In

- Click "Sign in"
- Choose "Continue with GitHub"
- Authorize Streamlit

### 3. Deploy New App

Click **"New app"** button

### 4. Configure Deployment

Fill in:
- **Repository:** `Ashbinbiju/s.genie`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL:** Choose custom URL like `trading-dashboard` or keep default

### 5. Click "Deploy!"

Wait 2-3 minutes for deployment...

---

## Part 3: Configure Secrets (IMPORTANT!)

### 1. Open App Settings

After deployment:
- Click the hamburger menu (⋮) on your app
- Select "Settings"

### 2. Go to Secrets Tab

Click on "Secrets" in the left sidebar

### 3. Add Your Credentials

Copy and paste this into the secrets box:

```toml
[angel_one]
CLIENT_ID = "AAAG399109"
PASSWORD = "1503"
TOTP_SECRET = "OLRQ3CYBLPN2XWQPHLKMB7WEKI"
HISTORICAL_API_KEY = "c3C0tMGn"
TRADING_API_KEY = "ruseeaBq"
MARKET_API_KEY = "PflRFXyd"
```

### 4. Save Secrets

Click "Save" button

### 5. Reboot App

App will automatically restart with new secrets

---

## Part 4: Verify Deployment

### 1. Wait for Build

Watch the logs - should see:
```
✓ App is ready
✓ You can now view your app
```

### 2. Test Login

- Open your app URL
- Click "Login to Angel One"
- Should see "✅ Connected"

### 3. Test Functionality

- Select a stock (TATASTEEL)
- Choose timeframe (30min)
- Verify data loads
- Check signals appear

---

## Your App URLs

After deployment, you'll have:

**Public URL:** 
`https://ashbinbiju-s-genie-app-xxxxx.streamlit.app`

**Custom URL (if available):**
`https://trading-dashboard.streamlit.app`

---

## Updating Your Deployment

### For Code Changes:

```powershell
cd e:\abcd\webtest

# Make your changes to files

git add .
git commit -m "Description of changes"
git push origin main
```

Streamlit Cloud will **auto-deploy** within 1-2 minutes! 🚀

### For Secret Changes:

1. Go to app settings
2. Click "Secrets"
3. Update values
4. Click "Save"
5. App reboots automatically

---

## Troubleshooting

### Issue: Git push denied ❌

**Solution:**
```powershell
# Create Personal Access Token on GitHub
# Settings → Developer settings → Personal access tokens → Generate new token
# Use token as password or:
git remote set-url origin https://YOUR_TOKEN@github.com/Ashbinbiju/s.genie.git
git push origin main
```

### Issue: Streamlit build fails ❌

**Check:**
- Requirements.txt has all dependencies
- No syntax errors in Python files
- Secrets are properly configured

**View logs:**
Click "Manage app" → "Logs" to see error details

### Issue: Login fails on cloud ❌

**Check:**
1. Secrets are properly set
2. TOTP secret is correct
3. Angel One API keys are active
4. No extra spaces in secrets.toml

### Issue: App crashes on data fetch ❌

**Possible causes:**
- Market is closed
- API rate limits
- Invalid symbol tokens

**Solution:**
- Check logs for exact error
- Try during market hours
- Verify symbol tokens in config

---

## Security Best Practices

### ✅ DO:
- Use Streamlit Secrets for credentials
- Keep secrets.toml out of Git (.gitignore)
- Use environment variables locally
- Regularly rotate API keys

### ❌ DON'T:
- Commit credentials to GitHub
- Share your app URL with untrusted users
- Use production keys in public repos
- Store passwords in plain text files

---

## Monitoring Your App

### View Logs:
```
Manage app → Logs
```

### View Analytics:
```
Manage app → Analytics
```

### Reboot App:
```
Manage app → Reboot app
```

### Update Settings:
```
Manage app → Settings
```

---

## Sharing Your App

### Public Access:
Your app URL is public by default. Anyone can access it.

### Private Access:
Upgrade to Streamlit Cloud Teams for:
- Password protection
- Private repos
- More resources
- Priority support

---

## Next Steps

### 1. Test Thoroughly ✅
- Paper trade for 1-2 weeks
- Monitor all signals
- Verify data accuracy

### 2. Customize ✅
- Add more stocks
- Adjust parameters
- Enhance UI

### 3. Monitor Performance ✅
- Track win rates
- Calculate P&L
- Optimize strategy

### 4. Share ✅
- Add to portfolio
- Share on LinkedIn
- Get feedback

---

## Cost

**Streamlit Cloud:**
- ✅ FREE tier: 1 app, public repos
- 💰 Teams: $250/month (private repos, more apps)

**Angel One API:**
- ✅ FREE for trading clients
- Data charges may apply

---

## Support

### Streamlit Issues:
- Docs: https://docs.streamlit.io
- Forum: https://discuss.streamlit.io
- GitHub: https://github.com/streamlit/streamlit

### Angel One API Issues:
- Docs: https://smartapi.angelbroking.com/docs
- Support: support@angelbroking.com

### Repository Issues:
- GitHub Issues: https://github.com/Ashbinbiju/s.genie/issues

---

## Success Checklist ✓

Before going live:

- [ ] Code pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] Secrets configured correctly
- [ ] Login working
- [ ] Data fetching properly
- [ ] Signals generating correctly
- [ ] Charts rendering
- [ ] Auto-refresh working
- [ ] Tested with multiple stocks
- [ ] Verified against TradingView
- [ ] Documented any custom changes
- [ ] README updated with app URL

---

## 🎉 Congratulations!

Your trading dashboard is now live on the internet! 🚀

**Next:** Start paper trading and monitor performance!

**Remember:** This is a demo system. Always use proper risk management when trading real money.

---

**Questions?** Open an issue on GitHub or check the documentation.

**Happy Trading! 📈💰**
