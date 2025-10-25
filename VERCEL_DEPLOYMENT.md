# Vercel Deployment Guide for StockGenie Pro

## 🚀 Quick Deployment Steps

### 1. Connect to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with your GitHub account
2. Click **"Add New Project"**
3. Import your repository: `Ashbinbiju/s.genie`
4. Vercel will automatically detect it's a Python project

### 2. Configure Environment Variables

⚠️ **CRITICAL**: You must add these environment variables in Vercel before deploying!

Go to **Project Settings → Environment Variables** and add:

```bash
# SmartAPI Credentials
CLIENT_ID=your_client_id
PASSWORD=your_password
TOTP_SECRET=your_totp_secret

# API Keys
HISTORICAL_API_KEY=your_historical_api_key
TRADING_API_KEY=your_trading_api_key
MARKET_API_KEY=your_market_api_key

# Telegram (Optional)
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Flask Configuration
FLASK_SECRET_KEY=generate-a-random-secret-key-here
FLASK_DEBUG=false

# Redis (Optional - recommended for production)
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_PREFIX=stockgenie:
```

### 3. Deploy

1. Click **"Deploy"**
2. Wait for the build to complete (~2-3 minutes)
3. Your app will be live at `https://your-project.vercel.app`

---

## 🔧 Advanced Configuration

### Using Redis (Recommended for Production)

For better performance, use Redis for caching:

1. Create a Redis database (free options):
   - [Upstash Redis](https://upstash.com/) - Free tier available
   - [Redis Cloud](https://redis.com/try-free/) - Free 30MB

2. Add Redis environment variables in Vercel:
   ```bash
   REDIS_ENABLED=true
   REDIS_HOST=your-redis-host.upstash.io
   REDIS_PORT=6379
   REDIS_PASSWORD=your-redis-password
   REDIS_PREFIX=stockgenie:
   ```

### Custom Domain

1. Go to **Project Settings → Domains**
2. Add your custom domain
3. Follow Vercel's DNS configuration instructions

---

## 📝 Important Notes

### Vercel Limitations

1. **Serverless Functions**: Vercel runs Python apps as serverless functions
   - Each request has a 10-second timeout on free tier
   - Background tasks may not work as expected
   - WebSocket connections have limited support

2. **Best Practices**:
   - Use Redis for caching (improves performance)
   - Keep API requests fast
   - Consider using Vercel's Edge Functions for critical endpoints

### Alternative Deployment Options

If you need long-running background processes or WebSocket support:

1. **Railway**: [railway.app](https://railway.app)
   - Better for long-running apps
   - Full WebSocket support
   - Similar deployment process

2. **Render**: [render.com](https://render.com)
   - Free tier with persistent services
   - Good WebSocket support

3. **Heroku**: [heroku.com](https://heroku.com)
   - Classic PaaS platform
   - More configuration options

---

## 🔒 Security Checklist

Before deploying:

- ✅ All secrets are in environment variables (not in code)
- ✅ `.env` file is in `.gitignore`
- ✅ `FLASK_DEBUG=false` in production
- ✅ Generate a strong `FLASK_SECRET_KEY`
- ✅ Rotate all exposed credentials (see SECURITY_NOTICE.md)

---

## 🐛 Troubleshooting

### Build Fails

**Problem**: "Could not find requirements.txt"
- **Solution**: Ensure `requirements.txt` is in the root directory

**Problem**: "Module not found"
- **Solution**: Check all dependencies are in `requirements.txt`

### App Crashes on Start

**Problem**: "KeyError: 'FLASK_SECRET_KEY'"
- **Solution**: Add all required environment variables in Vercel settings

**Problem**: "Connection timeout"
- **Solution**: Check if API endpoints are accessible from Vercel's servers

### Slow Performance

- **Solution 1**: Enable Redis caching
- **Solution 2**: Optimize API calls (use caching)
- **Solution 3**: Consider upgrading Vercel plan for better performance

---

## 📞 Support

- Check logs in Vercel Dashboard → Your Project → Logs
- Review `IMPROVEMENTS_COMPLETED.md` for recent changes
- Check `README.md` for API documentation

---

## 🎉 Post-Deployment

After successful deployment:

1. Test all endpoints:
   - `/` - Dashboard
   - `/api/market-health` - Market health
   - `/api/swing-scan` - Swing trading scan
   - `/api/intraday-scan` - Intraday scan

2. Monitor performance in Vercel Dashboard

3. Set up monitoring/alerts if needed

4. Consider adding a custom domain

---

**Your app is now live! 🚀**

Share your deployment URL: `https://your-project.vercel.app`
