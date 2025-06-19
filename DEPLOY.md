# 🚀 Railway Deployment Guide

Complete step-by-step guide to deploy MyMirror Backend on Railway.

## 📋 Prerequisites

- [Railway](https://railway.app) account
- [Supabase](https://supabase.com) project
- GitHub account
- Git installed locally

## 🛠️ Step 1: Prepare Your Repository

### 1.1 Fork the Repository
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mymirro-backend.git
cd mymirro-backend
```

### 1.2 Verify Project Structure
Ensure these files exist:
- ✅ `Dockerfile`
- ✅ `railway.toml`
- ✅ `start.sh`
- ✅ `requirements.txt`
- ✅ `env.example`

## 🗄️ Step 2: Setup Supabase Database

### 2.1 Create Supabase Project
1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Wait for setup to complete (~2 minutes)

### 2.2 Get Your Credentials
From your Supabase dashboard:
- **URL**: `https://your-project-ref.supabase.co`
- **Anon Key**: Settings → API → `anon` public key
- **Service Role Key**: Settings → API → `service_role` secret key

### 2.3 Setup Tables (Optional)
Run the setup script if needed:
```bash
python setup_supabase_tables.py
```

## 🚂 Step 3: Deploy to Railway

### 3.1 Connect Repository
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your forked repository

### 3.2 Configure Environment Variables
In Railway dashboard, go to **Variables** tab and add:

```bash
# Required
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
SECRET_KEY=your_random_secret_key_here

# Production Settings
FLASK_ENV=production
FLASK_DEBUG=false

# AI Configuration
MODEL_NAME=all-MiniLM-L6-v2
MAIN_OUTFITS_COUNT=100
TOPS_PER_OUTFIT=20
BOTTOMS_PER_OUTFIT=20

# Optional
API_HOST=0.0.0.0
```

### 3.3 Generate Secret Key
Use this Python snippet to generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3.4 Deploy
1. Railway will automatically detect your `Dockerfile`
2. Click **"Deploy"** if it doesn't start automatically
3. Wait 5-10 minutes for the build to complete

## ✅ Step 4: Verify Deployment

### 4.1 Check Health
Your app will be available at: `https://your-project-name.railway.app`

Test the health endpoint:
```bash
curl https://your-project-name.railway.app/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "environment": "production"
}
```

### 4.2 Test API Documentation
Visit: `https://your-project-name.railway.app/swagger/`

### 4.3 Test Color Analysis
```bash
curl -X POST https://your-project-name.railway.app/api/v1/color/analyze \
  -H "Content-Type: application/json" \
  -d '{"hex_color": "#FDB4A6"}'
```

### 4.4 Test Similar Products
```bash
curl -X POST https://your-project-name.railway.app/api/v1/products/3790/similar \
  -H "Content-Type: application/json" \
  -d '{
    "user_preferences": {
      "preferred_styles": ["Business Formal"],
      "preferred_colors": ["Black", "Navy"],
      "price_range": [800, 2500]
    }
  }'
```

## 🔧 Step 5: Configuration & Optimization

### 5.1 Custom Domain (Optional)
1. In Railway dashboard: Settings → Domains
2. Add your custom domain
3. Configure DNS as instructed

### 5.2 Monitor Resources
- Check **Metrics** tab for CPU/Memory usage
- Monitor **Logs** tab for errors
- Set up **Alerts** if needed

### 5.3 Scaling
- **Starter Plan**: 512MB RAM, 1GB storage
- **Pro Plan**: 8GB RAM, 100GB storage
- Scale up if you see memory issues

## 🐛 Troubleshooting

### Issue: Build Fails
**Solution**: Check the **Logs** tab for specific errors:
```bash
# Common issues:
- Missing environment variables
- Dockerfile syntax errors
- Requirements.txt conflicts
```

### Issue: "Missing Supabase configuration"
**Solution**: Verify environment variables:
1. Check Variables tab in Railway
2. Ensure no extra spaces in keys/values
3. Restart deployment after changes

### Issue: Models Downloading Slowly
**Solution**: 
- First deployment takes longer (~10-15 minutes)
- Models (~500MB) are downloaded on first run
- Subsequent deployments are faster

### Issue: Memory Errors
**Solution**:
- Upgrade Railway plan
- Optimize model loading in code
- Monitor memory usage in Metrics

### Issue: App Not Responding
**Solution**:
```bash
# Check logs
railway logs

# Restart deployment
railway redeploy
```

## 📊 Performance Tips

### 5.1 Optimize for Production
- Models are cached after first load
- Use `gunicorn` for better concurrency
- Monitor response times in Railway metrics

### 5.2 Database Optimization
- Index frequently queried columns in Supabase
- Use connection pooling
- Monitor query performance

### 5.3 Memory Management
- Railway automatically manages memory
- Consider upgrading if hitting limits
- Monitor memory usage trends

## 🔄 Continuous Deployment

### Automatic Deploys
Railway automatically deploys when you push to main branch:
```bash
git add .
git commit -m "Update API"
git push origin main
# Railway will automatically deploy
```

### Manual Deploy
Force a redeploy in Railway dashboard:
1. Go to your project
2. Click **"Redeploy"**
3. Wait for completion

## 🎯 Next Steps

1. **Monitor**: Set up alerts for downtime
2. **Scale**: Upgrade plan based on usage
3. **Secure**: Add rate limiting and authentication
4. **Optimize**: Profile and optimize slow endpoints
5. **Backup**: Regular Supabase backups

## 📞 Support

- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Supabase**: [supabase.com/docs](https://supabase.com/docs)
- **Project Issues**: GitHub Issues tab

---

**🎉 Congratulations! Your MyMirror Backend is now live on Railway!** 