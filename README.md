# 🎨 MyMirror Backend API

A comprehensive fashion recommendation system with AI-powered color analysis, outfit generation, and personalized styling recommendations.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

## 🚀 Features

### ✨ **Phase 1: Outfit Generation**
- **AI-Powered Outfit Creation**: Generate complete outfits using advanced machine learning
- **Personalized Recommendations**: Based on user preferences and skin tone analysis
- **Supabase Integration**: Real-time data synchronization and storage

### 🎨 **Phase 2: Similar Outfits**
- **Intelligent Outfit Matching**: Find similar outfits using semantic search
- **FAISS-Powered Search**: Fast and accurate similarity calculations
- **8-Factor Fashion Intelligence**: Style, color, formality, pattern compatibility

### 🛍️ **Phase 3: Similar Products (NEW)**
- **Same-Category Focus**: Products within the same category (shirts → shirts)
- **Color Diversity Intelligence**: Harmonious but different color suggestions
- **Design Diversity**: Different patterns, fits, and design variations
- **User Personalization**: Style, color, and price preference integration

### 🌈 **Advanced Color Analysis**
- **Dual-Mode Analysis**: Photo upload OR manual hex color selection
- **Skin Tone Detection**: Automatic skin region detection and analysis
- **Fitzpatrick Classification**: Professional skin type analysis (Type I-VI)
- **Category-Specific Recommendations**: Formal, Streetwear, Athleisure colors

## 🛠️ Tech Stack

- **Backend**: Flask, Flask-RESTX
- **AI/ML**: SentenceTransformers, FAISS, scikit-learn
- **Computer Vision**: OpenCV, Pillow
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Docker, Railway
- **Documentation**: Swagger UI

## 📚 API Documentation

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/color/analyze` | POST | Unified color analysis (photo/hex) |
| `/api/v1/outfits/generate` | POST | Generate user outfits |
| `/api/v1/outfits/{user_id}` | GET | Get user's outfits |
| `/api/v1/outfits/{outfit_id}/similar` | GET | Find similar outfits |
| `/api/v1/products/{product_id}/similar` | POST | Find similar products |

### **Interactive Documentation**
Visit `/swagger/` for complete API documentation and testing interface.

## 🐳 Quick Deploy with Railway

### **1. One-Click Deploy**
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

### **2. Manual Railway Deployment**

1. **Fork this repository**
2. **Connect to Railway**:
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and link project
   railway login
   railway link
   ```

3. **Set Environment Variables** in Railway Dashboard:
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   SECRET_KEY=your_random_secret_key
   FLASK_ENV=production
   MODEL_NAME=all-MiniLM-L6-v2
   ```

4. **Deploy**:
   ```bash
   railway up
   ```

Your API will be available at `https://your-project.railway.app`

## 🔧 Local Development

### **Prerequisites**
- Python 3.11+
- Supabase account and project
- Git

### **Setup**

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/mymirro-backend.git
   cd mymirro-backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**:
   ```bash
   cp env.example .env
   # Edit .env with your Supabase credentials
   ```

5. **Run application**:
   ```bash
   python app.py
   ```

Access the API at `http://localhost:8000`

## 🐳 Docker Development

### **Build and run locally**:
```bash
# Build image
docker build -t mymirro-backend .

# Run container
docker run -p 8000:8000 \
  -e SUPABASE_URL=your_url \
  -e SUPABASE_ANON_KEY=your_key \
  mymirro-backend
```

### **Docker Compose** (with environment file):
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
```

## 🏗️ Project Structure

```
mymirro-backend/
├── 📁 Core Application
│   ├── app.py                              # Main Flask application
│   ├── config.py                           # Configuration management
│   └── database.py                         # Supabase integration
├── 📁 AI/ML Modules
│   ├── color_analysis_api.py               # Color analysis engine
│   ├── skin_tone_analyzer.py               # Skin tone detection
│   ├── phase1_supabase_outfits_generator.py    # Outfit generation
│   ├── phase2_supabase_similar_outfits_api.py  # Similar outfits
│   └── phase3_supabase_similar_products_api.py # Similar products
├── 📁 Data
│   ├── Colour map.xlsx                     # Color mapping data
│   └── local_processed_results_enhanced.csv    # Product database
├── 📁 Deployment
│   ├── Dockerfile                          # Container configuration
│   ├── railway.toml                        # Railway deployment config
│   ├── start.sh                            # Production startup script
│   └── requirements.txt                    # Python dependencies
└── 📁 Documentation
    ├── README.md                           # This file
    └── env.example                         # Environment template
```

## 🔐 Environment Variables

### **Required Variables**
| Variable | Description | Example |
|----------|-------------|---------|
| `SUPABASE_URL` | Your Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | `eyJhbGciOiJIUzI1NiIs...` |
| `SECRET_KEY` | Flask secret key | `your-random-secret-key` |

### **Optional Variables**
| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode |
| `API_PORT` | `8000` | Server port |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | AI model for embeddings |

## 🧪 Testing the API

### **Health Check**
```bash
curl https://your-api.railway.app/api/v1/health
```

### **Color Analysis**
```bash
curl -X POST https://your-api.railway.app/api/v1/color/analyze \
  -H "Content-Type: application/json" \
  -d '{"hex_color": "#FDB4A6"}'
```

### **Similar Products**
```bash
curl -X POST https://your-api.railway.app/api/v1/products/3790/similar \
  -H "Content-Type: application/json" \
  -d '{
    "user_preferences": {
      "preferred_styles": ["Business Formal"],
      "preferred_colors": ["Black", "Navy"],
      "price_range": [800, 2500]
    }
  }'
```

## 📊 Performance & Scaling

### **Current Capabilities**
- **Response Time**: ~2-12 seconds for complex recommendations
- **Concurrent Users**: Optimized for 50+ concurrent requests
- **Database**: Supabase handles 500+ requests/second
- **AI Models**: Cached embeddings for faster responses

### **Scaling Recommendations**
- **Memory**: 2GB+ RAM recommended for production
- **CPU**: 2+ cores for optimal performance
- **Storage**: 1GB+ for model caches and temporary data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Common Issues**

**Q: "Missing Supabase configuration" error?**
A: Ensure `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set in your environment variables.

**Q: Models downloading slowly?**
A: First run downloads AI models (~500MB). Subsequent starts are faster.

**Q: Memory issues during deployment?**
A: Railway's free tier has memory limits. Consider upgrading for production use.

### **Get Help**
- 📧 Email: support@yourdomain.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/mymirro-backend/issues)
- 📖 Docs: [Full Documentation](https://your-docs-site.com)

## 🚀 What's Next?

- **Phase 4**: Cross-category recommendations (tops + bottoms)
- **Phase 5**: Seasonal and occasion-based styling
- **Phase 6**: AR/VR integration for virtual try-ons
- **Advanced Analytics**: User behavior insights and trends

---

**Built with ❤️ for the fashion-forward community** 