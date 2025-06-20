# Railway-optimized Dockerfile with FAISS support
FROM python:3.11-slim

# Environment variables for Railway
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy full requirements for complete functionality
COPY requirements.txt .

# Install dependencies with optimizations for Railway build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py config.py database.py ./
COPY color_analysis_api.py skin_tone_analyzer.py ./
COPY phase1_supabase_outfits_generator.py ./
COPY phase2_supabase_similar_outfits_api.py ./
COPY phase3_supabase_similar_products_api.py ./
COPY start.sh ./
COPY ["Colour map.xlsx", "./"]

# Create directories and user
RUN mkdir -p data/user_recommendations && \
    adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app && \
    chmod +x start.sh

USER appuser

EXPOSE 8000

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Use direct Python for faster startup during development
CMD ["python", "app.py"] 