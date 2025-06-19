# Railway-optimized Dockerfile
FROM python:3.11-slim

# Environment variables for Railway
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for faster build
COPY requirements-minimal.txt requirements.txt

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Use direct Python for faster startup during development
CMD ["python", "app.py"] 