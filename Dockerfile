# Multi-stage Railway-optimized Dockerfile for size optimization
FROM python:3.11-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements in builder stage
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - much smaller
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy app files
COPY app.py config.py database.py ./
COPY color_analysis_api.py skin_tone_analyzer.py ./
COPY phase1_supabase_outfits_generator.py ./
COPY phase2_supabase_similar_outfits_api.py ./
COPY phase3_supabase_similar_products_api.py ./
COPY start.sh ./
COPY ["Colour map.xlsx", "./"]

# Create directories and set permissions
RUN mkdir -p data/user_recommendations && \
    chown -R appuser:appuser /app && \
    chmod +x start.sh

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "app.py"] 