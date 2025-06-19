#!/bin/bash

# Production startup script for MyMirro Backend

echo "🚀 Starting MyMirro Backend API..."

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=false

# Get port from Railway environment or default to 8000
PORT=${PORT:-8000}

echo "📡 Starting on port $PORT"

# Start with gunicorn for production
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --worker-class sync \
    --worker-connections 1000 \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app 