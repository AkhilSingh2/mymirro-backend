#!/usr/bin/env python3
"""
Simple server runner that avoids terminal suspension issues
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

# Import and run the app
from app import app

if __name__ == '__main__':
    print("🚀 Starting MyMirror Backend API on port 8000...")
    print("📝 Swagger UI will be available at: http://localhost:8000/swagger/")
    print("🎨 Color analysis with Excel color map loaded!")
    
    # Run the app
    app.run(
        debug=False,
        host='0.0.0.0',
        port=8000,
        use_reloader=False,
        threaded=True
    ) 