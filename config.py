"""
Configuration module for MyMirro Backend
Handles Supabase connection and application settings
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY') 
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    # Fashion AI Configuration
    MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    MAIN_OUTFITS_COUNT = int(os.getenv('MAIN_OUTFITS_COUNT', 100))
    TOPS_PER_OUTFIT = int(os.getenv('TOPS_PER_OUTFIT', 20))
    BOTTOMS_PER_OUTFIT = int(os.getenv('BOTTOMS_PER_OUTFIT', 20))
    
    # Data paths (for backward compatibility)
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    USER_FILE = os.getenv('USER_FILE', 'data/User_Data.xlsx')
    PRODUCTS_FILE = os.getenv('PRODUCTS_FILE', 'data/local_processed_results_enhanced.csv')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/user_recommendations')
    
    @classmethod
    def validate_supabase_config(cls) -> bool:
        """Validate that required Supabase configuration is present."""
        required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print(f"❌ Missing required Supabase configuration: {', '.join(missing_vars)}")
            print("Please set these environment variables or create a .env file")
            return False
        
        return True
    
    @classmethod
    def get_supabase_config(cls) -> Dict[str, str]:
        """Get Supabase configuration as a dictionary."""
        return {
            'url': cls.SUPABASE_URL,
            'anon_key': cls.SUPABASE_ANON_KEY,
            'service_role_key': cls.SUPABASE_SERVICE_ROLE_KEY
        }

# Development configuration with fallbacks
class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    TESTING = False
    
    # Use service role key for development if available
    @classmethod
    def get_supabase_key(cls) -> str:
        """Get the appropriate Supabase key for development."""
        return cls.SUPABASE_SERVICE_ROLE_KEY or cls.SUPABASE_ANON_KEY

# Production configuration
class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    TESTING = False
    
    @classmethod
    def get_supabase_key(cls) -> str:
        """Get the appropriate Supabase key for production."""
        return cls.SUPABASE_ANON_KEY

# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config() -> Config:
    """Get the appropriate configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig) 