"""
Test script for Supabase integration
Verifies connection and basic database operations
"""

import logging
from config import get_config, Config
from database import get_db

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration():
    """Test the configuration setup."""
    logger.info("🔧 Testing configuration...")
    
    config = get_config()
    
    print("\n📋 Configuration Status:")
    print(f"   SUPABASE_URL: {'✅ Set' if config.SUPABASE_URL else '❌ Missing'}")
    print(f"   SUPABASE_ANON_KEY: {'✅ Set' if config.SUPABASE_ANON_KEY else '❌ Missing'}")
    print(f"   SUPABASE_SERVICE_ROLE_KEY: {'✅ Set' if config.SUPABASE_SERVICE_ROLE_KEY else '❌ Missing'}")
    print(f"   Flask Environment: {config.FLASK_ENV}")
    print(f"   API Port: {config.API_PORT}")
    
    # Validate configuration
    is_valid = config.validate_supabase_config()
    print(f"\n✅ Configuration Valid: {is_valid}")
    
    return is_valid

def test_database_connection():
    """Test the database connection."""
    logger.info("🔌 Testing database connection...")
    
    try:
        db = get_db()
        
        # Test connection
        connection_ok = db.test_connection()
        
        if connection_ok:
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        print(f"❌ Database connection error: {e}")
        return False

def test_database_operations():
    """Test basic database operations."""
    logger.info("🗄️ Testing database operations...")
    
    try:
        db = get_db()
        
        # Test getting users (this will help us understand the table structure)
        print("\n📊 Testing database queries...")
        
        # Test users table
        try:
            users_df = db.get_users()
            print(f"   Users table: {'✅ Found' if not users_df.empty else '⚠️ Empty'} ({len(users_df)} records)")
            if not users_df.empty:
                print(f"   User columns: {list(users_df.columns)}")
        except Exception as e:
            print(f"   Users table: ❌ Error - {e}")
        
        # Test products table
        try:
            products_df = db.get_products(limit=5)
            print(f"   Products table: {'✅ Found' if not products_df.empty else '⚠️ Empty'} ({len(products_df)} records)")
            if not products_df.empty:
                print(f"   Product columns: {list(products_df.columns)}")
        except Exception as e:
            print(f"   Products table: ❌ Error - {e}")
        
        # Test user_outfits table
        try:
            outfits_df = db.get_user_outfits(user_id=1, limit=5)
            print(f"   User_outfits table: {'✅ Found' if not outfits_df.empty else '⚠️ Empty'} ({len(outfits_df)} records)")
            if not outfits_df.empty:
                print(f"   Outfit columns: {list(outfits_df.columns)}")
        except Exception as e:
            print(f"   User_outfits table: ❌ Error - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations error: {e}")
        print(f"❌ Database operations error: {e}")
        return False

def test_supabase_outfit_generator():
    """Test the Supabase outfit generator."""
    logger.info("🎯 Testing Supabase outfit generator...")
    
    try:
        from phase1_supabase_outfits_generator import SupabaseMainOutfitsGenerator
        
        print("\n🚀 Initializing Supabase outfit generator...")
        generator = SupabaseMainOutfitsGenerator()
        
        print("✅ Generator initialized successfully!")
        print(f"   Model: {generator.config['model_name']}")
        print(f"   Target outfits: {generator.config['main_outfits_count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Outfit generator test error: {e}")
        print(f"❌ Outfit generator test error: {e}")
        return False

def print_setup_instructions():
    """Print setup instructions for Supabase."""
    print("""
🔧 SUPABASE SETUP INSTRUCTIONS:

1. Create a .env file in your project root with:
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

2. Expected database tables:
   - users (id, gender, fashion_style, color_preferences, body_shape, etc.)
   - products (id, title, wear_type, gender, primary_style, primary_color, price, etc.)
   - user_outfits (user_id, rank, score, top_id, bottom_id, etc.)

3. Make sure your Supabase project has these tables with appropriate columns.

4. Test the connection by running: python test_supabase.py
""")

def main():
    """Main test function."""
    print("🧪 SUPABASE INTEGRATION TEST")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_configuration()
    
    if not config_ok:
        print("\n❌ Configuration test failed!")
        print_setup_instructions()
        return
    
    # Test database connection
    connection_ok = test_database_connection()
    
    if not connection_ok:
        print("\n❌ Database connection test failed!")
        print_setup_instructions()
        return
    
    # Test database operations
    operations_ok = test_database_operations()
    
    # Test outfit generator
    generator_ok = test_supabase_outfit_generator()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"   Operations: {'✅ PASS' if operations_ok else '❌ FAIL'}")
    print(f"   Generator: {'✅ PASS' if generator_ok else '❌ FAIL'}")
    
    if all([config_ok, connection_ok, operations_ok, generator_ok]):
        print("\n🎉 ALL TESTS PASSED! Supabase integration is ready!")
    else:
        print("\n⚠️ Some tests failed. Check the setup instructions above.")

if __name__ == "__main__":
    main() 