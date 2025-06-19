#!/usr/bin/env python3
"""
Test Enhanced Database Functionality
Tests the new user data mapping and style quiz integration
"""

import sys
import json
from database import get_db
from phase1_supabase_outfits_generator import SupabaseMainOutfitsGenerator

def test_database_connection():
    """Test basic database connectivity"""
    print("🔍 Testing database connection...")
    
    db = get_db()
    if db.client:
        print("✅ Database connection successful")
        return True
    else:
        print("❌ Database connection failed")
        return False

def test_user_data_mapping(user_id: int = 2):
    """Test user data mapping functionality"""
    print(f"\n🧪 Testing user data mapping for user {user_id}...")
    
    db = get_db()
    
    # Test the mapping function
    test_results = db.test_user_data_mapping(user_id)
    
    print(f"📊 Test Results:")
    print(f"   User ID: {test_results.get('user_id')}")
    print(f"   Mapping Success: {test_results.get('mapping_success')}")
    print(f"   Has Style Quiz: {test_results.get('has_style_quiz')}")
    
    # Check required fields
    required_fields = test_results.get('required_fields_present', {})
    print(f"   Required Fields Present:")
    for field, present in required_fields.items():
        status = "✅" if present else "❌"
        print(f"      {status} {field}")
    
    # Show mapped data structure
    mapped_data = test_results.get('mapped_data', {})
    if mapped_data:
        print(f"\n📋 Mapped User Data Fields ({len(mapped_data)} total):")
        for key in sorted(mapped_data.keys()):
            value = mapped_data[key]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"      {key}: {value}")
    
    return test_results

def test_outfit_generation(user_id: int = 2):
    """Test the complete outfit generation pipeline"""
    print(f"\n🎯 Testing enhanced outfit generation for user {user_id}...")
    
    try:
        generator = SupabaseMainOutfitsGenerator()
        
        # Test user data loading
        print("📥 Testing user data loading...")
        user_data = generator.load_user_data_enhanced(user_id)
        
        if user_data:
            print(f"✅ User data loaded successfully")
            print(f"   User: {user_data.get('User')}")
            print(f"   Gender: {user_data.get('Gender')}")
            print(f"   Fashion Style: {user_data.get('Fashion Style', 'Not specified')}")
            print(f"   Body Shape: {user_data.get('Body Shape', 'Not specified')}")
            print(f"   Upper Wear Caption: {user_data.get('Upper Wear Caption', 'Not provided')[:60]}...")
        else:
            print("❌ Failed to load user data")
            return False
        
        # Test outfit generation (just the data loading part)
        print("\n🔄 Testing outfit generation process...")
        success = generator.generate_and_save_outfits(user_id)
        
        if success:
            print("✅ Outfit generation completed successfully")
            return True
        else:
            print("❌ Outfit generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in outfit generation test: {e}")
        return False

def test_table_creation():
    """Instructions for creating the enhanced table"""
    print(f"\n📋 Database Table Setup Instructions:")
    print(f"   1. Drop the existing user_outfits table (if it exists)")
    print(f"   2. Run the SQL script: create_user_outfits_table_enhanced.sql")
    print(f"   3. This will create a new table with all required columns")
    print(f"   4. The new table supports all enhanced features from the main generator")

def main():
    """Run all tests"""
    print("🚀 Enhanced Database Testing Suite")
    print("=" * 50)
    
    # Test 1: Database connection
    if not test_database_connection():
        print("❌ Database connection failed. Please check your configuration.")
        return
    
    # Test 2: User data mapping
    user_id = 2  # Test with user 2
    mapping_results = test_user_data_mapping(user_id)
    
    if not mapping_results.get('mapping_success'):
        print("❌ User data mapping failed. Please check your database schema.")
        return
    
    # Test 3: Show table creation instructions
    test_table_creation()
    
    # Test 4: Full outfit generation (optional)
    print(f"\n❓ Would you like to test the complete outfit generation? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        success = test_outfit_generation(user_id)
        if success:
            print("\n🎉 All tests passed! The enhanced system is working correctly.")
        else:
            print("\n⚠️ Outfit generation had issues. Check the error messages above.")
            print("   You may need to update the user_outfits table schema first.")
    else:
        print("\n✅ Basic tests completed successfully!")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Run the enhanced SQL script to update your database schema")
    print(f"   2. Test outfit generation with: python phase1_supabase_outfits_generator.py")
    print(f"   3. Use the Swagger API to test the endpoints")

if __name__ == "__main__":
    main() 