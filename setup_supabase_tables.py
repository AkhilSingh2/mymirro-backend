"""
Setup script for Supabase tables
Creates the user_outfits table and any other missing structures
"""

import logging
from database import get_db

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_user_outfits_table():
    """Create the user_outfits table in Supabase."""
    logger.info("🔧 Setting up user_outfits table...")
    
    db = get_db()
    
    if not db.client:
        logger.error("❌ Database connection failed")
        return False
    
    try:
        # SQL to create the user_outfits table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_outfits (
            id SERIAL PRIMARY KEY,
            main_outfit_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            score FLOAT NOT NULL,
            explanation TEXT,
            
            -- Top product details
            top_id TEXT,
            top_title TEXT,
            top_image TEXT,
            top_price FLOAT,
            top_style TEXT,
            top_color TEXT,
            top_semantic_score FLOAT,
            
            -- Bottom product details
            bottom_id TEXT,
            bottom_title TEXT,
            bottom_image TEXT,
            bottom_price FLOAT,
            bottom_style TEXT,
            bottom_color TEXT,
            bottom_semantic_score FLOAT,
            
            -- Combined details
            total_price FLOAT,
            generated_at TIMESTAMP DEFAULT NOW(),
            generation_method TEXT DEFAULT 'supabase_faiss_semantic',
            
            -- Constraints
            UNIQUE(user_id, rank),
            FOREIGN KEY (user_id) REFERENCES users_updated(id) ON DELETE CASCADE
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_user_outfits_user_id ON user_outfits(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_outfits_rank ON user_outfits(rank);
        CREATE INDEX IF NOT EXISTS idx_user_outfits_score ON user_outfits(score);
        """
        
        # Execute the SQL using Supabase RPC (if available) or try to create via insert
        # Note: Direct SQL execution might need to be done through Supabase dashboard
        # For now, let's try to create a dummy record to test if table exists
        
        try:
            # Try to query the table first
            result = db.client.table('user_outfits').select('id').limit(1).execute()
            logger.info("✅ user_outfits table already exists")
            return True
            
        except Exception as e:
            if "does not exist" in str(e):
                logger.info("📋 user_outfits table does not exist. Creating it...")
                
                # Since we can't execute DDL directly, let's create a sample structure
                # and inform the user to run the SQL manually
                print("\n" + "="*60)
                print("🔧 MANUAL SETUP REQUIRED")
                print("="*60)
                print("Please run this SQL in your Supabase SQL Editor:")
                print()
                print(create_table_sql)
                print("="*60)
                
                return False
            else:
                logger.error(f"❌ Error checking user_outfits table: {e}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error setting up user_outfits table: {e}")
        return False

def test_table_creation():
    """Test if we can insert a sample outfit record."""
    logger.info("🧪 Testing user_outfits table...")
    
    db = get_db()
    
    try:
        # Try to insert a test record
        test_outfit = {
            'main_outfit_id': 'test_outfit_1',
            'user_id': 1,
            'rank': 999,  # Use high rank to avoid conflicts
            'score': 0.85,
            'explanation': 'Test outfit',
            'top_id': 'test_top',
            'top_title': 'Test Top',
            'top_price': 1000.0,
            'bottom_id': 'test_bottom',
            'bottom_title': 'Test Bottom',
            'bottom_price': 1500.0,
            'total_price': 2500.0
        }
        
        # Try to insert
        result = db.client.table('user_outfits').insert(test_outfit).execute()
        
        if result.data:
            logger.info("✅ Successfully inserted test outfit")
            
            # Clean up - delete the test record
            db.client.table('user_outfits').delete().eq('rank', 999).execute()
            logger.info("✅ Test record cleaned up")
            
            return True
        else:
            logger.error("❌ Failed to insert test outfit")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing user_outfits table: {e}")
        return False

def check_product_categories():
    """Check what product categories we have to understand the data structure."""
    logger.info("📊 Analyzing product categories...")
    
    db = get_db()
    
    try:
        products_df = db.get_products(limit=50)
        
        if products_df.empty:
            logger.warning("⚠️ No products found")
            return
        
        # Analyze categories
        if 'category' in products_df.columns:
            categories = products_df['category'].value_counts()
            print("\n📋 Product Categories:")
            for category, count in categories.items():
                print(f"   {category}: {count} products")
        
        # Show sample products
        print("\n📦 Sample Products:")
        for idx, row in products_df.head(5).iterrows():
            print(f"   {row.get('title', 'No title')} | {row.get('category', 'No category')} | ₹{row.get('price', 0)}")
            
    except Exception as e:
        logger.error(f"❌ Error analyzing products: {e}")

def main():
    """Main setup function."""
    print("🚀 SUPABASE TABLE SETUP")
    print("=" * 50)
    
    # Check product categories first
    check_product_categories()
    
    # Try to create user_outfits table
    table_created = create_user_outfits_table()
    
    if table_created:
        # Test the table
        test_success = test_table_creation()
        
        if test_success:
            print("\n✅ ALL SETUP COMPLETE!")
            print("🎯 Ready to generate outfits!")
        else:
            print("\n⚠️ Table exists but testing failed")
    else:
        print("\n⚠️ Manual setup required - see SQL above")

if __name__ == "__main__":
    main() 