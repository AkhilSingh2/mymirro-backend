#!/usr/bin/env python3
"""
Precompute and store product embeddings in the database
This script computes embeddings for all products and stores them in the tagged_products table
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Graceful imports for optional ML dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import database functionality
from database import get_db
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingPrecomputer:
    """Precompute and store product embeddings in the database."""
    
    def __init__(self):
        """Initialize the embedding precomputer."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embedding precomputation")
        
        # Load configuration
        app_config = get_config()
        self.model_name = app_config.MODEL_NAME
        
        # Initialize database connection
        self.db = get_db()
        if not self.db.test_connection():
            raise ConnectionError("Failed to connect to Supabase database")
        
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Model loaded successfully")
        
        # Batch processing settings
        self.batch_size = 50
        self.max_products = None  # Set to None to process all products
    
    def get_all_products(self) -> pd.DataFrame:
        """Get all products from the database for embedding precomputation."""
        try:
            logger.info("Loading all products from database...")
            
            # Use direct query instead of chunked loading to avoid timeouts
            result = self.db.client.table('tagged_products').select('*').execute()
            
            if result.data:
                products_df = pd.DataFrame(result.data)
                logger.info(f"✅ Loaded {len(products_df)} products from database")
                return products_df
            else:
                logger.warning("No products found in database")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading products: {e}")
            return pd.DataFrame()
    
    def compute_embeddings_batch(self, captions: List[str]) -> np.ndarray:
        """Compute embeddings for a batch of captions."""
        try:
            embeddings = self.model.encode(captions, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise
    
    def precompute_embeddings(self) -> Dict:
        """Precompute embeddings for all products and store in database."""
        try:
            # Get all products
            products_df = self.get_all_products()
            
            if products_df.empty:
                logger.warning("No products found in database")
                return {"status": "no_products", "processed": 0}
            
            # Limit products if specified
            if self.max_products:
                products_df = products_df.head(self.max_products)
                logger.info(f"Processing first {len(products_df)} products")
            
            # Prepare captions for embedding
            captions = []
            product_ids = []
            valid_indices = []
            
            for idx, row in products_df.iterrows():
                # Use full_caption as primary, fallback to final_caption, then title
                caption = row.get('full_caption', '') or row.get('final_caption', '') or row.get('title', '')
                
                if caption.strip():
                    captions.append(caption)
                    product_ids.append(str(row['id']))
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Product {row['id']} has no valid caption")
            
            logger.info(f"Computing embeddings for {len(captions)} products with valid captions")
            
            # Compute embeddings in batches
            all_embeddings = []
            processed_count = 0
            
            for i in range(0, len(captions), self.batch_size):
                batch_captions = captions[i:i + self.batch_size]
                batch_product_ids = product_ids[i:i + self.batch_size]
                
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(captions) + self.batch_size - 1)//self.batch_size}")
                
                # Compute embeddings for this batch
                batch_embeddings = self.compute_embeddings_batch(batch_captions)
                all_embeddings.extend(batch_embeddings)
                
                # Store embeddings in database for this batch
                self._store_embeddings_batch(batch_product_ids, batch_embeddings)
                processed_count += len(batch_captions)
                
                logger.info(f"✅ Processed {processed_count}/{len(captions)} products")
            
            logger.info(f"🎉 Successfully precomputed embeddings for {processed_count} products")
            
            return {
                "status": "success",
                "processed": processed_count,
                "total_products": len(products_df),
                "valid_captions": len(captions)
            }
            
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {e}")
            raise
    
    def _store_embeddings_batch(self, product_ids: List[str], embeddings: List[np.ndarray]) -> None:
        """Store embeddings for a batch of products in the database."""
        try:
            # Prepare data for database update
            update_data = []
            
            for product_id, embedding in zip(product_ids, embeddings):
                # Convert embedding to JSON string for storage
                embedding_json = json.dumps(embedding.tolist())
                
                update_data.append({
                    'id': product_id,
                    'product_embedding': embedding_json,
                    'embedding_updated_at': datetime.now().isoformat()
                })
            
            # Update database in batches
            batch_size = 10  # Smaller batch size for database updates
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i:i + batch_size]
                
                # Update each product individually to avoid conflicts
                for item in batch:
                    try:
                        result = self.db.client.table('tagged_products').update({
                            'product_embedding': item['product_embedding'],
                            'embedding_updated_at': item['embedding_updated_at']
                        }).eq('id', item['id']).execute()
                        
                        if not result.data:
                            logger.warning(f"No rows updated for product {item['id']}")
                            
                    except Exception as e:
                        if "column tagged_products.product_embedding does not exist" in str(e):
                            logger.error(f"❌ Database schema error: product_embedding column does not exist!")
                            logger.error(f"Please run the SQL commands in add_embedding_columns.sql in your Supabase SQL Editor")
                            raise Exception("Database schema needs to be updated. Please add the embedding columns first.")
                        elif "embedding_updated_at" in str(e):
                            logger.error(f"❌ Database schema error: embedding_updated_at column does not exist!")
                            logger.error(f"Please run the SQL commands in add_embedding_columns.sql in your Supabase SQL Editor")
                            raise Exception("Database schema needs to be updated. Please add the embedding columns first.")
                        else:
                            logger.error(f"Error updating product {item['id']}: {e}")
                        continue
            
            logger.info(f"✅ Stored embeddings for {len(product_ids)} products in database")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def verify_embeddings(self) -> Dict:
        """Verify that embeddings are stored correctly in the database."""
        try:
            logger.info("Verifying stored embeddings...")
            
            # Get count of products with embeddings
            result = self.db.client.table('tagged_products').select('id, product_embedding').not_.is_('product_embedding', 'null').execute()
            
            products_with_embeddings = len(result.data)
            
            # Get total product count
            total_result = self.db.client.table('tagged_products').select('id').execute()
            total_products = len(total_result.data)
            
            logger.info(f"📊 Embedding verification results:")
            logger.info(f"  - Total products: {total_products}")
            logger.info(f"  - Products with embeddings: {products_with_embeddings}")
            logger.info(f"  - Coverage: {products_with_embeddings/total_products*100:.1f}%")
            
            return {
                "total_products": total_products,
                "products_with_embeddings": products_with_embeddings,
                "coverage_percentage": products_with_embeddings/total_products*100
            }
            
        except Exception as e:
            logger.error(f"Error verifying embeddings: {e}")
            raise

def main():
    """Main function to run embedding precomputation."""
    try:
        logger.info("🚀 Starting embedding precomputation...")
        
        # Initialize precomputer
        precomputer = EmbeddingPrecomputer()
        
        # Precompute embeddings
        result = precomputer.precompute_embeddings()
        
        if result["status"] == "success":
            logger.info("✅ Embedding precomputation completed successfully!")
            
            # Verify embeddings
            verification = precomputer.verify_embeddings()
            
            logger.info("🎉 All done! Embeddings are now stored in the database.")
            logger.info("You can now use the optimized Phase 2 API with precomputed embeddings.")
            
        else:
            logger.warning(f"⚠️ Precomputation completed with status: {result['status']}")
            
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 