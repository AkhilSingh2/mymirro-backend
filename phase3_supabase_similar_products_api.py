# Phase 3 ENHANCED: Same-Category Similar Products with Supabase Integration
# Focus: Same product type with color/design diversity + user preference integration + Supabase DB
# ✅ ENHANCED: Now with Phase 2 optimizations + Precomputed embeddings + Same-category filtering

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Graceful imports for optional ML dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import Supabase database functionality
from database import get_db, SupabaseDB
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseEnhancedSimilarProductsGenerator:
    """
    Phase 3 ENHANCED: Same-category similar products with diversity, user preferences, and Supabase integration
    ✅ ENHANCED: Now with Supabase database integration + Advanced Fashion Intelligence
    ✅ ENHANCED: Now with Phase 2 optimizations + Precomputed embeddings + Same-category filtering
    """
    
    # ✅ OPTIMIZATION: Class-level model cache to avoid reloading
    _model_cache = None
    _model_cache_ready = False
    
    def __init__(self, config: Dict = None, db: SupabaseDB = None, is_railway: bool = False):
        """Initialize the Supabase-enabled enhanced similar products generator."""
        self.config = config or self._default_config()
        
        # Railway CPU optimization - delay until needed
        self.is_railway = is_railway
        if self.is_railway:
            logger.info("🏭 Railway environment detected - will apply CPU optimizations when needed")
        
        # Check for required dependencies
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available. Similar products service requires FAISS for similarity search.")
            raise ImportError("FAISS is required for similar products but not installed")
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ SentenceTransformers not available. Similar products service requires sentence-transformers.")
            raise ImportError("sentence-transformers is required for similar products but not installed")
        
        # Initialize Supabase database connection
        self.db = db
        if not self.db.test_connection():
            logger.error("❌ Database connection failed. Please check your Supabase configuration.")
            raise ConnectionError("Failed to connect to Supabase database")
        
        # ✅ OPTIMIZATION: Lazy load model only when needed
        self.model = None
        self.embedding_cache = {}
        
        # FAISS indexes for different wear types
        self.faiss_indexes = {}
        self.product_mappings = {}
        
        # Enhanced similarity configuration
        self.similarity_config = {
            'semantic_weight': 3.5,          # Core product similarity importance
            'style_weight': 2.5,             # Style matching importance  
            'color_diversity_weight': 2.0,   # Color diversity bonus
            'design_diversity_weight': 1.8,  # Design diversity bonus
            'outfit_context_weight': 2.8,    # NEW: Outfit context intelligence
            'brand_weight': 1.2,             # Brand consistency (lower for diversity)
            'user_preference_weight': 2.5,   # User preference boost
            'wear_caption_weight': 2.2,      # Lower/upper wear caption matching
            'diversity_bonus': 0.3,          # Bonus for diverse but relevant products
            'min_similarity_score': 0.15,    # Lowered for more candidates (was 0.25)
            'max_similar_products': 20,      # Maximum products to return
            'color_diversity_threshold': 0.7, # Threshold for color diversity bonus
            'confidence_threshold': 0.2,      # Minimum similarity score for inclusion (lowered from 0.3)
            'enable_diversity_filtering': False,  # Temporarily disable diversity filtering
        }
        
        # Initialize enhanced mappings (focused on same-category)
        self.color_diversity_matrix = self._initialize_color_diversity_matrix()
        self.style_compatibility = self._initialize_same_category_style_compatibility()
        self.design_variation_keywords = self._initialize_design_variation_keywords()
        
        # Load context mappings (outfit intelligence) if available
        self.context_mappings = {}
        
        self.current_product_id = None  # Track current product being processed
        
        logger.info("✅ Enhanced Similar Products Generator initialized")
        
    def _ensure_model_loaded(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            try:
                if self.is_railway:
                    logger.info("🔧 Loading model with Railway CPU optimizations")
                
                # ✅ OPTIMIZATION: Use cached model if available
                if SupabaseEnhancedSimilarProductsGenerator._model_cache is not None:
                    self.model = SupabaseEnhancedSimilarProductsGenerator._model_cache
                    logger.info(f"✅ Using cached model: {self.config['model_name']}")
                else:
                    self.model = SentenceTransformer(self.config['model_name'])
                    # Cache the model for future use
                    SupabaseEnhancedSimilarProductsGenerator._model_cache = self.model
                    SupabaseEnhancedSimilarProductsGenerator._model_cache_ready = True
                    logger.info(f"✅ Model loaded and cached: {self.config['model_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
        
    def _default_config(self) -> Dict:
        """Enhanced default configuration for same-category focus with Supabase."""
        app_config = get_config()
        return {
            'model_name': app_config.MODEL_NAME,
            'cache_embeddings': True,
            'batch_size': 50,
            'search_expansion_factor': 2.5,  # More candidates for better diversity
            'enable_color_diversity': True,   # NEW: Encourage color diversity
            'enable_design_diversity': True,  # NEW: Encourage design diversity
            'enable_context_intelligence': True,  # NEW: Use outfit context intelligence
            'same_category_only': True,       # NEW: Strict same-category only
            'diversity_threshold': 0.2,      # Similarity threshold for diversity filtering (lowered from 0.4)
        }
    
    def _initialize_color_diversity_matrix(self) -> Dict:
        """Initialize color diversity scoring for same-category products."""
        return {
            'Black': {
                'diverse_colors': ['White', 'Gray', 'Navy', 'Red', 'Blue'],
                'similar_colors': ['Dark Gray', 'Charcoal'],
                'avoid_colors': []
            },
            'White': {
                'diverse_colors': ['Black', 'Navy', 'Blue', 'Red', 'Green'],
                'similar_colors': ['Off White', 'Cream', 'Light Gray'],
                'avoid_colors': []
            },
            'Blue': {
                'diverse_colors': ['White', 'Black', 'Gray', 'Yellow', 'Orange'],
                'similar_colors': ['Navy', 'Light Blue', 'Dark Blue'],
                'avoid_colors': []
            },
            'Red': {
                'diverse_colors': ['Black', 'White', 'Gray', 'Navy'],
                'similar_colors': ['Maroon', 'Burgundy', 'Dark Red'],
                'avoid_colors': ['Pink', 'Orange']  # Avoid clashing reds
            },
            'Green': {
                'diverse_colors': ['White', 'Black', 'Brown', 'Beige'],
                'similar_colors': ['Olive Green', 'Dark Green', 'Forest Green'],
                'avoid_colors': ['Red']  # Avoid Christmas colors
            },
            'Gray': {
                'diverse_colors': ['Black', 'White', 'Blue', 'Red'],
                'similar_colors': ['Light Gray', 'Dark Gray', 'Charcoal'],
                'avoid_colors': []
            },
            'Navy': {
                'diverse_colors': ['White', 'Red', 'Yellow', 'Pink'],
                'similar_colors': ['Dark Blue', 'Blue'],
                'avoid_colors': []
            }
        }
    
    def _initialize_same_category_style_compatibility(self) -> Dict:
        """Initialize style compatibility within same category."""
        return {
            'Casual': {
                'same_category_styles': ['Streetwear', 'Everyday Casual', 'Relaxed Fit'],
                'diverse_within_category': ['Graphic', 'Plain', 'Printed']
            },
            'Formal': {
                'same_category_styles': ['Business Formal', 'Professional', 'Dress'],
                'diverse_within_category': ['Solid', 'Striped', 'Checked']
            },
            'Business': {
                'same_category_styles': ['Business Formal', 'Business Casual', 'Professional'],
                'diverse_within_category': ['Single Breasted', 'Double Breasted', 'Slim Fit']
            },
            'Streetwear': {
                'same_category_styles': ['Urban', 'Hip-hop', 'Casual'],
                'diverse_within_category': ['Graphic', 'Logo', 'Plain']
            }
        }
    
    def _initialize_design_variation_keywords(self) -> Dict:
        """Initialize design variation keywords for diversity."""
        return {
            'patterns': {
                'solid': ['solid', 'plain', 'basic'],
                'striped': ['striped', 'stripes', 'pinstripe'],
                'checked': ['checked', 'checkered', 'plaid'],
                'printed': ['printed', 'print', 'graphic'],
                'floral': ['floral', 'flower', 'botanical'],
                'geometric': ['geometric', 'abstract', 'pattern']
            },
            'fits': {
                'slim': ['slim', 'fitted', 'tight'],
                'regular': ['regular', 'standard', 'classic'],
                'loose': ['loose', 'relaxed', 'oversized'],
                'comfort': ['comfort', 'easy', 'flexible']
            },
            'necklines': {
                'round': ['round neck', 'crew neck', 'round'],
                'v_neck': ['v neck', 'v-neck', 'vneck'],
                'collar': ['collar', 'polo', 'button'],
                'henley': ['henley', 'henley neck']
            }
        }
    
    def get_product_by_id_direct(self, product_id: str) -> Optional[pd.Series]:
        """Directly fetch a specific product by product_id from the database without loading all products."""
        try:
            logger.info(f"🔍 Directly fetching product {product_id} from database...")
            # Query only the specific product - use 'product_id' column
            result = self.db.client.table('tagged_products').select(
                'product_id,title,product_type,gender,primary_style,primary_color,image_url,full_caption,product_embedding,scraped_category,style_category'
            ).eq('product_id', product_id).execute()
            if result.data and len(result.data) > 0:
                product_data = result.data[0]
                logger.info(f"✅ Found product {product_id} directly from database")
                return pd.Series(product_data)
            else:
                logger.warning(f"❌ Product {product_id} not found in database")
                return None
        except Exception as e:
            logger.error(f"❌ Error fetching product {product_id} directly: {e}")
            return None

    def load_products_from_supabase(self, limit: int = None) -> pd.DataFrame:
        """Load products from Supabase using chunked loading with smart filtering."""
        import pandas as pd
        logger.info(f"Loading products from Supabase using chunked loading with smart filtering...")
        
        try:
            # Get source product info first to determine filters
            source_product = self.get_product_by_id_direct(self.current_product_id)
            if source_product is None:
                logger.error("❌ Cannot load products: source product not found")
                return pd.DataFrame()
            
            # Extract filter criteria from source product
            source_product_type = source_product.get('product_type', '').strip()  # Keep original case
            source_gender = source_product.get('gender', '').strip()  # Keep original case
            source_style_category = source_product.get('style_category', '').strip()  # Keep original case
            
            # Normalize style category: "Business/Formal" and "Business Casual" -> "Business Casual"
            if source_style_category in ['Business/Formal', 'Business Casual']:
                source_style_category = 'Business Casual'
            
            # DEBUG: Log the actual product data
            logger.info(f"🔍 DEBUG: Source product data for {self.current_product_id}:")
            logger.info(f"   product_type: '{source_product_type}'")
            logger.info(f"   gender: '{source_gender}'")
            logger.info(f"   style_category: '{source_style_category}'")
            logger.info(f"   title: '{source_product.get('title', 'Unknown')}'")
            
            logger.info(f"🔍 Smart filtering by: product_type='{source_product_type}', gender='{source_gender}', style_category='{source_style_category}'")
            
            # TEMPORARILY: Try simple approach without chunking first
            logger.info("🔄 TEMPORARILY: Loading all products with filters (no chunking)...")
            
            # FIRST: Test if we can query the database at all
            logger.info("🔍 TESTING: Basic database query without filters...")
            try:
                test_query = self.db.client.table('tagged_products').select('id,title,product_type,gender,style_category').limit(5)
                test_result = test_query.execute()
                if test_result.data:
                    logger.info(f"✅ Basic query works: found {len(test_result.data)} products")
                    for product in test_result.data:
                        logger.info(f"   Sample: id={product.get('id')}, title='{product.get('title')}', product_type='{product.get('product_type')}', gender='{product.get('gender')}', style_category='{product.get('style_category')}'")
                else:
                    logger.error("❌ Basic query failed: no products found")
            except Exception as test_e:
                logger.error(f"❌ Basic query failed: {test_e}")
            
            # SECOND: Test with exact product_type filter
            logger.info(f"🔍 TESTING: Query with product_type='{source_product_type}' filter...")
            try:
                type_query = self.db.client.table('tagged_products').select('id,title,product_type,gender,style_category').eq('product_type', source_product_type).limit(5)
                type_result = type_query.execute()
                if type_result.data:
                    logger.info(f"✅ Product type filter works: found {len(type_result.data)} products")
                    for product in type_result.data:
                        logger.info(f"   Sample: id={product.get('id')}, title='{product.get('title')}', gender='{product.get('gender')}', style_category='{product.get('style_category')}'")
                else:
                    logger.error(f"❌ Product type filter failed: no products with product_type='{source_product_type}'")
            except Exception as type_e:
                logger.error(f"❌ Product type filter failed: {type_e}")
            
            # THIRD: Test with exact gender filter
            logger.info(f"🔍 TESTING: Query with gender='{source_gender}' filter...")
            try:
                gender_query = self.db.client.table('tagged_products').select('id,title,product_type,gender,style_category').eq('gender', source_gender).limit(5)
                gender_result = gender_query.execute()
                if gender_result.data:
                    logger.info(f"✅ Gender filter works: found {len(gender_result.data)} products")
                    for product in gender_result.data:
                        logger.info(f"   Sample: id={product.get('id')}, title='{product.get('title')}', product_type='{product.get('product_type')}', style_category='{product.get('style_category')}'")
                else:
                    logger.error(f"❌ Gender filter failed: no products with gender='{source_gender}'")
            except Exception as gender_e:
                logger.error(f"❌ Gender filter failed: {gender_e}")
            
            # FOURTH: Test with style_category filter (including normalization)
            logger.info(f"🔍 TESTING: Query with style_category='{source_style_category}' filter...")
            try:
                # Test both original and normalized style categories
                style_query = self.db.client.table('tagged_products').select('id,title,product_type,gender,style_category').in_('style_category', [source_style_category, 'Business/Formal', 'Business Casual']).limit(5)
                style_result = style_query.execute()
                if style_result.data:
                    logger.info(f"✅ Style category filter works: found {len(style_result.data)} products")
                    for product in style_result.data:
                        logger.info(f"   Sample: id={product.get('id')}, title='{product.get('title')}', product_type='{product.get('product_type')}', gender='{product.get('gender')}', style_category='{product.get('style_category')}'")
                else:
                    logger.error(f"❌ Style category filter failed: no products with style_category='{source_style_category}'")
            except Exception as style_e:
                logger.error(f"❌ Style category filter failed: {style_e}")
            
            try:
                # Build query with smart filtering
                query = self.db.client.table('tagged_products').select(
                    'id,title,product_type,gender,primary_style,primary_color,image_url,full_caption,product_embedding,scraped_category,style_category'
                )
                
                # Apply smart filters with exact case matching
                if source_product_type:
                    query = query.eq('product_type', source_product_type)
                if source_gender:
                    query = query.eq('gender', source_gender)
                if source_style_category:
                    # Handle style category normalization in the query
                    if source_style_category == 'Business Casual':
                        query = query.in_('style_category', ['Business/Formal', 'Business Casual'])
                    else:
                        query = query.eq('style_category', source_style_category)
                
                # Apply limit if specified
                if limit:
                    query = query.limit(limit)
                
                # DEBUG: Log the actual query being executed
                logger.info(f"🔍 DEBUG: Executing query with filters: product_type='{source_product_type}', gender='{source_gender}', style_category='{source_style_category}', limit={limit}")
                
                # Execute query
                result = query.execute()
                
                if result.data:
                    all_products = result.data
                    logger.info(f"✅ Successfully loaded {len(all_products)} products with filters")
                else:
                    logger.error("❌ No products found with filters")
                    
            except Exception as e:
                logger.error(f"❌ Error loading products with filters: {e}")
                # Try without any filters as ultimate fallback
                logger.info("🔄 Trying ultimate fallback: loading all products without any filters...")
                try:
                    fallback_query = self.db.client.table('tagged_products').select(
                        'id,title,product_type,gender,primary_style,primary_color,image_url,full_caption,product_embedding,scraped_category,style_category'
                    )
                    if limit:
                        fallback_query = fallback_query.limit(limit)
                    fallback_result = fallback_query.execute()
                    if fallback_result.data:
                        all_products = fallback_result.data
                        logger.info(f"✅ Ultimate fallback successful: loaded {len(all_products)} products")
                    else:
                        logger.error("❌ Ultimate fallback also failed: no products at all")
                except Exception as ultimate_e:
                    logger.error(f"❌ Ultimate fallback also failed: {ultimate_e}")
            
            if all_products:
                products_df = pd.DataFrame(all_products)
                logger.info(f"✅ Successfully loaded {len(products_df)} products using chunked loading with smart filtering.")
                return products_df
            else:
                logger.error("❌ No products found with smart filtering")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error in chunked product loading: {e}")
            return pd.DataFrame()
    
    def validate_products_data(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation with price integration."""
        required_columns = ["title", "product_type"]
        missing_columns = [col for col in required_columns if col not in products_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        products_df = products_df.dropna(subset=["title", "product_type"])
        
        # Use existing price column or estimate
        if 'price' not in products_df.columns:
            products_df['price'] = products_df.apply(
                lambda row: self._estimate_price_from_title(row.get('title', '')), axis=1
            )
        
        # Add missing columns with defaults
        if 'final_caption' not in products_df.columns:
            if 'full_caption' in products_df.columns:
                products_df['final_caption'] = products_df['full_caption']
            else:
                products_df['final_caption'] = products_df['title']
        
        # Handle product_id column name variations
        if 'product_id' not in products_df.columns:
            if 'id' in products_df.columns:
                products_df['product_id'] = products_df['id']
            else:
                products_df['product_id'] = range(len(products_df))
        
        logger.info(f"✅ Products validation complete. Price range: ₹{products_df['price'].min():.0f} - ₹{products_df['price'].max():.0f}")
        
        return products_df
    
    def _estimate_price_from_title(self, title: str) -> float:
        """Estimate price based on product title keywords."""
        title_lower = title.lower()
        
        # Premium indicators
        if any(word in title_lower for word in ['premium', 'luxury', 'designer', 'silk', 'leather']):
            return 2500
        elif any(word in title_lower for word in ['cotton', 'linen', 'wool', 'formal']):
            return 1500
        elif any(word in title_lower for word in ['basic', 'essential', 'simple']):
            return 800
        else:
            return 1200  # Default mid-range price
    
    def find_similar_products(self, product_id: str, num_similar: int = 10, 
                            user_preferences: Dict = None, filters: Dict = None) -> List[Dict]:
        """
        Enhanced same-category similar products with diversity, caching, and pre-filtering using Supabase.
        ✅ ENHANCED: Now enforces strict same-category filtering by product_type and style_category.
        """
        from database import get_db
        db = get_db()
        import time
        start_time = time.time()
        
        # 1. Check for cached results first
        cached = db.get_cached_similar_products(product_id, user_preferences, filters)
        if cached and len(cached) >= num_similar:
            logger.info(f"✅ Returning {len(cached)} cached similar products for {product_id}")
            return cached[:num_similar]

        # 2. Try direct product lookup first (more efficient)
        source_product = self.get_product_by_id_direct(product_id)
        if source_product is None:
            logger.error(f"❌ Product {product_id} not found in database")
            return []
        
        logger.info(f"✅ Found source product: {source_product.get('title', 'Unknown')}")

        # 3. Load products for similarity search (REMOVED LIMIT for better coverage)
        self.current_product_id = product_id
        try:
            products_df = self.load_products_from_supabase()  # Load all products
            if products_df.empty:
                logger.error("No products available from Supabase")
                return []
        except Exception as e:
            logger.error(f"❌ Error loading products from Supabase: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []
        
        products_df = self.validate_products_data(products_df)

        # 4. Ensure source product has all required fields by processing it
        # Convert source_product to DataFrame, process it, then back to Series
        source_df = pd.DataFrame([source_product])
        source_df = self.validate_products_data(source_df)
        source_product = source_df.iloc[0] if not source_df.empty else source_product
        
        logger.info(f"✅ Processed source product - product_type: {source_product.get('product_type', 'NOT_FOUND')}")

        # ✅ ENHANCED: Strict same-category filtering
        source_product_type = source_product.get('product_type', '').strip().lower()
        source_style_category = source_product.get('style_category', source_product.get('primary_style', '')).strip().lower()
        
        logger.info(f"🔍 Source product type: {source_product_type}, style category: {source_style_category}")

        # 5. Simple filtering with only essential filters: gender, product_type, style_category
        filtered_df = products_df.copy()
        logger.info(f"🔍 Starting filtering. Initial products: {len(filtered_df)}")
        
        # ✅ SIMPLIFIED: Only apply the three essential filters
        # 1. Product type filter (same category)
        if source_product_type:
            filtered_df = filtered_df[filtered_df['product_type'].str.lower() == source_product_type]
            logger.info(f"🔍 After product_type filter: {len(filtered_df)} products")
        
        # 2. Style category filter (same category)
        if source_style_category:
            filtered_df = filtered_df[filtered_df['style_category'].str.lower() == source_style_category]
            logger.info(f"🔍 After style_category filter: {len(filtered_df)} products")
        
        # 3. Gender filter
        if user_preferences and user_preferences.get('gender'):
            gender = user_preferences['gender'].lower()
            filtered_df = filtered_df[filtered_df['gender'].str.lower().isin([gender, 'unisex'])]
            logger.info(f"🔍 After gender filter: {len(filtered_df)} products")
        
        logger.info(f"🔍 Final filtered products: {len(filtered_df)}")
        
        # 6. Build FAISS indexes on filtered set (by product_type) - CACHE THIS
        if not hasattr(self, 'faiss_indexes') or not self.faiss_indexes:
            logger.info("🔄 Building FAISS indexes (first time or cache miss)...")
            self.build_faiss_indexes(filtered_df, by_field='product_type')
        else:
            logger.info("✅ Using cached FAISS indexes")

        # 7. Generate same-category candidates only (from filtered set)
        candidates = self._generate_same_category_candidates(source_product, filtered_df, user_preferences)
        if not candidates:
            logger.warning("No candidate products found after filtering")
            logger.warning("⚠️ Skipping database storage due to no results")
            return []

        # 8. Apply enhanced scoring and filtering for diversity
        similar_products = self._score_and_filter_same_category_candidates(
            source_product, candidates, user_preferences, filters, num_similar
        )
        logger.info(f"Found {len(similar_products)} diverse same-category products after filtering and FAISS")

        # 9. Store results in cache
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"💾 Storing {len(similar_products)} results in database cache...")
        db.store_similar_products(product_id, similar_products, user_preferences, filters, processing_time_ms)

        return similar_products[:num_similar]
    
    def _generate_same_category_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                         user_preferences: Dict = None) -> List[Dict]:
        """Generate candidates from same category only with diversity focus (by product_type)."""
        source_product_type = source_product.get('product_type', '')
        source_text = source_product.get('final_caption', '') or source_product.get('title', '')
        all_candidates = []
        # 1. Core similar products (same category, similar features)
        core_candidates = self.search_similar_products_faiss(
            source_text, source_product_type, k=100
        )
        for candidate in core_candidates:
            candidate['candidate_type'] = 'core_similar'
            candidate['boost_factor'] = 1.0
        all_candidates.extend(core_candidates)
        # 2. Color-diverse candidates (same category, different colors)
        if self.config['enable_color_diversity']:
            color_diverse_candidates = self._get_color_diverse_candidates(
                source_product, products_df, user_preferences
            )
            all_candidates.extend(color_diverse_candidates)
        # 3. Design-diverse candidates (same category, different designs)
        if self.config['enable_design_diversity']:
            design_diverse_candidates = self._get_design_diverse_candidates(
                source_product, products_df, user_preferences
            )
            all_candidates.extend(design_diverse_candidates)
        # 4. User preference enhanced candidates (same category)
        if user_preferences:
            preference_candidates = self._get_user_preference_candidates(
                source_product, products_df, user_preferences
            )
            all_candidates.extend(preference_candidates)
        return all_candidates
    
    def _get_color_diverse_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                    user_preferences: Dict = None) -> List[Dict]:
        """Get color-diverse candidates within same category."""
        
        candidates = []
        source_color = source_product.get('primary_color', '')
        source_product_type = source_product.get('product_type', '')
        
        if source_color in self.color_diversity_matrix:
            diverse_colors = self.color_diversity_matrix[source_color]['diverse_colors']
            
            for color in diverse_colors:
                # Search for products with diverse colors in same category
                color_query = f"{color} {source_product_type}"
                color_candidates = self.search_similar_products_faiss(
                    color_query, source_product_type, k=8
                )
                for candidate in color_candidates:
                    candidate['candidate_type'] = 'color_diverse'
                    candidate['boost_factor'] = 1.1  # Slight boost for color diversity
                candidates.extend(color_candidates)
        
        return candidates[:30]  # Limit color diverse candidates (increased from 15 to 30)
    
    def _get_design_diverse_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                     user_preferences: Dict = None) -> List[Dict]:
        """Get design-diverse candidates within same category."""
        
        candidates = []
        source_title = source_product.get('title', '').lower()
        source_product_type = source_product.get('product_type', '')
        
        # Identify source design characteristics
        source_patterns = self._identify_design_patterns(source_title)
        
        # Search for different design patterns in same category
        for pattern_type, patterns in self.design_variation_keywords['patterns'].items():
            if pattern_type not in source_patterns:  # Different from source
                for pattern in patterns:
                    design_query = f"{pattern} {source_product_type}"
                    design_candidates = self.search_similar_products_faiss(
                        design_query, source_product_type, k=5
                    )
                    for candidate in design_candidates:
                        candidate['candidate_type'] = 'design_diverse'
                        candidate['boost_factor'] = 1.05  # Small boost for design diversity
                    candidates.extend(design_candidates)
        
        return candidates[:20]  # Limit design diverse candidates (increased from 10 to 20)
    
    def _get_user_preference_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                      user_preferences: Dict) -> List[Dict]:
        """Get candidates based on user preferences within same category."""
        
        candidates = []
        source_product_type = source_product.get('product_type', '')
        
        # User preferred styles within same category
        preferred_styles = user_preferences.get('preferred_styles', [])
        for style in preferred_styles:
            style_query = f"{style} {source_product_type}"
            style_candidates = self.search_similar_products_faiss(
                style_query, source_product_type, k=6
            )
            for candidate in style_candidates:
                candidate['candidate_type'] = 'user_preference'
                candidate['boost_factor'] = 1.15  # Boost for user preferences
            candidates.extend(style_candidates)
        
        # User preferred colors within same category
        preferred_colors = user_preferences.get('preferred_colors', [])
        for color in preferred_colors:
            color_query = f"{color} {source_product_type}"
            color_candidates = self.search_similar_products_faiss(
                color_query, source_product_type, k=6
            )
            for candidate in color_candidates:
                candidate['candidate_type'] = 'user_preference'
                candidate['boost_factor'] = 1.12
            candidates.extend(color_candidates)
        
        return candidates
    
    def _identify_design_patterns(self, title: str) -> List[str]:
        """Identify design patterns from product title."""
        patterns = []
        title_lower = title.lower()
        
        for pattern_type, keywords in self.design_variation_keywords['patterns'].items():
            if any(keyword in title_lower for keyword in keywords):
                patterns.append(pattern_type)
        
        return patterns
    
    def _score_and_filter_same_category_candidates(self, source_product: pd.Series, candidates: List[Dict], 
                                                 user_preferences: Dict, filters: Dict, num_similar: int) -> List[Dict]:
        """Enhanced scoring for same-category diversity."""
        
        similar_products = []
        source_product_id = str(source_product.get('product_id', ''))
        seen_products = set()
        
        for candidate in candidates:
            candidate_product = candidate['product']
            candidate_id = str(candidate_product.get('product_id', ''))
            
            if candidate_id in seen_products or candidate_id == source_product_id:
                continue
            
            if filters and not self._passes_filters(candidate_product, filters):
                continue
            
            similarity_score, score_breakdown = self._calculate_enhanced_same_category_similarity(
                source_product, candidate_product, candidate, user_preferences
            )
            
            if similarity_score < self.similarity_config['min_similarity_score']:
                continue
            
            similar_products.append({
                'product_id': candidate_id,
                'title': candidate_product.get('title', ''),
                'image_url': candidate_product.get('image_url', ''),
                'price': candidate_product.get('price', 1000),
                'brand': candidate_product.get('brand', ''),
                'style': candidate_product.get('enhanced_primary_style', candidate_product.get('primary_style', '')),
                'color': candidate_product.get('primary_color', ''),
                'product_type': candidate_product.get('product_type', ''),
                'occasion': candidate_product.get('enhanced_occasion', candidate_product.get('occasion', '')),
                'similarity_score': similarity_score,
                'score_breakdown': score_breakdown,
                'candidate_type': candidate.get('candidate_type', 'unknown'),
                'source_product_id': source_product_id,
                'generated_at': datetime.now().isoformat()
            })
            
            seen_products.add(candidate_id)
        
        diverse_products = self._apply_same_category_diversity_filtering(similar_products, source_product)
        
        diverse_products.sort(key=lambda x: x['similarity_score'], reverse=True)
        return diverse_products[:num_similar]
    
    def _calculate_enhanced_same_category_similarity(self, source_product: pd.Series, candidate_product: pd.Series, 
                                                   candidate: Dict, user_preferences: Dict = None) -> Tuple[float, Dict]:
        """Calculate similarity with same-category diversity factors."""
        
        # Base semantic similarity
        semantic_score = candidate.get('semantic_score', 0.5)
        
        # Style compatibility (within same category)
        style_score = self._calculate_same_category_style_compatibility(
            source_product.get('enhanced_primary_style', ''),
            candidate_product.get('enhanced_primary_style', '')
        )
        
        # Color diversity bonus
        color_diversity_score = self._calculate_color_diversity_bonus(
            source_product.get('primary_color', ''),
            candidate_product.get('primary_color', '')
        )
        
        # Design diversity bonus
        design_diversity_score = self._calculate_design_diversity_bonus(
            source_product.get('title', ''),
            candidate_product.get('title', '')
        )
        
        # User preference boost
        preference_score = 1.0
        if user_preferences:
            preference_score = self._calculate_user_preference_boost(
                candidate_product, user_preferences
            )
        
        # Candidate type boost
        type_boost = candidate.get('boost_factor', 1.0)
        
        # Calculate weighted final score
        weights = self.similarity_config
        final_score = (
            semantic_score * weights['semantic_weight'] +
            style_score * weights['style_weight'] +
            color_diversity_score * weights['color_diversity_weight'] +
            design_diversity_score * weights['design_diversity_weight'] +
            preference_score * weights['user_preference_weight']
        ) * type_boost
        
        # Normalize to 0-1 range
        total_weight = (weights['semantic_weight'] + weights['style_weight'] + 
                       weights['color_diversity_weight'] + weights['design_diversity_weight'] + 
                       weights['user_preference_weight'])
        
        final_score = min(final_score / total_weight, 1.0)
        
        score_breakdown = {
            'semantic_similarity': semantic_score,
            'style_compatibility': style_score,
            'color_diversity': color_diversity_score,
            'design_diversity': design_diversity_score,
            'user_preference_boost': preference_score,
            'type_boost': type_boost,
            'final_score': final_score
        }
        
        return final_score, score_breakdown
    
    def _calculate_same_category_style_compatibility(self, style1: str, style2: str) -> float:
        """Calculate style compatibility within same category."""
        if not style1 or not style2:
            return 0.5
        
        style1, style2 = style1.lower(), style2.lower()
        
        # Exact match
        if style1 == style2:
            return 1.0
        
        # Check same category compatibility
        for main_style, compatibility in self.style_compatibility.items():
            if main_style.lower() in style1:
                same_category_styles = compatibility.get('same_category_styles', [])
                if any(compatible.lower() in style2 for compatible in same_category_styles):
                    return 0.8
        
        return 0.6  # Default within-category compatibility
    
    def _calculate_color_diversity_bonus(self, color1: str, color2: str) -> float:
        """Calculate color diversity bonus (higher for diverse but harmonious colors)."""
        if not color1 or not color2:
            return 0.5
        
        # Same color gets neutral score
        if color1.lower() == color2.lower():
            return 0.6
        
        # Check for diverse but harmonious colors
        if color1 in self.color_diversity_matrix:
            color_info = self.color_diversity_matrix[color1]
            if color2 in color_info['diverse_colors']:
                return 0.9  # High score for good color diversity
            elif color2 in color_info['similar_colors']:
                return 0.7  # Medium score for similar colors
            elif color2 in color_info['avoid_colors']:
                return 0.3  # Low score for clashing colors
        
        return 0.5  # Default diversity score
    
    def _calculate_design_diversity_bonus(self, title1: str, title2: str) -> float:
        """Calculate design diversity bonus."""
        if not title1 or not title2:
            return 0.5
        
        patterns1 = self._identify_design_patterns(title1)
        patterns2 = self._identify_design_patterns(title2)
        
        # If products have different design patterns, give diversity bonus
        if patterns1 and patterns2:
            if set(patterns1).isdisjoint(set(patterns2)):
                return 0.8  # Good diversity bonus
            elif len(set(patterns1).intersection(set(patterns2))) < len(patterns1):
                return 0.7  # Partial diversity
            else:
                return 0.6  # Similar designs
        
        return 0.6  # Default design score
    
    def _calculate_user_preference_boost(self, product: pd.Series, user_preferences: Dict) -> float:
        """Calculate boost based on user preferences."""
        boost = 1.0
        
        # Style preferences
        preferred_styles = user_preferences.get('preferred_styles', [])
        product_style = product.get('enhanced_primary_style', '').lower()
        if any(style.lower() in product_style for style in preferred_styles):
            boost += 0.2
        
        # Color preferences
        preferred_colors = user_preferences.get('preferred_colors', [])
        product_color = product.get('primary_color', '')
        if product_color in preferred_colors:
            boost += 0.15
        
        return min(boost, 1.4)  # Cap at 1.4x boost
    
    def _apply_same_category_diversity_filtering(self, similar_products: List[Dict], 
                                               source_product: pd.Series) -> List[Dict]:
        """Apply diversity filtering for same-category products."""
        # TEMPORARILY DISABLED: Return all products without filtering
        logger.info(f"🔄 TEMPORARILY DISABLED: Returning all {len(similar_products)} products without diversity filtering")
        return similar_products
        
        # Original diversity filtering code (commented out)
        """
        if not self.config.get('diversity_threshold'):
            return similar_products
        
        diverse_products = []
        threshold = self.config['diversity_threshold']
        
        for product in similar_products:
            is_diverse = True
            
            # Check diversity against source product
            source_title_similarity = self._calculate_text_similarity(
                product['title'], source_product.get('title', '')
            )
            
            if source_title_similarity > threshold:
                is_diverse = False
            
            # Check diversity against already selected products
            if is_diverse:
                for existing in diverse_products:
                    title_similarity = self._calculate_text_similarity(
                        product['title'], existing['title']
                    )
                    
                    if title_similarity > threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_products.append(product)
        
        return diverse_products
        """
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _passes_filters(self, product: pd.Series, filters: Dict) -> bool:
        """Check if product passes user-defined filters."""
        
        # ✅ MODIFIED: Only apply styles filter, removed price_range and colors filters
        
        # Style filter
        if 'styles' in filters:
            product_style = product.get('enhanced_primary_style', product.get('primary_style', ''))
            if not any(style.lower() in product_style.lower() for style in filters['styles']):
                return False
        
        return True
    
    def build_faiss_indexes(self, products_df: pd.DataFrame, by_field: str = 'product_type') -> None:
        """Build FAISS indexes for each product_type (instead of wear_type)."""
        import faiss
        import time
        start_time = time.time()
        
        self.faiss_indexes = {}
        self.product_mappings = {}
        
        # Group by product_type and process each group
        product_types = products_df[by_field].unique()
        logger.info(f"🔄 Building FAISS indexes for {len(product_types)} product types...")
        
        for product_type in product_types:
            group_start = time.time()
            group = products_df[products_df[by_field] == product_type].copy()
            
            if group.empty:
                continue
            
            embeddings = []
            product_indices = []
            valid_products = []
            
            # Process embeddings more efficiently
            for idx, row in group.iterrows():
                if 'product_embedding' in row and row['product_embedding']:
                    try:
                        emb = row['product_embedding']
                        if isinstance(emb, str):
                            emb = json.loads(emb)
                        emb = np.array(emb)
                        embeddings.append(emb)
                        product_indices.append(idx)
                        valid_products.append(row)
                    except Exception as e:
                        continue
            
            if embeddings:
                embeddings = np.vstack(embeddings)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings.astype('float32'))
                self.faiss_indexes[product_type] = index
                self.product_mappings[product_type] = {
                    'indices': product_indices,
                    'products': pd.DataFrame(valid_products)
                }
                group_time = time.time() - group_start
                logger.info(f"✅ Built FAISS index for {by_field}={product_type}: {len(embeddings)} products indexed in {group_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"🎉 FAISS indexing completed in {total_time:.2f}s for {len(self.faiss_indexes)} product types")

    def get_embedding_cached(self, text: str, cache_key: str = None, product_id: str = None) -> np.ndarray:
        """Get embedding with caching. Now uses precomputed embeddings from tagged_products table."""
        if not cache_key:
            cache_key = text[:100]
        
        # ✅ ENHANCED: Try to get precomputed embedding from tagged_products table first
        if product_id:
            try:
                # Query the tagged_products table for precomputed embedding
                result = self.db.client.table('tagged_products').select('product_embedding').eq('product_id', product_id).execute()
                
                if result.data and result.data[0].get('product_embedding'):
                    embedding_json = result.data[0]['product_embedding']
                    if isinstance(embedding_json, str):
                        embedding = np.array(json.loads(embedding_json))
                    else:
                        embedding = np.array(embedding_json)
                    
                    # Cache the result
                    if self.config['cache_embeddings']:
                        self.embedding_cache[cache_key] = embedding
                    
                    logger.debug(f"✅ Retrieved precomputed embedding for product {product_id}")
                    return embedding
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get precomputed embedding for product {product_id}: {e}")
                # Fall back to computing embedding
        
        # Check cache first
        if self.config['cache_embeddings'] and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # ✅ ENHANCED: Ensure model is loaded before computing embeddings
        self._ensure_model_loaded()
        
        embedding = self.model.encode([text])[0]
        
        if self.config['cache_embeddings']:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_product_by_id(self, product_id, products_df):
        """Find a product by product_id only, robust to type mismatches and whitespace."""
        logger.info(f"Looking for product_id: {product_id} (type: {type(product_id)})")
        if 'product_id' not in products_df.columns:
            raise ValueError("'product_id' column not found in products DataFrame")
        
        # Clean and convert all product_ids to string, strip whitespace
        products_df['product_id_str'] = products_df['product_id'].astype(str).str.strip()
        product_id_str = str(product_id).strip()
        
        logger.info(f"Available product_ids (first 10): {products_df['product_id_str'].head(10).tolist()}")
        logger.info(f"Total products loaded: {len(products_df)}")
        logger.info(f"Product ID being searched: '{product_id_str}'")
        
        match = products_df[products_df['product_id_str'] == product_id_str]
        
        if not match.empty:
            logger.info(f"✅ Found product {product_id} in DataFrame")
            return match.iloc[0]
        else:
            # Print first 100 unique product_ids for debugging (not all to avoid log overflow)
            unique_ids = products_df['product_id_str'].unique()
            logger.error(f"❌ Product {product_id} not found in DataFrame")
            logger.error(f"Type of product_id being searched: {type(product_id)} value: {product_id}")
            logger.error(f"First 100 unique product_ids: {unique_ids[:100].tolist()}")
            logger.error(f"Total unique product_ids: {len(unique_ids)}")
            
            # Check if it exists with different formatting
            if product_id_str in unique_ids:
                logger.error(f"⚠️ Product {product_id} exists but match failed - possible whitespace/type issue")
            else:
                logger.error(f"❌ Product {product_id} does not exist in the loaded data")
            
            raise ValueError(f"Product with product_id '{product_id}' not found")
    
    def search_similar_products_faiss(self, query_text: str, product_type: str, k: int = 20) -> List[Dict]:
        """Search for similar products using FAISS with precomputed embeddings by product_type."""
        if product_type not in self.faiss_indexes:
            logger.warning(f"No FAISS index available for product_type: {product_type}")
            return []
        self._ensure_model_loaded()
        query_embedding = self.get_embedding_cached(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        index = self.faiss_indexes[product_type]
        scores, indices = index.search(query_embedding.astype('float32'), k)
        product_mapping = self.product_mappings[product_type]
        candidates = []
        for i, (score, faiss_idx) in enumerate(zip(scores[0], indices[0])):
            if faiss_idx >= len(product_mapping['indices']):
                continue
            product_idx = product_mapping['indices'][faiss_idx]
            product = product_mapping['products'].iloc[faiss_idx]
            candidates.append({
                'product_idx': product_idx,
                'product': product,
                'semantic_score': float(score),
                'faiss_rank': i + 1
            })
        return candidates

def main():
    """Test the Supabase enhanced same-category similar products generator."""
    
    # Initialize enhanced generator
    generator = SupabaseEnhancedSimilarProductsGenerator()
    
    # Sample user preferences
    user_preferences = {
        'preferred_styles': ['Business Formal', 'Formal'],
        'preferred_colors': ['Black', 'Navy', 'White'],
    }
    
    # Sample filters
    filters = {
        # ✅ REMOVED: No longer using styles filter, only essential filters
    }
    
    # Test with enhanced same-category features - using product ID 2859 as requested
    test_product_id = "2859"
    
    logger.info(f"🔍 Testing Enhanced Same-Category Phase 3 with Supabase for product: {test_product_id}")
    
    try:
        similar_products = generator.find_similar_products(
            test_product_id, 
            num_similar=8,
            user_preferences=user_preferences,
            filters=filters
        )
        
        if similar_products:
            print(f"\n✅ SUCCESS: Found {len(similar_products)} diverse same-category products")
            print("=" * 80)
            
            for i, product in enumerate(similar_products, 1):
                score = product['similarity_score']
                price = product['price']
                ptype = product.get('candidate_type', 'unknown')
                
                print(f"\n{i}. {product['title'][:55]}...")
                print(f"   Score: {score:.3f} | Type: {ptype}")
                print(f"   Price: ₹{price} | Style: {product['style']}")
                print(f"   Color: {product['color']} | Wear: {product['product_type']}")
                
                # Show score breakdown for first few
                if i <= 3:
                    breakdown = product['score_breakdown']
                    print(f"   Breakdown: Semantic={breakdown['semantic_similarity']:.2f}, "
                          f"Style={breakdown['style_compatibility']:.2f}, "
                          f"Color_Div={breakdown['color_diversity']:.2f}, "
                          f"Design_Div={breakdown['design_diversity']:.2f}")
            
            print(f"\n🔗 Enhanced Same-Category API: GET /api/products/{test_product_id}/similar?count=8&diverse=true&personalized=true")
        else:
            print(f"\n❌ No enhanced similar products found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get more specific error information
        if "timeout" in str(e).lower():
            print(f"\n⚠️ Database timeout detected. This might be due to:")
            print(f"   - Large number of products in the database")
            print(f"   - Network connectivity issues")
            print(f"   - Database server load")
            print(f"\n💡 Suggestions:")
            print(f"   - Try with a smaller product subset")
            print(f"   - Check database connection")
            print(f"   - Consider adding pagination to product loading")

if __name__ == "__main__":
    main() 