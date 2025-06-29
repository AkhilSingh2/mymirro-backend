# Phase 2: Similar Outfits API with Supabase Integration (On-demand)
# This generates 10 similar outfits for any given outfit in real-time using Supabase

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import random
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
from database import get_db
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseSimilarOutfitsGenerator:
    """
    Phase 2: Generate 10 similar outfits for any given outfit on-demand using Supabase
    ✅ ENHANCED: Now with Supabase database integration + Advanced Fashion Intelligence
    """
    
    # ✅ OPTIMIZATION: Class-level model cache to avoid reloading
    _model_cache = None
    _model_cache_ready = False
    
    def __init__(self, config: Dict = None):
        """Initialize the Supabase-enabled similar outfits generator."""
        self.config = config or self._default_config()
        
        # Railway CPU optimization - delay until needed
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        if self.is_railway:
            logger.info("🏭 Railway environment detected - will apply CPU optimizations when needed")
        
        # Check for required dependencies
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available. Similar outfits service requires FAISS for similarity search.")
            raise ImportError("FAISS is required for similar outfits but not installed")
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ SentenceTransformers not available. Similar outfits service requires sentence-transformers.")
            raise ImportError("sentence-transformers is required for similar outfits but not installed")
        
        # Initialize Supabase database connection
        self.db = get_db()
        if not self.db.test_connection():
            logger.error("❌ Database connection failed. Please check your Supabase configuration.")
            raise ConnectionError("Failed to connect to Supabase database")
        
        # ✅ OPTIMIZATION: Lazy load model only when needed
        self.model = None
        self.embedding_cache = {}
        
        # FAISS indexes for different wear types
        self.faiss_indexes = {}
        self.product_mappings = {}
        
        # ✅ ENHANCED: Sophisticated similarity configuration with fashion intelligence
        self.similarity_config = {
            'semantic_weight': 4.0,           # Core AI matching
            'style_harmony_weight': 3.5,      # Advanced style compatibility  
            'color_harmony_weight': 3.0,      # Sophisticated color theory
            'pattern_compatibility_weight': 2.0,  # Pattern mixing intelligence
            'occasion_weight': 2.2,           # Occasion-specific matching
            'diversity_bonus': 0.8,           # Encourage variety in results
            'confidence_threshold': 0.1,      # ✅ FIX: Lowered for more outfits
            'min_similar_outfits': 5,         # Minimum outfits to return
            'max_similar_outfits': 20,        # Maximum outfits to return
            'candidate_pool_size': 500,       # ✅ FIX: Increased for more candidates
            'fallback_strategy': 'best_available'  # ✅ NEW: Best Available fallback
        }
        
        # Advanced fashion intelligence systems
        self.color_harmony = self._initialize_color_harmony()
        self.style_formality = self._initialize_style_formality()
        self.seasonal_preferences = self._initialize_seasonal_preferences()
        self.pattern_compatibility = self._initialize_pattern_compatibility()
    
    def _ensure_model_loaded(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            try:
                if self.is_railway:
                    logger.info("🔧 Loading model with Railway CPU optimizations")
                
                # ✅ OPTIMIZATION: Use cached model if available
                if SupabaseSimilarOutfitsGenerator._model_cache is not None:
                    self.model = SupabaseSimilarOutfitsGenerator._model_cache
                    logger.info(f"✅ Using cached model: {self.config['model_name']}")
                else:
                    self.model = SentenceTransformer(self.config['model_name'])
                    # Cache the model for future use
                    SupabaseSimilarOutfitsGenerator._model_cache = self.model
                    SupabaseSimilarOutfitsGenerator._model_cache_ready = True
                    logger.info(f"✅ Model loaded and cached: {self.config['model_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
    
    def _default_config(self) -> Dict:
        """Default configuration for the similar outfits generator."""
        app_config = get_config()
        return {
            'model_name': app_config.MODEL_NAME,
            'cache_embeddings': True,
            'batch_size': 50
        }
    
    def _initialize_color_harmony(self) -> Dict:
        """Initialize sophisticated color harmony rules from fashion designer data."""
        try:
            # Load fashion designer color harmony data
            color_harmony_file = "Fashion designer input/Color Harmony.csv"
            if os.path.exists(color_harmony_file):
                logger.info("🎨 Loading fashion designer color harmony data...")
                return self._load_fashion_designer_color_harmony(color_harmony_file)
            else:
                logger.warning("⚠️ Fashion designer color harmony file not found, using fallback rules")
                return self._get_fallback_color_harmony()
        except Exception as e:
            logger.error(f"Error loading fashion designer color harmony: {e}")
            return self._get_fallback_color_harmony()
    
    def _load_fashion_designer_color_harmony(self, file_path: str) -> Dict:
        """Load color harmony rules from fashion designer CSV data."""
        color_harmony = {}
        
        try:
            df = pd.read_csv(file_path, skiprows=3)  # Skip header rows
            
            current_base_color = None
            
            for _, row in df.iterrows():
                # Check if this is a base color row
                if pd.notna(row.iloc[0]) and 'BASE COLOR:' in str(row.iloc[0]):
                    current_base_color = str(row.iloc[0]).replace('BASE COLOR:', '').strip()
                    color_harmony[current_base_color] = {
                        'perfect': [],
                        'excellent': [],
                        'good': [],
                        'avoid': []
                    }
                    continue
                
                # Check if this is a color combination row
                if current_base_color and pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    color_combination = str(row.iloc[0]).strip()
                    rating = row.iloc[1]
                    
                    # Extract the second color from combination (e.g., "Black + Light Pink" -> "Light Pink")
                    if ' + ' in color_combination:
                        second_color = color_combination.split(' + ')[1].strip()
                        
                        # Categorize based on rating
                        if pd.notna(rating):
                            rating_val = float(rating)
                            if rating_val >= 9:
                                color_harmony[current_base_color]['perfect'].append(second_color)
                            elif rating_val >= 7:
                                color_harmony[current_base_color]['excellent'].append(second_color)
                            elif rating_val >= 5:
                                color_harmony[current_base_color]['good'].append(second_color)
                            else:
                                color_harmony[current_base_color]['avoid'].append(second_color)
            
            logger.info(f"✅ Loaded fashion designer color harmony for {len(color_harmony)} base colors")
            return color_harmony
            
        except Exception as e:
            logger.error(f"Error parsing fashion designer color harmony: {e}")
            return self._get_fallback_color_harmony()
    
    def _get_fallback_color_harmony(self) -> Dict:
        """Fallback color harmony rules if fashion designer data is not available."""
        return {
            'Black': {
                'perfect': ['White', 'Gray', 'Silver'],
                'excellent': ['Red', 'Gold', 'Navy', 'Cream'],
                'good': ['Blue', 'Yellow', 'Pink', 'Green'],
                'avoid': ['Brown', 'Orange', 'Purple']
            },
            'White': {
                'perfect': ['Black', 'Navy', 'Gray'],
                'excellent': ['Blue', 'Red', 'Green', 'Brown'],
                'good': ['Yellow', 'Pink', 'Purple', 'Orange'],
                'avoid': ['Cream', 'Beige', 'Light Gray']
            },
            'Blue': {
                'perfect': ['White', 'Navy', 'Light Blue'],
                'excellent': ['Gray', 'Yellow', 'Orange', 'Brown'],
                'good': ['Black', 'Green', 'Pink', 'Silver'],
                'avoid': ['Red', 'Purple', 'Maroon']
            },
            'Red': {
                'perfect': ['Black', 'White', 'Navy'],
                'excellent': ['Gray', 'Blue', 'Yellow', 'Gold'],
                'good': ['Green', 'Pink', 'Orange', 'Brown'],
                'avoid': ['Purple', 'Maroon', 'Pink']
            },
            'Green': {
                'perfect': ['White', 'Beige', 'Brown'],
                'excellent': ['Navy', 'Yellow', 'Orange', 'Gray'],
                'good': ['Black', 'Blue', 'Red', 'Pink'],
                'avoid': ['Purple', 'Maroon', 'Bright Pink']
            },
            'Yellow': {
                'perfect': ['Navy', 'Blue', 'Purple'],
                'excellent': ['Gray', 'Black', 'White', 'Brown'],
                'good': ['Green', 'Orange', 'Red', 'Pink'],
                'avoid': ['Light Green', 'Cream', 'Beige']
            },
            'Navy': {
                'perfect': ['White', 'Yellow', 'Red'],
                'excellent': ['Gray', 'Blue', 'Orange', 'Pink'],
                'good': ['Green', 'Brown', 'Gold', 'Silver'],
                'avoid': ['Black', 'Purple', 'Maroon']
            },
            'Gray': {
                'perfect': ['Black', 'White', 'Navy'],
                'excellent': ['Blue', 'Red', 'Yellow', 'Pink'],
                'good': ['Green', 'Orange', 'Purple', 'Brown'],
                'avoid': ['Beige', 'Cream', 'Light Gray']
            },
            'Brown': {
                'perfect': ['Beige', 'White', 'Green'],
                'excellent': ['Blue', 'Orange', 'Yellow', 'Navy'],
                'good': ['Red', 'Gray', 'Pink', 'Gold'],
                'avoid': ['Black', 'Purple', 'Bright Colors']
            },
            'Pink': {
                'perfect': ['Gray', 'Navy', 'White'],
                'excellent': ['Blue', 'Green', 'Yellow', 'Brown'],
                'good': ['Black', 'Red', 'Orange', 'Purple'],
                'avoid': ['Bright Pink', 'Maroon', 'Dark Colors']
            },
            'Purple': {
                'perfect': ['Yellow', 'Green', 'Gray'],
                'excellent': ['White', 'Pink', 'Orange', 'Silver'],
                'good': ['Blue', 'Navy', 'Brown', 'Gold'],
                'avoid': ['Red', 'Black', 'Maroon']
            },
            'Orange': {
                'perfect': ['Blue', 'Navy', 'Brown'],
                'excellent': ['Green', 'Yellow', 'White', 'Gray'],
                'good': ['Black', 'Red', 'Pink', 'Purple'],
                'avoid': ['Bright Orange', 'Bright Red', 'Maroon']
            },
            'Beige': {
                'perfect': ['Brown', 'White', 'Navy'],
                'excellent': ['Blue', 'Green', 'Orange', 'Gray'],
                'good': ['Red', 'Yellow', 'Pink', 'Purple'],
                'avoid': ['Black', 'Bright Colors', 'Neon']
            }
        }
    
    def _initialize_style_formality(self) -> Dict:
        """Initialize style formality hierarchy (1=most casual, 10=most formal)."""
        return {
            'Tank Top': 1, 'Sleeveless': 1, 'Crop Top': 1,
            'T-Shirt': 2, 'Casual Shirt': 2, 'Polo': 2,
            'Henley': 3, 'Long Sleeve': 3, 'Hoodie': 3,
            'Blouse': 4, 'Tunic': 4, 'Cardigan': 4,
            'Button-Up': 5, 'Dress Shirt': 5, 'Sweater': 5,
            'Business Casual': 6, 'Smart Casual': 6, 'Blazer': 6,
            'Sports Coat': 7, 'Dress Coat': 7, 'Formal Shirt': 7,
            'Business Formal': 8, 'Suit Jacket': 8, 'Dinner Jacket': 8,
            'Formal Blazer': 9, 'Tuxedo Shirt': 9, 'Evening Wear': 9,
            'Black Tie': 10, 'White Tie': 10, 'Gala': 10,
            
            # Bottoms
            'Shorts': 1, 'Mini Skirt': 1, 'Yoga Pants': 1,
            'Jeans': 2, 'Casual Pants': 2, 'Sweatpants': 2,
            'Khakis': 3, 'Chinos': 3, 'Midi Skirt': 3,
            'Dress Pants': 4, 'Slacks': 4, 'A-Line Skirt': 4,
            'Business Pants': 5, 'Pencil Skirt': 5, 'Straight Pants': 5,
            'Formal Pants': 6, 'Suit Pants': 6, 'Long Skirt': 6,
            'Dress Trousers': 7, 'Formal Skirt': 7, 'Evening Pants': 7,
            'Cocktail Pants': 8, 'Formal Trousers': 8, 'Gala Skirt': 8,
            'Tuxedo Pants': 9, 'Evening Trousers': 9, 'Ball Gown': 9,
            'White Tie Pants': 10, 'Formal Evening': 10
        }
    
    def _initialize_seasonal_preferences(self) -> Dict:
        """Initialize seasonal color, fabric, and pattern preferences."""
        return {
            'Spring': {
                'colors': ['Light Pink', 'Yellow', 'Light Green', 'Sky Blue', 'Coral', 'Mint'],
                'patterns': ['Floral', 'Light Stripes', 'Small Dots', 'Pastels'],
                'fabrics': ['Cotton', 'Linen', 'Light Knit', 'Chiffon'],
                'avoid_colors': ['Dark Brown', 'Black', 'Heavy Colors']
            },
            'Summer': {
                'colors': ['White', 'Light Blue', 'Yellow', 'Coral', 'Turquoise', 'Lime'],
                'patterns': ['Tropical', 'Beach', 'Light Patterns', 'Bright Stripes'],
                'fabrics': ['Linen', 'Cotton', 'Breathable', 'Light'],
                'avoid_colors': ['Black', 'Dark Colors', 'Heavy Patterns']
            },
            'Autumn': {
                'colors': ['Brown', 'Orange', 'Burgundy', 'Forest Green', 'Mustard', 'Rust'],
                'patterns': ['Plaid', 'Earth Tones', 'Rich Patterns', 'Warm Stripes'],
                'fabrics': ['Wool', 'Flannel', 'Corduroy', 'Thick Knit'],
                'avoid_colors': ['Bright Pink', 'Neon', 'Light Pastels']
            },
            'Winter': {
                'colors': ['Black', 'Navy', 'Gray', 'Burgundy', 'Forest Green', 'Deep Purple'],
                'patterns': ['Solid', 'Dark Patterns', 'Rich Textures', 'Classic'],
                'fabrics': ['Wool', 'Cashmere', 'Heavy Knit', 'Thermal'],
                'avoid_colors': ['Light Pink', 'Yellow', 'Light Colors']
            }
        }
    
    def _initialize_pattern_compatibility(self) -> Dict:
        """Initialize pattern and texture mixing rules from fashion designer data."""
        try:
            # Load fashion designer style mixing data
            style_mixing_file = "Fashion designer input/Style Mixing.csv"
            if os.path.exists(style_mixing_file):
                logger.info("🎭 Loading fashion designer style mixing data...")
                return self._load_fashion_designer_style_mixing(style_mixing_file)
            else:
                logger.warning("⚠️ Fashion designer style mixing file not found, using fallback rules")
                return self._get_fallback_pattern_compatibility()
        except Exception as e:
            logger.error(f"Error loading fashion designer style mixing: {e}")
            return self._get_fallback_pattern_compatibility()
    
    def _load_fashion_designer_style_mixing(self, file_path: str) -> Dict:
        """Load style mixing rules from fashion designer CSV data."""
        style_mixing = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the style mixing rules
            lines = content.split('\n')
            current_style = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for style headers
                if 'BUSINESS/FORMAL Style:' in line:
                    current_style = 'formal'
                    style_mixing[current_style] = {'compatible': [], 'avoid': [], 'special_rules': []}
                elif 'CASUAL Style:' in line:
                    current_style = 'casual'
                    style_mixing[current_style] = {'compatible': [], 'avoid': [], 'special_rules': []}
                elif 'STREETWEAR Style:' in line:
                    current_style = 'streetwear'
                    style_mixing[current_style] = {'compatible': [], 'avoid': [], 'special_rules': []}
                
                # Parse compatibility rules
                if current_style and 'Pairs perfectly with:' in line:
                    compatible = line.split('Pairs perfectly with:')[1].strip().strip('"')
                    style_mixing[current_style]['compatible'].extend([s.strip() for s in compatible.split(',')])
                elif current_style and 'Never pair with:' in line:
                    avoid = line.split('Never pair with:')[1].strip().strip('"')
                    style_mixing[current_style]['avoid'].extend([s.strip() for s in avoid.split(',')])
                elif current_style and 'Special mixing rule:' in line:
                    rule = line.split('Special mixing rule:')[1].strip().strip('"')
                    style_mixing[current_style]['special_rules'].append(rule)
            
            logger.info(f"✅ Loaded fashion designer style mixing for {len(style_mixing)} styles")
            return style_mixing
            
        except Exception as e:
            logger.error(f"Error parsing fashion designer style mixing: {e}")
            return self._get_fallback_pattern_compatibility()
    
    def _get_fallback_pattern_compatibility(self) -> Dict:
        """Fallback pattern compatibility rules if fashion designer data is not available."""
        return {
            'Solid': ['Stripes', 'Dots', 'Floral', 'Plaid', 'Geometric'],
            'Stripes': ['Solid', 'Small Dots', 'Subtle Patterns'],
            'Dots': ['Solid', 'Stripes', 'Subtle Patterns'],
            'Floral': ['Solid', 'Subtle Stripes', 'Plain'],
            'Plaid': ['Solid', 'Plain', 'Subtle Textures'],
            'Geometric': ['Solid', 'Plain', 'Minimal Patterns'],
            'Abstract': ['Solid', 'Plain', 'Neutral'],
            'Animal Print': ['Solid', 'Plain', 'Neutral'],
            'Paisley': ['Solid', 'Plain', 'Subtle'],
            'Textured': ['Solid', 'Plain', 'Smooth']
        }
    
    def validate_products_data(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Validate, clean, and enhance products data with proper price mapping."""
        required_columns = ["title", "product_type", "gender"]
        missing_columns = [col for col in required_columns if col not in products_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in products data: {missing_columns}")
        
        # Clean data
        products_df = products_df.dropna(subset=["title", "product_type"])
        
        # ✅ CRITICAL FIX: Merge actual price data from scraped products
        try:
            scraped_products = pd.read_csv("data/products_scraped.csv")
            if 'price' in scraped_products.columns and 'id' in scraped_products.columns:
                price_mapping = scraped_products[['id', 'price']].dropna()
                price_mapping = price_mapping[price_mapping['price'] > 0]
                
                # Create mapping dictionary for faster lookup
                price_dict = dict(zip(price_mapping['id'].astype(str), price_mapping['price']))
                
                # Map prices using product ID
                def get_actual_price(row):
                    product_id = str(row.get('product_id', ''))
                    if product_id in price_dict:
                        return float(price_dict[product_id])
                    
                    # Smart default pricing based on product_type and style
                    product_type = row.get('product_type', '')
                    style = row.get('enhanced_primary_style', row.get('primary_style', ''))
                    
                    if product_type.lower() in ['t-shirt', 'shirt', 'top', 'blouse', 'sweater', 'hoodie']:
                        if any(x in style.lower() for x in ['formal', 'business', 'blazer']):
                            return 2500
                        elif any(x in style.lower() for x in ['casual', 't-shirt', 'tank']):
                            return 800
                        else:
                            return 1500
                    elif product_type.lower() in ['trousers', 'pants', 'jeans', 'shorts', 'skirt']:
                        if any(x in style.lower() for x in ['formal', 'trouser', 'chino']):
                            return 2000
                        elif any(x in style.lower() for x in ['casual', 'jeans', 'short']):
                            return 1200
                        else:
                            return 1400
                    return 1000
                
                products_df['price'] = products_df.apply(get_actual_price, axis=1)
                logger.info(f"✅ Loaded {len(price_dict)} price mappings from scraped data")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load price mappings: {e}")
            products_df['price'] = products_df.get('price', 1000)
        
        # Ensure we have caption data for FAISS
        if 'final_caption' not in products_df.columns and 'full_caption' not in products_df.columns:
            logger.warning("No caption columns found, using title for FAISS embeddings")
            products_df['final_caption'] = products_df['title']
        
        # Fix product_id mapping
        if 'product_id' not in products_df.columns and 'id' in products_df.columns:
            products_df['product_id'] = products_df['id']
        
        logger.info(f"✅ Products validation complete. Price range: ₹{products_df['price'].min():.0f} - ₹{products_df['price'].max():.0f}")
        return products_df
    
    def get_embedding_cached(self, text: str, cache_key: str = None) -> np.ndarray:
        """Get embedding with caching for better performance."""
        # ✅ OPTIMIZATION: Ensure model is loaded
        self._ensure_model_loaded()
        
        if not cache_key:
            cache_key = text[:100]
        
        if self.config['cache_embeddings'] and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.model.encode([text])[0]
        
        if self.config['cache_embeddings']:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def calculate_color_harmony_score(self, color1: str, color2: str) -> float:
        """Calculate sophisticated color harmony score between two colors."""
        if not color1 or not color2:
            return 0.5
        
        color1 = str(color1).strip().title()
        color2 = str(color2).strip().title()
        
        # Same color gets perfect score
        if color1 == color2:
            return 1.0
        
        # Check sophisticated color harmony rules
        def get_color_score(c1, c2):
            if c1 in self.color_harmony:
                color_rules = self.color_harmony[c1]
                if c2 in color_rules.get('perfect', []):
                    return 1.0
                elif c2 in color_rules.get('excellent', []):
                    return 0.9
                elif c2 in color_rules.get('good', []):
                    return 0.7
                elif c2 in color_rules.get('avoid', []):
                    return 0.2
            return 0.5
        
        # Check both directions for color harmony
        score1 = get_color_score(color1, color2)
        score2 = get_color_score(color2, color1)
        
        # Return the better score (most harmonious)
        return max(score1, score2)
    
    def build_faiss_indexes(self, products_df: pd.DataFrame) -> None:
        """Build FAISS indexes for different wear types using precomputed embeddings from database."""
        logger.info("Building FAISS indexes for similar outfit search...")
        self._ensure_model_loaded()
        if self.is_railway:
            logger.info("🏭 Applying Railway CPU limits for FAISS indexing operations")
            for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                os.environ[var] = '1'
        
        # ✅ FIX: Use product_type to categorize as topwear/bottomwear instead of non-existent wear_type
        def categorize_wear_type(product_type):
            """Categorize product type as topwear or bottomwear."""
            topwear_keywords = [
                'shirt', 't-shirt', 'polo', 'henley', 'tank', 'crop', 'sleeveless', 
                'blouse', 'tunic', 'sweater', 'hoodie', 'jacket', 'blazer', 'cardigan',
                'coat', 'vest', 'top', 'tee', 'tshirt'
            ]
            bottomwear_keywords = [
                'pants', 'jeans', 'shorts', 'chinos', 'trousers', 'slacks', 'dress_pants',
                'casual_pants', 'formal_pants', 'joggers', 'leggings', 'skirt', 'dress',
                'jumpsuit', 'overall', 'denim', 'cargo', 'khaki', 'sweatpants'
            ]
            
            product_type_lower = product_type.lower()
            
            # Check for topwear keywords
            if any(keyword in product_type_lower for keyword in topwear_keywords):
                return 'Upperwear'
            # Check for bottomwear keywords
            elif any(keyword in product_type_lower for keyword in bottomwear_keywords):
                return 'Bottomwear'
            else:
                # Default categorization based on common patterns
                if any(word in product_type_lower for word in ['shirt', 'top', 'tee', 'blouse']):
                    return 'Upperwear'
                elif any(word in product_type_lower for word in ['pants', 'shorts', 'jeans', 'skirt']):
                    return 'Bottomwear'
                else:
                    return 'Other'
        
        # Add wear_type column based on product_type
        products_df['wear_type'] = products_df['product_type'].apply(categorize_wear_type)
        
        # Use unique wear types
        wear_types = products_df['wear_type'].dropna().unique()
        logger.info(f"🔍 Found wear types: {wear_types}")
        
        for wear_type in wear_types:
            type_products = products_df[products_df['wear_type'] == wear_type].copy()
            if type_products.empty:
                logger.warning(f"No products found for wear_type: {wear_type}")
                continue
            embeddings = []
            product_indices = []
            missing_embeddings = []
            for idx, row in type_products.iterrows():
                product_id = str(row.get('product_id', ''))
                if 'product_embedding' in row and pd.notna(row['product_embedding']):
                    try:
                        import json
                        embedding_list = json.loads(row['product_embedding'])
                        embedding = np.array(embedding_list, dtype=np.float32)
                        embeddings.append(embedding)
                        product_indices.append(idx)
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to parse embedding for product {product_id}: {e}")
                caption = row.get('full_caption', '') or row.get('final_caption', '') or row.get('title', '')
                if caption.strip():
                    missing_embeddings.append((idx, caption))
            if missing_embeddings:
                logger.info(f"Computing {len(missing_embeddings)} missing embeddings for {wear_type}...")
                missing_indices, missing_captions = zip(*missing_embeddings)
                batch_size = min(50, len(missing_captions))
                for i in range(0, len(missing_captions), batch_size):
                    batch_captions = missing_captions[i:i + batch_size]
                    batch_embeddings = self.model.encode(batch_captions, show_progress_bar=False)
                    for j, embedding in enumerate(batch_embeddings):
                        embeddings.append(embedding)
                        product_indices.append(missing_indices[i + j])
            if not embeddings:
                logger.warning(f"No valid embeddings found for wear_type: {wear_type}")
                continue
            logger.info(f"Using {len(embeddings)} embeddings for {wear_type} (precomputed: {len(embeddings) - len(missing_embeddings)}, computed: {len(missing_embeddings)})")
            embeddings = np.array(embeddings)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index.add(embeddings.astype('float32'))
            self.faiss_indexes[wear_type] = index
            self.product_mappings[wear_type] = {
                'indices': product_indices,
                'products': type_products.iloc[[type_products.index.get_loc(idx) for idx in product_indices]].copy()
            }
            logger.info(f"Built FAISS index for {wear_type}: {len(embeddings)} products indexed (PRECOMPUTED)")
    
    def load_outfit_from_supabase(self, main_outfit_id: str) -> Dict:
        """Load outfit data from Supabase database."""
        try:
            logger.info(f"Loading outfit {main_outfit_id} from Supabase...")
            
            # Query the user_outfits table
            result = self.db.client.table('user_outfits').select('*').eq('main_outfit_id', main_outfit_id).execute()
            
            if not result.data:
                logger.error(f"❌ Outfit {main_outfit_id} not found")
                raise ValueError(f"Outfit {main_outfit_id} not found in database")
            
            outfit_data = result.data[0]
            
            # Get user data for gender-based filtering using the database method
            user_id = outfit_data.get('user_id')
            if user_id:
                user_data = self.db.get_user_with_style_quiz(user_id)
                
                if user_data:
                    outfit_data['user_data'] = user_data
                    logger.info(f"✅ Loaded user data for user {user_id} (gender: {user_data.get('Gender', 'unknown')})")
                else:
                    logger.warning(f"⚠️ Could not load user data for user {user_id}")
            
            logger.info(f"✅ Successfully loaded outfit {main_outfit_id} from Supabase")
            
            return outfit_data
                
        except Exception as e:
            logger.error(f"Error loading outfit {main_outfit_id} from Supabase: {e}")
            raise
    
    def load_products_from_supabase(self, user_data: Dict = None, exclude_outfit_ids: List[str] = None, main_outfit_product_types: List[str] = None) -> pd.DataFrame:
        """Load ALL products data from Supabase database with proper filtering."""
        try:
            logger.info("Loading ALL products from Supabase (no limit)...")
            products_df = self.db.get_products_phase2()
            
            # ✅ FIX: Add gender filtering to match main outfit's user
            if user_data and user_data.get('Gender'):
                user_gender = user_data['Gender'].lower()
                logger.info(f"🔍 Filtering products by user gender: {user_gender}")
                products_df = products_df[products_df['gender'].str.lower() == user_gender]
                logger.info(f"✅ After gender filtering: {len(products_df)} products remaining")
            
            # ✅ FIX: Add product type filtering to match main outfit's product types
            if main_outfit_product_types:
                logger.info(f"🔍 Filtering products by main outfit product types: {main_outfit_product_types}")
                # Get related product types for diversity
                related_types = self._get_related_product_types(main_outfit_product_types)
                all_allowed_types = main_outfit_product_types + related_types
                logger.info(f"🔍 Allowed product types: {all_allowed_types}")
                
                # Filter products by allowed product types
                products_df = products_df[products_df['product_type'].isin(all_allowed_types)]
                logger.info(f"✅ After product type filtering: {len(products_df)} products remaining")
            
            if exclude_outfit_ids:
                logger.info(f"Excluding {len(exclude_outfit_ids)} main outfit products (exact IDs only)")
                products_df = products_df[~products_df['product_id'].isin(exclude_outfit_ids)]
                logger.info(f"After ID exclusion: {len(products_df)} products remaining")
            
            if len(products_df) > 0:
                logger.info("Shuffling products while maintaining type priority...")
                products_df = products_df.sample(frac=1, random_state=42).reset_index(drop=True)
                logger.info(f"Final dataset: {len(products_df)} products ready for similar outfit generation")
                return products_df
            else:
                logger.warning("⚠️ No products available after filtering")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading products from Supabase: {e}")
            return pd.DataFrame()
    
    def _get_related_product_types(self, main_product_types: List[str]) -> List[str]:
        """Get related product types for diversity while maintaining similarity."""
        related_types = []
        
        # Define related product type mappings
        type_relationships = {
            # Formal shirts related types
            'formal_shirt': ['casual_shirt', 'polo_shirt', 'henley', 'tunic', 'blouse'],
            'casual_shirt': ['formal_shirt', 'polo_shirt', 'henley', 'tunic', 'blouse'],
            'polo_shirt': ['formal_shirt', 'casual_shirt', 'henley', 't-shirt'],
            'henley': ['casual_shirt', 'polo_shirt', 't-shirt', 'sweater'],
            
            # Tops related types
            't-shirt': ['henley', 'casual_shirt', 'polo_shirt', 'tank_top'],
            'tank_top': ['crop_top', 'sleeveless', 't-shirt', 'blouse'],
            'crop_top': ['tank_top', 'sleeveless', 't-shirt', 'blouse'],
            'sleeveless': ['tank_top', 'crop_top', 't-shirt', 'blouse'],
            'blouse': ['casual_shirt', 'formal_shirt', 'tunic', 'sleeveless'],
            'tunic': ['blouse', 'casual_shirt', 'formal_shirt', 'dress'],
            
            # Formal bottoms related types
            'formal_pants': ['chinos', 'dress_pants', 'slacks', 'casual_pants', 'trousers'],
            'chinos': ['formal_pants', 'dress_pants', 'casual_pants', 'jeans', 'trousers'],
            'dress_pants': ['formal_pants', 'chinos', 'slacks', 'trousers'],
            'slacks': ['formal_pants', 'dress_pants', 'chinos', 'trousers'],
            'trousers': ['formal_pants', 'chinos', 'dress_pants', 'slacks'],
            
            # Casual bottoms related types
            'casual_pants': ['chinos', 'jeans', 'formal_pants', 'shorts'],
            'jeans': ['casual_pants', 'chinos', 'shorts', 'denim_pants'],
            'shorts': ['jeans', 'casual_pants', 'chinos', 'denim_shorts'],
            'denim_pants': ['jeans', 'casual_pants', 'shorts'],
            
            # Skirts related types
            'pencil_skirt': ['a_line_skirt', 'midi_skirt', 'mini_skirt', 'maxi_skirt'],
            'a_line_skirt': ['pencil_skirt', 'midi_skirt', 'mini_skirt', 'maxi_skirt'],
            'midi_skirt': ['a_line_skirt', 'pencil_skirt', 'mini_skirt', 'maxi_skirt'],
            'mini_skirt': ['a_line_skirt', 'midi_skirt', 'pencil_skirt'],
            'maxi_skirt': ['a_line_skirt', 'midi_skirt', 'pencil_skirt'],
            
            # Outerwear related types
            'blazer': ['jacket', 'cardigan', 'sweater', 'coat'],
            'jacket': ['blazer', 'cardigan', 'sweater', 'coat'],
            'cardigan': ['sweater', 'blazer', 'jacket'],
            'sweater': ['cardigan', 'blazer', 'jacket', 'hoodie'],
            'hoodie': ['sweater', 'jacket', 'sweatshirt'],
            'coat': ['blazer', 'jacket', 'overcoat', 'trench_coat'],
            
            # Special categories
            'dress': ['tunic', 'blouse', 'maxi_dress', 'mini_dress'],
            'maxi_dress': ['dress', 'tunic', 'gown'],
            'mini_dress': ['dress', 'tunic', 'blouse'],
            'gown': ['maxi_dress', 'dress', 'formal_dress'],
            'jumpsuit': ['dress', 'pants', 'overall'],
            'overall': ['jumpsuit', 'pants', 'dungarees']
        }
        
        # Get related types for each main product type
        for product_type in main_product_types:
            if product_type in type_relationships:
                related_types.extend(type_relationships[product_type])
        
        # Remove duplicates and main types
        related_types = list(set(related_types) - set(main_product_types))
        
        return related_types
    
    def calculate_outfit_similarity(self, source_outfit: Dict, candidate_outfit: Dict) -> Tuple[float, Dict]:
        """Calculate sophisticated similarity score with advanced fashion intelligence."""
        
        scores = {}
        explanations = []
        
        # 1. SEMANTIC SIMILARITY (Core AI matching)
        source_text = f"{source_outfit.get('top_title', '')} {source_outfit.get('bottom_title', '')}"
        candidate_text = f"{candidate_outfit.get('top_title', '')} {candidate_outfit.get('bottom_title', '')}"
        
        source_embedding = self.get_embedding_cached(source_text)
        candidate_embedding = self.get_embedding_cached(candidate_text)
        
        semantic_score = float(np.dot(source_embedding, candidate_embedding))
        scores['semantic_similarity'] = semantic_score
        explanations.append(f"Semantic match: {semantic_score:.3f}")
        
        # 2. ADVANCED STYLE HARMONY (Not just exact matching)
        source_top_style = source_outfit.get('top_style', '')
        source_bottom_style = source_outfit.get('bottom_style', '')
        candidate_top_style = candidate_outfit.get('top_style', '')
        candidate_bottom_style = candidate_outfit.get('bottom_style', '')
        
        # Calculate style compatibility rather than exact matching
        def calculate_style_compatibility(style1, style2):
            if not style1 or not style2:
                return 0.5
            if style1 == style2:
                return 1.0
            
            # ✅ ENHANCED: Use fashion designer style mixing data
            style1_lower = style1.lower()
            style2_lower = style2.lower()
            
            # Check fashion designer compatibility rules
            for style_category, rules in self.pattern_compatibility.items():
                if style_category in style1_lower or style1_lower in style_category:
                    # Check if style2 is compatible with this category
                    if any(compatible in style2_lower for compatible in rules.get('compatible', [])):
                        return 0.9  # Fashion designer approved
                    elif any(avoid in style2_lower for avoid in rules.get('avoid', [])):
                        return 0.3  # Fashion designer disapproved
            
            # Fallback to formality compatibility
            form1 = self.style_formality.get(style1, 5)
            form2 = self.style_formality.get(style2, 5)
            formality_diff = abs(form1 - form2)
            
            if formality_diff <= 1:
                return 0.8  # Very compatible
            elif formality_diff <= 2:
                return 0.6  # Somewhat compatible  
            elif formality_diff <= 3:
                return 0.4  # Moderately compatible
            else:
                return 0.2  # Less compatible
        
        top_style_compat = calculate_style_compatibility(source_top_style, candidate_top_style)
        bottom_style_compat = calculate_style_compatibility(source_bottom_style, candidate_bottom_style)
        style_score = (top_style_compat + bottom_style_compat) / 2
        
        scores['style_harmony'] = style_score
        explanations.append(f"Style compatibility: {style_score:.3f}")
        
        # 3. SOPHISTICATED COLOR HARMONY
        source_top_color = source_outfit.get('top_color', '')
        source_bottom_color = source_outfit.get('bottom_color', '')
        candidate_top_color = candidate_outfit.get('top_color', '')
        candidate_bottom_color = candidate_outfit.get('bottom_color', '')
        
        # Compare outfit color harmony
        source_color_harmony = self.calculate_color_harmony_score(source_top_color, source_bottom_color)
        candidate_color_harmony = self.calculate_color_harmony_score(candidate_top_color, candidate_bottom_color)
        
        # Compare individual piece color similarity
        top_color_similarity = self.calculate_color_harmony_score(source_top_color, candidate_top_color)
        bottom_color_similarity = self.calculate_color_harmony_score(source_bottom_color, candidate_bottom_color)
        
        # Weighted color score
        color_score = (
            source_color_harmony * 0.3 +
            candidate_color_harmony * 0.3 + 
            top_color_similarity * 0.2 +
            bottom_color_similarity * 0.2
        )
        
        scores['color_harmony'] = color_score
        explanations.append(f"Color harmony: {color_score:.3f}")
        
        # 4. FORMALITY MATCHING (Simplified - no tolerance)
        source_top_formality = self.style_formality.get(source_top_style, 5)
        source_bottom_formality = self.style_formality.get(source_bottom_style, 5)
        candidate_top_formality = self.style_formality.get(candidate_top_style, 5)
        candidate_bottom_formality = self.style_formality.get(candidate_bottom_style, 5)
        
        source_avg_formality = (source_top_formality + source_bottom_formality) / 2
        candidate_avg_formality = (candidate_top_formality + candidate_bottom_formality) / 2
        
        formality_diff = abs(source_avg_formality - candidate_avg_formality)
        # ✅ FIX: Simplified formality scoring without tolerance
        formality_score = max(0.1, 1 - (formality_diff / 10))  # Gradual decrease
        
        scores['formality_matching'] = formality_score
        explanations.append(f"Formality match: {formality_score:.3f}")
        
        # 5. PATTERN COMPATIBILITY
        source_top_pattern = source_outfit.get('top_pattern', 'Solid')
        source_bottom_pattern = source_outfit.get('bottom_pattern', 'Solid')
        candidate_top_pattern = candidate_outfit.get('top_pattern', 'Solid')
        candidate_bottom_pattern = candidate_outfit.get('bottom_pattern', 'Solid')
        
        def check_pattern_compatibility(p1, p2):
            if p1 == p2:
                return 1.0
            if p1 in self.pattern_compatibility and p2 in self.pattern_compatibility[p1]:
                return 0.8
            if p2 in self.pattern_compatibility and p1 in self.pattern_compatibility[p2]:
                return 0.8
            return 0.6
        
        pattern_score = (
            check_pattern_compatibility(source_top_pattern, candidate_top_pattern) +
            check_pattern_compatibility(source_bottom_pattern, candidate_bottom_pattern)
        ) / 2
        
        scores['pattern_compatibility'] = pattern_score
        explanations.append(f"Pattern compatibility: {pattern_score:.3f}")
        
        # 6. PRICE COMPATIBILITY (Simplified - no tolerance)
        source_price = float(source_outfit.get('total_price', 0))
        candidate_price = float(candidate_outfit.get('total_price', 0))
        
        if source_price > 0 and candidate_price > 0:
            # ✅ FIX: Simplified price scoring without tolerance
            price_ratio = min(source_price, candidate_price) / max(source_price, candidate_price)
            price_score = price_ratio  # Direct ratio (0.5 = 50% difference, 1.0 = same price)
        else:
            price_score = 0.8  # Neutral if missing data
        
        scores['price_similarity'] = price_score
        explanations.append(f"Price compatibility: {price_score:.3f}")
        
        # 7. OCCASION MATCHING
        source_occasion = source_outfit.get('top_occasion', '') or source_outfit.get('bottom_occasion', '')
        candidate_occasion = candidate_outfit.get('top_occasion', '') or candidate_outfit.get('bottom_occasion', '')
        
        if source_occasion and candidate_occasion:
            occasion_score = 1.0 if source_occasion == candidate_occasion else 0.6
        else:
            occasion_score = 0.8  # Neutral if missing data
        
        scores['occasion_matching'] = occasion_score
        explanations.append(f"Occasion match: {occasion_score:.3f}")
        
        # 8. SEASONAL APPROPRIATENESS (Simplified)
        # ✅ FIX: Simplified seasonal scoring
        scores['seasonal_appropriateness'] = 0.8  # Neutral score
        
        # Calculate sophisticated weighted final score
        weighted_score = (
            scores['semantic_similarity'] * self.similarity_config['semantic_weight'] +
            scores['style_harmony'] * self.similarity_config['style_harmony_weight'] +
            scores['color_harmony'] * self.similarity_config['color_harmony_weight'] +
            scores['formality_matching'] * 2.0 +  # ✅ FIX: Fixed weight
            scores['pattern_compatibility'] * self.similarity_config['pattern_compatibility_weight'] +
            scores['price_similarity'] * 1.5 +  # ✅ FIX: Fixed weight
            scores['occasion_matching'] * self.similarity_config['occasion_weight'] +
            scores['seasonal_appropriateness'] * 1.0  # ✅ FIX: Fixed weight
        )
        
        # Normalize by total weights
        total_weights = (
            self.similarity_config['semantic_weight'] + 
            self.similarity_config['style_harmony_weight'] +
            self.similarity_config['color_harmony_weight'] +
            2.0 +  # formality weight
            self.similarity_config['pattern_compatibility_weight'] +
            1.5 +  # price weight
            self.similarity_config['occasion_weight'] +
            1.0   # seasonal weight
        )
        
        final_score = weighted_score / total_weights
        
        # ✅ NEW: Best Available fallback - ensure minimum score for diversity
        if self.similarity_config.get('fallback_strategy') == 'best_available':
            final_score = max(final_score, 0.1)  # Minimum score to ensure inclusion
        
        # Add detailed breakdown for debugging
        scores['final_score'] = final_score
        scores['explanation'] = " | ".join(explanations)
        
        return final_score, scores
    
    def save_similar_outfits_to_database(self, main_outfit_id: str, similar_outfits: List[Dict]) -> bool:
        """Save similar outfits to the similar_outfits table."""
        try:
            logger.info(f"Saving {len(similar_outfits)} similar outfits to database for {main_outfit_id}")
            
            # Get the why_picked_explanation from the main outfit
            main_outfit_result = self.db.client.table('user_outfits').select('why_picked_explanation').eq('main_outfit_id', main_outfit_id).execute()
            why_picked_explanation = ""
            if main_outfit_result.data:
                why_picked_explanation = main_outfit_result.data[0].get('why_picked_explanation', '')
            
            # Prepare data for insertion
            similar_outfits_data = []
            for i, outfit in enumerate(similar_outfits, 1):
                outfit_data = outfit['outfit_data']
                similarity_score = float(outfit['similarity_score'])  # Convert to float
                
                similar_outfit_id = f"similar_{main_outfit_id.replace('main_', '')}_{i}"
                
                similar_outfits_data.append({
                    'main_outfit_id': main_outfit_id,
                    'similar_outfit_id': similar_outfit_id,
                    'similar_top_id': str(outfit_data['top_id']),  # Convert to string
                    'similar_bottom_id': str(outfit_data['bottom_id']),  # Convert to string
                    'similar_top_image': outfit_data.get('top_image', ''),
                    'similar_bottom_image': outfit_data.get('bottom_image', ''),
                    'outfit_name': outfit_data.get('outfit_name', ''),
                    'outfit_description': outfit_data.get('outfit_description', ''),
                    'why_picked_explanation': why_picked_explanation,
                    'similarity_score': similarity_score
                })
            
            # ✅ FIX: Use upsert to handle existing records
            # First, delete existing similar outfits for this main outfit
            delete_result = self.db.client.table('similar_outfits').delete().eq('main_outfit_id', main_outfit_id).execute()
            logger.info(f"Deleted {len(delete_result.data) if delete_result.data else 0} existing similar outfits for {main_outfit_id}")
            
            # Then insert the new similar outfits
            result = self.db.client.table('similar_outfits').insert(similar_outfits_data).execute()
            
            if result.data:
                logger.info(f"✅ Successfully saved {len(result.data)} similar outfits to database")
                return True
            else:
                logger.error("❌ Failed to save similar outfits to database")
                return False
                
        except Exception as e:
            logger.error(f"Error saving similar outfits to database: {e}")
            return False

    def find_similar_outfits(self, outfit_id: str, num_similar: int = 20) -> List[Dict]:
        # Apply Railway CPU optimization before heavy computation
        if self.is_railway:
            for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                os.environ[var] = '2'
            logger.info("🔧 Applied CPU limits for similar outfits computation")
        
        """Find similar outfits for a given outfit using Supabase data."""
        
        try:
            # Stage 1: Load source outfit
            source_outfit = self.load_outfit_from_supabase(outfit_id)
            
            logger.info(f"Finding similar outfits for outfit: {outfit_id}")
            
            # Get user data for gender-based filtering
            user_data = source_outfit.get('user_data', {})
            
            # Extract main outfit product IDs to exclude them from similar outfits
            main_top_id = str(source_outfit['top_id'])
            main_bottom_id = str(source_outfit['bottom_id'])
            exclude_outfit_ids = [main_top_id, main_bottom_id]
            
            # Get product types of main outfit products to exclude same types
            main_outfit_product_types = []
            if main_top_id and main_bottom_id:
                # ✅ FIX: Use product_id instead of id for queries
                top_result = self.db.client.table('tagged_products').select('product_type').eq('product_id', main_top_id).execute()
                bottom_result = self.db.client.table('tagged_products').select('product_type').eq('product_id', main_bottom_id).execute()
                
                if top_result.data:
                    main_outfit_product_types.append(top_result.data[0].get('product_type'))
                if bottom_result.data:
                    main_outfit_product_types.append(bottom_result.data[0].get('product_type'))
                
                logger.info(f"📊 Main outfit product types: {main_outfit_product_types}")
            
            logger.info(f"Excluding main outfit products: top_id={main_top_id}, bottom_id={main_bottom_id}")
            logger.info(f"Main outfit product types: {main_outfit_product_types}")
            
            # Stage 2: Load and filter products
            products_df = self.load_products_from_supabase(user_data, exclude_outfit_ids, main_outfit_product_types)
            
            if products_df.empty:
                logger.error("❌ No products available for similar outfit generation")
                return []
            
            # Stage 3: Build FAISS indexes (OPTIMIZED - only filtered products)
            self.build_faiss_indexes(products_df)
            
            # Stage 4: Compute similar outfits
            similar_outfits = self._compute_similar_outfits(source_outfit, products_df, num_similar)
            
            # Stage 5: Save to database
            if similar_outfits:
                self.save_similar_outfits_to_database(outfit_id, similar_outfits)
            
            logger.info(f"✅ Found {len(similar_outfits)} similar outfits for {outfit_id}")
            return similar_outfits
            
        except Exception as e:
            logger.error(f"Error finding similar outfits: {e}")
            return []
    
    def generate_candidate_outfits(self, source_outfit: Dict, products_df: pd.DataFrame) -> List[Dict]:
        """Generate candidate outfits for similarity comparison with proper product type filtering."""
        
        candidates = []
        
        # Get source outfit details
        source_top_id = source_outfit.get('top_id', '')
        source_bottom_id = source_outfit.get('bottom_id', '')
        
        # ✅ FIX: Get source outfit product types for better filtering
        source_top_type = None
        source_bottom_type = None
        
        if source_top_id:
            top_result = self.db.client.table('tagged_products').select('product_type').eq('product_id', source_top_id).execute()
            if top_result.data:
                source_top_type = top_result.data[0].get('product_type')
        
        if source_bottom_id:
            bottom_result = self.db.client.table('tagged_products').select('product_type').eq('product_id', source_bottom_id).execute()
            if bottom_result.data:
                source_bottom_type = bottom_result.data[0].get('product_type')
        
        logger.info(f"🔍 Source outfit types - Top: {source_top_type}, Bottom: {source_bottom_type}")
        
        # Use FAISS to find similar tops and bottoms separately
        source_top_title = source_outfit.get('top_title', '')
        source_bottom_title = source_outfit.get('bottom_title', '')
        
        # Find similar tops and bottoms with larger pool for better diversity
        similar_tops = self.find_similar_products(source_top_title, 'Upperwear', k=15)  # Increased pool
        similar_bottoms = self.find_similar_products(source_bottom_title, 'Bottomwear', k=15)  # Increased pool
        
        # ✅ FIX: Filter bottoms by product type to match source bottom type
        if source_bottom_type:
            logger.info(f"🔍 Filtering bottoms to match source type: {source_bottom_type}")
            filtered_bottoms = []
            for bottom in similar_bottoms:
                bottom_type = bottom['product'].get('product_type', '').lower()
                source_bottom_type_lower = source_bottom_type.lower()
                
                # Exact match or related types
                if (bottom_type == source_bottom_type_lower or 
                    self._are_product_types_compatible(bottom_type, source_bottom_type_lower)):
                    filtered_bottoms.append(bottom)
            
            similar_bottoms = filtered_bottoms
            logger.info(f"✅ Filtered to {len(similar_bottoms)} compatible bottoms")
        
        # Generate outfit combinations with enhanced candidate pool
        combination_count = 0
        max_combinations = self.similarity_config['candidate_pool_size']
        
        for top_candidate in similar_tops:
            for bottom_candidate in similar_bottoms:
                if combination_count >= max_combinations:
                    break
                
                # Skip if same as source outfit
                if (top_candidate['product'].get('product_id', '') == source_top_id and
                    bottom_candidate['product'].get('product_id', '') == source_bottom_id):
                    continue
                
                # Create candidate outfit
                candidate_outfit = self.create_outfit_from_products(
                    top_candidate['product'], 
                    bottom_candidate['product']
                )
                
                candidates.append(candidate_outfit)
                combination_count += 1
            
            if combination_count >= max_combinations:
                break
        
        logger.info(f"Generated {len(candidates)} candidate outfits")
        return candidates
    
    def find_similar_products(self, query_text: str, wear_type: str, k: int = 20) -> List[Dict]:
        """Find similar products using FAISS."""
        if wear_type not in self.faiss_indexes:
            logger.warning(f"No FAISS index available for wear_type: {wear_type}")
            return []
        query_embedding = self.get_embedding_cached(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        index = self.faiss_indexes[wear_type]
        scores, indices = index.search(query_embedding.astype('float32'), k)
        product_mapping = self.product_mappings[wear_type]
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
    
    def create_outfit_from_products(self, top_product: pd.Series, bottom_product: pd.Series) -> Dict:
        """Create outfit data from top and bottom products."""
        try:
            # Generate outfit name and description
            outfit_name = self._generate_outfit_name(top_product, bottom_product, "Casual")
            outfit_description = self._generate_outfit_description(top_product, bottom_product, "Casual")
            
            outfit_data = {
                'top_id': str(top_product.get('product_id', '')),
                'top_title': str(top_product.get('title', '')),
                'top_image': str(top_product.get('image_url', '')),
                'top_price': float(top_product.get('price', 1000.0)),
                'top_style': str(top_product.get('primary_style', 'Casual')),
                'top_color': self._extract_color(top_product),
                'top_semantic_score': 0.8,
                
                'bottom_id': str(bottom_product.get('product_id', '')),
                'bottom_title': str(bottom_product.get('title', '')),
                'bottom_image': str(bottom_product.get('image_url', '')),
                'bottom_price': float(bottom_product.get('price', 1000.0)),
                'bottom_style': str(bottom_product.get('primary_style', 'Casual')),
                'bottom_color': self._extract_color(bottom_product),
                'bottom_semantic_score': 0.8,
                
                'total_price': float(top_product.get('price', 1000.0)) + float(bottom_product.get('price', 1000.0)),
                'outfit_name': outfit_name,
                'outfit_description': outfit_description,
                'generation_method': 'similar_outfits_api'
            }
            
            return outfit_data
            
        except Exception as e:
            logger.error(f"Error creating outfit from products: {e}")
            return {}

    def _compute_similar_outfits(self, source_outfit: Dict, products_df: pd.DataFrame, num_similar: int) -> List[Dict]:
        """Compute similar outfits using the filtered product data with parallel processing."""
        try:
            # Validate products data
            products_df = self.validate_products_data(products_df)
            
            # Build FAISS indexes with filtered data
            self.build_faiss_indexes(products_df)
            
            # Generate candidate outfits
            candidates = self.generate_candidate_outfits(source_outfit, products_df)
            
            if not candidates:
                logger.warning("No candidate outfits found")
                return []
            
            # ✅ OPTIMIZATION: Parallel candidate scoring
            similar_outfits = self._score_candidates_parallel(source_outfit, candidates, num_similar)
            
            logger.info(f"Found {len(similar_outfits)} unique similar outfits")
            return similar_outfits
            
        except Exception as e:
            logger.error(f"Error computing similar outfits: {e}")
            return []
    
    def _score_candidates_parallel(self, source_outfit: Dict, candidates: List[Dict], num_similar: int) -> List[Dict]:
        """Score candidates in parallel using joblib."""
        try:
            from joblib import Parallel, delayed
            import multiprocessing
            
            logger.info(f"Scoring {len(candidates)} candidates in parallel...")
            
            # Determine number of jobs (use fewer cores to avoid overwhelming the system)
            n_jobs = min(multiprocessing.cpu_count() - 1, 4, len(candidates))
            if n_jobs < 1:
                n_jobs = 1
            
            logger.info(f"Using {n_jobs} parallel jobs for candidate scoring")
            
            # Score candidates in parallel
            def score_candidate(candidate):
                try:
                    similarity_score, score_breakdown = self.calculate_outfit_similarity(source_outfit, candidate)
                    return {
                        'candidate': candidate,
                        'similarity_score': similarity_score,
                        'score_breakdown': score_breakdown
                    }
                except Exception as e:
                    logger.warning(f"Error scoring candidate: {e}")
                    return None
            
            # Process candidates in parallel
            scored_candidates = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(score_candidates_batch)(source_outfit, batch) 
                for batch in self._chunk_candidates(candidates, max(1, len(candidates) // n_jobs))
            )
            
            # Flatten results
            all_scored = []
            for batch_result in scored_candidates:
                if batch_result:
                    all_scored.extend(batch_result)
            
            # Filter and sort by similarity score
            valid_candidates = [sc for sc in all_scored if sc and sc['similarity_score'] > self.similarity_config['confidence_threshold']]
            valid_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Apply diversity filtering
            similar_outfits = []
            used_top_ids = set()
            used_bottom_ids = set()
            
            # ✅ OPTIMIZATION: Early termination counter
            high_quality_count = 0
            max_high_quality = num_similar  # Use the actual requested number instead of hardcoded 8
            
            for scored_candidate in valid_candidates:
                candidate = scored_candidate['candidate']
                similarity_score = scored_candidate['similarity_score']
                
                # Check for unique product IDs to avoid repetition
                top_id = str(candidate.get('top_id', ''))
                bottom_id = str(candidate.get('bottom_id', ''))
                
                # Skip if we've already used these product IDs
                if top_id in used_top_ids or bottom_id in used_bottom_ids:
                    continue
                
                similar_outfits.append({
                    'outfit_data': candidate,
                    'similarity_score': similarity_score,
                    'score_breakdown': scored_candidate['score_breakdown'],
                    'source_outfit_id': source_outfit['main_outfit_id'],
                    'generated_at': datetime.now().isoformat()
                })
                
                # Track used product IDs
                used_top_ids.add(top_id)
                used_bottom_ids.add(bottom_id)
                
                # ✅ OPTIMIZATION: Early termination for high-quality outfits
                if similarity_score > 0.7:  # High quality threshold
                    high_quality_count += 1
                    if high_quality_count >= max_high_quality:
                        logger.info(f"✅ Early termination: Found {high_quality_count} high-quality outfits")
                        break
                
                # Stop if we have enough unique outfits
                if len(similar_outfits) >= num_similar:
                    break
            
            logger.info(f"✅ Parallel scoring completed: {len(similar_outfits)} outfits selected from {len(valid_candidates)} valid candidates")
            return similar_outfits
            
        except ImportError:
            logger.warning("joblib not available, falling back to sequential processing")
            return self._score_candidates_sequential(source_outfit, candidates, num_similar)
        except Exception as e:
            logger.error(f"Error in parallel scoring: {e}, falling back to sequential")
            return self._score_candidates_sequential(source_outfit, candidates, num_similar)
    
    def _score_candidates_sequential(self, source_outfit: Dict, candidates: List[Dict], num_similar: int) -> List[Dict]:
        """Fallback sequential scoring method."""
        similar_outfits = []
        used_top_ids = set()
        used_bottom_ids = set()
        
        # ✅ OPTIMIZATION: Early termination counter
        high_quality_count = 0
        max_high_quality = num_similar  # Use the actual requested number instead of hardcoded 8
        
        for candidate in candidates:
            similarity_score, score_breakdown = self.calculate_outfit_similarity(source_outfit, candidate)
            
            # Apply sophisticated confidence threshold
            if similarity_score > self.similarity_config['confidence_threshold']:
                # Check for unique product IDs to avoid repetition
                top_id = str(candidate.get('top_id', ''))
                bottom_id = str(candidate.get('bottom_id', ''))
                
                # Skip if we've already used these product IDs
                if top_id in used_top_ids or bottom_id in used_bottom_ids:
                    continue
                
                similar_outfits.append({
                    'outfit_data': candidate,
                    'similarity_score': similarity_score,
                    'score_breakdown': score_breakdown,
                    'source_outfit_id': source_outfit['main_outfit_id'],
                    'generated_at': datetime.now().isoformat()
                })
                
                # Track used product IDs
                used_top_ids.add(top_id)
                used_bottom_ids.add(bottom_id)
                
                # ✅ OPTIMIZATION: Early termination for high-quality outfits
                if similarity_score > 0.7:  # High quality threshold
                    high_quality_count += 1
                    if high_quality_count >= max_high_quality:
                        logger.info(f"✅ Early termination: Found {high_quality_count} high-quality outfits")
                        break
                
                # Stop if we have enough unique outfits
                if len(similar_outfits) >= num_similar:
                    break
        
        logger.info(f"Found {len(similar_outfits)} unique similar outfits (used {len(used_top_ids)} unique tops, {len(used_bottom_ids)} unique bottoms)")
        return similar_outfits
    
    def _chunk_candidates(self, candidates: List[Dict], chunk_size: int) -> List[List[Dict]]:
        """Split candidates into chunks for parallel processing."""
        return [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    def _extract_color(self, product: pd.Series) -> str:
        """Extract dominant color from product data."""
        try:
            # Try different color fields in order of preference
            color_fields = ['primary_color', 'color', 'dominant_color']
            for field in color_fields:
                if field in product and pd.notna(product[field]) and str(product[field]).strip():
                    return str(product[field]).strip()
            
            # Fallback to title analysis
            title = str(product.get('title', '')).lower()
            color_keywords = {
                'black': 'Black', 'white': 'White', 'blue': 'Blue', 'red': 'Red',
                'green': 'Green', 'yellow': 'Yellow', 'pink': 'Pink', 'purple': 'Purple',
                'brown': 'Brown', 'gray': 'Gray', 'grey': 'Grey', 'orange': 'Orange',
                'navy': 'Navy', 'beige': 'Beige', 'cream': 'Cream', 'maroon': 'Maroon'
            }
            
            for keyword, color in color_keywords.items():
                if keyword in title:
                    return color
            
            return "Unknown"
        except Exception as e:
            logger.error(f"Error extracting color: {e}")
            return "Unknown"

    def _generate_outfit_name(self, top: pd.Series, bottom: pd.Series, style: str) -> str:
        """Generate outfit name using the proper mood + accent + noun + occasion template."""
        try:
            # Word banks for outfit naming
            MOOD_WORDS = [
                "Urban", "Luxe", "Coastal", "Athleisure", "Retro", "Boho",
                "Minimal", "Neo-Noir", "Heritage", "Streetline", "Artsy",
                "Safari", "Midnight", "Electric", "Pastel", "Monochrome",
                "Vintage", "Modern", "Classic", "Edgy", "Sophisticated",
                "Casual", "Elegant", "Bold", "Subtle", "Dynamic"
            ]
            
            ACCENT_WORDS = [
                "Sage", "Scarlet", "Indigo", "Denim", "Linen", "Houndstooth",
                "Tweed", "Velvet", "Floral", "Graphic", "Gingham", "Leather",
                "Navy", "Coral", "Olive", "Cream", "Charcoal", "Burgundy",
                "Camel", "Emerald", "Rose", "Slate", "Amber", "Teal"
            ]
            
            NOUN_WORDS = [
                "Shift", "Edit", "Ensemble", "Layer", "Remix", "Story",
                "Set", "Collective", "Combo", "Twist", "Look", "Style",
                "Vibe", "Mood", "Statement", "Signature", "Essence"
            ]
            
            OCCASION_WORDS = [
                "Brunch", "Boardroom", "Festival", "Weekend", "Soirée",
                "Runway", "Getaway", "Studio", "Commute", "Date Night",
                "Office", "Dinner", "Party", "Travel", "Meeting", "Lunch"
            ]
            
            # Extract outfit components
            top_data = {
                'title': str(top.get('title', '')),
                'primary_style': str(top.get('primary_style', '')),
                'dominant_color': self._extract_color(top)
            }
            bottom_data = {
                'title': str(bottom.get('title', '')),
                'primary_style': str(bottom.get('primary_style', '')),
                'dominant_color': self._extract_color(bottom)
            }
            
            # Analyze outfit components to determine mood
            mood = self._determine_outfit_mood(top_data, bottom_data, {}, MOOD_WORDS)
            
            # Determine accent (color/fabric/print)
            accent = self._determine_outfit_accent(top_data, bottom_data, ACCENT_WORDS)
            
            # Select noun anchor
            noun = random.choice(NOUN_WORDS)
            
            # Determine occasion/context
            occasion = self._determine_outfit_occasion(top_data, bottom_data, {}, OCCASION_WORDS)
            
            # Build outfit name (max 3 words)
            outfit_name = self._build_outfit_name(mood, accent, noun, occasion)
            
            return outfit_name
            
        except Exception as e:
            logger.error(f"❌ Error generating outfit name: {e}")
            return "URBAN SHIFT"

    def _generate_outfit_description(self, top: pd.Series, bottom: pd.Series, style: str) -> str:
        """Generate detailed outfit description based on the outfit name and products."""
        try:
            # Generate outfit name first
            outfit_name = self._generate_outfit_name(top, bottom, style)
            
            # Extract product details
            top_title = str(top.get('title', ''))
            bottom_title = str(bottom.get('title', ''))
            top_color = self._extract_color(top)
            bottom_color = self._extract_color(bottom)
            
            # Create detailed description based on the outfit name
            if "Urban" in outfit_name:
                return f"Urban {outfit_name.lower()} featuring a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This contemporary street-ready ensemble combines modern edge with cultural influence, perfect for city life and urban exploration. The layered approach creates depth while maintaining comfort and movement."
            elif "Luxe" in outfit_name:
                return f"Luxurious {outfit_name.lower()} showcasing a {top_color} {top_title} with {bottom_color} {bottom_title}. This sophisticated combination elevates your frame with structured lines and premium appeal, making it ideal for special occasions and refined gatherings. The balanced proportions create an elegant silhouette."
            elif "Coastal" in outfit_name:
                return f"Coastal {outfit_name.lower()} featuring a {top_color} {top_title} and {bottom_color} {bottom_title}. This relaxed ensemble captures beach vibes with breathable fabrics and casual elegance, perfect for vacation days and outdoor activities. The easygoing structure mirrors a low-effort, high-comfort vibe."
            elif "Athleisure" in outfit_name:
                return f"Athleisure {outfit_name.lower()} with a {top_color} {top_title} and {bottom_color} {bottom_title}. This active lifestyle ensemble combines sporty comfort with everyday style, featuring athletic-inspired lines and laid-back structure that matches your on-the-move rhythm. Perfect for gym sessions and active daily routines."
            elif "Retro" in outfit_name:
                return f"Retro {outfit_name.lower()} showcasing a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This vintage-inspired ensemble brings nostalgic charm with time-washed tones and retro detailing that adds character to the present. The combination creates a timeless appeal with modern comfort."
            elif "Boho" in outfit_name:
                return f"Bohemian {outfit_name.lower()} featuring a {top_color} {top_title} with {bottom_color} {bottom_title}. This free-spirited ensemble combines artistic elements with eclectic style, perfect for creative gatherings and artistic events. The relaxed fits and warm tones reflect your out-of-office energy."
            elif "Minimal" in outfit_name:
                return f"Minimalist {outfit_name.lower()} with a {top_color} {top_title} and {bottom_color} {bottom_title}. This clean ensemble features refined silhouettes and intentional pieces that speak to a quiet, less-is-more style approach. The balanced proportions and precise details keep your fit polished without trying too hard."
            elif "Neo-Noir" in outfit_name:
                return f"Neo-noir {outfit_name.lower()} featuring a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This dark sophistication ensemble brings modern edge with bold contrasts and crisp saturation that highlight your sharp clarity. Perfect for evening elegance with contemporary appeal."
            elif "Heritage" in outfit_name:
                return f"Heritage {outfit_name.lower()} showcasing a {top_color} {top_title} with {bottom_color} {bottom_title}. This timeless ensemble features classic pieces with traditional appeal, perfect for heritage events and classic social occasions. The structured lines and fit-aware pieces elevate your frame while staying wearable."
            elif "Streetline" in outfit_name:
                return f"Streetline {outfit_name.lower()} featuring a {top_color} {top_title} and {bottom_color} {bottom_title}. This urban fashion ensemble combines street culture influence with everyday edge, confident and culturally aware. The modern layers and pace-driven fits reflect your city-first lifestyle."
            elif "Artsy" in outfit_name:
                return f"Artsy {outfit_name.lower()} with a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This creative expression ensemble features artistic elements and bold visuals that make your outfit speak before you do. Perfect for artistic events and creative gatherings with expressive style."
            elif "Safari" in outfit_name:
                return f"Safari {outfit_name.lower()} featuring a {top_color} {top_title} and {bottom_color} {bottom_title}. This adventure-ready ensemble combines exploration vibes with functional styling, perfect for outdoor activities and travel adventures. The structured pieces provide both style and practicality."
            elif "Midnight" in outfit_name:
                return f"Midnight {outfit_name.lower()} showcasing a {top_color} {top_title} with {bottom_color} {bottom_title}. This evening elegance ensemble features dark allure with mood-lit colors and tailored silhouettes for after-dark sharpness. Perfect for night events and sophisticated evening occasions."
            elif "Electric" in outfit_name:
                return f"Electric {outfit_name.lower()} featuring a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This bold energy ensemble combines vibrant impact with dynamic styling, perfect for statement moments and high-energy events. The bold visuals and print-led design create confident energy."
            elif "Pastel" in outfit_name:
                return f"Pastel {outfit_name.lower()} with a {top_color} {top_title} and {bottom_color} {bottom_title}. This soft elegance ensemble features gentle charm with calming tones that bring softness into form, feel, and finish. Perfect for relaxed occasions and gentle styling."
            elif "Monochrome" in outfit_name:
                return f"Monochrome {outfit_name.lower()} featuring a {top_color} {top_title} with {bottom_color} {bottom_title}. This tonal sophistication ensemble creates a unified palette with balanced proportions and precise details. The clean aesthetic provides versatile styling for various occasions."
            else:
                return f"{outfit_name} featuring a {top_color} {top_title} paired with {bottom_color} {bottom_title}. This versatile ensemble combines contemporary appeal with smart styling, perfect for daily activities and social occasions. The balanced combination creates a polished look that adapts to your lifestyle."
            
        except Exception as e:
            logger.error(f"❌ Error generating outfit description: {e}")
            return "A stylish outfit combination featuring carefully selected pieces for various occasions with contemporary appeal and balanced proportions."

    def _determine_outfit_mood(self, top_data: Dict, bottom_data: Dict, user_data: Dict, mood_words: List[str]) -> str:
        """Determine the mood of the outfit based on product characteristics."""
        try:
            # Analyze colors for mood
            top_color = top_data.get('dominant_color', '').lower()
            bottom_color = bottom_data.get('dominant_color', '').lower()
            
            # Color-based mood mapping
            color_mood_map = {
                'black': ['Neo-Noir', 'Midnight', 'Urban'],
                'white': ['Minimal', 'Coastal', 'Clean'],
                'blue': ['Coastal', 'Heritage', 'Urban'],
                'red': ['Electric', 'Bold', 'Streetline'],
                'green': ['Safari', 'Heritage', 'Urban'],
                'yellow': ['Electric', 'Bold', 'Artsy'],
                'pink': ['Pastel', 'Soft', 'Boho'],
                'purple': ['Luxe', 'Artsy', 'Electric'],
                'brown': ['Heritage', 'Boho', 'Vintage'],
                'gray': ['Minimal', 'Urban', 'Modern'],
                'grey': ['Minimal', 'Urban', 'Modern'],
                'orange': ['Electric', 'Bold', 'Streetline'],
                'navy': ['Heritage', 'Urban', 'Sophisticated'],
                'beige': ['Minimal', 'Coastal', 'Soft'],
                'cream': ['Minimal', 'Coastal', 'Soft'],
                'maroon': ['Luxe', 'Heritage', 'Sophisticated']
            }
            
            # Get mood candidates from colors
            mood_candidates = []
            for color in [top_color, bottom_color]:
                if color in color_mood_map:
                    mood_candidates.extend(color_mood_map[color])
            
            # Analyze styles for mood
            top_style = top_data.get('primary_style', '').lower()
            bottom_style = bottom_data.get('primary_style', '').lower()
            
            style_mood_map = {
                'casual': ['Urban', 'Coastal', 'Relaxed'],
                'formal': ['Luxe', 'Heritage', 'Sophisticated'],
                'sporty': ['Athleisure', 'Urban', 'Dynamic'],
                'vintage': ['Vintage', 'Retro', 'Heritage'],
                'streetwear': ['Streetline', 'Urban', 'Edgy'],
                'minimalist': ['Minimal', 'Clean', 'Modern'],
                'bohemian': ['Boho', 'Artsy', 'Relaxed'],
                'elegant': ['Luxe', 'Sophisticated', 'Heritage']
            }
            
            for style in [top_style, bottom_style]:
                for style_key, moods in style_mood_map.items():
                    if style_key in style:
                        mood_candidates.extend(moods)
            
            # Count mood frequencies and select the most common
            mood_counts = {}
            for mood in mood_candidates:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            if mood_counts:
                # Get the mood with highest count, or random if tied
                max_count = max(mood_counts.values())
                top_moods = [mood for mood, count in mood_counts.items() if count == max_count]
                return random.choice(top_moods)
            
            # Fallback to random mood
            return random.choice(mood_words)
            
        except Exception as e:
            logger.error(f"Error determining outfit mood: {e}")
            return random.choice(mood_words)

    def _determine_outfit_accent(self, top_data: Dict, bottom_data: Dict, accent_words: List[str]) -> str:
        """Determine the accent of the outfit based on product characteristics."""
        try:
            # Extract colors
            top_color = top_data.get('dominant_color', '').lower()
            bottom_color = bottom_data.get('dominant_color', '').lower()
            
            # Color to accent mapping
            color_accent_map = {
                'black': ['Charcoal', 'Navy'],
                'white': ['Cream', 'Linen'],
                'blue': ['Navy', 'Indigo', 'Denim'],
                'red': ['Scarlet', 'Burgundy'],
                'green': ['Olive', 'Sage', 'Emerald'],
                'yellow': ['Amber'],
                'pink': ['Rose', 'Coral'],
                'purple': ['Burgundy', 'Slate'],
                'brown': ['Camel', 'Burgundy'],
                'gray': ['Slate', 'Charcoal'],
                'grey': ['Slate', 'Charcoal'],
                'orange': ['Amber', 'Coral'],
                'navy': ['Navy', 'Indigo'],
                'beige': ['Camel', 'Cream'],
                'cream': ['Cream', 'Linen'],
                'maroon': ['Burgundy', 'Scarlet']
            }
            
            # Get accent from colors
            for color in [top_color, bottom_color]:
                if color in color_accent_map:
                    return random.choice(color_accent_map[color])
            
            # Fallback to random accent
            return random.choice(accent_words)
            
        except Exception as e:
            logger.error(f"Error determining outfit accent: {e}")
            return random.choice(accent_words)

    def _determine_outfit_occasion(self, top_data: Dict, bottom_data: Dict, user_data: Dict, occasion_words: List[str]) -> str:
        """Determine the occasion of the outfit based on product characteristics."""
        try:
            # Analyze styles for occasion
            top_style = top_data.get('primary_style', '').lower()
            bottom_style = bottom_data.get('primary_style', '').lower()
            
            style_occasion_map = {
                'casual': ['Weekend', 'Commute', 'Lunch'],
                'formal': ['Office', 'Boardroom', 'Meeting'],
                'sporty': ['Studio', 'Commute'],
                'elegant': ['Dinner', 'Soirée', 'Date Night'],
                'party': ['Party', 'Festival', 'Date Night'],
                'business': ['Office', 'Boardroom', 'Meeting'],
                'evening': ['Dinner', 'Soirée', 'Date Night'],
                'day': ['Brunch', 'Lunch', 'Weekend']
            }
            
            for style in [top_style, bottom_style]:
                for style_key, occasions in style_occasion_map.items():
                    if style_key in style:
                        return random.choice(occasions)
            
            # Fallback to random occasion
            return random.choice(occasion_words)
            
        except Exception as e:
            logger.error(f"Error determining outfit occasion: {e}")
            return random.choice(occasion_words)

    def _build_outfit_name(self, mood: str, accent: str, noun: str, occasion: str) -> str:
        """Build the final outfit name from components."""
        try:
            # Create outfit name with mood + accent + noun pattern
            outfit_name = f"{mood.upper()} {noun.upper()}"
            
            # Limit to reasonable length
            if len(outfit_name) > 20:
                outfit_name = f"{mood.upper()} {noun.upper()}"[:20]
            
            return outfit_name
            
        except Exception as e:
            logger.error(f"Error building outfit name: {e}")
            return "URBAN SHIFT"

    def _get_product_type_from_id(self, product_id: str) -> str:
        """Get product type from product ID by querying the database."""
        try:
            if not product_id:
                return 'T-Shirt'  # Default fallback
            
            # Query the database to get product type
            result = self.db.client.table('tagged_products').select('product_type').eq('product_id', product_id).execute()
            
            if result.data:
                return result.data[0].get('product_type', 'T-Shirt')
            else:
                logger.warning(f"Product {product_id} not found in database")
                return 'T-Shirt'  # Default fallback
                
        except Exception as e:
            logger.error(f"Error getting product type for {product_id}: {e}")
            return 'T-Shirt'  # Default fallback
    
    def _map_to_broad_category(self, product_type: str) -> str:
        """Map specific product types to broad categories (Upperwear/Bottomwear) for FAISS search."""
        if not product_type:
            return 'Upperwear'  # Default fallback
        
        # Define upperwear product types
        upperwear_types = [
            'T-Shirt', 'Shirt', 'Blouse', 'Top', 'Sweater', 'Hoodie', 'Jacket', 
            'Blazer', 'Cardigan', 'Tank Top', 'Crop Top', 'Sleeveless', 'Tunic',
            'Polo Shirt', 'Henley', 'Dress', 'Maxi Dress', 'Mini Dress', 'Gown'
        ]
        
        # Define bottomwear product types
        bottomwear_types = [
            'Trousers', 'Jeans', 'Pants', 'Shorts', 'Skirt', 'Pencil Skirt',
            'A-Line Skirt', 'Midi Skirt', 'Mini Skirt', 'Maxi Skirt', 'Joggers',
            'Chinos', 'Dress Pants', 'Slacks', 'Cargos', 'Denim Pants', 'Denim Shorts'
        ]
        
        # Check if product type is in upperwear or bottomwear categories
        if product_type in upperwear_types:
            return 'Upperwear'
        elif product_type in bottomwear_types:
            return 'Bottomwear'
        else:
            # If unknown, try to guess based on common patterns
            product_type_lower = product_type.lower()
            if any(word in product_type_lower for word in ['shirt', 'top', 'blouse', 'sweater', 'jacket', 'dress']):
                return 'Upperwear'
            elif any(word in product_type_lower for word in ['pants', 'trousers', 'jeans', 'shorts', 'skirt']):
                return 'Bottomwear'
            else:
                logger.warning(f"Unknown product type '{product_type}', defaulting to Upperwear")
                return 'Upperwear'  # Default fallback

    def _are_product_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two product types are compatible for recommendations."""
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        # Define compatible product type groups
        compatible_groups = {
            # Shorts group
            'shorts': ['shorts', 'denim_shorts', 'cargo_shorts', 'athletic_shorts'],
            # Pants group
            'pants': ['pants', 'casual_pants', 'formal_pants', 'dress_pants', 'trousers'],
            # Jeans group
            'jeans': ['jeans', 'denim_pants', 'skinny_jeans', 'straight_jeans'],
            # Chinos group
            'chinos': ['chinos', 'khaki_pants', 'casual_pants'],
            # Joggers group
            'joggers': ['joggers', 'sweatpants', 'athletic_pants'],
            # T-shirts group
            't-shirt': ['t-shirt', 'tshirt', 'tee', 'casual_shirt'],
            # Shirts group
            'shirt': ['shirt', 'casual_shirt', 'formal_shirt', 'polo_shirt'],
            # Sweaters group
            'sweater': ['sweater', 'cardigan', 'hoodie', 'pullover']
        }
        
        # Check if types are in the same compatible group
        for group_name, compatible_types in compatible_groups.items():
            if type1_lower in compatible_types and type2_lower in compatible_types:
                return True
        
        # Check if one type is a subset of another (e.g., "shorts" in "denim_shorts")
        if type1_lower in type2_lower or type2_lower in type1_lower:
            return True
        
        return False

def score_candidates_batch(source_outfit: Dict, candidates_batch: List[Dict]) -> List[Dict]:
    """Score a batch of candidates (for parallel processing)."""
    results = []
    for candidate in candidates_batch:
        try:
            # Create a temporary generator instance for scoring
            temp_generator = SupabaseSimilarOutfitsGenerator()
            similarity_score, score_breakdown = temp_generator.calculate_outfit_similarity(source_outfit, candidate)
            
            results.append({
                'candidate': candidate,
                'similarity_score': similarity_score,
                'score_breakdown': score_breakdown
            })
        except Exception as e:
            continue
    return results

def main():
    """Main function to test similar outfits generation."""
    
    # Initialize generator
    generator = SupabaseSimilarOutfitsGenerator()
    
    # Test with a main outfit ID from Phase 1
    test_outfit_id = "main_2_1"  # Top scoring outfit for user 2 (male)
    
    logger.info(f"🔍 Finding similar outfits for: {test_outfit_id}")
    
    try:
        similar_outfits = generator.find_similar_outfits(test_outfit_id, num_similar=20)
        
        if similar_outfits:
            print(f"\n✅ SUCCESS: Found {len(similar_outfits)} similar outfits")
            print("=" * 80)
            
            for i, outfit in enumerate(similar_outfits, 1):
                data = outfit['outfit_data']
                score = outfit['similarity_score']
                
                print(f"\nSimilar Outfit #{i}: Score {score:.3f}")
                print(f"Top: {data['top_title'][:50]}... (₹{data['top_price']})")
                print(f"Bottom: {data['bottom_title'][:50]}... (₹{data['bottom_price']})")
                print(f"Total: ₹{data['total_price']}")
                print(f"Styles: {data['top_style']} + {data['bottom_style']}")
                print(f"Colors: {data['top_color']} + {data['bottom_color']}")
                print("-" * 40)
            
            print(f"\n🔗 API Endpoint: GET /api/outfit/{test_outfit_id}/similar?count=20")
        else:
            print(f"\n❌ No similar outfits found for {test_outfit_id}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 