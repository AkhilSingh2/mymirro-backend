# Phase 2: Similar Outfits API with Supabase Integration (On-demand)
# This generates 10 similar outfits for any given outfit in real-time using Supabase

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
    
    def __init__(self, config: Dict = None):
        """Initialize the Supabase-enabled similar outfits generator."""
        self.config = config or self._default_config()
        
        # Initialize Supabase database connection
        self.db = get_db()
        if not self.db.test_connection():
            logger.error("❌ Database connection failed. Please check your Supabase configuration.")
            raise ConnectionError("Failed to connect to Supabase database")
        
        # Load model
        try:
            self.model = SentenceTransformer(self.config['model_name'])
            logger.info(f"✅ Model loaded: {self.config['model_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        self.embedding_cache = {}
        
        # FAISS indexes for different wear types
        self.faiss_indexes = {}
        self.product_mappings = {}
        
        # ✅ ENHANCED: Sophisticated similarity configuration with fashion intelligence
        self.similarity_config = {
            'semantic_weight': 4.0,           # Core AI matching
            'style_harmony_weight': 3.5,      # Advanced style compatibility  
            'color_harmony_weight': 3.0,      # Sophisticated color theory
            'formality_weight': 2.5,          # Style formality matching
            'pattern_compatibility_weight': 2.0,  # Pattern mixing intelligence
            'seasonal_weight': 1.5,           # Seasonal appropriateness
            'price_similarity_weight': 1.8,   # Price range compatibility
            'occasion_weight': 2.2,           # Occasion-specific matching
            'diversity_bonus': 0.8,           # Encourage variety in results
            'confidence_threshold': 0.4,      # Minimum similarity threshold
            'price_tolerance': 0.35,          # ±35% price range tolerance
            'formality_tolerance': 2,         # ±2 levels in formality scale
            'min_similar_outfits': 5,         # Minimum outfits to return
            'max_similar_outfits': 10,        # Maximum outfits to return
            'candidate_pool_size': 150        # Larger candidate pool for diversity
        }
        
        # Advanced fashion intelligence systems
        self.color_harmony = self._initialize_color_harmony()
        self.style_formality = self._initialize_style_formality()
        self.seasonal_preferences = self._initialize_seasonal_preferences()
        self.pattern_compatibility = self._initialize_pattern_compatibility()
        
    def _default_config(self) -> Dict:
        """Default configuration for the similar outfits generator."""
        app_config = get_config()
        return {
            'model_name': app_config.MODEL_NAME,
            'cache_embeddings': True,
            'batch_size': 50
        }
    
    def _initialize_color_harmony(self) -> Dict:
        """Initialize sophisticated color harmony rules based on advanced color theory."""
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
        """Initialize pattern and texture mixing rules."""
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
        required_columns = ["title", "wear_type", "gender"]
        missing_columns = [col for col in required_columns if col not in products_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in products data: {missing_columns}")
        
        # Clean data
        products_df = products_df.dropna(subset=["title", "wear_type"])
        
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
                    product_id = str(row.get('id', ''))
                    if product_id in price_dict:
                        return float(price_dict[product_id])
                    
                    # Smart default pricing based on wear_type and style
                    wear_type = row.get('wear_type', '')
                    style = row.get('enhanced_primary_style', row.get('primary_style', ''))
                    
                    if wear_type == 'Upperwear':
                        if any(x in style.lower() for x in ['formal', 'business', 'blazer']):
                            return 2500
                        elif any(x in style.lower() for x in ['casual', 't-shirt', 'tank']):
                            return 800
                        else:
                            return 1500
                    elif wear_type == 'Bottomwear':
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
            logger.warning(f"Could not load scraped prices: {e}, using defaults")
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
        """Build FAISS indexes for different wear types."""
        logger.info("Building FAISS indexes for similar outfit search...")
        
        wear_types = ['Upperwear', 'Bottomwear']
        
        for wear_type in wear_types:
            wear_products = products_df[products_df['wear_type'] == wear_type].copy()
            
            if wear_products.empty:
                logger.warning(f"No products found for wear_type: {wear_type}")
                continue
            
            captions = []
            product_indices = []
            
            for idx, row in wear_products.iterrows():
                caption = row.get('final_caption', '') or row.get('full_caption', '') or row.get('title', '')
                if caption.strip():
                    captions.append(caption)
                    product_indices.append(idx)
            
            if not captions:
                logger.warning(f"No valid captions found for wear_type: {wear_type}")
                continue
            
            logger.info(f"Generating embeddings for {len(captions)} {wear_type} products...")
            
            embeddings = []
            for caption in captions:
                embedding = self.get_embedding_cached(caption)
                embeddings.append(embedding)
            embeddings = np.array(embeddings)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            index.add(embeddings.astype('float32'))
            
            # Store index and mapping
            self.faiss_indexes[wear_type] = index
            self.product_mappings[wear_type] = {
                'indices': product_indices,
                'products': wear_products.iloc[[wear_products.index.get_loc(idx) for idx in product_indices]].copy()
            }
            
            logger.info(f"Built FAISS index for {wear_type}: {len(captions)} products indexed")
    
    def load_outfit_from_supabase(self, main_outfit_id: str) -> Dict:
        """Load outfit data from Supabase database."""
        try:
            logger.info(f"Loading outfit {main_outfit_id} from Supabase...")
            
            # Query the user_outfits table
            result = self.db.client.table('user_outfits').select('*').eq('main_outfit_id', main_outfit_id).execute()
            
            if not result.data:
                raise ValueError(f"Outfit {main_outfit_id} not found in database")
            
            outfit_data = result.data[0]
            logger.info(f"✅ Successfully loaded outfit {main_outfit_id} from Supabase")
            
            return outfit_data
                
        except Exception as e:
            logger.error(f"Error loading outfit {main_outfit_id} from Supabase: {e}")
            raise
    
    def load_products_from_supabase(self) -> pd.DataFrame:
        """Load products data from Supabase database."""
        try:
            logger.info("Loading products from Supabase...")
            
            # Use the database method to get products
            products_df = self.db.get_products()
            
            if products_df.empty:
                logger.error("❌ No products data available from Supabase")
                return pd.DataFrame()
            
            logger.info(f"✅ Successfully loaded {len(products_df)} products from Supabase")
            return products_df
                
        except Exception as e:
            logger.error(f"Error loading products from Supabase: {e}")
            return pd.DataFrame()
    
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
            
            # Check formality compatibility
            form1 = self.style_formality.get(style1, 5)
            form2 = self.style_formality.get(style2, 5)
            formality_diff = abs(form1 - form2)
            
            if formality_diff <= 1:
                return 0.9  # Very compatible
            elif formality_diff <= 2:
                return 0.7  # Somewhat compatible  
            elif formality_diff <= 3:
                return 0.5  # Moderately compatible
            else:
                return 0.3  # Less compatible
        
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
        
        # 4. FORMALITY MATCHING
        source_top_formality = self.style_formality.get(source_top_style, 5)
        source_bottom_formality = self.style_formality.get(source_bottom_style, 5)
        candidate_top_formality = self.style_formality.get(candidate_top_style, 5)
        candidate_bottom_formality = self.style_formality.get(candidate_bottom_style, 5)
        
        source_avg_formality = (source_top_formality + source_bottom_formality) / 2
        candidate_avg_formality = (candidate_top_formality + candidate_bottom_formality) / 2
        
        formality_diff = abs(source_avg_formality - candidate_avg_formality)
        formality_score = max(0, 1 - (formality_diff / self.similarity_config['formality_tolerance']))
        
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
        
        # 6. PRICE COMPATIBILITY 
        source_price = float(source_outfit.get('total_price', 0))
        candidate_price = float(candidate_outfit.get('total_price', 0))
        
        if source_price > 0:
            price_diff = abs(source_price - candidate_price) / source_price
            price_score = max(0, 1 - (price_diff / self.similarity_config['price_tolerance']))
        else:
            price_score = 1.0
        
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
        
        # 8. SEASONAL APPROPRIATENESS (if available)
        # For now, give neutral score - will be enhanced with fashion designer input
        scores['seasonal_appropriateness'] = 0.8
        
        # Calculate sophisticated weighted final score
        weighted_score = (
            scores['semantic_similarity'] * self.similarity_config['semantic_weight'] +
            scores['style_harmony'] * self.similarity_config['style_harmony_weight'] +
            scores['color_harmony'] * self.similarity_config['color_harmony_weight'] +
            scores['formality_matching'] * self.similarity_config['formality_weight'] +
            scores['pattern_compatibility'] * self.similarity_config['pattern_compatibility_weight'] +
            scores['price_similarity'] * self.similarity_config['price_similarity_weight'] +
            scores['occasion_matching'] * self.similarity_config['occasion_weight'] +
            scores['seasonal_appropriateness'] * self.similarity_config['seasonal_weight']
        )
        
        # Normalize by total weights
        total_weights = (
            self.similarity_config['semantic_weight'] + 
            self.similarity_config['style_harmony_weight'] +
            self.similarity_config['color_harmony_weight'] +
            self.similarity_config['formality_weight'] +
            self.similarity_config['pattern_compatibility_weight'] +
            self.similarity_config['price_similarity_weight'] +
            self.similarity_config['occasion_weight'] +
            self.similarity_config['seasonal_weight']
        )
        
        final_score = weighted_score / total_weights
        
        # Add detailed breakdown for debugging
        scores['final_score'] = final_score
        scores['explanation'] = " | ".join(explanations)
        
        return final_score, scores
    
    def find_similar_outfits(self, outfit_id: str, num_similar: int = 10) -> List[Dict]:
        """Find similar outfits for a given outfit using Supabase data."""
        
        try:
            # Load source outfit from Supabase
            source_outfit = self.load_outfit_from_supabase(outfit_id)
            
            logger.info(f"Finding similar outfits for outfit: {outfit_id}")
            
            # Load and validate products from Supabase
            products_df = self.load_products_from_supabase()
            
            if products_df.empty:
                logger.error("No products available from Supabase")
                return []
            
            products_df = self.validate_products_data(products_df)
            
            # Build FAISS indexes
            self.build_faiss_indexes(products_df)
            
            # Generate candidate outfits
            candidates = self.generate_candidate_outfits(source_outfit, products_df)
            
            if not candidates:
                logger.warning("No candidate outfits found")
                return []
            
            # Calculate similarity scores for all candidates
            similar_outfits = []
            
            for candidate in candidates:
                similarity_score, score_breakdown = self.calculate_outfit_similarity(source_outfit, candidate)
                
                # Apply sophisticated confidence threshold
                if similarity_score > self.similarity_config['confidence_threshold']:
                    similar_outfits.append({
                        'outfit_data': candidate,
                        'similarity_score': similarity_score,
                        'score_breakdown': score_breakdown,
                        'source_outfit_id': outfit_id,
                        'generated_at': datetime.now().isoformat()
                    })
            
            # Sort by similarity score and return top matches
            similar_outfits.sort(key=lambda x: x['similarity_score'], reverse=True)
            result = similar_outfits[:num_similar]
            
            logger.info(f"Found {len(result)} similar outfits")
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar outfits: {e}")
            raise
    
    def generate_candidate_outfits(self, source_outfit: Dict, products_df: pd.DataFrame) -> List[Dict]:
        """Generate candidate outfits for similarity comparison."""
        
        candidates = []
        
        # Get source outfit details
        source_top_id = source_outfit.get('top_id', '')
        source_bottom_id = source_outfit.get('bottom_id', '')
        
        # Use FAISS to find similar tops and bottoms separately
        source_top_title = source_outfit.get('top_title', '')
        source_bottom_title = source_outfit.get('bottom_title', '')
        
        # Find similar tops and bottoms with larger pool for better diversity
        similar_tops = self.find_similar_products(source_top_title, 'Upperwear', k=20)
        similar_bottoms = self.find_similar_products(source_bottom_title, 'Bottomwear', k=20)
        
        # Generate outfit combinations with enhanced candidate pool
        combination_count = 0
        max_combinations = self.similarity_config['candidate_pool_size']
        
        for top_candidate in similar_tops:
            for bottom_candidate in similar_bottoms:
                if combination_count >= max_combinations:
                    break
                
                # Skip if same as source outfit
                if (top_candidate['product'].get('product_id', top_candidate['product'].get('id', '')) == source_top_id and 
                    bottom_candidate['product'].get('product_id', bottom_candidate['product'].get('id', '')) == source_bottom_id):
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
        
        # Get query embedding
        query_embedding = self.get_embedding_cached(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        index = self.faiss_indexes[wear_type]
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        # Get corresponding products
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
        """Create outfit data structure from top and bottom products."""
        
        return {
            'top_id': top_product.get('product_id', top_product.get('id', '')),
            'top_title': top_product.get('title', ''),
            'top_image': top_product.get('image_url', ''),
            'top_price': float(top_product.get('price', 0)),
            'top_style': top_product.get('enhanced_primary_style', top_product.get('primary_style', '')),
            'top_color': top_product.get('primary_color', ''),
            'top_occasion': top_product.get('enhanced_occasion', top_product.get('occasion', '')),
            
            'bottom_id': bottom_product.get('product_id', bottom_product.get('id', '')),
            'bottom_title': bottom_product.get('title', ''),
            'bottom_image': bottom_product.get('image_url', ''),
            'bottom_price': float(bottom_product.get('price', 0)),
            'bottom_style': bottom_product.get('enhanced_primary_style', bottom_product.get('primary_style', '')),
            'bottom_color': bottom_product.get('primary_color', ''),
            'bottom_occasion': bottom_product.get('enhanced_occasion', bottom_product.get('occasion', '')),
            
            'total_price': float(top_product.get('price', 0)) + float(bottom_product.get('price', 0)),
        }

def main():
    """Main function to test similar outfits generation."""
    
    # Initialize generator
    generator = SimilarOutfitsGenerator()
    
    # Test with a main outfit ID from Phase 1
    test_outfit_id = "main_1_1"  # Top scoring outfit for user 1
    
    logger.info(f"🔍 Finding similar outfits for: {test_outfit_id}")
    
    try:
        similar_outfits = generator.find_similar_outfits(test_outfit_id, num_similar=10)
        
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
            
            print(f"\n🔗 API Endpoint: GET /api/outfit/{test_outfit_id}/similar?count=10")
        else:
            print(f"\n❌ No similar outfits found for {test_outfit_id}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 