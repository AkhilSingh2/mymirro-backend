"""
Phase 1: Enhanced Supabase Outfits Generator with Professional Fashion Intelligence

Generates 20 main outfits per user using Supabase database with all enhancements from main generator
✅ ENHANCED: Now with Phase 2 optimizations + Precomputed embeddings + Same-category filtering
"""

import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 🚀 PARALLEL PROCESSING IMPORTS
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

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

# Import our Supabase database module
from database import get_db
from config import get_config

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the Why Picked Feature
try:
    from why_picked_feature import WhyPickedFeature
    WHY_PICKED_AVAILABLE = True
except ImportError:
    WHY_PICKED_AVAILABLE = False
    logger.warning("WhyPickedFeature not available - 'Why was this picked for you' feature will be disabled")

class SupabaseMainOutfitsGenerator:
    """
    Phase 1: Generate and store 20 main outfits per user using Supabase database
    ✅ ENHANCED: Now with Professional Fashion Designer Intelligence + Database Integration
    ✅ ENHANCED: Now with Phase 2 optimizations + Precomputed embeddings + Same-category filtering
    ✅ COMPLETE: Full feature parity with main generator
    """
    
    # ✅ OPTIMIZATION: Class-level model cache to avoid reloading
    _model_cache = None
    _model_cache_ready = False
    
    def __init__(self, config: Dict = None):
        """Initialize the Supabase-enabled outfits generator with fashion designer intelligence."""
        self.config = config or self._default_config()
        self.db = get_db()
        
        # Railway CPU optimization - delay until needed
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        if self.is_railway:
            logger.info("🏭 Railway environment detected - will apply CPU optimizations when needed")
        
        # Check for required dependencies
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available. Outfit generation requires FAISS for similarity search.")
            raise ImportError("FAISS is required for outfit generation but not installed")
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ SentenceTransformers not available. Outfit generation requires sentence-transformers.")
            raise ImportError("sentence-transformers is required for outfit generation but not installed")
        
        # Test database connection
        if not self.db.test_connection():
            logger.error("❌ Database connection failed. Please check your Supabase configuration.")
            raise ConnectionError("Failed to connect to Supabase database")
        
        # ✅ OPTIMIZATION: Lazy load model only when needed
        self.model = None
        
        # 🚀 ENHANCED: Advanced embedding cache with statistics
        self.embedding_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'max_size': self.config.get('cache_embeddings_limit', 1000)
        }
        
        # FAISS indexes for different wear types
        self.faiss_indexes = {}
        self.product_mappings = {}
        
        # Load color harmony from designer CSV
        self.color_harmony_map = self._load_color_harmony_from_csv()
        
        # ✅ ENHANCED: Initialize all professional fashion intelligence from main generator
        self._initialize_professional_fashion_intelligence()
        
        # 🎯 NEW: Initialize Why Picked Feature
        if WHY_PICKED_AVAILABLE:
            try:
                self.why_picked_feature = WhyPickedFeature()
                logger.info("✅ Why Picked Feature initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Why Picked Feature: {e}")
                self.why_picked_feature = None
        else:
            self.why_picked_feature = None

    def _ensure_model_loaded(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            try:
                if self.is_railway:
                    logger.info("🔧 Loading model with Railway CPU optimizations")
                    # Set conservative CPU limits for ML operations
                    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                        os.environ[var] = '2'
                
                # ✅ OPTIMIZATION: Use cached model if available
                if SupabaseMainOutfitsGenerator._model_cache is not None:
                    self.model = SupabaseMainOutfitsGenerator._model_cache
                    logger.info(f"✅ Using cached model: {self.config['model_name']}")
                else:
                    self.model = SentenceTransformer(self.config['model_name'])
                    # Cache the model for future use
                    SupabaseMainOutfitsGenerator._model_cache = self.model
                    SupabaseMainOutfitsGenerator._model_cache_ready = True
                    logger.info(f"✅ Model loaded and cached: {self.config['model_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise

    def _default_config(self) -> Dict:
        """Default configuration for the Supabase outfits generator."""
        app_config = get_config()
        return {
            'model_name': app_config.MODEL_NAME,
            'query_expansion': True,
            'reranking_enabled': True,
            'cache_embeddings': True,
            'main_outfits_count': 20,  # Changed from 50 to 20 for faster generation
            'tops_per_outfit': 30,  # Increased from 20 to 30 for more combinations
            'bottoms_per_outfit': 30,  # Increased from 20 to 30 for more combinations
            'semantic_boost_factors': {
                'style_match': 1.2,
                'color_match': 1.1,
                'occasion_match': 1.15
            }
        }

    def _load_color_harmony_from_csv(self) -> dict:
        """
        Load color harmony rules from the designer's CSV into a mapping:
        (base_color, pair_color) -> {'rating': int, 'notes': str}
        """
        import csv
        import os
        color_harmony_map = {}
        csv_path = os.path.join(
            'Fashion designer input',
            'Color Harmony.csv')
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                base_color = None
                for row in reader:
                    if not row or not any(row):
                        continue
                    
                    # Check for base color headers
                    if len(row) > 0 and 'BASE COLOR:' in str(row[0]):
                        base_color = str(row[0]).replace('BASE COLOR:', '').replace('*a color that can have it all*', '').strip()
                        continue
                    
                    # Skip header rows
                    if len(row) > 0 and any(keyword in str(row[0]).lower() for keyword in ['color combination', 'rating', 'notes', 'section']):
                        continue
                    
                    # Parse color combinations
                    if base_color and len(row) >= 2 and '+' in str(row[0]):
                        try:
                            # Parse color pair
                            color_pair = str(row[0]).strip()
                            if '+' in color_pair:
                                parts = color_pair.split('+')
                                if len(parts) == 2:
                                    color1 = base_color.title()
                                    color2 = parts[1].strip().title()
                                    
                                    # Parse rating
                                    rating = None
                                    if len(row) > 1 and row[1]:
                                        try:
                                            rating = int(str(row[1]).strip())
                                        except ValueError:
                                            # Try to extract number from text
                                            import re
                                            numbers = re.findall(r'\d+', str(row[1]))
                                            if numbers:
                                                rating = int(numbers[0])
                                    
                                    # Parse notes
                                    notes = str(row[2]).strip() if len(row) > 2 and row[2] else ""
                                    
                                    if rating is not None:
                                        color_harmony_map[(color1, color2)] = {
                                            'rating': rating, 
                                            'notes': notes,
                                            'base_color': color1,
                                            'pair_color': color2
                                        }
                        except Exception as e:
                            logger.debug(f"Could not parse color combination row: {row}, error: {e}")
                            continue
                            
            logger.info(f"✅ Loaded {len(color_harmony_map)} color harmony rules from designer CSV")
            
            # Add seasonal rules
            seasonal_rules = {
                'spring_summer': [
                    ('White', 'Blue'), ('Gray', 'Pink'), ('White', 'Yellow'), 
                    ('Black', 'Sage'), ('Beige', 'Pastels')
                ],
                'fall_winter': [
                    ('Black', 'Deep Red'), ('Navy', 'Emerald Green'), 
                    ('Brown', 'Cream'), ('Gray', 'Burgundy')
                ]
            }
            
            # Add special rules
            special_rules = {
                'black_safe': True,
                'white_universal': True,
                'no_bright_bright': True,
                'professional_dark_hues': True,
                'casual_pop_colors': True
            }
            
            color_harmony_map['_seasonal_rules'] = seasonal_rules
            color_harmony_map['_special_rules'] = special_rules
            
        except Exception as e:
            logger.warning(f"Could not load color harmony CSV: {e}")
            # Fallback to basic rules
            color_harmony_map = self._get_fallback_color_harmony()
            
        return color_harmony_map
    
    def _get_fallback_color_harmony(self) -> dict:
        """Fallback color harmony rules if CSV loading fails."""
        return {
            ('Black', 'White'): {'rating': 6, 'notes': 'Basic contrast'},
            ('Black', 'Navy'): {'rating': 9, 'notes': 'Elegant'},
            ('White', 'Blue'): {'rating': 9, 'notes': 'Classic'},
            ('Navy', 'White'): {'rating': 8, 'notes': 'Rich look'},
            '_seasonal_rules': {
                'spring_summer': [('White', 'Blue'), ('Gray', 'Pink')],
                'fall_winter': [('Black', 'Deep Red'), ('Navy', 'Emerald Green')]
            },
            '_special_rules': {
                'black_safe': True,
                'white_universal': True,
                'no_bright_bright': True
            }
        }
    
    def _initialize_professional_fashion_intelligence(self):
        """✅ ENHANCED: Initialize all professional fashion intelligence from main generator."""
        
        # ✅ ENHANCED: Professional scoring weights with cultural context
        self.scoring_weights = {
            'semantic_similarity': 3.0,           # Core AI matching
            'fit_compatibility': 2.8,            # NEW: Based on fit_confidence and body_shape_compatibility
            'comfort_metrics': 2.5,              # NEW: Based on comfort_level and movement_restriction
            'style_intelligence': 2.5,           # ENHANCED: Using new style attributes
            'color_harmony': 2.3,                # ENHANCED: Using new color attributes
            'quality_metrics': 2.0,              # NEW: Based on quality_indicators and durability
            'occasion_context': 2.0,             # ENHANCED: Using detailed occasion attributes
            'cultural_relevance': 1.8,           # ENHANCED: Better cultural context matching
            'versatility_score': 1.5,            # ENHANCED: Using style_versatility and adaptability
            'price_coherence': 1.3,              # Price point harmony
            'trend_relevance': 1.0               # Fashion forward
        }

        # ✅ ENHANCED: Style formality hierarchy for intelligent mixing
        self.style_formality = {
            'Ultra Formal': 10,
            'Business Formal': 9,
            'Evening Formal': 8,
            'Business Casual': 7,
            'Smart Casual': 6,
            'Contemporary': 5,
            'Casual': 4,
            'Streetwear': 3,
            'Athleisure': 2,
            'Activewear': 1
        }

        # ✅ ENHANCED: Professional color harmony with expert ratings
        self.color_harmony = self._initialize_professional_color_harmony()
        self.quick_rules = self._initialize_professional_quick_rules()
        self.body_shape_intelligence = self._initialize_body_shape_intelligence()
        
        # ✅ NEW: Load designer body shape rules
        self.body_shape_rules = self._load_body_shape_rules()
        
        # ✅ NEW: Load style mixing rules
        self.style_mixing_rules = self._load_style_mixing_rules()
        
        # ✅ NEW: Load quick styling rules
        self.quick_styling_rules = self._load_quick_styling_rules()
        
        # ✅ ENHANCED: Seasonal intelligence with cultural context
        self.seasonal_preferences = {
            'Spring': {
                'colors': ['Pink', 'Light Blue', 'Yellow', 'Green', 'White'],
                'fabrics': ['Cotton', 'Linen', 'Silk'],
                'patterns': ['Floral', 'Stripe', 'Geometric'],
                'cultural_favorites': ['Sage Green', 'Light Pink', 'White']
            },
            'Summer': {
                'colors': ['White', 'Light Blue', 'Yellow', 'Pink', 'Orange'],
                'fabrics': ['Cotton', 'Linen', 'Chambray'],
                'patterns': ['Solid', 'Stripe', 'Small Print'],
                'cultural_favorites': ['White', 'Sky Blue', 'Mint Green']
            },
            'Fall': {
                'colors': ['Brown', 'Orange', 'Navy', 'Burgundy', 'Green'],
                'fabrics': ['Wool', 'Cotton', 'Denim'],
                'patterns': ['Plaid', 'Check', 'Solid'],
                'cultural_favorites': ['Navy', 'Olive Green', 'Dark Brown']
            },
            'Winter': {
                'colors': ['Black', 'Gray', 'Navy', 'Burgundy', 'Purple'],
                'fabrics': ['Wool', 'Cashmere', 'Thick Cotton'],
                'patterns': ['Solid', 'Check', 'Herringbone'],
                'cultural_favorites': ['Black', 'Navy', 'Maroon']
            }
        }
        
        logger.info("✅ Enhanced fashion intelligence initialized with professional designer rules")

    def _initialize_professional_color_harmony(self) -> Dict:
        """✅ NEW: Initialize professional color harmony with expert ratings and cultural context."""
        return {
            'Black': {
                'perfect': [
                    {'color': 'Deep Red', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Maroon', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Navy Blue', 'rating': 9, 'context': 'Elegant'},
                    {'color': 'Emerald Green', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Olive Green', 'rating': 9, 'context': 'New n fresh, very flattering on Indian skin'}
                ],
                'excellent': [
                    {'color': 'Light Pink', 'rating': 8, 'context': 'Soft, delicate'},
                    {'color': 'Mint Green', 'rating': 8, 'context': 'Very clean look, breezy'},
                    {'color': 'Sky Blue', 'rating': 8, 'context': 'Casual formal'},
                    {'color': 'Charcoal Gray', 'rating': 8, 'context': 'Close to monochrome'}
                ]
            },
            'White': {
                'perfect': [
                    {'color': 'Light Pink', 'rating': 9, 'context': 'Dreamy'},
                    {'color': 'Bright Red', 'rating': 9, 'context': 'Classic contrast'},
                    {'color': 'Mint Green', 'rating': 9, 'context': 'Spring, soft, light, calm'},
                    {'color': 'Sky Blue', 'rating': 9, 'context': 'Best summer combo'},
                    {'color': 'Navy Blue', 'rating': 9, 'context': 'Classy'}
                ]
            },
            'Navy': {
                'perfect': [
                    {'color': 'Emerald Green', 'rating': 9, 'context': 'Bold yet classy'},
                    {'color': 'White', 'rating': 8, 'context': 'Rich look'},
                    {'color': 'Gray', 'rating': 8, 'context': 'Elegant n new'},
                    {'color': 'Burgundy', 'rating': 8, 'context': 'New, rich, colorful'}
                ]
            }
        }

    def _initialize_professional_quick_rules(self) -> Dict:
        """✅ NEW: Initialize professional quick rules from fashion designer."""
        return {
            'validated_rules': {
                'all_black_everything': {
                    'allowed': True,
                    'rating': 8,
                    'condition': 'Safe play for majority. Elevate with textures, patterns, shapes.'
                },
                'fit_balance': {
                    'rule': 'tight_top_loose_bottom',
                    'allowed': True,
                    'rating': 9,
                    'condition': 'Balances body shape'
                }
            },
            'never_break_rules': [
                'Color balance',
                'Top wear and bottom wear should never have same silhouette',
                'Right sizing - avoid too tight or too loose'
            ]
        }

    def _initialize_body_shape_intelligence(self) -> Dict:
        """✅ NEW: Initialize professional body shape intelligence for both genders."""
        return {
            'male': {
                'hourglass': {
                    'best_tops': 'Structured shirts, polos, t-shirts fitted',
                    'best_bottoms': 'Straight or tapered, not baggy',
                    'perfect_fit': 'Blazers with cinched waists, fitted crewnecks, knit polos',
                    'show_off': 'Natural upper-lower balance',
                    'never_wear': 'Boxy tops/bottoms',
                    'wrong_fit': 'Too baggy fits',
                    'pro_tip': 'Highlight snatched waists and good neckline to frame face and upper body',
                    'score_multiplier': 1.0
                },
                'rectangle': {
                    'best_tops': 'Layered tops to create definition',
                    'best_bottoms': 'Straight, pleated pants to flow with lower body curves',
                    'create_curves': 'Layering or textural contrast to create upper-lower division',
                    'best_details': 'Shaped fitted hem to break symmetry',
                    'never_wear': 'Straight fits',
                    'wrong_lines': 'Monotone looks',
                    'pro_tip': 'Add shapes and volume, visually divide body with belt, jacket',
                    'score_multiplier': 0.9
                },
                'pear': {
                    'best_tops': 'Add volume or detail to upper body',
                    'best_bottoms': 'Straight, wide-leg, bootcut - no skinny pants or hip-boxy',
                    'balance_trick': 'Highlight upper body with statement details, emphasize waist',
                    'enhance': 'Draw attention to waist and upper body, soft flare at bottom',
                    'never_wear': 'Tightly fitted bottoms',
                    'wrong_emphasis': 'Visible volume or detailing at hip',
                    'pro_tip': 'Volume at top, keep lower half clean and minimal',
                    'score_multiplier': 0.8
                },
                'apple': {
                    'best_tops': 'Loose but not oversized',
                    'best_bottoms': 'Mid-rise straight-leg or pleated trousers, clean drape without tight waist',
                    'flattering_lines': 'Monochrome vertical elongated look',
                    'best_assets': 'Show off legs, hands, neck',
                    'never_wear': 'Fitted on waist and hip, oversized clothing',
                    'wrong_areas': 'Do not draw attention to waist/hip, avoid bold belts, heavy prints',
                    'pro_tip': 'Elongate the torso',
                    'score_multiplier': 0.7
                }
            },
            'female': {
                'hourglass': {
                    'best_tops': 'Always fitted tops',
                    'best_bottoms': 'Straight or flare, not baggy',
                    'perfect_fit': 'Bodycon dresses, crop top + pants, mini skirts + fitted tops',
                    'show_off': 'Always highlight waist',
                    'never_wear': 'Boxy tops/bottoms',
                    'wrong_fit': 'Too baggy fits',
                    'pro_tip': 'Highlight snatched waists and good neckline to frame face and upper body',
                    'score_multiplier': 1.0
                },
                'rectangle': {
                    'best_tops': 'Cropped tops with shape/volume, flare/puff sleeves, peplum tops, loose but not oversized',
                    'best_bottoms': 'Boot cut, wide leg pants, skirts',
                    'create_curves': 'Add volume/details at sleeves and hem',
                    'best_details': 'Shaped fitted hem to break symmetry',
                    'never_wear': 'Straight fits',
                    'wrong_lines': 'Coords - matching top and pants',
                    'pro_tip': 'Add shapes and volume, visually divide body with bold belt',
                    'score_multiplier': 0.9
                },
                'pear': {
                    'best_tops': 'Add volume or detail to upper body, try deep necklines',
                    'best_bottoms': 'Straight, wide-leg, bootcut - no skinny pants or hip-boxy',
                    'balance_trick': 'Highlight upper body with statement tops, deep necklines, emphasize waist',
                    'enhance': 'Draw attention to waist and upper body, soft flare at bottom',
                    'never_wear': 'Tightly fitted bottoms or hip detailing like cowls in skirts',
                    'wrong_emphasis': 'Hip area emphasis',
                    'pro_tip': 'Fit and flare silhouettes, layered tops, strong necklines, crop jackets',
                    'score_multiplier': 0.8
                },
                'apple': {
                    'best_tops': 'Loose but not oversized',
                    'best_bottoms': 'Straight, wide-leg, bootcut - avoid tight waistbands or high-rise. Skirts are great.',
                    'flattering_lines': 'Empire waist and vertical elongated look',
                    'best_assets': 'Show off legs, hands, neck - short fits, smaller/no sleeves, deeper necklines',
                    'never_wear': 'Fitted on waist and hip, oversized clothing',
                    'wrong_areas': 'Do not draw attention to waist/hip',
                    'pro_tip': 'Elongate torso, shift focus to legs or neckline, avoid heavy prints',
                    'score_multiplier': 0.7
                }
            },
            'universal_rules': {
                'fit_same_way': False,
                'loose_top_rule': 'Bottom should be well fitted from waist/hip - straight, flare, or wide',
                'fitted_top_rule': 'Bottom can be straight, flare, wide, oversized',
                'most_flattering': 'Fitted top with straight or flare bottom',
                'universal_flattering': 'Fit and flare silhouettes work for every body shape',
                'most_important': 'Add accessories',
                'when_in_doubt': 'Fitted top + loose bottoms'
            }
        }

    def load_user_data_enhanced(self, user_id: int) -> Dict:
        """✅ ENHANCED: Load user data with style quiz flow from Supabase using enhanced database methods."""
        logger.info(f"📥 Loading enhanced user data for user {user_id} from Supabase...")
        
        try:
            # Use the enhanced database method with proper column mapping
            user_data = self.db.get_user_with_style_quiz(user_id)
            
            if not user_data:
                raise ValueError(f"User {user_id} not found or has no accessible data")
            
            logger.info(f"✅ Successfully loaded user data for user {user_id}")
            
            # Validate that we have required fields
            required_fields = ['User', 'Gender', 'Upper Wear Caption', 'Lower Wear Caption']
            missing_fields = [field for field in required_fields if field not in user_data]
            
            if missing_fields:
                logger.warning(f"⚠️ Missing required fields: {missing_fields}, adding defaults")
                for field in missing_fields:
                    if field == 'User':
                        user_data['User'] = user_id
                    elif field == 'Gender':
                        user_data['Gender'] = 'Unisex'
                    elif field == 'Upper Wear Caption':
                        user_data['Upper Wear Caption'] = self.db._generate_default_upper_wear_caption(user_data)
                    elif field == 'Lower Wear Caption':
                        user_data['Lower Wear Caption'] = self.db._generate_default_lower_wear_caption(user_data)
            
            return user_data
            
        except Exception as e:
            logger.error(f"❌ Error loading user data: {e}")
            # Return basic user profile as fallback
            return self._create_basic_user_profile({'id': user_id})

    def _create_basic_user_profile(self, user_data: Dict) -> Dict:
        """Create basic user profile when style quiz is not available."""
        return {
            'User': user_data.get('id'),
            'id': user_data.get('id'),
            'Gender': 'Unisex',
            'Fashion Style': 'Contemporary',
            'Body Shape': 'Rectangle',
            'Colors family': 'Blue',
            'occasion_preference': 'Daily Activities',
            'budget_preference': 'Mid-Range',
            'Upper Wear Caption': 'Contemporary casual style',
            'Lower Wear Caption': 'Comfortable everyday wear'
        }

    def _generate_upper_wear_caption(self, quiz_data: Dict) -> str:
        """Generate upper wear caption from style quiz data."""
        style = quiz_data.get('fashion_style', 'Contemporary')
        colors = quiz_data.get('color_preferences', 'Blue')
        occasion = quiz_data.get('occasion_preference', 'Daily')
        
        return f"{style} style upper wear in {colors} tones for {occasion} occasions"

    def _generate_lower_wear_caption(self, quiz_data: Dict) -> str:
        """Generate lower wear caption from style quiz data."""
        style = quiz_data.get('fashion_style', 'Contemporary')
        colors = quiz_data.get('color_preferences', 'Blue')
        fit = quiz_data.get('preferred_fit', 'Regular')
        
        return f"{style} style {fit} fit lower wear in complementary colors"

    def load_products_data_enhanced(self, filters: Dict = None, user_data: Dict = None) -> pd.DataFrame:
        """✅ ENHANCED: Load products data from Supabase with pre-filtering and enhanced validation."""
        logger.info("📥 Loading enhanced products data from Supabase using pre-filtering...")
        
        try:
            # ✅ FIX: Use simpler, more reliable method to avoid timeouts
            if user_data:
                # Extract gender from user data for filtering
                gender = user_data.get('Gender', '').lower()
                if gender in ['male', 'female']:
                    # Use chunking to load all products with gender filter
                    products_df = self.db._get_all_products_chunked(gender=gender)
                    logger.info(f"✅ Loaded {len(products_df)} products for {gender} users using chunking method")
                else:
                    # Fallback to basic products without gender filter using chunking
                    products_df = self.db._get_all_products_chunked()
                    logger.info(f"✅ Loaded {len(products_df)} products using chunking method (no gender filter)")
            else:
                # Fallback to basic products loading using chunking
                products_df = self.db._get_all_products_chunked()
                logger.info(f"✅ Loaded {len(products_df)} products using chunking method (no user data)")
            
            if products_df.empty:
                logger.error("❌ No products data found in Supabase")
                return pd.DataFrame()
            
            # ✅ ENHANCED: Apply comprehensive product validation and enhancement
            products_df = self.validate_products_data_enhanced(products_df)
            
            logger.info(f"✅ Enhanced products data loaded: {len(products_df)} products ready")
            return products_df
            
        except Exception as e:
            logger.error(f"❌ Error loading enhanced products data: {e}")
            return pd.DataFrame()

    def validate_products_data_enhanced(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate, clean and enhance product data with improved vocabulary matching.
        Returns enhanced DataFrame with normalized values and confidence scores.
        """
        if products_df.empty:
            raise ValueError("Empty products DataFrame provided")

        # Track data quality metrics
        metrics = {
            'total_products': len(products_df),
            'missing_title': products_df['title'].isna().sum(),
            'missing_category': products_df['scraped_category'].isna().sum() if 'scraped_category' in products_df.columns else 0
        }

        # Drop rows with missing critical fields
        products_df = products_df.dropna(subset=['title'])

        # ✅ ENHANCED: Infer wear_type from scraped_category and other information
        def infer_wear_type(row):
            """Infer wear type from scraped_category and title"""
            category = str(row.get('scraped_category', '')).lower()
            title = str(row.get('title', '')).lower()

            # Check for bottom wear keywords
            bottom_keywords = [
                'pant',
                'jean',
                'trouser',
                'short',
                'skirt',
                'legging',
                'jogger',
                'track',
                'bottom']
            if any(
                    keyword in category or keyword in title for keyword in bottom_keywords):
                return 'Bottomwear'

            # Check for top wear keywords
            top_keywords = [
                'shirt',
                't-shirt',
                'top',
                'blouse',
                'sweater',
                'sweatshirt',
                'hoodie',
                'jacket',
                'kurta',
                'kameez']
            if any(
                    keyword in category or keyword in title for keyword in top_keywords):
                return 'Upperwear'

            # Default to Upperwear if no clear match
            return 'Upperwear'

        products_df['wear_type'] = products_df.apply(infer_wear_type, axis=1)

        # Normalize and validate vocabulary
        for col in [
            'primary_style',
            'secondary_style',
            'primary_color',
            'secondary_color',
                'fit_analysis']:
            if col in products_df.columns:
                products_df[f'{col}_normalized'] = products_df[col].apply(lambda x: self.normalize_vocabulary(
                    x, col.split('_')[1]) if pd.notna(x) else self._get_default_value(col.split('_')[1]))
                products_df[f'{col}_confidence'] = products_df[col].apply(
                    lambda x: self._calculate_vocabulary_confidence(x, col.split('_')[1]) if pd.notna(x) else 0.0
                )

        # ✅ ENHANCED: Smart default pricing based on product context
        def get_smart_default_price(row):
            base_price = 1000  # Base price in INR

            # Adjust based on style
            if pd.notna(row.get('primary_style_normalized')):
                style = row['primary_style_normalized']
                if style in ['luxury', 'premium']:
                    base_price *= 2.5
                elif style in ['casual', 'basic']:
                    base_price *= 0.8

            # Adjust based on quality indicators
            if pd.notna(row.get('quality_indicators')):
                indicators = str(row['quality_indicators']).lower()
                if 'premium' in indicators or 'luxury' in indicators:
                    base_price *= 1.5
                elif 'basic' in indicators or 'standard' in indicators:
                    base_price *= 0.7

            # Adjust based on wear type
            if row['wear_type'] == 'Upperwear':
                base_price *= 1.2  # Tops generally cost more than bottoms

            return round(base_price)

        # Apply smart pricing
        if 'price' not in products_df.columns:
            products_df['price'] = products_df.apply(get_smart_default_price, axis=1)

        # Generate budget tags
        def get_budget_tag(price):
            if price <= 1000:
                return 'Budget'
            elif price <= 3000:
                return 'Mid-Range'
            elif price <= 8000:
                return 'Premium'
            else:
                return 'Luxury'

        products_df['budget_tag'] = products_df['price'].apply(get_budget_tag)

        # Ensure unique product IDs
        if 'product_id' not in products_df.columns:
            # Use the actual product_id column from the database, not create from id
            logger.warning("No product_id column found in products data - this should not happen")
            products_df['product_id'] = [f"PROD_{i:06d}" for i in range(len(products_df))]
        else:
            # Ensure product_id is string type
            products_df['product_id'] = products_df['product_id'].astype(str)

        # Calculate overall data quality score
        def calculate_quality_score(row):
            score = 0.0
            total_fields = 0

            # Required fields
            required_fields = ['title', 'scraped_category', 'wear_type']
            for field in required_fields:
                if pd.notna(row.get(field)):
                    score += 1.0
                total_fields += 1

            # Optional fields with confidence scores
            optional_fields = [
                'primary_style',
                'secondary_style',
                'primary_color',
                'secondary_color',
                'fit_analysis']
            for field in optional_fields:
                if f'{field}_confidence' in row:
                    score += row[f'{field}_confidence']
                    total_fields += 1

            return score / total_fields if total_fields > 0 else 0.0

        products_df['data_quality_score'] = products_df.apply(
            calculate_quality_score, axis=1)

        logger.info(f"Product validation complete. Average quality score: {products_df['data_quality_score'].mean():.2f}")
        logger.info(f"Price range: ₹{products_df['price'].min():.0f} - ₹{products_df['price'].max():.0f}")

        # Add missing default columns if not present
        if 'enhanced_primary_style' not in products_df.columns:
            products_df['enhanced_primary_style'] = products_df.get('primary_style', 'Contemporary')
        
        if 'enhanced_occasion' not in products_df.columns:
            products_df['enhanced_occasion'] = 'Daily'
        
        if 'gender' not in products_df.columns:
            products_df['gender'] = 'Unisex'

        # Create final_caption for FAISS if missing
        if 'final_caption' not in products_df.columns:
            products_df['final_caption'] = products_df.apply(lambda row: 
                f"{row.get('title', '')} {row.get('scraped_category', '')} {row.get('primary_style', '')} {row.get('primary_color', '')}".strip(), axis=1)

        return products_df

    def get_embedding_cached(self, text: str, cache_key: str = None, product_id: str = None) -> np.ndarray:
        """Get embedding with enhanced caching for better performance. Now uses precomputed embeddings from tagged_products table."""
        if not cache_key:
            cache_key = text[:100]
        
        # ✅ ENHANCED: Try to get precomputed embedding from tagged_products table first
        if product_id:
            try:
                # Query the tagged_products table for precomputed embedding
                result = self.db.client.table('tagged_products').select('product_embedding').eq('id', product_id).execute()
                
                if result.data and result.data[0].get('product_embedding'):
                    embedding_json = result.data[0]['product_embedding']
                    if isinstance(embedding_json, str):
                        embedding = np.array(json.loads(embedding_json))
                    else:
                        embedding = np.array(embedding_json)
                    
                    # Cache the result
                    if self.config['cache_embeddings']:
                        self.embedding_cache[cache_key] = embedding
                        self.cache_stats['size'] += 1
                        self.cache_stats['hits'] += 1
                    
                    logger.debug(f"✅ Retrieved precomputed embedding for product {product_id}")
                    return embedding
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get precomputed embedding for product {product_id}: {e}")
                # Fall back to computing embedding
        
        # Check cache first
        if self.config['cache_embeddings'] and cache_key in self.embedding_cache:
            self.cache_stats['hits'] += 1
            return self.embedding_cache[cache_key]
        
        # ✅ ENHANCED: Ensure model is loaded before computing embeddings
        self._ensure_model_loaded()
        
        # Generate embedding
        embedding = self.model.encode([text])[0]
        
        # 🚀 ENHANCED: Smart cache management with size limits
        if self.config['cache_embeddings']:
            # Check if cache is full and clean up if needed
            if self.cache_stats['size'] >= self.cache_stats['max_size']:
                self._cleanup_embedding_cache()
            
            # Add to cache
            self.embedding_cache[cache_key] = embedding
            self.cache_stats['size'] += 1
        
        self.cache_stats['misses'] += 1
        return embedding
    
    def _cleanup_embedding_cache(self):
        """Clean up embedding cache to prevent memory issues."""
        if len(self.embedding_cache) > self.cache_stats['max_size']:
            # Remove oldest entries (simple FIFO approach)
            items_to_remove = len(self.embedding_cache) - self.cache_stats['max_size'] + 100
            keys_to_remove = list(self.embedding_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.embedding_cache[key]
            
            self.cache_stats['size'] = len(self.embedding_cache)
            logger.info(f"🧹 Cleaned up embedding cache. Removed {items_to_remove} entries. Current size: {self.cache_stats['size']}")
    
    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics."""
        hit_rate = (self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])) * 100 if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'size': self.cache_stats['size'],
            'max_size': self.cache_stats['max_size']
        }

    def build_faiss_indexes(self, products_df: pd.DataFrame) -> None:
        """Build FAISS indexes for different wear types using enhanced Supabase data with precomputed embeddings."""
        logger.info("🔄 Building FAISS indexes for product recommendations using precomputed embeddings...")
        
        wear_types = ['Upperwear', 'Bottomwear']
        
        # 🚀 PARALLEL: Process wear types in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for wear_type in wear_types:
                wear_products = products_df[products_df['wear_type'] == wear_type].copy()
                if not wear_products.empty:
                    future = executor.submit(self._build_faiss_index_for_wear_type, wear_type, wear_products)
                    futures.append(future)
            
            # Wait for all indexes to be built
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"❌ Error building FAISS index: {e}")
    
    def _build_faiss_index_for_wear_type(self, wear_type: str, wear_products: pd.DataFrame) -> None:
        """Build FAISS index for a specific wear type using precomputed embeddings."""
        if wear_products.empty:
            logger.warning(f"No products found for wear_type: {wear_type}")
            return
        
        # ✅ ENHANCED: Use precomputed embeddings from tagged_products table
        embeddings = []
        product_indices = []
        valid_products = []
        
        logger.info(f"🔄 Loading precomputed embeddings for {len(wear_products)} {wear_type} products...")
        
        for idx, row in wear_products.iterrows():
            product_id = str(row['id'])
            
            try:
                # Get precomputed embedding from tagged_products table
                result = self.db.client.table('tagged_products').select('product_embedding').eq('id', product_id).execute()
                
                if result.data and result.data[0].get('product_embedding'):
                    embedding_json = result.data[0]['product_embedding']
                    if isinstance(embedding_json, str):
                        embedding = np.array(json.loads(embedding_json))
                    else:
                        embedding = np.array(embedding_json)
                    
                    embeddings.append(embedding)
                    product_indices.append(idx)
                    valid_products.append(row)
                else:
                    logger.warning(f"No precomputed embedding found for product {product_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to get embedding for product {product_id}: {e}")
                continue
        
        if not embeddings:
            logger.warning(f"No valid embeddings found for wear_type: {wear_type}")
            return

        embeddings = np.array(embeddings)
        logger.info(f"✅ Loaded {len(embeddings)} precomputed embeddings for {wear_type}")

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
            'products': pd.DataFrame(valid_products)
        }

        logger.info(f"✅ Built FAISS index for {wear_type}: {len(embeddings)} products indexed")
        
        # Log cache statistics
        cache_stats = self.get_cache_stats()
        logger.info(f"📊 Embedding cache stats: {cache_stats['hit_rate']} hit rate ({cache_stats['hits']} hits, {cache_stats['misses']} misses)")

    def filter_products_enhanced(self, products_df: pd.DataFrame, user: Dict, wear_type: str = None) -> pd.DataFrame:
        """✅ ENHANCED: Enhanced manual filtering on Gender, Fashion Style, and Body Shape with seasonal considerations."""
        logger.info(f"Starting enhanced filtering - Initial products: {len(products_df)}")

        # ✅ NEW: Seasonal filtering - exclude winter items during summer
        winter_keywords = [
            'sweater', 'pullover', 'hoodie', 'hooded', 'sweatshirt', 'cardigan', 'jumper',
            'wool', 'woollen', 'knit', 'thermal', 'fleece', 'winter', 'cold', 'warm', 'insulated',
            'turtleneck', 'turtle neck', 'turtle-neck', 'mock neck', 'high neck', 'cable knit',
            'chunky', 'thick', 'heavy', 'winter jacket', 'coat', 'blazer', 'jumper',
            'angora', 'cashmere', 'merino', 'alpaca', 'fuzzy', 'furry', 'thermal',
            'long sleeve', 'longsleeve', 'full sleeve', 'fullsleeve', 'warm jacket',
            'winter coat', 'overcoat', 'pea coat', 'duffle coat', 'parka', 'anorak'
        ]
        
        # Filter out winter items - check multiple fields
        winter_mask = products_df['title'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        winter_mask |= products_df['scraped_category'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        winter_mask |= products_df['primary_style'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        winter_mask |= products_df['style_category'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        winter_mask |= products_df['product_type'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        
        # Also check for winter indicators in the description or other fields
        if 'description' in products_df.columns:
            winter_mask |= products_df['description'].str.lower().str.contains('|'.join(winter_keywords), na=False)
        
        # Remove winter items
        products_df = products_df[~winter_mask]
        logger.info(f"🌞 Removed {winter_mask.sum()} winter items, remaining: {len(products_df)} products")

        def match(row):
            # 1. GENDER FILTERING (ENHANCED - Women only get female clothes, no unisex)
            user_gender = user.get('Gender', user.get('gender', '')).lower()
            product_gender = row.get('gender', 'Unisex').lower()

            if user_gender and product_gender:
                # Updated gender filtering logic
                if user_gender in ['male', 'men']:
                    acceptable_genders = ['men', 'male', 'unisex']
                elif user_gender in ['female', 'women']:
                    # Women only get female-specific clothes, no unisex
                    acceptable_genders = ['women', 'female']
                elif user_gender in ['unisex']:
                    acceptable_genders = ['men', 'male', 'women', 'female', 'unisex']
                else:
                    acceptable_genders = [user_gender, 'unisex']

                if product_gender not in acceptable_genders:
                    return False

            # 2. FASHION STYLE FILTERING (Flexible)
            user_style = user.get('Fashion Style', '').strip()
            product_style = row.get('enhanced_primary_style', row.get('primary_style', '')).strip()

            # --- Style Matching Logic ---
            # Supports user_style as JSON array or plain string
            # Matches if ANY user style (case-insensitive, partial match) is present in product_style
            if user_style and product_style:
                import json
                style_match = False
                user_styles = []
                # Try to parse user_style as JSON array
                if user_style.startswith('[') and user_style.endswith(']'):
                    try:
                        user_styles = json.loads(user_style)
                        if not isinstance(user_styles, list):
                            user_styles = [str(user_styles)]
                    except Exception:
                        user_styles = [user_style]
                else:
                    user_styles = [user_style]
                # For each user style, check if it appears in product_style (case-insensitive, partial match)
                for style in user_styles:
                    if not isinstance(style, str):
                        continue
                    if style.strip().lower() in product_style.lower():
                        style_match = True
                        break
                # If no direct match, try partial/word match
                if not style_match:
                    for style in user_styles:
                        if not isinstance(style, str):
                            continue
                        style_words = style.strip().lower().split()
                        for word in style_words:
                            if word and word in product_style.lower():
                                style_match = True
                                break
                        if style_match:
                            break
                # If still no match, fail the filter
                if not style_match:
                    return False
            # --- End Style Matching Logic ---

            # 3. WEAR TYPE FILTERING (Enhanced - based on apparel preferences)
            user_gender = user.get('Gender', user.get('gender', '')).lower()
            user_style = user.get('Fashion Style', '').strip()
            import json
            if user_style and user_style.startswith('[') and user_style.endswith(']'):
                try:
                    selected_styles = json.loads(user_style)
                    if not isinstance(selected_styles, list):
                        selected_styles = [str(selected_styles)]
                except Exception:
                    selected_styles = [user_style]
            else:
                selected_styles = [user_style] if user_style else []

            # Map style names to preference columns
            style_to_pref_map = {
                'business casual': 'Apparel Pref Business Casual',
                'streetwear': 'Apparel Pref Streetwear',
                'athleisure': 'Apparel Pref Athleisure'
            }

            # Collect all product types from all selected styles
            allowed_product_types = set()
            for style in selected_styles:
                if isinstance(style, str):
                    style_lower = style.strip().lower()
                    pref_key = style_to_pref_map.get(style_lower)
                    if pref_key:
                        pref_value = user.get(pref_key, '').strip()
                        if pref_value:
                            # Try to parse as JSON array
                            try:
                                pref_list = json.loads(pref_value)
                                if isinstance(pref_list, list):
                                    for pt in pref_list:
                                        if isinstance(pt, str) and pt.strip():
                                            allowed_product_types.add(pt.strip().lower())
                                else:
                                    if isinstance(pref_list, str) and pref_list.strip():
                                        allowed_product_types.add(pref_list.strip().lower())
                            except Exception:
                                # If not JSON, treat as single string
                                allowed_product_types.add(pref_value.lower())

            # If no apparel preferences found, use fallback based on gender
            if not allowed_product_types:
                if user_gender in ['male', 'men']:
                    allowed_product_types = {'tshirt', 'jeans', 'cargo'}
                elif user_gender in ['female', 'women']:
                    allowed_product_types = {'dress'}
                else:
                    allowed_product_types = {'tshirt', 'jeans'}

            # Check if product matches any of the allowed product types (exact match, case-insensitive)
            if allowed_product_types:
                product_type = str(row.get('product_type', '')).strip().lower()
                if product_type not in allowed_product_types:
                    return False

            return True

        filtered_df = products_df[products_df.apply(match, axis=1)]
        logger.info(f"Enhanced filtering complete - Remaining products: {len(filtered_df)}")
        return filtered_df

    def get_semantic_recommendations(self, user_profile: str, wear_type: str, 
                                   gender_filter: str = None, k: int = 20, user: Dict = None,
                                   target_product_type: str = None, target_style_category: str = None) -> List[Dict]:
        """✅ ENHANCED: Get semantic recommendations using FAISS with improved diversity and context awareness.
        Now supports same-category filtering for similar products."""

        if wear_type not in self.faiss_indexes:
            logger.warning(f"No FAISS index available for wear_type: {wear_type}")
            return []

        # Enhanced query expansion
        if user and self.config.get('query_expansion', False):
            expanded_profile = self.expand_user_query(user_profile, user, wear_type)
            query_embedding = self.get_embedding_cached(expanded_profile)
        else:
            query_embedding = self.get_embedding_cached(user_profile)

        # Search FAISS index
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        search_k = k * 3  # Get more candidates for diversity
        index = self.faiss_indexes[wear_type]
        scores, indices = index.search(query_embedding.astype('float32'), search_k)

        # Get corresponding products with enhanced filtering
        product_mapping = self.product_mappings[wear_type]
        candidates = []
        seen_styles = set()
        seen_colors = set()

        for i, (score, faiss_idx) in enumerate(zip(scores[0], indices[0])):
            if faiss_idx >= len(product_mapping['indices']):
                continue

            product_idx = product_mapping['indices'][faiss_idx]
            product = product_mapping['products'].iloc[faiss_idx]

            # ✅ ENHANCED: Same-category filtering for similar products
            if target_product_type and target_style_category:
                product_type = str(product.get('product_type', '')).strip().lower()
                style_category = str(product.get('style_category', product.get('primary_style', ''))).strip().lower()
                
                target_product_type_lower = target_product_type.strip().lower()
                target_style_category_lower = target_style_category.strip().lower()
                
                # Check if product matches the same category and style
                if product_type != target_product_type_lower or style_category != target_style_category_lower:
                    continue

            # Apply gender filter if specified
            if gender_filter and gender_filter != 'Unisex':
                product_gender = product.get('gender', 'Unisex').lower()
                gender_filter_lower = gender_filter.lower()

                # Updated gender filtering logic - consistent with main filtering
                if gender_filter_lower in ['male', 'men']:
                    acceptable_genders = ['men', 'male', 'unisex']
                elif gender_filter_lower in ['female', 'women']:
                    # Women only get female-specific clothes, no unisex
                    acceptable_genders = ['women', 'female']
                else:
                    acceptable_genders = [gender_filter_lower, 'unisex']

                if product_gender not in acceptable_genders:
                    continue

            # Check style and color diversity
            style = product.get('enhanced_primary_style', product.get('primary_style', ''))
            color = product.get('primary_color', '')
            
            if style in seen_styles and len(seen_styles) >= k // 2:
                continue
            if color in seen_colors and len(seen_colors) >= k // 2:
                continue

            candidates.append({
                'product_idx': product_idx,
                'product': product,
                'semantic_score': float(score),
                'faiss_rank': i + 1
            })

            seen_styles.add(style)
            seen_colors.add(color)

            if len(candidates) >= k:
                break

        # Sort by score
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
        return candidates[:k]

    def expand_user_query(self, user_profile: str, user: Dict, wear_type: str) -> str:
        """✅ ENHANCED: Expand user query with comprehensive fashion context."""
        expanded_segments = [user_profile]

        # Style context expansion
        style = user.get('Fashion Style', '')
        if style:
            style_expansions = {
                'Streetwear': 'urban fashion street style casual contemporary edgy youthful trendy modern',
                'Athleisure': 'sporty casual athletic wear active lifestyle performance casual comfort versatile',
                'Contemporary': 'modern style current fashion updated classic fresh aesthetic sophisticated trendy',
                'Business': 'professional attire work appropriate office wear corporate fashion polished formal',
                'Formal': 'elegant sophisticated professional business formal evening wear polished refined'
            }

            if style in style_expansions:
                expanded_segments.append(f"Style context: {style_expansions[style]}")

        # Color intelligence expansion
        colors = user.get('Colors family', '')
        if colors:
            color_expansions = {
                'Blue': 'navy blue sky blue cobalt blue royal blue',
                'Black': 'jet black charcoal black deep black',
                'White': 'pure white ivory white cream white',
                'Red': 'deep red burgundy red crimson red',
                'Green': 'olive green emerald green sage green'
            }

            if colors in color_expansions:
                expanded_segments.append(f"Color preferences: {color_expansions[colors]}")

        # Wear type specific context
        if wear_type == 'Upperwear':
            expanded_segments.append("top wear shirt blouse t-shirt")
        elif wear_type == 'Bottomwear':
            expanded_segments.append("bottom wear pants trousers jeans")

        return ". ".join(expanded_segments)



    def score_outfit_enhanced(self, top: pd.Series, bottom: pd.Series, user: Dict, 
                            top_semantic: float, bottom_semantic: float) -> Tuple[float, str]:
        """✅ ENHANCED: Score outfit using comprehensive designer rule engine."""
        try:
            total_score = 0.0
            explanations = []
            
            # 1. SEMANTIC SIMILARITY (Core AI matching)
            semantic_score = (top_semantic + bottom_semantic) / 2
            total_score += semantic_score * self.scoring_weights['semantic_similarity']
            explanations.append(f"AI style match: {semantic_score:.2f}")
            
            # 2. ✅ ENHANCED: Comprehensive Designer Rule Engine
            rule_scores = self._apply_designer_rule_engine(top, bottom, user)
            
            # Apply rule scores with weights
            rule_weights = {
                'color_harmony': self.scoring_weights['color_harmony'],
                'body_shape': self.scoring_weights['fit_compatibility'],
                'style_mixing': self.scoring_weights['style_intelligence'],
                'quick_rules': self.scoring_weights['quality_metrics'],
                'fit_balance': self.scoring_weights['fit_compatibility'],
                'silhouette': self.scoring_weights['style_intelligence'],
                'accessories': self.scoring_weights['quality_metrics'],
                'seasonal': self.scoring_weights['occasion_context'],
                'cultural': self.scoring_weights['cultural_relevance'],
                'professional': self.scoring_weights['occasion_context']
            }
            
            for rule_type, score in rule_scores.items():
                if rule_type in rule_weights:
                    weighted_score = score * rule_weights[rule_type]
                    total_score += weighted_score
                    if score > 6.0:  # Only mention high-scoring rules
                        explanations.append(f"{rule_type.replace('_', ' ').title()}: {score:.2f}")
            
            # 3. Additional scoring factors
            # Fit Compatibility (Enhanced)
            fit_score = self._calculate_fit_compatibility_score(top, user) + self._calculate_fit_compatibility_score(bottom, user)
            fit_score /= 2
            total_score += fit_score * self.scoring_weights['fit_compatibility']
            explanations.append(f"Fit compatibility: {fit_score:.2f}")
            
            # Quality Metrics
            quality_score = self._calculate_quality_metrics_score(top, bottom)
            total_score += quality_score * self.scoring_weights['quality_metrics']
            explanations.append(f"Quality: {quality_score:.2f}")
            
            # Price Coherence
            price_score = self._calculate_price_coherence(top, bottom)
            total_score += price_score * self.scoring_weights['price_coherence']
            explanations.append(f"Price harmony: {price_score:.2f}")
            
            # Versatility Score
            versatility_score = self._calculate_versatility_score(top, bottom)
            total_score += versatility_score * self.scoring_weights['versatility_score']
            explanations.append(f"Versatility: {versatility_score:.2f}")
            
            # Trend Relevance
            trend_score = self._calculate_trend_relevance_score(top, bottom)
            total_score += trend_score * self.scoring_weights['trend_relevance']
            explanations.append(f"Trend relevance: {trend_score:.2f}")
            
            # Normalize final score
            final_score = min(10.0, max(0.0, total_score / sum(self.scoring_weights.values())))
            
            # Generate comprehensive explanation with rule insights
            explanation = self._generate_enhanced_explanation(explanations, final_score, rule_scores, top, bottom, user)
            
            return final_score, explanation
            
        except Exception as e:
            logger.error(f"❌ Error scoring outfit: {e}")
            return 5.0, "Standard outfit combination"
    
    def _generate_enhanced_explanation(self, explanations: List[str], final_score: float, 
                                     rule_scores: Dict[str, float], top: pd.Series, bottom: pd.Series, user: Dict) -> str:
        """Generate enhanced explanation using rule engine insights."""
        try:
            # Get top performing rules
            top_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get rule-specific insights
            insights = []
            
            # Color harmony insight
            if rule_scores.get('color_harmony', 0) > 7.0:
                top_color = self._extract_color(top).title()
                bottom_color = self._extract_color(bottom).title()
                color_pair = (bottom_color, top_color)
                if color_pair in self.color_harmony_map:
                    rule = self.color_harmony_map[color_pair]
                    insights.append(f"Professional color pairing: {rule.get('notes', '')}")
            
            # Body shape insight
            if rule_scores.get('body_shape', 0) > 7.0:
                body_shape = user.get('Body Shape', '').lower()
                if body_shape:
                    insights.append(f"Perfect for {body_shape} body shape")
            
            # Style mixing insight
            if rule_scores.get('style_mixing', 0) > 7.0:
                insights.append("Excellent style balance")
            
            # Quick rules insight
            if rule_scores.get('quick_rules', 0) > 7.0:
                insights.append("Follows professional styling rules")
            
            # Seasonal insight
            if rule_scores.get('seasonal', 0) > 7.0:
                import datetime
                month = datetime.datetime.now().month
                season = "Spring/Summer" if month in [3, 4, 5, 6] else "Fall/Winter"
                insights.append(f"Perfect for {season}")
            
            # Cultural insight
            if rule_scores.get('cultural', 0) > 7.0:
                insights.append("Culturally appropriate styling")
            
            # Professional insight
            if rule_scores.get('professional', 0) > 7.0:
                insights.append("Professional setting appropriate")
            
            # Combine explanations
            base_explanation = f"Outfit score: {final_score:.1f}/10. "
            
            if insights:
                base_explanation += " ".join(insights[:2])  # Limit to 2 insights
            else:
                # Fallback to rule scores
                top_rule = top_rules[0] if top_rules else None
                if top_rule and top_rule[1] > 6.0:
                    base_explanation += f"Strong {top_rule[0].replace('_', ' ')} compatibility."
                else:
                    base_explanation += "Stylish combination following professional fashion guidelines."
            
            return base_explanation
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced explanation: {e}")
            return f"Professional outfit combination with score {final_score:.1f}/10"

    def generate_main_outfits_for_user(self, user_id: int) -> List[Dict]:
        """✅ ENHANCED: Generate 20 main outfits for a user using enhanced Supabase data."""
        try:
            logger.info(f"🎯 Generating enhanced main outfits for user {user_id} using Supabase...")
            
            # ✅ ENHANCED: Load user data with style quiz flow
            user = self.load_user_data_enhanced(user_id)
            
            # Validate user data
            self.validate_user_data(user)
            
            # Load and validate products
            products_df = self.load_products_data_enhanced(user_data=user)
            
            if products_df.empty:
                logger.error("❌ No products data available")
                return []
            
            # ✅ FIX 1: Ensure only 20 outfits
            target_outfits = 20
            
            # ✅ FIX 3: Get user's style preferences for proper distribution
            user_styles = self._get_user_style_preferences(user)
            logger.info(f"🎨 User style preferences: {user_styles}")
            
            # ✅ FIX 2 & 3: Generate outfits with product diversity and style distribution
            outfits = self._generate_diverse_outfits_with_style_distribution(
                products_df, user, target_outfits, user_styles
            )
            
            if not outfits:
                logger.warning("❌ No outfits generated")
                return []
            
            logger.info(f"✅ Generated {len(outfits)} diverse outfits with proper style distribution")
            return outfits
            
        except Exception as e:
            logger.error(f"❌ Error generating outfits: {e}")
            return []
    
    def _get_user_style_preferences(self, user_data: Dict) -> List[str]:
        """Get user's ACTUAL style preferences from their Fashion Style selection only."""
        import json
        styles = []
        
        # Helper to parse possible JSON array or comma-separated string
        def parse_styles(val):
            if not val:
                return []
            if isinstance(val, list):
                return [s.lower().strip() for s in val if s]
            if isinstance(val, str):
                v = val.strip()
                if v.startswith('[') and v.endswith(']'):
                    try:
                        arr = json.loads(v)
                        if isinstance(arr, list):
                            return [s.lower().strip() for s in arr if s]
                    except Exception:
                        pass
                # fallback: comma-separated
                return [s.lower().strip() for s in v.split(',') if s.strip()]
            return []
        
        # ✅ FIX: Use ONLY the Fashion Style field as the source of truth
        # The Fashion Style field contains what the user actually selected
        fashion_style = user_data.get('Fashion Style', '')
        if fashion_style:
            styles = parse_styles(fashion_style)
        
        # ✅ CRITICAL FIX: Only check apparel preferences if the user actually selected those styles
        # Don't automatically add styles just because apparel preferences exist
        # The apparel preferences are just product categories within the selected styles
        
        # Remove duplicates while preserving order
        seen = set()
        unique_styles = []
        for style in styles:
            if style not in seen and style:
                seen.add(style)
                unique_styles.append(style)
        
        # 4. If no styles found, use fallback based on occasion
        if not unique_styles:
            occasion = user_data.get('occasion_preference', '').lower()
            if 'work' in occasion or 'office' in occasion or 'professional' in occasion:
                unique_styles = ['business casual']
            elif 'gym' in occasion or 'sport' in occasion or 'active' in occasion:
                unique_styles = ['athleisure']
            else:
                unique_styles = ['streetwear']
        
        logger.info(f"🎨 User's actual style preferences: {unique_styles}")
        return unique_styles
    
    def _generate_diverse_outfits_with_style_distribution(self, 
                                                        products_df: pd.DataFrame, 
                                                        user_data: Dict, 
                                                        target_outfits: int,
                                                        user_styles: List[str]) -> List[Dict]:
        """Generate diverse outfits with proper distribution across user's selected styles."""
        try:
            outfits = []
            used_top_ids = set()
            used_bottom_ids = set()
            
            # ✅ FIX: Ensure exactly target_outfits (20) are generated
            logger.info(f"🎯 Target outfits: {target_outfits}")
            
            # Calculate outfits per style for equal distribution
            outfits_per_style = target_outfits // len(user_styles)
            remaining_outfits = target_outfits % len(user_styles)
            
            logger.info(f"🎯 Generating {outfits_per_style} outfits per style: {user_styles}")
            
            for i, style in enumerate(user_styles):
                # Add extra outfit to first style if there's remainder
                current_style_count = outfits_per_style + (1 if i < remaining_outfits else 0)
                
                logger.info(f"🎨 Generating {current_style_count} outfits for style: {style}")
                
                # ✅ FIX: More flexible style filtering
                style_products = self._filter_products_by_style(products_df, style)
                
                if style_products.empty:
                    logger.warning(f"⚠️ No products found for style: {style}")
                    # ✅ FIX: Try fallback to broader style matching
                    style_products = self._filter_products_by_style_fallback(products_df, style)
                    if style_products.empty:
                        logger.error(f"❌ No products available for style: {style} even with fallback")
                        continue
                
                # Generate outfits for this style
                style_outfits = self._generate_style_specific_outfits(
                    style_products, user_data, current_style_count, 
                    used_top_ids, used_bottom_ids, style
                )
                
                outfits.extend(style_outfits)
                logger.info(f"✅ Generated {len(style_outfits)} outfits for {style}")
            
            # ✅ FIX: If we don't have enough outfits, try to generate more with available products
            if len(outfits) < target_outfits:
                logger.warning(f"⚠️ Only generated {len(outfits)} outfits, need {target_outfits}")
                remaining_needed = target_outfits - len(outfits)
                
                # Try to generate remaining outfits with any available products
                remaining_outfits = self._generate_remaining_outfits(
                    products_df, user_data, remaining_needed, used_top_ids, used_bottom_ids
                )
                outfits.extend(remaining_outfits)
                logger.info(f"✅ Generated {len(remaining_outfits)} additional outfits")
            
            # Ensure we have exactly target_outfits
            if len(outfits) > target_outfits:
                outfits = outfits[:target_outfits]
                logger.info(f"✅ Trimmed to exactly {target_outfits} outfits")
            
            logger.info(f"🎯 Final outfit count: {len(outfits)}")
            logger.info(f"🎯 Unique top IDs used: {len(used_top_ids)}")
            logger.info(f"🎯 Unique bottom IDs used: {len(used_bottom_ids)}")
            
            return outfits
            
        except Exception as e:
            logger.error(f"❌ Error generating diverse outfits: {e}")
            return []
    
    def _filter_products_by_style_fallback(self, products_df: pd.DataFrame, target_style: str) -> pd.DataFrame:
        """Fallback style filtering with broader matching when strict filtering fails."""
        try:
            if products_df.empty:
                return pd.DataFrame()
            
            # Broader style mappings for fallback
            broad_style_mappings = {
                'business casual': ['business', 'casual', 'professional', 'office', 'work', 'smart'],
                'streetwear': ['street', 'urban', 'contemporary', 'modern', 'hip-hop', 'edgy'],
                'athleisure': ['athletic', 'sport', 'active', 'performance', 'gym', 'workout', 'sporty'],
                'casual': ['casual', 'relaxed', 'everyday', 'comfortable', 'informal'],
                'formal': ['formal', 'elegant', 'sophisticated', 'business formal', 'evening'],
                'contemporary': ['contemporary', 'modern', 'current', 'trendy', 'fashion-forward'],
                'vintage': ['vintage', 'retro', 'classic', 'heritage', 'nostalgic'],
                'bohemian': ['bohemian', 'boho', 'free-spirited', 'artistic', 'eclectic'],
                'minimalist': ['minimalist', 'minimal', 'simple', 'clean', 'essential']
            }
            
            # Get broader keywords for the target style
            style_keywords = broad_style_mappings.get(target_style.lower(), [target_style.lower()])
            
            # Broader filtering using multiple fields that actually exist
            filtered_products = products_df[
                # Check primary_style field
                products_df['primary_style'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check full_caption field
                products_df['full_caption'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check title field for style keywords
                products_df['title'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check style_category field (exists)
                products_df['style_category'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check product_type field (exists)
                products_df['product_type'].str.contains('|'.join(style_keywords), case=False, na=False)
            ].copy()
            
            logger.info(f"🎨 Fallback filtering found {len(filtered_products)} products for style '{target_style}'")
            
            return filtered_products
            
        except Exception as e:
            logger.error(f"❌ Error in fallback style filtering: {e}")
            return pd.DataFrame()
    
    def _generate_remaining_outfits(self, products_df: pd.DataFrame, user_data: Dict, 
                                  count: int, used_top_ids: set, used_bottom_ids: set) -> List[Dict]:
        """Generate remaining outfits using any available products when style-specific filtering fails."""
        try:
            outfits = []
            
            # Use any available products that haven't been used
            available_products = products_df[
                ~products_df['product_id'].isin(used_top_ids | used_bottom_ids)
            ].copy()
            
            if available_products.empty:
                logger.warning("⚠️ No unused products available for remaining outfits")
                return []
            
            # Separate tops and bottoms
            tops = available_products[available_products['wear_type'] == 'Upperwear'].copy()
            bottoms = available_products[available_products['wear_type'] == 'Bottomwear'].copy()
            
            if tops.empty or bottoms.empty:
                logger.warning("⚠️ Missing tops or bottoms for remaining outfits")
                return []
            
            # Generate remaining outfits
            for i in range(count):
                if tops.empty or bottoms.empty:
                    logger.warning("⚠️ Ran out of products for remaining outfits")
                    break
                
                # Select random top and bottom
                top = tops.sample(n=1).iloc[0]
                bottom = bottoms.sample(n=1).iloc[0]
                
                # Remove selected products from available pool
                tops = tops[tops['product_id'] != top['product_id']]
                bottoms = bottoms[bottoms['product_id'] != bottom['product_id']]
                
                # Add to used sets
                used_top_ids.add(top['product_id'])
                used_bottom_ids.add(bottom['product_id'])
                
                # Generate outfit data with fallback style
                outfit = self._create_outfit_data(
                    top, bottom, user_data, len(outfits) + 1, self._get_user_style_preferences(user_data)[0] if self._get_user_style_preferences(user_data) else "Business Casual"
                )
                
                outfits.append(outfit)
            
            return outfits
            
        except Exception as e:
            logger.error(f"❌ Error generating remaining outfits: {e}")
            return []
    
    def _filter_products_by_style(self, products_df: pd.DataFrame, target_style: str) -> pd.DataFrame:
        """Filter products by style with intelligent matching that respects multi-category products."""
        try:
            if products_df.empty:
                return pd.DataFrame()
            
            target_style_lower = target_style.lower()
            
            # ✅ ADDITIONAL SEASONAL FILTER: Remove winter items that might have slipped through
            winter_keywords = [
                'sweater', 'pullover', 'hoodie', 'hooded', 'sweatshirt', 'cardigan', 'jumper',
                'wool', 'woollen', 'knit', 'thermal', 'fleece', 'winter', 'cold', 'warm', 'insulated',
                'turtleneck', 'turtle neck', 'turtle-neck', 'mock neck', 'high neck', 'cable knit',
                'chunky', 'thick', 'heavy', 'winter jacket', 'coat', 'blazer', 'jumper',
                'angora', 'cashmere', 'merino', 'alpaca', 'fuzzy', 'furry', 'thermal',
                'long sleeve', 'longsleeve', 'full sleeve', 'fullsleeve', 'warm jacket',
                'winter coat', 'overcoat', 'pea coat', 'duffle coat', 'parka', 'anorak'
            ]
            
            # Filter out winter items using columns that exist
            winter_mask = products_df['title'].str.lower().str.contains('|'.join(winter_keywords), na=False)
            winter_mask |= products_df['scraped_category'].str.lower().str.contains('|'.join(winter_keywords), na=False)
            winter_mask |= products_df['primary_style'].str.lower().str.contains('|'.join(winter_keywords), na=False)
            winter_mask |= products_df['style_category'].str.lower().str.contains('|'.join(winter_keywords), na=False)
            winter_mask |= products_df['product_type'].str.lower().str.contains('|'.join(winter_keywords), na=False)
            
            # Remove winter items
            products_df = products_df[~winter_mask]
            logger.info(f"🌞 Style filter: Removed {winter_mask.sum()} winter items for {target_style}")
            
            # ✅ PRACTICAL APPROACH: Use the actual style fields from the database
            # Check the style fields that actually exist in order of importance
            style_matches = []
            
            # 1. Check style_category (most important - exists)
            style_matches.append(products_df[
                products_df['style_category'].str.contains(target_style_lower, case=False, na=False)
            ])
            
            # 2. Check primary_style (exists)
            style_matches.append(products_df[
                products_df['primary_style'].str.contains(target_style_lower, case=False, na=False)
            ])
            
            # 3. Check title for style keywords
            style_matches.append(products_df[
                products_df['title'].str.contains(target_style_lower, case=False, na=False)
            ])
            
            # 4. Check product_type for style keywords
            style_matches.append(products_df[
                products_df['product_type'].str.contains(target_style_lower, case=False, na=False)
            ])
            
            # Combine all matches and remove duplicates
            if style_matches:
                filtered_products = pd.concat(style_matches, ignore_index=True).drop_duplicates(subset=['product_id'])
            else:
                filtered_products = pd.DataFrame()
            
            logger.info(f"🎨 Found {len(filtered_products)} products for style '{target_style}' using intelligent filtering")
            
            # Debug: Log some sample products to verify filtering
            if not filtered_products.empty:
                sample_products = filtered_products.head(3)
                for _, product in sample_products.iterrows():
                    logger.info(f"  ✅ Sample: {product.get('title', 'N/A')} | Primary Style: {product.get('primary_style', 'N/A')} | Style: {product.get('style_category', 'N/A')} | ID: {product.get('product_id', 'N/A')}")
            
            return filtered_products
            
        except Exception as e:
            logger.error(f"❌ Error filtering products by style: {e}")
            return pd.DataFrame()
    
    def _generate_style_specific_outfits(self, 
                                       style_products: pd.DataFrame,
                                       user_data: Dict,
                                       count: int,
                                       used_top_ids: set,
                                       used_bottom_ids: set,
                                       style: str) -> List[Dict]:
        """Generate outfits for a specific style with product diversity."""
        try:
            outfits = []
            
            # Separate tops and bottoms
            tops = style_products[style_products['wear_type'] == 'Upperwear'].copy()
            bottoms = style_products[style_products['wear_type'] == 'Bottomwear'].copy()
            
            if tops.empty or bottoms.empty:
                logger.warning(f"⚠️ Missing tops or bottoms for style: {style}")
                return []
            
            # Remove already used products
            tops = tops[~tops['product_id'].isin(used_top_ids)]
            bottoms = bottoms[~bottoms['product_id'].isin(used_bottom_ids)]
            
            if tops.empty or bottoms.empty:
                logger.warning(f"⚠️ No unused products available for style: {style}")
                return []
            
            # Generate outfits
            for i in range(count):
                if tops.empty or bottoms.empty:
                    logger.warning(f"⚠️ Ran out of products for style: {style}")
                    break
                
                # Select random top and bottom
                top = tops.sample(n=1).iloc[0]
                bottom = bottoms.sample(n=1).iloc[0]
                
                # Remove selected products from available pool
                tops = tops[tops['product_id'] != top['product_id']]
                bottoms = bottoms[bottoms['product_id'] != bottom['product_id']]
                
                # Add to used sets
                used_top_ids.add(top['product_id'])
                used_bottom_ids.add(bottom['product_id'])
                
                # Generate outfit data
                outfit = self._create_outfit_data(
                    top, bottom, user_data, len(outfits) + 1, style
                )
                
                outfits.append(outfit)
            
            return outfits
            
        except Exception as e:
            logger.error(f"❌ Error generating style-specific outfits: {e}")
            return []
    
    def _create_outfit_data(self, top: pd.Series, bottom: pd.Series, 
                           user_data: Dict, rank: int, style: str) -> Dict:
        """Create outfit data with proper naming and scoring."""
        try:
            # ✅ FIX 1: Improved type checking for pandas Series
            if not isinstance(top, pd.Series) or not isinstance(bottom, pd.Series):
                logger.warning(f"[SKIP] Invalid product type: top={type(top)}, bottom={type(bottom)}")
                return {}
            
            # ✅ FIX 2: Safe data access for pandas Series
            try:
                top_id = top.get('product_id', '')
                bottom_id = bottom.get('product_id', '')
            except Exception as e:
                logger.warning(f"[SKIP] Cannot access product ID: {e}")
                return {}
            
            # ✅ FIX 3: Strict wear type validation
            top_wear_type = str(top.get('wear_type', '')).lower()
            bottom_wear_type = str(bottom.get('wear_type', '')).lower()
            
            if top_wear_type != 'upperwear' or bottom_wear_type != 'bottomwear':
                logger.warning(f"[SKIP] Invalid wear type combination: top={top_wear_type}, bottom={bottom_wear_type}")
                return {}
            
            # Calculate scores
            top_score = self._calculate_product_score(top, user_data)
            bottom_score = self._calculate_product_score(bottom, user_data)
            total_score = (top_score + bottom_score) / 2
            
            # Ensure score is never null
            if total_score is None or pd.isna(total_score):
                total_score = 0.5
            
            # Generate explanation
            explanation = self._generate_outfit_explanation(top, bottom, user_data, style)
            
            # Generate why picked explanation with proper format
            why_picked = self._generate_why_picked_explanation_enhanced(top, bottom, user_data, style)
            
            # Generate outfit name and description
            outfit_name = self._generate_outfit_name(top, bottom, style)
            outfit_description = self._generate_outfit_description(top, bottom, style)
            
            # ✅ FIX 4: Proper color extraction from multiple possible fields
            top_color = self._extract_color(top)
            bottom_color = self._extract_color(bottom)
            
            # ✅ FIX 3: Safe data access with proper pandas Series methods
            outfit_data = {
                'main_outfit_id': f"main_{user_data.get('id', 'unknown')}_{rank}",
                'rank': rank,
                'score': float(total_score),  # Ensure score is a float
                'explanation': explanation or "Stylish outfit combination",
                'why_picked_explanation': why_picked or "Carefully selected based on your preferences",
                'outfit_name': outfit_name or f"{style.upper()} OUTFIT",
                'outfit_description': outfit_description or "A stylish outfit combination for various occasions",
                'top_id': str(top_id),
                'bottom_id': str(bottom_id),
                'top_title': str(top.get('title', '')),
                'bottom_title': str(bottom.get('title', '')),
                'top_color': top_color,
                'bottom_color': bottom_color,
                'top_style': str(style).title(),  # Use user's selected style
                'bottom_style': str(style).title(),  # Use user's selected style
                'top_price': float(top.get('price', 0)),
                'bottom_price': float(bottom.get('price', 0)),
                'top_image': str(top.get('image_url', '')),
                'bottom_image': str(bottom.get('image_url', '')),
                'top_semantic_score': float(top_score),
                'bottom_semantic_score': float(bottom_score),
                'user_id': int(user_data.get('id', 0)),
                'generation_method': 'enhanced_supabase'
            }
            
            # Calculate total price
            outfit_data['total_price'] = outfit_data['top_price'] + outfit_data['bottom_price']
            
            return outfit_data
            
        except Exception as e:
            logger.error(f"❌ Error creating outfit data: {e}")
            return {}
    
    def _extract_color(self, product: pd.Series) -> str:
        """Extract color from product data using multiple possible fields."""
        try:
            # Try multiple color fields in order of preference
            color_fields = [
                'primary_color',
                'dominant_color', 
                'color',
                'scraped_color',
                'llava_color'
            ]
            
            for field in color_fields:
                if field in product and product[field]:
                    color = str(product[field]).strip()
                    if color and color.lower() not in ['nan', 'none', 'null', '']:
                        return color.title()
            
            # Fallback: try to extract from title or caption
            title = str(product.get('title', '')).lower()
            caption = str(product.get('full_caption', '')).lower()
            
            # Common color keywords
            color_keywords = [
                'black', 'white', 'blue', 'red', 'green', 'yellow', 'purple', 'pink', 
                'orange', 'brown', 'gray', 'grey', 'navy', 'beige', 'cream', 'olive',
                'burgundy', 'maroon', 'coral', 'teal', 'mint', 'sage', 'camel', 'tan'
            ]
            
            for color in color_keywords:
                if color in title or color in caption:
                    return color.title()
            
            return "Black"  # Default color
            
        except Exception as e:
            logger.error(f"❌ Error extracting color: {e}")
            return "Black"
    
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
    
    def _generate_why_picked_explanation_enhanced(self, top: pd.Series, bottom: pd.Series, 
                                       user_data: Dict, style: str = None) -> str:
        """Generate why picked explanation using exact keywords and explanations from Excel file."""
        try:
            # Extract user data
            user_body_shape = user_data.get('Body Shape', 'Rectangle')
            user_gender = user_data.get('Gender', 'Unisex')
            
            # Use the individual outfit style if provided, otherwise fall back to user's Fashion Style
            if style:
                user_style = style
            else:
                user_style = user_data.get('Fashion Style', 'Contemporary')
            
            # Extract product data
            top_color = self._extract_color(top)
            bottom_color = self._extract_color(bottom)
            
            # 1. STYLE VIBE - Use exact keywords and explanations from Excel
            style_vibe = self._get_style_vibe_from_excel_exact(user_style)
            
            # 2. OCCASIONS - Use exact keywords and explanations from Excel
            occasions = self._get_occasions_from_excel_exact(user_style)
            
            # 3. SKIN TONE - Use exact keywords and explanations from Excel
            skin_tone = self._get_skin_tone_from_excel_exact(top_color, bottom_color)
            
            # 4. BODY SHAPE - Use exact keywords and explanations from Excel
            body_shape = self._get_body_shape_from_excel_exact(user_body_shape, user_gender)
            
            # Combine all sections with proper format
            why_picked = f"Style Vibe: {style_vibe} | Occasions: {occasions} | Skin Tone: {skin_tone} | Body Shape: {body_shape}"
            
            return why_picked
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced why picked explanation: {e}")
            return "Personalized outfit selection based on your style preferences"
    
    def _get_style_vibe_from_excel_exact(self, user_style: str) -> str:
        """Get style vibe using exact keywords and explanations from Excel file."""
        # Exact mapping from Excel file
        style_vibe_map = {
            'Minimalist': 'Minimalist - Refined silhouettes and intentional pieces that speak to a quiet, less-is-more style approach.',
            'Bold': 'Bold - Strong shapes or standout elements that add confident energy to your everyday fit.',
            'Relaxed': 'Relaxed - Easygoing structure and breathable layering that mirror a low-effort, high-comfort vibe.',
            'Edgy': 'Edgy - Modern cuts and subtle attitude that bring a sharper edge to your usual routine.',
            'Clean': 'Clean - Balanced proportions and precise details that keep your fit polished without trying too hard.',
            'Streetwear': 'Streetwear - Casual layers with cultural influence — styled for expression, movement, and comfort.',
            'Sporty': 'Sporty - Athletic-inspired lines and laid-back structure that match your on-the-move rhythm.',
            'Smart-Casual': 'Smart-Casual - Sleek elements with a relaxed twist — bridging polished and practical for daily wear.',
            'business casual': 'Smart-Casual - Sleek elements with a relaxed twist — bridging polished and practical for daily wear.',
            'Tailored': 'Tailored - Structured lines and fit-aware pieces that elevate your frame while staying wearable.',
            'Distressed': 'Distressed - Worn textures and raw finishes that bring a lived-in feel with a bold personality.',
            'Fitted': 'Fitted - Close-to-body shapes that offer structure without stiffness, tuned to highlight your form.',
            'Oversized': 'Oversized - Intentionally loose proportions that add comfort, attitude, and space to move.',
            'Graphic': 'Graphic - Bold visuals and print-led design that make your outfit speak before you do.',
            'Vintage': 'Vintage - Retro-inspired detailing and time-washed tones that bring character to the present.',
            'Washed': 'Washed - Softened colors and textures with a relaxed feel — styled for ease and repeat wear.',
            'Neutral': 'Neutral - Earth-based tones and quiet details that blend seamlessly into a versatile wardrobe.',
            'Soft': 'Soft - Gentle fabrics and calming tones that bring softness into form, feel, and finish.',
            'Casual': 'Casual - Low-pressure styling made for real routines — flexible, clean, and always wearable.',
            'athleisure': 'Sporty - Athletic-inspired lines and laid-back structure that match your on-the-move rhythm.',
            'Slim Fit': 'Slim Fit - Streamlined cuts with subtle tapering that keep the outfit neat, not rigid.',
            'Urban': 'Urban - Modern layers and pace-driven fits that reflect your city-first, culture-aware lifestyle.'
        }
        
        # Try to match user style to available keywords
        user_style_lower = user_style.lower()
        for keyword, explanation in style_vibe_map.items():
            if keyword.lower() in user_style_lower or user_style_lower in keyword.lower():
                return explanation
        
        # Default fallback
        return "Urban - Modern layers and pace-driven fits that reflect your city-first, culture-aware lifestyle."
    
    def _get_occasions_from_excel_exact(self, user_style: str) -> str:
        """Get occasions using exact keywords and explanations from Excel file."""
        # Exact mapping from Excel file
        occasion_map = {
            'Wedding': 'Wedding - Elegant picks with sharp styling to match the energy of the moment.',
            'Party': 'Party - Built for movement, compliments, and that main-character vibe.',
            'Office': 'Office - Clean and structured fits that keep things polished but never stiff.',
            'Work': 'Work - Everyday professional pieces with subtle personality and comfort.',
            'business casual': 'Office - Clean and structured fits that keep things polished but never stiff.',
            'Formal': 'Formal - Structured, elevated pieces meant for your dress-up days.',
            'Casual': 'Casual - Easy layers and breathable fabrics for your chill, everyday flow.',
            'Travel': 'Travel - Light, layered, and functional — perfect for moving through your day in style.',
            'Vacation': 'Vacation - Relaxed fits and warm tones that match your out-of-office energy.',
            'Dinner': 'Dinner - Refined looks with a little edge — polished enough to impress, easy enough to enjoy.',
            'Brunch': 'Brunch - Clean-casual outfits that land between laid-back and styled-up.',
            'Date': 'Date - Effort-meets-attitude looks that do the talking without trying too hard.',
            'Event': 'Event - Statement-ready fits that balance comfort and camera moments.',
            'Festival': 'Festival - Vibrant, expressive pieces built for movement, music, and heat.',
            'Street': 'Street - Urban fits with everyday edge — confident, casual, and culturally aware.',
            'Weekend': 'Weekend - Relaxed and put-together combos for those no-pressure plans.',
            'Daily': 'Daily - Go-to staples with smart styling — for repeat use without repeat looks.',
            'Beach': 'Beach - Airy fabrics and sun-ready tones made for breezy styling.',
            'Outing': 'Outing - Versatile picks for wherever the day takes you.',
            'College': 'College - Low-effort, on-trend fits that work from lecture hall to late night.',
            'Night': 'Night - Mood-lit colors and tailored silhouettes for after-dark sharpness.',
            'Gym': 'Gym - Movement-first pieces that flex with your routine and reflect your style.',
            'athleisure': 'Gym - Movement-first pieces that flex with your routine and reflect your style.'
        }
        
        # Try to match user style to available keywords
        user_style_lower = user_style.lower()
        for keyword, explanation in occasion_map.items():
            if keyword.lower() in user_style_lower or user_style_lower in keyword.lower():
                return explanation
        
        # Default fallback
        return "Daily - Go-to staples with smart styling — for repeat use without repeat looks."
    
    def _get_skin_tone_from_excel_exact(self, top_color: str, bottom_color: str) -> str:
        """Get skin tone using exact keywords and explanations from Excel file."""
        # Exact mapping from Excel file
        skin_tone_map = {
            'Spring (Warm Undertones)': 'Spring (Warm Undertones) - Soft warmth and golden hues bring out your natural glow and keep things light, fresh, and sunlit.',
            'Summer (Cool Undertones)': 'Summer (Cool Undertones) - Cool, breezy tones and muted clarity reflect your softness — balanced, calm, and effortlessly elegant.',
            'Autumn (Warm / Neutral-Warm Undertones)': 'Autumn (Warm / Neutral-Warm Undertones) - Earth-rich tones and deep warmth align with your grounded, textured presence and glowing undertone.',
            'Winter (Cool / Neutral-Cool Undertones)': 'Winter (Cool / Neutral-Cool Undertones) - Bold contrasts and crisp saturation highlight your sharp clarity and bring out your inner edge.'
        }
        
        # Determine skin tone based on colors (simplified logic)
        warm_colors = ['red', 'orange', 'yellow', 'brown', 'olive', 'coral', 'peach', 'gold']
        cool_colors = ['blue', 'purple', 'pink', 'gray', 'navy', 'teal', 'mint', 'lavender']
        
        top_color_lower = top_color.lower()
        bottom_color_lower = bottom_color.lower()
        
        warm_count = sum(1 for color in warm_colors if color in top_color_lower or color in bottom_color_lower)
        cool_count = sum(1 for color in cool_colors if color in top_color_lower or color in bottom_color_lower)
        
        if warm_count > cool_count:
            return skin_tone_map['Spring (Warm Undertones)']
        elif cool_count > warm_count:
            return skin_tone_map['Summer (Cool Undertones)']
        else:
            return skin_tone_map['Autumn (Warm / Neutral-Warm Undertones)']
    
    def _get_body_shape_from_excel_exact(self, user_body_shape: str, user_gender: str) -> str:
        """Get body shape using exact keywords and explanations from Excel file."""
        # Exact mapping from Excel file
        body_shape_map = {
            'Inverted Triangle': 'Inverted Triangle - Balanced layers and structured bottoms soften broad shoulders and bring symmetry.',
            'Rectangle': 'Rectangle - Visual shape is added through layering, texture, or waist-defining silhouettes.',
            'Oval': 'Oval - Streamlined fits and vertical details elongate and frame your midsection confidently.',
            'Hourglass': 'Hourglass - Balanced fits that follow your natural proportions without overcomplication.',
            'Triangle': 'Triangle - Elevated tops and structured shoulders bring harmony to a fuller lower frame.'
        }
        
        # Map user body shape to available keywords
        user_body_shape_lower = user_body_shape.lower()
        for keyword, explanation in body_shape_map.items():
            if keyword.lower() in user_body_shape_lower or user_body_shape_lower in keyword.lower():
                return explanation
        
        # Default fallback
        return "Rectangle - Visual shape is added through layering, texture, or waist-defining silhouettes."
    
    def _calculate_product_score(self, product: pd.Series, user_data: Dict) -> float:
        """Calculate product score based on user preferences."""
        try:
            # Base score
            score = 0.5
            
            # Style match bonus
            user_styles = []
            if user_data.get('Apparel Pref Business Casual', False):
                user_styles.append('business casual')
            if user_data.get('Apparel Pref Streetwear', False):
                user_styles.append('streetwear')
            if user_data.get('Apparel Pref Athleisure', False):
                user_styles.append('athleisure')
            
            product_style = product.get('primary_style', '').lower()
            if any(style in product_style for style in user_styles):
                score += 0.3
            
            # Color compatibility bonus
            if product.get('color'):
                score += 0.1
            
            # Price range bonus
            price = product.get('price', 0)
            if 500 <= price <= 2000:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating product score: {e}")
            return 0.5
    
    def _generate_outfit_explanation(self, top: pd.Series, bottom: pd.Series, 
                                   user_data: Dict, style: str) -> str:
        """Generate outfit explanation."""
        try:
            top_title = top.get('title', '')[:50]
            bottom_title = bottom.get('title', '')[:50]
            
            return f"{style.title()} recommendation | {top_title} | Matches your {style} style preference"
            
        except Exception as e:
            logger.error(f"❌ Error generating outfit explanation: {e}")
            return "Stylish outfit combination"

    def save_outfits_to_supabase(self, user_id: int, outfits_data: List[Dict]) -> bool:
        """Save generated outfits to Supabase database with robust collision handling."""
        try:
            if not outfits_data:
                logger.warning("No outfits data to save")
                return False

            logger.info(f"💾 Saving {len(outfits_data)} outfits to Supabase for user {user_id}")

            # ✅ FIX: Use simple sequential IDs to avoid conflicts with existing outfits
            
            # 🎯 FIRST: Prepare outfits with simple sequential IDs
            processed_outfits = []
            for i, outfit in enumerate(outfits_data):
                # Use simple sequential ID format: main_userid_rank
                unique_id = f"main_{user_id}_{i+1}"
                
                outfit_copy = outfit.copy()
                outfit_copy['main_outfit_id'] = unique_id
                # Ensure rank starts from 1 and is sequential
                outfit_copy['rank'] = i + 1
                # Add user_id to each outfit
                outfit_copy['user_id'] = user_id
                processed_outfits.append(outfit_copy)

            # 🚀 SECOND: Use simple insert since we have unique IDs
            batch_size = 10  # Insert in smaller batches to avoid timeouts
            total_processed = 0
            
            for i in range(0, len(processed_outfits), batch_size):
                batch = processed_outfits[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    logger.info(f"📦 Inserting batch {batch_num}: {len(batch)} outfits")
                    
                    # Use simple insert since we have unique IDs
                    result = self.db.client.table('user_outfits').insert(batch).execute()
                    
                    if result.data:
                        total_processed += len(batch)
                        logger.info(f"✅ Batch {batch_num} inserted successfully")
                    else:
                        logger.error(f"❌ Batch {batch_num} failed to insert")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Error inserting batch {batch_num}: {e}")
                    return False

            logger.info(f"✅ Successfully saved {total_processed}/{len(processed_outfits)} outfits to Supabase with unique IDs")
            return total_processed == len(processed_outfits)

        except Exception as e:
            logger.error(f"❌ Error saving outfits to Supabase: {e}")
            return False

    def _verify_unique_outfit_id(self, outfit_id: str) -> bool:
        """Verify that an outfit ID doesn't already exist in the database."""
        try:
            existing = self.db.client.table('user_outfits').select('main_outfit_id').eq('main_outfit_id', outfit_id).execute()
            return len(existing.data) == 0
        except Exception:
            return False

    def _clear_user_outfits_completely(self, user_id: int) -> bool:
        """Completely clear all outfits for a user with multiple strategies."""
        try:
            
            strategies_attempted = 0
            total_deleted = 0
            
            # Strategy 1: Delete by user_id
            try:
                result1 = self.db.client.table('user_outfits').delete().eq('user_id', user_id).execute()
                deleted1 = len(result1.data) if result1.data else 0
                total_deleted += deleted1
                strategies_attempted += 1
                logger.info(f"Strategy 1: Deleted {deleted1} outfits by user_id")
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Strategy 1 failed: {e}")
            
            # Strategy 2: Delete by main_outfit_id pattern
            try:
                result2 = self.db.client.table('user_outfits').delete().like('main_outfit_id', f'main_{user_id}_%').execute()
                deleted2 = len(result2.data) if result2.data else 0
                total_deleted += deleted2
                strategies_attempted += 1
                logger.info(f"Strategy 2: Deleted {deleted2} outfits by ID pattern")
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Strategy 2 failed: {e}")
            
            # Strategy 3: Check for any remaining and delete individually
            try:
                remaining = self.db.client.table('user_outfits').select('main_outfit_id').eq('user_id', user_id).execute()
                if remaining.data:
                    for item in remaining.data:
                        try:
                            self.db.client.table('user_outfits').delete().eq('main_outfit_id', item['main_outfit_id']).execute()
                            total_deleted += 1
                        except:
                            continue
                    strategies_attempted += 1
                    logger.info(f"Strategy 3: Individually deleted {len(remaining.data)} remaining outfits")
            except Exception as e:
                logger.warning(f"Strategy 3 failed: {e}")
            
            # Final verification
            time.sleep(0.5)
            final_check = self.db.client.table('user_outfits').select('id').eq('user_id', user_id).execute()
            remaining_count = len(final_check.data) if final_check.data else 0
            
            if remaining_count == 0:
                logger.info(f"✅ Successfully cleared all outfits for user {user_id} (total deleted: {total_deleted})")
                return True
            else:
                logger.warning(f"⚠️ {remaining_count} outfits still remain for user {user_id} after {strategies_attempted} strategies")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in complete cleanup for user {user_id}: {e}")
            return False

    def generate_and_save_outfits(self, user_id: int) -> bool:
        """✅ ENHANCED: Generate and save enhanced outfits for a user."""
        try:
            logger.info(f"🎯 Starting enhanced outfit generation and save for user {user_id}")
            
            # Generate outfits
            outfits_data = self.generate_main_outfits_for_user(user_id)
            
            if not outfits_data:
                logger.warning(f"No outfits generated for user {user_id}")
                return False
            
            # Save to Supabase
            success = self.save_outfits_to_supabase(user_id, outfits_data)
            
            if success:
                logger.info(f"✅ Successfully completed enhanced outfit generation for user {user_id}")
                return True
            else:
                logger.error(f"❌ Failed to save outfits for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced generate and save process: {e}")
            return False

    def validate_user_data(self, user_data: Dict) -> bool:
        """Enhanced validation of user data with vocabulary normalization."""
        required_fields = [
            'User',
            'Gender',
            'Upper Wear Caption',
            'Lower Wear Caption']
        missing_fields = [
            field for field in required_fields if field not in user_data]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in user data: {missing_fields}")

        # Track data quality metrics
        quality_metrics = {
            "has_body_shape": 'Body Shape' in user_data,
            "has_style_preference": 'Style Preference' in user_data,
            "has_occasion": 'Occasion' in user_data,
            "has_budget": 'Budget Preference' in user_data
        }

        logger.info("User Data Quality Metrics:")
        for metric, value in quality_metrics.items():
            logger.info(f"{metric}: {value}")

        # Normalize user data using vocabulary system
        if 'Gender' in user_data:
            user_data['Gender'] = str(user_data['Gender']).title()

        if 'Body Shape' in user_data:
            user_data['Body Shape'] = self.normalize_vocabulary(
                user_data['Body Shape'], 'body_shape_mapping')

        if 'Style Preference' in user_data:
            user_data['Style Preference'] = self.normalize_vocabulary(
                user_data['Style Preference'], 'style_mapping')

        if 'Occasion' in user_data:
            user_data['Occasion'] = self.normalize_vocabulary(
                user_data['Occasion'], 'occasion_mapping')

        # Validate and normalize captions
        if 'Upper Wear Caption' in user_data:
            user_data['Upper Wear Caption'] = str(
                user_data['Upper Wear Caption']).strip()
            if not user_data['Upper Wear Caption']:
                raise ValueError("Upper Wear Caption cannot be empty")

        if 'Lower Wear Caption' in user_data:
            user_data['Lower Wear Caption'] = str(
                user_data['Lower Wear Caption']).strip()
            if not user_data['Lower Wear Caption']:
                raise ValueError("Lower Wear Caption cannot be empty")

        # Validate gender
        valid_genders = ['Male', 'Female', 'Unisex']
        if user_data['Gender'] not in valid_genders:
            raise ValueError(
                f"Invalid gender: {user_data['Gender']}. Must be one of {valid_genders}")

        # Add confidence scores for user preferences
        confidence_scores = {
            'body_shape': 1.0 if 'Body Shape' in user_data else 0.7,
            'style_preference': 1.0 if 'Style Preference' in user_data else 0.7,
            'occasion': 1.0 if 'Occasion' in user_data else 0.7,
            'budget': 1.0 if 'Budget Preference' in user_data else 0.7}

        user_data['confidence_scores'] = confidence_scores
        logger.info(f"User data validation complete. Average confidence: {sum(confidence_scores.values()) / len(confidence_scores):.2f}")

        return True

    def _initialize_vocabulary_mappings(self) -> Dict:
        """Initialize comprehensive vocabulary mappings for consistent matching."""
        return {
            'style_mapping': {
                # Formal styles
                'formal': ['Formal', 'Business Formal', 'Evening Formal', 'Ultra Formal', 'Professional', 'Corporate', 'Business'],
                'business': ['Business', 'Business Formal', 'Business Casual', 'Professional', 'Corporate', 'Office', 'Work'],
                'casual': ['Casual', 'Smart Casual', 'Business Casual', 'Relaxed', 'Everyday', 'Informal', 'Comfortable'],
                'streetwear': ['Streetwear', 'Street', 'Urban', 'Contemporary', 'Modern', 'Hip-hop', 'Urban Style'],
                'athleisure': ['Athleisure', 'Activewear', 'Sport', 'Athletic', 'Performance', 'Gym', 'Workout'],
                'contemporary': ['Contemporary', 'Modern', 'Current', 'Trendy', 'Fashion-forward', 'Latest', 'New'],
                'classic': ['Classic', 'Traditional', 'Timeless', 'Heritage', 'Conventional', 'Traditional', 'Elegant'],
                'vintage': ['Vintage', 'Retro', 'Classic', 'Heritage', 'Nostalgic', 'Old-school', 'Throwback'],
                'bohemian': ['Bohemian', 'Boho', 'Free-spirited', 'Artistic', 'Eclectic', 'Hippie', 'Free'],
                'minimalist': ['Minimalist', 'Minimal', 'Simple', 'Clean', 'Essential', 'Basic', 'Plain']
            },
            'color_mapping': {
                'black': ['Black', 'Jet Black', 'Onyx', 'Ebony', 'Charcoal', 'Dark', 'Deep Black'],
                'white': ['White', 'Ivory', 'Cream', 'Off-white', 'Pearl', 'Snow', 'Pure White'],
                'blue': ['Blue', 'Navy', 'Royal Blue', 'Sky Blue', 'Teal', 'Azure', 'Cobalt', 'Sapphire'],
                'red': ['Red', 'Crimson', 'Burgundy', 'Maroon', 'Ruby', 'Scarlet', 'Cherry', 'Wine'],
                'green': ['Green', 'Olive', 'Emerald', 'Forest', 'Sage', 'Mint', 'Jade', 'Lime'],
                'yellow': ['Yellow', 'Gold', 'Mustard', 'Amber', 'Honey', 'Sunshine', 'Bright Yellow'],
                'purple': ['Purple', 'Violet', 'Lavender', 'Plum', 'Mauve', 'Lilac', 'Amethyst'],
                'pink': ['Pink', 'Rose', 'Coral', 'Salmon', 'Blush', 'Fuchsia', 'Magenta'],
                'orange': ['Orange', 'Peach', 'Coral', 'Rust', 'Terracotta', 'Apricot', 'Tangerine'],
                'brown': ['Brown', 'Tan', 'Beige', 'Khaki', 'Camel', 'Chocolate', 'Bronze', 'Coffee'],
                'gray': ['Gray', 'Grey', 'Silver', 'Charcoal', 'Slate', 'Ash', 'Smoke']
            },
            'body_shape_mapping': {
                'hourglass': ['Hourglass', 'Curvy', 'Proportional', 'Balanced', 'Symmetrical'],
                'rectangle': ['Rectangle', 'Straight', 'Athletic', 'Column', 'Linear', 'Boxy'],
                'pear': ['Pear', 'Triangle', 'A-shape', 'Bottom-heavy', 'Curvy Bottom'],
                'apple': ['Apple', 'Oval', 'Round', 'Top-heavy', 'Full Figure'],
                'inverted triangle': ['Inverted Triangle', 'V-shape', 'Top-heavy', 'Athletic', 'Broad Shoulders']
            },
            'fit_mapping': {
                'regular': ['Regular Fit', 'Standard Fit', 'Classic Fit', 'Traditional Fit', 'Normal Fit'],
                'slim': ['Slim Fit', 'Fitted', 'Tailored', 'Close Fit', 'Trim Fit', 'Slim-cut'],
                'loose': ['Loose Fit', 'Relaxed', 'Comfortable', 'Easy Fit', 'Roomy', 'Baggy'],
                'oversized': ['Oversized', 'Oversize', 'Loose', 'Baggy', 'Extra Large', 'XL'],
                'fitted': ['Fitted', 'Bodycon', 'Form-fitting', 'Close-fitting', 'Tight', 'Snug']
            },
            'occasion_mapping': {
                'formal': ['Formal', 'Business', 'Professional', 'Corporate', 'Elegant', 'Sophisticated'],
                'casual': ['Casual', 'Everyday', 'Relaxed', 'Informal', 'Comfortable', 'Leisure'],
                'sport': ['Sport', 'Athletic', 'Active', 'Performance', 'Gym', 'Workout', 'Exercise'],
                'evening': ['Evening', 'Night', 'Party', 'Special Occasion', 'Dinner', 'Event'],
                'work': ['Work', 'Office', 'Professional', 'Business', 'Corporate', 'Career']
            }
        }

    def normalize_vocabulary(self, value: str, category: str) -> str:
        """Normalize vocabulary terms to ensure consistent matching with fuzzy matching support."""
        if not value or pd.isna(value):
            return self._get_default_value(category)

        value = str(value).lower().strip()
        mappings = self._initialize_vocabulary_mappings()

        if category not in mappings:
            return value.title()

        # First try exact matching
        for standard_term, variations in mappings[category].items():
            if any(var.lower() in value for var in variations):
                return standard_term.title()

        # If no exact match, try fuzzy matching
        from difflib import SequenceMatcher

        best_match = None
        best_ratio = 0.6  # Minimum similarity threshold

        for standard_term, variations in mappings[category].items():
            for var in variations:
                ratio = SequenceMatcher(None, value, var.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = standard_term

        if best_match:
            return best_match.title()

        # If no match found, return the original value with proper casing
        return value.title()

    def _get_default_value(self, category: str) -> str:
        """Get default value for each category with enhanced fallback logic."""
        defaults = {
            'style': 'Casual',
            'color': 'Black',
            'body_shape': 'Regular',
            'fit': 'Regular',
            'occasion': 'Casual'
        }
        return defaults.get(category, 'Unknown')

    def _calculate_vocabulary_confidence(
            self, value: str, category: str) -> float:
        """Calculate confidence score for vocabulary matching."""
        if not value or pd.isna(value):
            return 0.0

        value = str(value).lower().strip()
        mappings = self._initialize_vocabulary_mappings()

        if category not in mappings:
            return 0.5  # Medium confidence for unknown categories

        # Check for exact matches
        for standard_term, variations in mappings[category].items():
            if any(var.lower() == value for var in variations):
                return 1.0  # High confidence for exact matches
            if any(var.lower() in value for var in variations):
                return 0.9  # High confidence for partial matches

        # Check for fuzzy matches
        from difflib import SequenceMatcher
        best_ratio = 0.0

        for standard_term, variations in mappings[category].items():
            for var in variations:
                ratio = SequenceMatcher(None, value, var.lower()).ratio()
                best_ratio = max(best_ratio, ratio)

        if best_ratio > 0.8:
            return 0.8  # High confidence for very close fuzzy matches
        elif best_ratio > 0.6:
            return 0.6  # Medium confidence for fuzzy matches

        return 0.3  # Low confidence for poor matches

    def map_user_body_shape_to_designer(
            self,
            user_body_shape: str,
            user_gender: str) -> str:
        """
        Map user body shape terms to designer rule terms, based on gender.
        - For women: Triangle → Pear, Oval → Apple
        - For men: Inverted Triangle → Pear, Oval → Apple
        - All others: direct mapping (lowercase)
        """
        if not user_body_shape or not user_gender or user_body_shape is None or user_gender is None:
            return ""
        
        shape = str(user_body_shape).strip().lower()
        gender = str(user_gender).strip().lower()
        if gender in ['female', 'women']:
            if shape == 'triangle':
                return 'pear'
            elif shape == 'oval':
                return 'apple'
        elif gender in ['male', 'men']:
            if shape == 'inverted triangle':
                return 'pear'
            elif shape == 'oval':
                return 'apple'
        return shape

    def get_explanation_for_outfit(
            self,
            explanations: list,
            color_score: float) -> str:
        """
        Combine all rule-based explanations into a single, user-friendly string for API/UI.
        Highlights positive matches and any penalties.
        """
        color_quality = "Perfect" if color_score > 0.9 else "Good" if color_score > 0.7 else "Fair"
        main_explanation = []
        for exp in explanations:
            if exp and isinstance(exp, str):
                main_explanation.append(exp)
        # Remove duplicates and empty
        main_explanation = [e for i, e in enumerate(
            main_explanation) if e and e not in main_explanation[:i]]
        # Join with separators
        return f"Color: {color_quality} | " + " | ".join(main_explanation)

    def _check_basic_outfit_compatibility(
            self, top: pd.Series, bottom: pd.Series) -> Tuple[bool, str]:
        """
        Check basic outfit compatibility based on fashion designer rules.
        Returns (is_compatible, reason)
        """
        # Get categories and colors with safe null handling
        top_category = str(top.get('category', '') or '').lower()
        bottom_category = str(bottom.get('category', '') or '').lower()
        top_color = str(top.get('primary_color', '') or '').lower()
        bottom_color = str(bottom.get('primary_color', '') or '').lower()

        # Enhanced color combinations from fashion designer
        safe_color_combinations = {
            'black': [
                'light pink',
                'medium pink',
                'deep red',
                'maroon',
                'mint green',
                'sage green',
                'emerald green',
                'olive green',
                'sky blue',
                'cobalt blue',
                'navy blue',
                'baby blue',
                'charcoal gray',
                'light brown',
                'dark brown',
                'burgundy',
                'mustard yellow',
                'camel',
                'tan'],
            'white': [
                'light pink',
                'medium pink',
                'hot pink',
                'bright red',
                'maroon',
                'lemon yellow',
                'bright yellow',
                'mint green',
                'sage green',
                'emerald green',
                'sky blue',
                'cobalt blue',
                'navy blue',
                'baby blue',
                'light gray',
                'dark brown',
                'burgundy',
                'mustard yellow',
                'camel',
                'tan',
                'black',
                'navy'],
            'navy blue': [
                'light pink',
                'burgundy',
                'lemon yellow',
                'mustard yellow',
                'olive green',
                'emerald green',
                'white',
                'gray',
                'black',
                'camel',
                'tan',
                'light blue',
                'sky blue',
                'cobalt blue'],
        }

        # Basic incompatible combinations (only the most obvious ones)
        incompatible_combinations = {
            'formal shirt': ['denim shorts', 'cargo shorts', 'athletic shorts'],
            't-shirt': ['formal trousers', 'formal pants'],
            'sweatshirt': ['formal trousers', 'formal pants'],
            'hoodie': ['formal trousers', 'formal pants']
        }

        # Check for incompatible combinations
        for top_type, incompatible_bottoms in incompatible_combinations.items():
            if top_type in top_category and any(
                    bottom in bottom_category for bottom in incompatible_bottoms):
                return False, f"Incompatible combination: {top_type} with {bottom_category}"

        # Check color harmony
        def check_color_harmony(color1, color2):
            # Check if colors are in safe combinations
            for base_color, compatible_colors in safe_color_combinations.items():
                if color1 == base_color and color2 in compatible_colors:
                    return True
                if color2 == base_color and color1 in compatible_colors:
                    return True

            # Check for monochrome/tonal dressing
            if color1 in color2 or color2 in color1:
                return True

            # Check for neutral combinations
            neutral_colors = [
                'black',
                'white',
                'gray',
                'navy',
                'beige',
                'camel',
                'tan']
            if color1 in neutral_colors or color2 in neutral_colors:
                return True

            return False

        # Check color harmony
        if not check_color_harmony(top_color, bottom_color):
            return False, f"Color combination {top_color} + {bottom_color} may not be harmonious"

        return True, "Outfit is compatible"

    def _calculate_diversity_score(self, product: pd.Series, seen_styles: set,
                                   seen_colors: set, wear_type: str) -> float:
        """Calculate diversity score for a product based on seen styles and colors."""
        style = str(product.get(
            'enhanced_primary_style', product.get(
                'primary_style', '')) or '')
        color = str(product.get('primary_color', '') or '')

        # Base score
        score = 1.0

        # Style diversity penalty
        if style in seen_styles:
            score *= 0.8

        # Color diversity penalty
        if color in seen_colors:
            score *= 0.8

        # Quality boost
        quality = product.get('quality_indicator1', '')
        if quality and 'High' in quality:
            score *= 1.1

        # Versatility boost
        versatility = product.get('versatility_analysis', '')
        if versatility and 'High' in versatility:
            score *= 1.1

        # Fit confidence boost
        fit_confidence = product.get('fit_confidence_score', 0)
        if fit_confidence and fit_confidence > 0.8:
            score *= 1.1

        return min(max(score, 0.5), 1.5)  # Keep score between 0.5 and 1.5

    def diversify_outfit_recommendations(
            self,
            recommendations: list,
            top_n: int = 20) -> list:
        """Diversify outfit recommendations to ensure variety in styles and colors."""
        if not recommendations:
            return []

        # Track seen combinations
        seen_styles = set()
        seen_colors = set()
        seen_outfits = set()
        diverse_recs = []

        # Get user's fashion style preferences
        user_styles = set()
        for rec in recommendations:
            if 'user' in rec and 'Fashion Style' in rec['user']:
                user_styles = set(rec['user']['Fashion Style'].split(','))
                break

        # Initialize style distribution tracking with minimum targets
        style_distribution = {style: 0 for style in user_styles}
        min_style_target = max(3, top_n // (len(user_styles) * 2)) if user_styles else 5
        target_distribution = {style: min_style_target for style in user_styles} if user_styles else {}

        # Sort by score first
        sorted_recs = sorted(
            recommendations,
            key=lambda x: x['score'],
            reverse=True)

        # First pass: Ensure top 10 has minimum distribution
        top_10_styles = {style: 0 for style in user_styles} if user_styles else {}
        top_10_recs = []

        for rec in sorted_recs:
            if len(top_10_recs) >= 10:
                break

            # Get top and bottom products using correct keys
            top = {
                'style': rec['top_style'],
                'color': rec['top_color'],
                'title': rec['top_title']
            }
            bottom = {
                'style': rec['bottom_style'],
                'color': rec['bottom_color'],
                'title': rec['bottom_title']
            }

            # Create outfit hash
            outfit_hash = self._create_outfit_hash(top, bottom)

            if outfit_hash in seen_outfits:
                continue

            # Check if this style combination matches user preferences
            outfit_styles = {top['style'], bottom['style']}
            matching_styles = outfit_styles & user_styles if user_styles else outfit_styles

            if not matching_styles and user_styles:
                continue

            # Add to top 10 recommendations
            top_10_recs.append(rec)
            seen_outfits.add(outfit_hash)
            
            # Update top 10 style distribution
            for style in matching_styles:
                if style in top_10_styles:
                    top_10_styles[style] += 1

        # Second pass: Collect remaining outfits while maintaining overall distribution
        for rec in sorted_recs:
            if len(diverse_recs) >= top_n:
                break

            # Get top and bottom products using correct keys
            top = {
                'style': rec['top_style'],
                'color': rec['top_color'],
                'title': rec['top_title'],
                'product_id': rec.get('top_id', '')
            }
            bottom = {
                'style': rec['bottom_style'],
                'color': rec['bottom_color'],
                'title': rec['bottom_title'],
                'product_id': rec.get('bottom_id', '')
            }

            # Create outfit hash
            outfit_hash = self._create_outfit_hash(top, bottom)

            if outfit_hash in seen_outfits:
                continue

            # Check if this style combination matches user preferences
            outfit_styles = {top['style'], bottom['style']}
            matching_styles = outfit_styles & user_styles if user_styles else outfit_styles

            if not matching_styles and user_styles:
                continue

            # Add to diverse recommendations
            diverse_recs.append(rec)

            # Update seen sets and style distribution
            seen_styles.add(top['style'])
            seen_styles.add(bottom['style'])
            seen_colors.add(top['color'])
            seen_colors.add(bottom['color'])
            seen_outfits.add(outfit_hash)
            
            # Update style distribution
            for style in matching_styles:
                if style in style_distribution:
                    style_distribution[style] += 1

        # Combine top 10 with remaining recommendations
        return top_10_recs + diverse_recs

    def _calculate_attribute_similarity(
            self,
            attr1: str,
            attr2: str,
            seen_attrs: set) -> float:
        """Calculate similarity score for an attribute pair."""
        if not attr1 or not attr2:
            return 0.0

        # Direct match
        if attr1 == attr2:
            return 1.0

        # Seen before penalty
        seen_penalty = 0.0
        if attr1 in seen_attrs:
            seen_penalty += 0.5
        if attr2 in seen_attrs:
            seen_penalty += 0.5

        # Fuzzy match for similar attributes
        attr1 = str(attr1).lower()
        attr2 = str(attr2).lower()

        if any(word in attr1 for word in attr2.split()) or any(
                word in attr2 for word in attr1.split()):
            return 0.7 - seen_penalty

        return 0.0

    def _create_outfit_hash(self, top: Dict, bottom: Dict) -> tuple:
        """Create a hash tuple for an outfit based on product IDs to detect duplicates."""
        # Primary key: product IDs (most important for duplicate detection)
        top_id = str(top.get('product_id', top.get('id', '')))
        bottom_id = str(bottom.get('product_id', bottom.get('id', '')))
        
        # Secondary attributes for additional uniqueness
        top_style = str(top.get('style', '')).lower()
        bottom_style = str(bottom.get('style', '')).lower()
        top_color = str(top.get('color', '')).lower()
        bottom_color = str(bottom.get('color', '')).lower()
        
        return (
            top_id,
            bottom_id,
            top_style,
            bottom_style,
            top_color,
            bottom_color
        )

    def calculate_color_harmony_score(self, color1: str, color2: str) -> float:
        """ENHANCED: Calculate color harmony score using designer's CSV ratings if available."""
        if not color1 or not color2 or pd.isna(color1) or pd.isna(color2):
            return 0.5
        color1 = str(color1).strip().title()
        color2 = str(color2).strip().title()
        # Same color - perfect match
        if color1 == color2:
            return 1.0
        # Try designer CSV mapping
        if hasattr(self, 'color_harmony_map') and self.color_harmony_map:
            pair1 = (color1, color2)
            pair2 = (color2, color1)
            rating = None
            if pair1 in self.color_harmony_map and self.color_harmony_map[
                    pair1]['rating'] is not None:
                rating = self.color_harmony_map[pair1]['rating']
            elif pair2 in self.color_harmony_map and self.color_harmony_map[pair2]['rating'] is not None:
                rating = self.color_harmony_map[pair2]['rating']
            if rating is not None:
                return rating / 10.0
            else:
                logger.debug(
                    f"No designer rating for color pair: {color1} + {color2}")
        # Fallback to professional harmony rules

        def get_professional_harmony_score(c1, c2):
            if c1 not in self.color_harmony:
                return 0.5
            harmony_rules = self.color_harmony[c1]
            for combo in harmony_rules.get('perfect', []):
                if c2 == combo['color']:
                    return combo['rating'] / 10.0
            for combo in harmony_rules.get('excellent', []):
                if c2 == combo['color']:
                    return combo['rating'] / 10.0
            for combo in harmony_rules.get('good', []):
                if c2 == combo['color']:
                    return combo['rating'] / 10.0
            for combo in harmony_rules.get('avoid', []):
                if c2 == combo['color']:
                    return combo['rating'] / 10.0
            return 0.5  # Default neutral score
        score1 = get_professional_harmony_score(color1, color2)
        score2 = get_professional_harmony_score(color2, color1)
        return max(score1, score2)

    def generate_outfit_name_and_description(self, outfit_data: Dict, user_data: Dict = None) -> Tuple[str, str]:
        """
        Generate outfit name and description using fashion intelligence.
        
        Args:
            outfit_data: Dictionary containing top and bottom product data
            user_data: User data for personalization
            
        Returns:
            Tuple[str, str]: (outfit_name, outfit_description)
        """
        try:
            # Extract outfit components
            top_data = outfit_data.get('top', {})
            bottom_data = outfit_data.get('bottom', {})
            
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
            
            # Analyze outfit components to determine mood
            mood = self._determine_outfit_mood(top_data, bottom_data, user_data, MOOD_WORDS)
            
            # Determine accent (color/fabric/print)
            accent = self._determine_outfit_accent(top_data, bottom_data, ACCENT_WORDS)
            
            # Select noun anchor
            noun = random.choice(NOUN_WORDS)
            
            # Determine occasion/context
            occasion = self._determine_outfit_occasion(top_data, bottom_data, user_data, OCCASION_WORDS)
            
            # Build outfit name (max 3 words)
            outfit_name = self._build_outfit_name(mood, accent, noun, occasion)
            
            # Generate short description
            outfit_description = self._generate_outfit_description(outfit_data, outfit_name, user_data)
            
            return outfit_name, outfit_description
            
        except Exception as e:
            logger.error(f"❌ Error generating outfit name/description: {e}")
            return "Classic Edit", "A timeless combination perfect for any occasion."
    
    def _determine_outfit_mood(self, top_data: Dict, bottom_data: Dict, user_data: Dict, mood_words: List[str]) -> str:
        """Determine the overall mood/vibe of the outfit."""
        try:
            # Extract style information
            top_style = top_data.get('primary_style', '').lower()
            bottom_style = bottom_data.get('primary_style', '').lower()
            top_title = top_data.get('title', '').lower()
            bottom_title = bottom_data.get('title', '').lower()
            
            # Style-based mood mapping
            style_mood_map = {
                'streetwear': ['Urban', 'Streetline', 'Edgy', 'Bold'],
                'business casual': ['Sophisticated', 'Classic', 'Heritage', 'Minimal'],
                'athleisure': ['Athleisure', 'Dynamic', 'Modern', 'Casual'],
                'casual': ['Casual', 'Coastal', 'Relaxed', 'Minimal'],
                'formal': ['Luxe', 'Elegant', 'Sophisticated', 'Classic'],
                'vintage': ['Vintage', 'Retro', 'Heritage', 'Classic'],
                'bohemian': ['Boho', 'Artsy', 'Coastal', 'Relaxed'],
                'minimalist': ['Minimal', 'Subtle', 'Clean', 'Modern']
            }
            
            # Determine mood from styles
            possible_moods = []
            for style, moods in style_mood_map.items():
                if style in top_style or style in bottom_style:
                    possible_moods.extend(moods)
            
            # Add user personality influence
            if user_data:
                personality = user_data.get('Personality Tag 1', '').lower()
                if 'minimalistic' in personality:
                    possible_moods.extend(['Minimal', 'Subtle', 'Clean'])
                elif 'adventurous' in personality:
                    possible_moods.extend(['Bold', 'Dynamic', 'Edgy'])
                elif 'elegant' in personality:
                    possible_moods.extend(['Luxe', 'Elegant', 'Sophisticated'])
            
            # Filter to available mood words and select
            available_moods = [m for m in possible_moods if m in mood_words]
            if available_moods:
                return random.choice(available_moods)
            
            # Fallback to common moods
            fallback_moods = ['Urban', 'Classic', 'Modern', 'Casual']
            return random.choice([m for m in fallback_moods if m in mood_words])
            
        except Exception as e:
            logger.error(f"❌ Error determining outfit mood: {e}")
            return "Classic"
    
    def _determine_outfit_accent(self, top_data: Dict, bottom_data: Dict, accent_words: List[str]) -> str:
        """Determine accent color/fabric/print for the outfit."""
        try:
            # Extract color information
            top_color = top_data.get('dominant_color', '').lower()
            bottom_color = bottom_data.get('dominant_color', '').lower()
            
            # Color to accent word mapping
            color_accent_map = {
                'black': ['Charcoal', 'Midnight', 'Neo-Noir'],
                'white': ['Cream', 'Monochrome'],
                'blue': ['Indigo', 'Navy', 'Sage'],
                'red': ['Scarlet', 'Coral', 'Rose'],
                'green': ['Sage', 'Olive', 'Emerald'],
                'brown': ['Camel', 'Leather', 'Tweed'],
                'gray': ['Slate', 'Charcoal', 'Monochrome'],
                'pink': ['Rose', 'Coral', 'Pastel'],
                'yellow': ['Amber', 'Pastel'],
                'purple': ['Burgundy', 'Indigo'],
                'orange': ['Amber', 'Coral'],
                'teal': ['Teal', 'Emerald']
            }
            
            # Check for fabric/print patterns
            top_title = top_data.get('title', '').lower()
            bottom_title = bottom_data.get('title', '').lower()
            
            fabric_accent_map = {
                'denim': ['Denim'],
                'linen': ['Linen'],
                'leather': ['Leather'],
                'velvet': ['Velvet'],
                'tweed': ['Tweed'],
                'floral': ['Floral'],
                'graphic': ['Graphic'],
                'gingham': ['Gingham'],
                'houndstooth': ['Houndstooth']
            }
            
            # Check for fabrics/prints first
            for fabric, accents in fabric_accent_map.items():
                if fabric in top_title or fabric in bottom_title:
                    available_accents = [a for a in accents if a in accent_words]
                    if available_accents:
                        return random.choice(available_accents)
            
            # Check for colors
            for color, accents in color_accent_map.items():
                if color in top_color or color in bottom_color:
                    available_accents = [a for a in accents if a in accent_words]
                    if available_accents:
                        return random.choice(available_accents)
            
            # Fallback to neutral accents
            fallback_accents = ['Cream', 'Charcoal', 'Monochrome']
            return random.choice([a for a in fallback_accents if a in accent_words])
            
        except Exception as e:
            logger.error(f"❌ Error determining outfit accent: {e}")
            return "Cream"
    
    def _determine_outfit_occasion(self, top_data: Dict, bottom_data: Dict, user_data: Dict, occasion_words: List[str]) -> str:
        """Determine the occasion/context for the outfit."""
        try:
            # Extract style and formality information
            top_style = top_data.get('primary_style', '').lower()
            bottom_style = bottom_data.get('primary_style', '').lower()
            
            # Style to occasion mapping
            style_occasion_map = {
                'business casual': ['Boardroom', 'Office', 'Meeting'],
                'formal': ['Soirée', 'Dinner', 'Party'],
                'casual': ['Weekend', 'Brunch', 'Commute'],
                'streetwear': ['Street', 'Weekend', 'Commute'],
                'athleisure': ['Studio', 'Weekend', 'Commute'],
                'party': ['Party', 'Soirée', 'Date Night'],
                'travel': ['Travel', 'Getaway', 'Commute']
            }
            
            # Determine occasion from styles
            possible_occasions = []
            for style, occasions in style_occasion_map.items():
                if style in top_style or style in bottom_style:
                    possible_occasions.extend(occasions)
            
            # Add user context
            if user_data:
                workspace_style = user_data.get('Workspace Style', '').lower()
                if 'formal' in workspace_style:
                    possible_occasions.extend(['Boardroom', 'Office'])
                elif 'casual' in workspace_style:
                    possible_occasions.extend(['Office', 'Meeting'])
            
            # Filter to available occasion words and select
            available_occasions = [o for o in possible_occasions if o in occasion_words]
            if available_occasions:
                return random.choice(available_occasions)
            
            # Fallback to common occasions
            fallback_occasions = ['Weekend', 'Office', 'Commute']
            return random.choice([o for o in fallback_occasions if o in occasion_words])
            
        except Exception as e:
            logger.error(f"❌ Error determining outfit occasion: {e}")
            return "Weekend"
    
    def _build_outfit_name(self, mood: str, accent: str, noun: str, occasion: str) -> str:
        """Build outfit name with max 3 words."""
        try:
            # Different name patterns (max 3 words)
            patterns = [
                f"{mood} {noun}",  # Urban Shift
                f"{accent} {noun}",  # Scarlet Edit
                f"{mood} {occasion}",  # Urban Weekend
                f"{accent} {mood} {noun}",  # Scarlet Urban Edit
                f"{mood} {noun} {occasion}"  # Urban Shift Weekend
            ]
            
            # Select a pattern that fits 3-word limit
            valid_patterns = [p for p in patterns if len(p.split()) <= 3]
            
            if valid_patterns:
                outfit_name = random.choice(valid_patterns)
                return outfit_name.upper()  # Convert to uppercase
            else:
                # Fallback to simple 2-word pattern
                outfit_name = f"{mood} {noun}"
                return outfit_name.upper()  # Convert to uppercase
                
        except Exception as e:
            logger.error(f"❌ Error building outfit name: {e}")
            outfit_name = f"{mood} {noun}"
            return outfit_name.upper()  # Convert to uppercase
        full_caption = str(product.get('full_caption', '')).lower()
        return (style_keyword in primary_style) or (style_keyword in full_caption)

    def _load_body_shape_rules(self) -> Dict:
        """Load body shape styling rules from designer CSV files."""
        body_shape_rules = {
            'male': self._load_male_body_shape_rules(),
            'female': self._load_female_body_shape_rules()
        }
        logger.info(f"✅ Loaded body shape rules for male and female")
        return body_shape_rules
    
    def _load_male_body_shape_rules(self) -> Dict:
        """Load male body shape rules from CSV."""
        import csv
        import os
        
        male_rules = {}
        csv_path = os.path.join('Fashion designer input', 'Body Shape - male.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the content by body shape sections
            sections = content.split('\n\n')
            current_shape = None
            
            for section in sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                    
                # Check if this is a body shape header
                if any(shape in lines[0].upper() for shape in ['HOURGLASS', 'RECTANGLE', 'PEAR', 'APPLE', 'TRIANGLE', 'OVAL']):
                    current_shape = lines[0].replace('BODY SHAPE', '').strip()
                    male_rules[current_shape.lower()] = {}
                    continue
                
                # Parse rules for current shape
                if current_shape and current_shape.lower() in male_rules:
                    for line in lines:
                        if ':' in line and not line.startswith('BODY SHAPE'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower().replace(' ', '_')
                                value = parts[1].strip().strip('"')
                                male_rules[current_shape.lower()][key] = value
            
            # Add general fit rules
            general_rules = {}
            for section in sections:
                if 'GENERAL FIT RULES' in section:
                    lines = section.split('\n')
                    for line in lines:
                        if ':' in line and 'GENERAL FIT RULES' not in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower().replace(' ', '_')
                                value = parts[1].strip().strip('"')
                                general_rules[key] = value
                    break
            
            male_rules['general_fit_rules'] = general_rules
            
        except Exception as e:
            logger.warning(f"Could not load male body shape rules: {e}")
            male_rules = self._get_fallback_male_body_shape_rules()
            
        return male_rules
    
    def _load_female_body_shape_rules(self) -> Dict:
        """Load female body shape rules from CSV."""
        import csv
        import os
        
        female_rules = {}
        csv_path = os.path.join('Fashion designer input', 'Body Shape - female.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the content by body shape sections
            sections = content.split('\n\n')
            current_shape = None
            
            for section in sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                    
                # Check if this is a body shape header
                if any(shape in lines[0].upper() for shape in ['HOURGLASS', 'RECTANGLE', 'PEAR', 'APPLE', 'TRIANGLE', 'OVAL']):
                    current_shape = lines[0].replace('BODY SHAPE', '').strip()
                    female_rules[current_shape.lower()] = {}
                    continue
                
                # Parse rules for current shape
                if current_shape and current_shape.lower() in female_rules:
                    for line in lines:
                        if ':' in line and not line.startswith('BODY SHAPE'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower().replace(' ', '_')
                                value = parts[1].strip().strip('"')
                                female_rules[current_shape.lower()][key] = value
            
            # Add general fit rules
            general_rules = {}
            for section in sections:
                if 'GENERAL FIT RULES' in section:
                    lines = section.split('\n')
                    for line in lines:
                        if ':' in line and 'GENERAL FIT RULES' not in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower().replace(' ', '_')
                                value = parts[1].strip().strip('"')
                                general_rules[key] = value
                    break
            
            female_rules['general_fit_rules'] = general_rules
            
        except Exception as e:
            logger.warning(f"Could not load female body shape rules: {e}")
            female_rules = self._get_fallback_female_body_shape_rules()
            
        return female_rules
    
    def _get_fallback_male_body_shape_rules(self) -> Dict:
        """Fallback male body shape rules if CSV loading fails."""
        return {
            'hourglass': {
                'best_tops': 'structured shirts, polos, t-shirts fitted',
                'best_bottoms': 'straight or tapered, not baggy',
                'perfect_fit': 'blazers with cinched waists, fitted crewnecks, knit polos',
                'show_off': 'natural upper-lower balance',
                'never_wear': 'boxy tops/bottoms',
                'wrong_fit': 'too baggy fits',
                'pro_tip': 'highlight snatched waists and good neckline to frame face and upper body'
            },
            'general_fit_rules': {
                'fitted_top_with_straight_bottom': 'most flattering combination',
                'fit_and_flare': 'every body shape looks good in fit n flare silhouettes',
                'accessories': 'most important fit rule is add accessories'
            }
        }
    
    def _get_fallback_female_body_shape_rules(self) -> Dict:
        """Fallback female body shape rules if CSV loading fails."""
        return {
            'hourglass': {
                'best_tops': 'fitted, crop tops, snug tanks, bodycon tees',
                'best_bottoms': 'straight or fit n flare not baggy',
                'perfect_fit': 'bodycon dresses, crop top n pants, mini skirts n fitted tops',
                'show_off': 'always highlight your waist, belts, waist seams',
                'never_wear': 'boxy tops/bottoms',
                'wrong_fit': 'too baggy or oversized',
                'pro_tip': 'accentuate the waist and balance top and bottom evenly'
            },
            'general_fit_rules': {
                'fitted_top_with_straight_bottom': 'most flattering combination',
                'fit_and_flare': 'every body shape looks good in fit n flare silhouettes',
                'accessories': 'most important fit rule is add accessories'
            }
        }

    def _load_style_mixing_rules(self) -> Dict:
        """Load style mixing rules from designer CSV."""
        import csv
        import os
        
        style_rules = {}
        csv_path = os.path.join('Fashion designer input', 'Style Mixing.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse style compatibility sections
            sections = content.split('\n\n')
            
            for section in sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                # Parse style types
                if 'BUSINESS/FORMAL' in lines[0]:
                    style_rules['business_formal'] = self._parse_style_section(lines)
                elif 'CASUAL' in lines[0]:
                    style_rules['casual'] = self._parse_style_section(lines)
                elif 'STREETWEAR' in lines[0]:
                    style_rules['streetwear'] = self._parse_style_section(lines)
                elif 'FORMALITY MIXING RULES' in lines[0]:
                    style_rules['formality_mixing'] = self._parse_mixing_ratings(lines)
                elif 'MIXING GUIDELINES' in lines[0]:
                    style_rules['mixing_guidelines'] = self._parse_mixing_guidelines(lines)
            
        except Exception as e:
            logger.warning(f"Could not load style mixing rules: {e}")
            style_rules = self._get_fallback_style_mixing_rules()
            
        return style_rules
    
    def _parse_style_section(self, lines: List[str]) -> Dict:
        """Parse a style section from the CSV."""
        style_info = {}
        for line in lines[1:]:  # Skip header
            if ':' in line and not line.startswith('STYLE'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip().strip('"')
                    style_info[key] = value
        return style_info
    
    def _parse_mixing_ratings(self, lines: List[str]) -> Dict:
        """Parse formality mixing ratings."""
        ratings = {}
        for line in lines:
            if '/' in line and any(char.isdigit() for char in line):
                # Extract rating from format like "Formal Top + Casual Bottom: 7/10"
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        combination = parts[0].strip()
                        rating_part = parts[1].strip()
                        # Extract number before /10
                        import re
                        numbers = re.findall(r'\d+', rating_part)
                        if numbers:
                            ratings[combination] = int(numbers[0])
        return ratings
    
    def _parse_mixing_guidelines(self, lines: List[str]) -> List[str]:
        """Parse mixing guidelines."""
        guidelines = []
        for line in lines:
            if line.strip() and not line.startswith('MIXING GUIDELINES'):
                guidelines.append(line.strip())
        return guidelines
    
    def _load_quick_styling_rules(self) -> Dict:
        """Load quick styling rules from designer CSV."""
        import csv
        import os
        
        quick_rules = {}
        csv_path = os.path.join('Fashion designer input', 'Quick Rules.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse different sections
            sections = content.split('\n\n')
            
            # Parse TRUE/FALSE statements
            true_false_rules = {}
            fashion_rules = []
            biggest_mistakes = []
            quick_fixes = {}
            
            current_section = None
            
            for section in sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                if 'TRUE OR FALSE' in section:
                    current_section = 'true_false'
                    continue
                elif 'TOP 5 FASHION RULES' in section:
                    current_section = 'fashion_rules'
                    continue
                elif 'BIGGEST MISTAKES' in section:
                    current_section = 'mistakes'
                    continue
                elif 'QUICK FIXES' in section:
                    current_section = 'fixes'
                    continue
                
                if current_section == 'true_false':
                    for line in lines:
                        if ',' in line and any(keyword in line.lower() for keyword in ['true', 'false']):
                            parts = line.split(',')
                            if len(parts) >= 2:
                                statement = parts[0].strip()
                                true_false = parts[1].strip()
                                if len(parts) > 2:
                                    reason = parts[2].strip().strip('"')
                                else:
                                    reason = ""
                                true_false_rules[statement] = {'answer': true_false, 'reason': reason}
                
                elif current_section == 'fashion_rules':
                    for line in lines:
                        if line.strip() and not line.startswith('TOP 5'):
                            fashion_rules.append(line.strip())
                
                elif current_section == 'mistakes':
                    for line in lines:
                        if line.strip() and not line.startswith('BIGGEST'):
                            biggest_mistakes.append(line.strip())
                
                elif current_section == 'fixes':
                    for line in lines:
                        if ':' in line and not line.startswith('QUICK FIXES'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower().replace(' ', '_')
                                value = parts[1].strip().strip('"')
                                quick_fixes[key] = value
            
            quick_rules = {
                'true_false_rules': true_false_rules,
                'fashion_rules': fashion_rules,
                'biggest_mistakes': biggest_mistakes,
                'quick_fixes': quick_fixes
            }
            
        except Exception as e:
            logger.warning(f"Could not load quick styling rules: {e}")
            quick_rules = self._get_fallback_quick_styling_rules()
            
        return quick_rules
    
    def _get_fallback_style_mixing_rules(self) -> Dict:
        """Fallback style mixing rules if CSV loading fails."""
        return {
            'business_formal': {
                'description': 'Business style means professional and structured',
                'pairs_perfectly_with': 'semi formals like polos, statement shirts',
                'never_pair_with': 'heavy prints, bleached or ripped denims',
                'special_mixing_rule': 'keep it in solids, avoids prints'
            },
            'casual': {
                'description': 'Casual style means everyday fit, easy to style',
                'pairs_perfectly_with': 'almost all styles',
                'never_pair_with': 'super formal structure fit, tailored pants'
            },
            'streetwear': {
                'description': 'Streetwear style means bold, expressive, chic',
                'pairs_perfectly_with': 'casuals',
                'never_pair_with': 'super formal structure fit, tailored pants'
            },
            'formality_mixing': {
                'Formal Top + Casual Bottom': 7,
                'Casual Top + Formal Bottom': 8,
                'Business Top + Streetwear Bottom': 6
            },
            'mixing_guidelines': [
                'You can mix formal and casual IF you balance the fit style n silhouette',
                'The key to successful style mixing is don\'t mix the extremes',
                'One piece should be a good fit and the other should be loose'
            ]
        }
    
    def _get_fallback_quick_styling_rules(self) -> Dict:
        """Fallback quick styling rules if CSV loading fails."""
        return {
            'true_false_rules': {
                'Its okay to wear all black everything': {'answer': 'True', 'reason': 'Safe play for majority'},
                'You can mix brown and black': {'answer': 'True', 'reason': 'High end fashion combination'},
                'Patterns on top AND bottom is too much': {'answer': 'True', 'reason': 'Too many colors in small space'}
            },
            'fashion_rules': [
                'Color balance',
                'Top wear and bottom wear should never have the same silhouette',
                'Mindfully accessorised',
                'Be mindful of the occasion you are dressing up for'
            ],
            'biggest_mistakes': [
                'Skinny fit',
                'White shirt and black pants',
                'Wrong sizing clothing, too tight or too loose',
                'Too many colors put together'
            ],
            'quick_fixes': {
                'to_instantly_improve': 'add accessories',
                'most_versatile_piece': 'blue denims',
                'when_nothing_works': 'graphic tee n shorts for casual'
            }
        }

    def _calculate_comfort_metrics_score(self, top: pd.Series, bottom: pd.Series) -> float:
        """Calculate comfort metrics score based on product attributes."""
        try:
            score = 5.0  # Base score
            
            # Check comfort level attributes
            top_comfort = str(top.get('comfort_level', '')).lower()
            bottom_comfort = str(bottom.get('comfort_level', '')).lower()
            
            # Comfort level scoring
            comfort_scores = {
                'high': 8.0,
                'medium': 6.0,
                'low': 4.0,
                'very high': 9.0,
                'very low': 3.0
            }
            
            if top_comfort in comfort_scores:
                score += (comfort_scores[top_comfort] - 5.0) * 0.3
            if bottom_comfort in comfort_scores:
                score += (comfort_scores[bottom_comfort] - 5.0) * 0.3
            
            # Check movement restriction
            top_movement = str(top.get('movement_restriction', '')).lower()
            bottom_movement = str(bottom.get('movement_restriction', '')).lower()
            
            if 'low' in top_movement or 'minimal' in top_movement:
                score += 1.0
            elif 'high' in top_movement or 'restrictive' in top_movement:
                score -= 1.0
                
            if 'low' in bottom_movement or 'minimal' in bottom_movement:
                score += 1.0
            elif 'high' in bottom_movement or 'restrictive' in bottom_movement:
                score -= 1.0
            
            return min(10.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating comfort metrics: {e}")
            return 5.0
    
    def _calculate_occasion_appropriateness(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Calculate occasion appropriateness score."""
        try:
            user_occasion = user.get('occasion_preference', '').lower()
            if not user_occasion:
                return 5.0
            
            score = 5.0  # Base score
            
            # Get product occasion attributes
            top_occasion = str(top.get('occasion', '')).lower()
            bottom_occasion = str(bottom.get('occasion', '')).lower()
            
            # Occasion matching
            occasion_keywords = {
                'work': ['work', 'office', 'professional', 'business'],
                'casual': ['casual', 'everyday', 'relaxed', 'informal'],
                'formal': ['formal', 'elegant', 'sophisticated', 'evening'],
                'party': ['party', 'celebration', 'festive', 'glamorous'],
                'sport': ['sport', 'athletic', 'active', 'gym']
            }
            
            for occasion_type, keywords in occasion_keywords.items():
                if occasion_type in user_occasion:
                    # Check if products match the occasion
                    top_match = any(keyword in top_occasion for keyword in keywords)
                    bottom_match = any(keyword in bottom_occasion for keyword in keywords)
                    
                    if top_match and bottom_match:
                        score += 2.0
                    elif top_match or bottom_match:
                        score += 1.0
                    else:
                        score -= 1.0
                    break
            
            return min(10.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating occasion appropriateness: {e}")
            return 5.0
    
    def _calculate_cultural_context(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Calculate cultural relevance score."""
        try:
            score = 5.0  # Base score
            
            # Check for cultural preferences in user data
            user_culture = user.get('cultural_preference', '').lower()
            if not user_culture:
                return 5.0
            
            # Get product cultural attributes
            top_cultural = str(top.get('cultural_context', '')).lower()
            bottom_cultural = str(bottom.get('cultural_context', '')).lower()
            
            # Cultural matching
            if user_culture in top_cultural or user_culture in bottom_cultural:
                score += 2.0
            
            # Check for Indian skin tone considerations (from designer rules)
            if 'indian' in user_culture or 'south asian' in user_culture:
                # Check if colors are flattering for Indian skin tones
                top_color = self._extract_color(top).lower()
                bottom_color = self._extract_color(bottom).lower()
                
                indian_flattering_colors = ['olive green', 'deep red', 'maroon', 'navy', 'emerald green']
                if top_color in indian_flattering_colors or bottom_color in indian_flattering_colors:
                    score += 1.0
            
            return min(10.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating cultural context: {e}")
            return 5.0
    
    def _calculate_versatility_score(self, top: pd.Series, bottom: pd.Series) -> float:
        """Calculate versatility score based on product attributes."""
        try:
            score = 5.0  # Base score
            
            # Check versatility attributes
            top_versatility = str(top.get('style_versatility', '')).lower()
            bottom_versatility = str(bottom.get('style_versatility', '')).lower()
            
            # Versatility scoring
            versatility_scores = {
                'high': 8.0,
                'medium': 6.0,
                'low': 4.0,
                'very high': 9.0,
                'very low': 3.0
            }
            
            if top_versatility in versatility_scores:
                score += (versatility_scores[top_versatility] - 5.0) * 0.3
            if bottom_versatility in versatility_scores:
                score += (versatility_scores[bottom_versatility] - 5.0) * 0.3
            
            # Check adaptability
            top_adaptability = str(top.get('adaptability', '')).lower()
            bottom_adaptability = str(bottom.get('adaptability', '')).lower()
            
            if 'high' in top_adaptability or 'versatile' in top_adaptability:
                score += 1.0
            if 'high' in bottom_adaptability or 'versatile' in bottom_adaptability:
                score += 1.0
            
            return min(10.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating versatility score: {e}")
            return 5.0

    def _apply_designer_rule_engine(self, top: pd.Series, bottom: pd.Series, user: Dict) -> Dict[str, float]:
        """✅ ENHANCED: Apply comprehensive designer rule engine with programmatic logic."""
        rule_scores = {
            'color_harmony': 0.0,
            'body_shape': 0.0,
            'style_mixing': 0.0,
            'quick_rules': 0.0,
            'fit_balance': 0.0,
            'silhouette': 0.0,
            'accessories': 0.0,
            'seasonal': 0.0,
            'cultural': 0.0,
            'professional': 0.0
        }
        
        try:
            # 1. COLOR HARMONY RULE ENGINE
            rule_scores['color_harmony'] = self._apply_color_harmony_rules(top, bottom, user)
            
            # 2. BODY SHAPE RULE ENGINE
            rule_scores['body_shape'] = self._apply_body_shape_rules(top, bottom, user)
            
            # 3. STYLE MIXING RULE ENGINE
            rule_scores['style_mixing'] = self._apply_style_mixing_rules(top, bottom, user)
            
            # 4. QUICK RULES ENGINE
            rule_scores['quick_rules'] = self._apply_quick_rules_engine(top, bottom, user)
            
            # 5. FIT BALANCE RULE ENGINE
            rule_scores['fit_balance'] = self._apply_fit_balance_rules(top, bottom, user)
            
            # 6. SILHOUETTE RULE ENGINE
            rule_scores['silhouette'] = self._apply_silhouette_rules(top, bottom, user)
            
            # 7. ACCESSORIES RULE ENGINE
            rule_scores['accessories'] = self._apply_accessories_rules(top, bottom, user)
            
            # 8. SEASONAL RULE ENGINE
            rule_scores['seasonal'] = self._apply_seasonal_rules(top, bottom, user)
            
            # 9. CULTURAL RULE ENGINE
            rule_scores['cultural'] = self._apply_cultural_rules(top, bottom, user)
            
            # 10. PROFESSIONAL RULE ENGINE
            rule_scores['professional'] = self._apply_professional_rules(top, bottom, user)
            
        except Exception as e:
            logger.error(f"❌ Error in designer rule engine: {e}")
        
        return rule_scores
    
    def _apply_color_harmony_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply color harmony rules with programmatic logic."""
        score = 5.0
        top_color = self._extract_color(top).title()
        bottom_color = self._extract_color(bottom).title()
        
        if not top_color or not bottom_color:
            return score
        
        # Rule 1: Exact designer combinations
        color_pair = (bottom_color, top_color)
        if color_pair in self.color_harmony_map:
            rule = self.color_harmony_map[color_pair]
            rating = rule.get('rating', 5)
            score = rating / 10.0 * 10.0  # Convert to 0-10 scale
            logger.debug(f"🎨 Designer color rule applied: {bottom_color} + {top_color} = {score:.1f}")
            return score
        
        # Rule 2: Color family matching
        color_families = {
            'warm': ['red', 'orange', 'yellow', 'pink', 'coral'],
            'cool': ['blue', 'green', 'purple', 'teal', 'navy'],
            'neutral': ['black', 'white', 'gray', 'brown', 'beige', 'cream']
        }
        
        top_family = self._get_color_family(top_color)
        bottom_family = self._get_color_family(bottom_color)
        
        # Rule 2a: Monochromatic (same family)
        if top_family == bottom_family and top_family != 'neutral':
            score += 2.0
        
        # Rule 2b: Complementary (opposite families)
        if (top_family == 'warm' and bottom_family == 'cool') or (top_family == 'cool' and bottom_family == 'warm'):
            score += 1.5
        
        # Rule 2c: Neutral + Color (always safe)
        if top_family == 'neutral' or bottom_family == 'neutral':
            score += 1.0
        
        # Rule 3: Brightness balance
        top_brightness = self._get_color_brightness(top_color)
        bottom_brightness = self._get_color_brightness(bottom_color)
        
        if abs(top_brightness - bottom_brightness) <= 2:  # Similar brightness
            score += 1.0
        elif abs(top_brightness - bottom_brightness) >= 5:  # High contrast
            score += 0.5
        
        return min(10.0, max(0.0, score))
    
    def _apply_body_shape_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply body shape rules with programmatic logic."""
        score = 5.0
        user_gender = user.get('Gender', '').lower()
        body_shape = user.get('Body Shape', '').lower()
        
        if not body_shape or not user_gender:
            return score
        
        gender_rules = self.body_shape_rules.get(user_gender, {})
        shape_rules = gender_rules.get(body_shape, {})
        
        if not shape_rules:
            return score
        
        top_title = str(top.get('title', '')).lower()
        bottom_title = str(bottom.get('title', '')).lower()
        
        # Rule 1: Best tops matching
        best_tops = shape_rules.get('best_tops', '')
        if best_tops:
            keywords = self._extract_keywords_from_text(best_tops)
            matches = sum(1 for keyword in keywords if keyword in top_title)
            if matches > 0:
                score += (matches / len(keywords)) * 2.0
        
        # Rule 2: Best bottoms matching
        best_bottoms = shape_rules.get('best_bottoms', '')
        if best_bottoms:
            keywords = self._extract_keywords_from_text(best_bottoms)
            matches = sum(1 for keyword in keywords if keyword in bottom_title)
            if matches > 0:
                score += (matches / len(keywords)) * 2.0
        
        # Rule 3: Never wear violations
        never_wear = shape_rules.get('never_wear', '')
        if never_wear:
            keywords = self._extract_keywords_from_text(never_wear)
            violations = sum(1 for keyword in keywords if keyword in top_title or keyword in bottom_title)
            if violations > 0:
                score -= violations * 1.5
        
        # Rule 4: Body shape specific rules
        if body_shape == 'hourglass':
            # Check for waist emphasis
            waist_keywords = ['fitted', 'cinched', 'belt', 'waist', 'crop']
            if any(keyword in top_title for keyword in waist_keywords):
                score += 1.5
        
        elif body_shape == 'rectangle':
            # Check for volume creation
            volume_keywords = ['layered', 'puff', 'flare', 'volume', 'structured']
            if any(keyword in top_title for keyword in volume_keywords):
                score += 1.5
        
        elif body_shape == 'pear':
            # Check for upper body emphasis
            upper_keywords = ['statement', 'detail', 'volume', 'layered']
            if any(keyword in top_title for keyword in upper_keywords):
                score += 1.5
        
        return min(10.0, max(0.0, score))
    
    def _apply_style_mixing_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply style mixing rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Formality level calculation
        top_formality = self._calculate_formality_level(top)
        bottom_formality = self._calculate_formality_level(bottom)
        
        # Rule 1a: Formality mixing score
        formality_diff = abs(top_formality - bottom_formality)
        if formality_diff == 1:  # Slight difference (good)
            score += 2.0
        elif formality_diff == 2:  # Moderate difference (acceptable)
            score += 1.0
        elif formality_diff >= 3:  # Large difference (problematic)
            score -= 1.0
        
        # Rule 1b: Specific formality combinations
        if top_formality > bottom_formality:  # Formal top + casual bottom
            score += 1.0
        elif bottom_formality > top_formality:  # Casual top + formal bottom
            score += 1.5
        
        # Rule 2: Style compatibility
        top_style = str(top.get('primary_style', '')).lower()
        bottom_style = str(bottom.get('primary_style', '')).lower()
        
        # Check against designer style mixing rules
        for style_type, rules in self.style_mixing_rules.items():
            if isinstance(rules, dict):
                never_pair = rules.get('never_pair_with', '')
                if never_pair:
                    never_keywords = self._extract_keywords_from_text(never_pair)
                    violations = sum(1 for keyword in never_keywords 
                                   if keyword in top_style or keyword in bottom_style)
                    if violations > 0:
                        score -= violations * 1.0
        
        # Rule 3: Fit balance (from designer rules)
        top_fit = self._get_fit_type(top)
        bottom_fit = self._get_fit_type(bottom)
        
        if top_fit != bottom_fit:  # Different fits (good)
            score += 1.5
        elif top_fit == 'fitted' and bottom_fit == 'fitted':  # Both fitted (bad for most)
            score -= 1.0
        
        return min(10.0, max(0.0, score))
    
    def _apply_quick_rules_engine(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply quick styling rules with programmatic logic."""
        score = 5.0
        violations = []
        
        top_title = str(top.get('title', '')).lower()
        bottom_title = str(bottom.get('title', '')).lower()
        
        # Rule 1: Pattern balance
        pattern_keywords = ['print', 'pattern', 'floral', 'striped', 'checkered', 'plaid', 'geometric']
        top_patterns = sum(1 for keyword in pattern_keywords if keyword in top_title)
        bottom_patterns = sum(1 for keyword in pattern_keywords if keyword in bottom_title)
        
        if top_patterns > 0 and bottom_patterns > 0:
            score -= 2.0
            violations.append("Too many patterns")
        
        # Rule 2: Color balance
        top_color = self._extract_color(top).lower()
        bottom_color = self._extract_color(bottom).lower()
        
        bright_colors = ['red', 'yellow', 'orange', 'pink', 'purple', 'green', 'blue']
        top_bright = any(color in top_color for color in bright_colors)
        bottom_bright = any(color in bottom_color for color in bright_colors)
        
        if top_bright and bottom_bright:
            score -= 1.5
            violations.append("Too many bright colors")
        
        # Rule 3: Classic mistakes
        classic_mistakes = [
            ('white', 'shirt', 'black', 'pant'),
            ('skinny', 'skinny'),
            ('oversized', 'oversized'),
            ('baggy', 'baggy')
        ]
        
        for mistake in classic_mistakes:
            if all(keyword in top_title or keyword in bottom_title for keyword in mistake):
                score -= 2.0
                violations.append(f"Classic mistake: {' + '.join(mistake)}")
        
        # Rule 4: Silhouette balance
        top_silhouette = self._get_silhouette_type(top)
        bottom_silhouette = self._get_silhouette_type(bottom)
        
        if top_silhouette == bottom_silhouette and top_silhouette != 'unknown':
            score -= 1.0
            violations.append("Same silhouette top and bottom")
        
        # Rule 5: Positive rules
        positive_combinations = [
            ('fitted', 'loose'),
            ('structured', 'relaxed'),
            ('crop', 'high-waist'),
            ('oversized', 'fitted')
        ]
        
        for combo in positive_combinations:
            if (combo[0] in top_title and combo[1] in bottom_title) or \
               (combo[1] in top_title and combo[0] in bottom_title):
                score += 1.0
        
        if violations:
            logger.debug(f"🚨 Quick rules violations: {violations}")
        
        return min(10.0, max(0.0, score))
    
    def _apply_fit_balance_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply fit balance rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Fit contrast (designer rule: "fitted top + loose bottom")
        top_fit = self._get_fit_type(top)
        bottom_fit = self._get_fit_type(bottom)
        
        if top_fit == 'fitted' and bottom_fit in ['loose', 'relaxed', 'straight']:
            score += 2.0
        elif bottom_fit == 'fitted' and top_fit in ['loose', 'relaxed', 'oversized']:
            score += 1.5
        elif top_fit == bottom_fit and top_fit == 'fitted':
            score -= 1.0  # Both fitted (not ideal)
        
        # Rule 2: Proportion balance
        top_length = self._get_length_type(top)
        bottom_length = self._get_length_type(bottom)
        
        if top_length == 'crop' and bottom_length == 'high-waist':
            score += 1.5
        elif top_length == 'long' and bottom_length == 'mid-rise':
            score += 1.0
        
        return min(10.0, max(0.0, score))
    
    def _apply_silhouette_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply silhouette rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Silhouette contrast
        top_silhouette = self._get_silhouette_type(top)
        bottom_silhouette = self._get_silhouette_type(bottom)
        
        if top_silhouette != bottom_silhouette:
            score += 1.5
        
        # Rule 2: Body shape specific silhouettes
        body_shape = user.get('Body Shape', '').lower()
        
        if body_shape == 'hourglass':
            if top_silhouette == 'fitted' and bottom_silhouette in ['straight', 'flare']:
                score += 1.5
        elif body_shape == 'rectangle':
            if top_silhouette in ['loose', 'oversized'] and bottom_silhouette == 'straight':
                score += 1.5
        elif body_shape == 'pear':
            if top_silhouette in ['loose', 'oversized'] and bottom_silhouette == 'straight':
                score += 1.5
        
        return min(10.0, max(0.0, score))
    
    def _apply_accessories_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply accessories rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Accessory-friendly pieces
        accessory_keywords = ['collar', 'pocket', 'button', 'detail', 'embellishment']
        top_accessories = sum(1 for keyword in accessory_keywords if keyword in str(top.get('title', '')).lower())
        bottom_accessories = sum(1 for keyword in accessory_keywords if keyword in str(bottom.get('title', '')).lower())
        
        if top_accessories > 0 or bottom_accessories > 0:
            score += 1.0
        
        # Rule 2: Minimal pieces (need accessories)
        minimal_keywords = ['basic', 'plain', 'simple', 'solid']
        top_minimal = any(keyword in str(top.get('title', '')).lower() for keyword in minimal_keywords)
        bottom_minimal = any(keyword in str(bottom.get('title', '')).lower() for keyword in minimal_keywords)
        
        if top_minimal and bottom_minimal:
            score += 0.5  # Encourages accessories
        
        return min(10.0, max(0.0, score))
    
    def _apply_seasonal_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply seasonal rules with programmatic logic."""
        score = 5.0
        
        # Get current season (simplified)
        import datetime
        month = datetime.datetime.now().month
        
        if month in [3, 4, 5, 6]:  # Spring/Summer
            season = 'spring_summer'
        else:  # Fall/Winter
            season = 'fall_winter'
        
        # Check seasonal color combinations
        seasonal_rules = self.color_harmony_map.get('_seasonal_rules', {})
        season_combinations = seasonal_rules.get(season, [])
        
        top_color = self._extract_color(top).title()
        bottom_color = self._extract_color(bottom).title()
        
        for base, pair in season_combinations:
            if (bottom_color.lower() in base.lower() and top_color.lower() in pair.lower()) or \
               (top_color.lower() in base.lower() and bottom_color.lower() in pair.lower()):
                score += 2.0
                break
        
        return min(10.0, max(0.0, score))
    
    def _apply_cultural_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply cultural rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Indian skin tone considerations
        user_culture = user.get('cultural_preference', '').lower()
        if 'indian' in user_culture or 'south asian' in user_culture:
            top_color = self._extract_color(top).lower()
            bottom_color = self._extract_color(bottom).lower()
            
            indian_flattering = ['olive green', 'deep red', 'maroon', 'navy', 'emerald green', 'burgundy']
            if top_color in indian_flattering or bottom_color in indian_flattering:
                score += 1.5
        
        return min(10.0, max(0.0, score))
    
    def _apply_professional_rules(self, top: pd.Series, bottom: pd.Series, user: Dict) -> float:
        """Apply professional rules with programmatic logic."""
        score = 5.0
        
        # Rule 1: Professional setting considerations
        occasion = user.get('occasion_preference', '').lower()
        if 'work' in occasion or 'professional' in occasion or 'office' in occasion:
            # Check for professional colors
            professional_colors = ['black', 'navy', 'gray', 'white', 'brown']
            top_color = self._extract_color(top).lower()
            bottom_color = self._extract_color(bottom).lower()
            
            if top_color in professional_colors or bottom_color in professional_colors:
                score += 1.0
            
            # Check for professional styles
            professional_keywords = ['shirt', 'pant', 'blazer', 'trouser', 'formal']
            top_professional = any(keyword in str(top.get('title', '')).lower() for keyword in professional_keywords)
            bottom_professional = any(keyword in str(bottom.get('title', '')).lower() for keyword in professional_keywords)
            
            if top_professional and bottom_professional:
                score += 1.5
        
        return min(10.0, max(0.0, score))
    
    # Helper methods for rule engine
    def _get_color_family(self, color: str) -> str:
        """Get color family classification."""
        color = color.lower()
        warm = ['red', 'orange', 'yellow', 'pink', 'coral', 'peach']
        cool = ['blue', 'green', 'purple', 'teal', 'navy', 'mint']
        neutral = ['black', 'white', 'gray', 'brown', 'beige', 'cream', 'tan']
        
        if any(w in color for w in warm):
            return 'warm'
        elif any(c in color for c in cool):
            return 'cool'
        elif any(n in color for n in neutral):
            return 'neutral'
        return 'unknown'
    
    def _get_color_brightness(self, color: str) -> int:
        """Get color brightness level (1-10)."""
        color = color.lower()
        brightness_map = {
            'white': 10, 'yellow': 9, 'pink': 8, 'orange': 7, 'red': 6,
            'green': 5, 'blue': 4, 'purple': 3, 'brown': 2, 'black': 1
        }
        return brightness_map.get(color, 5)
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from natural language text."""
        if not text:
            return []
        
        # Remove common words and extract meaningful keywords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_formality_level(self, product: pd.Series) -> int:
        """Calculate formality level of a product (1-10)."""
        title = str(product.get('title', '')).lower()
        style = str(product.get('primary_style', '')).lower()
        
        formality_keywords = {
            'formal': 9, 'business': 8, 'professional': 8, 'elegant': 8,
            'casual': 5, 'relaxed': 4, 'streetwear': 3, 'athleisure': 2,
            'activewear': 2, 'sport': 2, 'oversized': 4, 'fitted': 6
        }
        
        max_level = 5  # Default
        for keyword, level in formality_keywords.items():
            if keyword in title or keyword in style:
                max_level = max(max_level, level)
        
        return max_level
    
    def _get_fit_type(self, product: pd.Series) -> str:
        """Get fit type of a product."""
        title = str(product.get('title', '')).lower()
        
        if any(keyword in title for keyword in ['fitted', 'slim', 'tight', 'bodycon']):
            return 'fitted'
        elif any(keyword in title for keyword in ['loose', 'relaxed', 'oversized', 'baggy']):
            return 'loose'
        elif any(keyword in title for keyword in ['straight', 'regular', 'classic']):
            return 'straight'
        return 'unknown'
    
    def _get_length_type(self, product: pd.Series) -> str:
        """Get length type of a product."""
        title = str(product.get('title', '')).lower()
        
        if any(keyword in title for keyword in ['crop', 'short', 'mini']):
            return 'crop'
        elif any(keyword in title for keyword in ['long', 'maxi', 'full']):
            return 'long'
        elif any(keyword in title for keyword in ['mid', 'regular']):
            return 'mid'
        return 'unknown'

def main():
    """Main function to test the enhanced Supabase outfit generator."""
    import argparse
    parser = argparse.ArgumentParser(description="Test the enhanced Supabase outfit generator.")
    parser.add_argument('--user_id', type=int, help='User ID to generate outfits for (for local testing only)')
    args = parser.parse_args()

    if args.user_id is None:
        print("Usage: python phase1_supabase_outfits_generator.py --user_id <USER_ID>")
        print("This script is intended for API use. For local testing, provide a user ID.")
        return

    try:
        # Initialize enhanced generator
        generator = SupabaseMainOutfitsGenerator()
        user_id = args.user_id
        logger.info(f"🎯 Testing enhanced outfit generation for User {user_id}")
        success = generator.generate_and_save_outfits(user_id)
        if success:
            print(f"\n✅ SUCCESS: Enhanced outfits generated and saved for user {user_id}")
            print(f"📁 Outfits are now available in Supabase database")
            print(f"🔗 API Endpoint: GET /api/user/{user_id}/outfits")
        else:
            print(f"\n❌ FAILED: Could not generate enhanced outfits for user {user_id}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main() 