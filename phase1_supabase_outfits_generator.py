"""
Phase 1: Enhanced Supabase Outfits Generator with Professional Fashion Intelligence

Generates 20 main outfits per user using Supabase database with all enhancements from main generator
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
    ✅ COMPLETE: Full feature parity with main generator
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the Supabase-enabled outfits generator with fashion designer intelligence."""
        self.config = config or self._default_config()
        self.db = get_db()
        
        # Railway CPU optimization
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        if self.is_railway:
            # Set conservative CPU limits for ML operations
            for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                os.environ[var] = '2'
        
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
        
        # Load model
        try:
            self.model = SentenceTransformer(self.config['model_name'])
            logger.info(f"✅ Model loaded: {self.config['model_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
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
            'Fashion_Designer_Templates_Simple.xlsx - 1. Color Harmony.csv')
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                base_color = None
                for row in reader:
                    if not row or not any(row):
                        continue
                    if row[0].startswith('BASE COLOR:'):
                        base_color = row[0].replace(
                            'BASE COLOR:', '').replace(
                            '*a color that can have it all*', '').strip()
                        continue
                    if base_color and len(row) >= 2 and '+' in row[0]:
                        # Parse color pair and rating
                        try:
                            pair = row[0].split('+')
                            if len(pair) == 2:
                                color1 = base_color.title()
                                color2 = pair[1].strip().title()
                                rating = int(
                                    row[1]) if row[1].strip().isdigit() else None
                                notes = row[2].strip() if len(row) > 2 else ''
                                color_harmony_map[(color1, color2)] = {
                                    'rating': rating, 'notes': notes}
                        except Exception as e:
                            continue
            logger.info(
                f"✅ Loaded {len(color_harmony_map)} color harmony rules from designer CSV")
        except Exception as e:
            logger.warning(f"Could not load color harmony CSV: {e}")
        return color_harmony_map
    
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
            # Use pre-filtered loading if user data is available
            if user_data:
                products_df = self.db.get_products_with_user_filters(user_data)
            else:
                # Fallback to chunked loading for backward compatibility
                products_df = self.db.get_products()
            
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
            'missing_category': products_df['category'].isna().sum() if 'category' in products_df.columns else 0
        }

        # Drop rows with missing critical fields
        products_df = products_df.dropna(subset=['title'])

        # ✅ ENHANCED: Infer wear_type from category and other information
        def infer_wear_type(row):
            """Infer wear type from category and title"""
            category = str(row.get('category', '')).lower()
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
            if 'id' in products_df.columns:
                products_df['product_id'] = products_df['id'].astype(str)
            else:
                products_df['product_id'] = [f"PROD_{i:06d}" for i in range(len(products_df))]

        # Calculate overall data quality score
        def calculate_quality_score(row):
            score = 0.0
            total_fields = 0

            # Required fields
            required_fields = ['title', 'category', 'wear_type']
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
                f"{row.get('title', '')} {row.get('category', '')} {row.get('primary_style', '')} {row.get('primary_color', '')}".strip(), axis=1)

        return products_df

    def get_embedding_cached(self, text: str, cache_key: str = None) -> np.ndarray:
        """Get embedding with enhanced caching for better performance."""
        if not cache_key:
            cache_key = text[:100]
        
        if self.config['cache_embeddings'] and cache_key in self.embedding_cache:
            self.cache_stats['hits'] += 1
            return self.embedding_cache[cache_key]
        
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
        """Build FAISS indexes for different wear types using enhanced Supabase data with parallel processing."""
        logger.info("🔄 Building FAISS indexes for product recommendations...")
        
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
        """Build FAISS index for a specific wear type with parallel embedding generation."""
        if wear_products.empty:
            logger.warning(f"No products found for wear_type: {wear_type}")
            return
        
        captions = []
        product_indices = []
        
        for idx, row in wear_products.iterrows():
            caption = row.get('final_caption', '') or row.get('title', '')
            if caption.strip():
                captions.append(caption)
                product_indices.append(idx)
        
        if not captions:
            logger.warning(f"No valid captions found for wear_type: {wear_type}")
            return

        logger.info(f"🚀 Generating embeddings for {len(captions)} {wear_type} products...")

        # 🚀 PARALLEL: Generate embeddings in batches
        batch_size = self.config.get('embedding_batch_size', 50)
        embeddings = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(batch_captions)
            embeddings.extend(batch_embeddings)

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

        logger.info(f"✅ Built FAISS index for {wear_type}: {len(captions)} products indexed")
        
        # Log cache statistics
        cache_stats = self.get_cache_stats()
        logger.info(f"📊 Embedding cache stats: {cache_stats['hit_rate']} hit rate ({cache_stats['hits']} hits, {cache_stats['misses']} misses)")

    def filter_products_enhanced(self, products_df: pd.DataFrame, user: Dict, wear_type: str = None) -> pd.DataFrame:
        """✅ ENHANCED: Enhanced manual filtering on Gender, Fashion Style, and Body Shape."""
        logger.info(f"Starting enhanced filtering - Initial products: {len(products_df)}")

        def match(row):
            # 1. GENDER FILTERING (SIMPLE - Match gender columns)
            user_gender = user.get('Gender', user.get('gender', '')).lower()
            product_gender = row.get('gender', 'Unisex').lower()

            if user_gender and product_gender:
                # Allow same gender products and unisex products
                if user_gender in ['male', 'men']:
                    acceptable_genders = ['men', 'male', 'unisex']
                elif user_gender in ['female', 'women']:
                    acceptable_genders = ['women', 'female', 'unisex']
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
                                   gender_filter: str = None, k: int = 20, user: Dict = None) -> List[Dict]:
        """✅ ENHANCED: Get semantic recommendations using FAISS with improved diversity and context awareness."""

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

            # Apply gender filter if specified
            if gender_filter and gender_filter != 'Unisex':
                product_gender = product.get('gender', 'Unisex').lower()
                gender_filter_lower = gender_filter.lower()

                # Allow same gender products and unisex products
                if gender_filter_lower in ['male', 'men']:
                    acceptable_genders = ['men', 'male', 'unisex']
                elif gender_filter_lower in ['female', 'women']:
                    acceptable_genders = ['women', 'female', 'unisex']
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
        """Enhanced outfit scoring with basic compatibility check."""

        # First check basic compatibility
        is_compatible, reason = self._check_basic_outfit_compatibility(
            top, bottom)
        if not is_compatible:
            return 0.0, f"Incompatible: {reason}"

        try:
            # Get confidence scores
            top_confidence = top.get('confidence_score', 1.0)
            bottom_confidence = bottom.get('confidence_score', 1.0)
            avg_confidence = (top_confidence + bottom_confidence) / 2

            # Initialize scoring components with confidence weighting
            scoring_components = {
                'semantic_similarity': {
                    'score': (top_semantic + bottom_semantic) / 2,
                    'weight': 2.0,
                    'confidence': avg_confidence
                },
                'fit_compatibility': {
                    'score': self._calculate_fit_compatibility_score(top, user) * top_confidence +
                    self._calculate_fit_compatibility_score(bottom, user) * bottom_confidence,
                    'weight': 2.8,
                    'confidence': avg_confidence
                },
                'comfort_metrics': {
                    'score': self._normalize_comfort_level(top.get('comfort_level', 'Medium')) * top_confidence +
                    self._normalize_comfort_level(bottom.get('comfort_level', 'Medium')) * bottom_confidence,
                    'weight': 2.5,
                    'confidence': avg_confidence
                },
                'style_intelligence': {
                    'score': self._calculate_style_intelligence_score(top, bottom),
                    'weight': 2.5,
                    'confidence': avg_confidence
                },
                'color_harmony': {
                    'score': self.calculate_color_harmony_score(
                        top.get('primary_color', 'Black'),
                        bottom.get('primary_color', 'Black')
                    ),
                    'weight': 2.3,
                    'confidence': avg_confidence
                },
                'quality_metrics': {
                    'score': self._calculate_quality_metrics_score(top, bottom),
                    'weight': 2.0,
                    'confidence': avg_confidence
                },
                'occasion_context': {
                    'score': self._calculate_occasion_appropriateness(top, bottom, user),
                    'weight': 1.8,
                    'confidence': avg_confidence
                },
                'cultural_relevance': {
                    'score': self._calculate_cultural_context(top, bottom, user),
                    'weight': 1.5,
                    'confidence': avg_confidence
                },
                'versatility': {
                    'score': self._calculate_versatility_score(top, bottom),
                    'weight': 1.8,
                    'confidence': avg_confidence
                },
                'price_coherence': {
                    'score': self._calculate_price_coherence(top, bottom),
                    'weight': 1.5,
                    'confidence': avg_confidence
                }
            }

            # Calculate weighted score with confidence adjustment
            total_weight = 0
            weighted_score = 0

            for component, data in scoring_components.items():
                # Adjust weight by confidence
                weight = data['weight'] * data['confidence']
                total_weight += weight
                weighted_score += data['score'] * weight

            final_score = weighted_score / total_weight if total_weight > 0 else 0

            # Generate explanation with confidence indicators
            explanations = []
            for component, data in scoring_components.items():
                if data['score'] > 0.7:
                    confidence_indicator = "✓" if data['confidence'] > 0.9 else "~"
                    explanations.append(
                        f"{confidence_indicator} Strong {component.replace('_', ' ').title()}")
                elif data['score'] < 0.3:
                    confidence_indicator = "✓" if data['confidence'] > 0.9 else "~"
                    explanations.append(
                        f"{confidence_indicator} Weak {component.replace('_', ' ').title()}")

            # Add overall confidence indicator
            if avg_confidence < 0.7:
                explanations.append("⚠️ Some attributes have low confidence")

            explanation = self.get_explanation_for_outfit(
                explanations, scoring_components['color_harmony']['score'])

            return final_score, explanation

        except Exception as e:
            logger.error(f"Error in outfit scoring: {str(e)}")
            return 0.5, "Error in scoring, using default score"

    def _calculate_style_intelligence_score(self, top: pd.Series, bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate style intelligence score using professional rules."""
        top_style = top.get('enhanced_primary_style', top.get('primary_style', ''))
        bottom_style = bottom.get('enhanced_primary_style', bottom.get('primary_style', ''))

        # Get formality levels
        top_formality = self.style_formality.get(top_style, 5)
        bottom_formality = self.style_formality.get(bottom_style, 5)

        # Check formality balance
        formality_diff = abs(top_formality - bottom_formality)
        if formality_diff > 3:  # Too much difference in formality
            return 0.4

        # Style compatibility check
        style_compatibility = {
            'Streetwear': ['Streetwear', 'Casual', 'Contemporary', 'Activewear', 'Athleisure'],
            'Athleisure': ['Athleisure', 'Activewear', 'Streetwear', 'Casual', 'Contemporary'],
            'Contemporary': ['Contemporary', 'Business Casual', 'Smart Casual', 'Casual'],
            'Business': ['Business', 'Business Formal', 'Business Casual', 'Professional'],
            'Formal': ['Formal', 'Business Formal', 'Evening Formal', 'Ultra Formal']
        }

        top_compatibility = style_compatibility.get(top_style, [top_style])
        
        if bottom_style in top_compatibility:
            return 0.8
        else:
            return 0.5

    def _calculate_price_coherence(self, top: pd.Series, bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate price coherence score with smart thresholds."""
        top_price = float(top.get('price', 0))
        bottom_price = float(bottom.get('price', 0))

        if not top_price or not bottom_price:
            return 0.5

        # Calculate price ratio
        price_ratio = max(top_price, bottom_price) / min(top_price, bottom_price)

        # Define price coherence thresholds
        if price_ratio <= 1.2:  # Very close prices
            return 1.0
        elif price_ratio <= 1.5:  # Slightly different
            return 0.9
        elif price_ratio <= 2.0:  # Moderately different
            return 0.7
        elif price_ratio <= 3.0:  # Significantly different
            return 0.5
        else:  # Very different
            return 0.3

    def _calculate_fit_compatibility_score(
            self, product: pd.Series, user: Dict) -> float:
        """✅ NEW: Calculate fit compatibility score using professional intelligence."""
        user_body_shape = self.map_user_body_shape_to_designer(
            user.get('Body Shape', ''), user.get('Gender', '')
        )
        user_gender = user.get('Gender', '').strip().lower()

        if not user_body_shape or not user_gender:
            return 0.5

        gender_key = 'male' if user_gender in ['male', 'men'] else 'female'

        if gender_key not in self.body_shape_intelligence or \
           user_body_shape not in self.body_shape_intelligence[gender_key]:
            return 0.5

        body_rules = self.body_shape_intelligence[gender_key][user_body_shape]
        base_multiplier = body_rules.get('score_multiplier', 0.7)

        # Analyze product fit against professional guidance
        product_fit = product.get('fit_analysis', '')
        llava_fit = product.get('llava_fit_type', '')
        product_title = product.get('title', '')

        fit_text = f"{product_fit} {llava_fit} {product_title}".lower()

        if product.get('wear_type', '') == 'Upperwear':
            best_guidance = body_rules.get('best_tops', '').lower()
        else:
            best_guidance = body_rules.get('best_bottoms', '').lower()

        # Calculate match score
        match_score = 0.5
        if fit_text and best_guidance:
            guidance_words = [
                word for word in best_guidance.split() if len(word) > 3]
            matches = sum(1 for word in guidance_words if word in fit_text)
            if guidance_words:
                match_score = min(
                    0.3 + (matches / len(guidance_words)) * 0.7, 1.0)

        return match_score * base_multiplier

    def _normalize_comfort_level(self, comfort_level: str) -> float:
        """Convert comfort level to a numerical scale."""
        comfort_levels = {
            'Very Comfortable': 1.0,
            'Moderately Comfortable': 0.8,
            'Somewhat Comfortable': 0.6,
            'Not Comfortable': 0.4,
            'Very Uncomfortable': 0.2
        }
        return comfort_levels.get(comfort_level, 0.5)

    def _calculate_quality_metrics_score(
            self, top: pd.Series, bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate quality metrics score using professional standards."""
        quality_indicators = [
            top.get('quality_indicator1', ''),
            top.get('quality_indicator2', ''),
            bottom.get('quality_indicator1', ''),
            bottom.get('quality_indicator2', '')
        ]

        # Calculate base quality score
        quality_scores = [
            self._calculate_quality_indicator_score(indicator)
            for indicator in quality_indicators
            if indicator
        ]

        if not quality_scores:
            return 0.5

        base_score = sum(quality_scores) / len(quality_scores)

        # Apply additional quality checks
        durability_scores = [
            self._normalize_durability_level(
                top.get(
                    'durability_level',
                    'Medium')),
            self._normalize_durability_level(
                bottom.get(
                    'durability_level',
                    'Medium'))]

        durability_score = sum(durability_scores) / len(durability_scores)

        # Combine scores with weights
        final_score = (base_score * 0.7) + (durability_score * 0.3)

        return min(final_score, 1.0)

    def _normalize_durability_level(self, durability_level: str) -> float:
        """Convert durability level to a numerical score."""
        durability_levels = {
            'Very High': 1.0,
            'High': 0.8,
            'Medium': 0.6,
            'Low': 0.4,
            'Very Low': 0.2
        }
        return durability_levels.get(durability_level, 0.5)

    def _calculate_quality_indicator_score(self, indicator: str) -> float:
        """✅ NEW: Calculate quality indicator score using professional intelligence."""
        quality_thresholds = {
            'High': 1.0,
            'Medium': 0.8,
            'Low': 0.6,
            'Poor': 0.4,
            'Very Poor': 0.2
        }
        for threshold, score in quality_thresholds.items():
            if threshold in indicator:
                return score
        return 0.5

    def _calculate_occasion_appropriateness(
            self,
            top: pd.Series,
            bottom: pd.Series,
            user: Dict) -> float:
        """Enhanced occasion matching logic."""
        top_occasion = top.get('enhanced_occasion', top.get('occasion', ''))
        bottom_occasion = bottom.get(
            'enhanced_occasion', bottom.get(
                'occasion', ''))
        user_occasion = user.get('occasion_preference', 'Daily Activities')

        # Both match user preference
        if top_occasion == user_occasion and bottom_occasion == user_occasion:
            return 1.0

        # One matches user preference
        if top_occasion == user_occasion or bottom_occasion == user_occasion:
            return 0.8

        # Both same occasion (but not user's)
        if top_occasion == bottom_occasion:
            return 0.7

        # Different occasions
        return 0.5

    def _calculate_cultural_context(
            self,
            top: pd.Series,
            bottom: pd.Series,
            user: Dict) -> float:
        """✅ NEW: Calculate cultural context score (Indian skin tone awareness)."""
        score = 0.5  # Base score

        # Get current season for cultural favorites
        import datetime
        month = datetime.datetime.now().month

        if month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        elif month in [9, 10, 11]:
            season = 'Fall'
        else:
            season = 'Winter'

        cultural_favorites = self.seasonal_preferences[season].get(
            'cultural_favorites', [])

        # Check if colors are culturally favorable
        top_color = top.get('primary_color', '')
        bottom_color = bottom.get('primary_color', '')

        if top_color in cultural_favorites:
            score += 0.2
        if bottom_color in cultural_favorites:
            score += 0.2

        # Special cultural awareness from designer inputs
        # Olive Green is "very flattering on Indian skin"
        if 'Olive Green' in [top_color, bottom_color]:
            score += 0.3

        # Avoid combinations that "may not work for Indian skin tones"
        if top_color == 'Black' and bottom_color == 'Bright Red':
            score -= 0.3
        if bottom_color == 'Black' and top_color == 'Bright Red':
            score -= 0.3

        return min(max(score, 0.0), 1.0)

    def _calculate_versatility_score(
            self,
            top: pd.Series,
            bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate versatility score using professional metrics."""
        # Get versatility metrics
        top_versatility = top.get('versatility_analysis', '')
        bottom_versatility = bottom.get('versatility_analysis', '')

        # Calculate base versatility
        base_score = 0.5
        if top_versatility and bottom_versatility:
            top_versatility_list = [
                word for word in top_versatility.split() if word.isalpha()]
            bottom_versatility_list = [
                word for word in bottom_versatility.split() if word.isalpha()]

            if top_versatility_list and bottom_versatility_list:
                common_words = set(top_versatility_list) & set(
                    bottom_versatility_list)
                if common_words:
                    base_score = 0.5 + (len(common_words) * 0.1)

        # Apply style versatility rules
        top_style = top.get('enhanced_primary_style', '')
        bottom_style = bottom.get('enhanced_primary_style', '')

        # Check for versatile style combinations
        versatile_combinations = [
            ('Casual', 'Smart Casual'),
            ('Business Casual', 'Contemporary'),
            ('Contemporary', 'Casual'),
            ('Smart Casual', 'Business Casual')
        ]

        for style1, style2 in versatile_combinations:
            if (top_style == style1 and bottom_style == style2) or \
               (top_style == style2 and bottom_style == style1):
                base_score *= 1.2
                break

        # Check for color versatility
        top_color = top.get('primary_color', '')
        bottom_color = bottom.get('primary_color', '')

        neutral_colors = ['Black', 'White', 'Gray', 'Navy', 'Beige']
        if top_color in neutral_colors or bottom_color in neutral_colors:
            base_score *= 1.1

        return min(base_score, 1.0)

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
        """Get user's style preferences for proper distribution."""
        styles = []
        
        # ✅ FIX: Extract fashion style from user data properly
        fashion_style = user_data.get('Fashion Style', '')
        
        if fashion_style:
            # Handle JSON array format
            if isinstance(fashion_style, str) and fashion_style.startswith('[') and fashion_style.endswith(']'):
                try:
                    import json
                    style_list = json.loads(fashion_style)
                    if isinstance(style_list, list):
                        styles = [style.lower().strip() for style in style_list if style]
                except:
                    styles = [fashion_style.lower().strip()]
            else:
                # Handle comma-separated string
                styles = [style.lower().strip() for style in str(fashion_style).split(',') if style.strip()]
        
        # ✅ FIX: Also check apparel preferences as backup
        if not styles:
            if user_data.get('Apparel Pref Business Casual', False):
                styles.append('business casual')
            if user_data.get('Apparel Pref Streetwear', False):
                styles.append('streetwear')
            if user_data.get('Apparel Pref Athleisure', False):
                styles.append('athleisure')
        
        # ✅ FIX: If still no styles, use a default based on user's data
        if not styles:
            # Try to infer from other user data
            if user_data.get('occasion_preference', '').lower() in ['work', 'office', 'professional']:
                styles = ['business casual']
            elif user_data.get('occasion_preference', '').lower() in ['gym', 'sport', 'active']:
                styles = ['athleisure']
            else:
                styles = ['streetwear']  # Default to streetwear
        
        logger.info(f"🎨 User fashion styles: {styles}")
        return styles
    
    def _generate_diverse_outfits_with_style_distribution(self, 
                                                        products_df: pd.DataFrame, 
                                                        user_data: Dict, 
                                                        target_outfits: int,
                                                        user_styles: List[str]) -> List[Dict]:
        """Generate diverse outfits with proper style distribution and no duplicate products."""
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
            
            # Broader filtering using multiple fields
            filtered_products = products_df[
                # Check primary_style field
                products_df['primary_style'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check full_caption field
                products_df['full_caption'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check title field for style keywords
                products_df['title'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check category field
                products_df['category'].str.contains('|'.join(style_keywords), case=False, na=False)
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
                ~products_df['id'].isin(used_top_ids | used_bottom_ids)
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
                tops = tops[tops['id'] != top['id']]
                bottoms = bottoms[bottoms['id'] != bottom['id']]
                
                # Add to used sets
                used_top_ids.add(top['id'])
                used_bottom_ids.add(bottom['id'])
                
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
        """Filter products by style with strict matching to user's fashion style preference."""
        try:
            if products_df.empty:
                return pd.DataFrame()
            
            # ✅ FIX: More precise style mapping for exact matching
            style_mappings = {
                'business casual': ['business casual', 'business', 'casual', 'professional', 'office', 'work', 'smart casual'],
                'streetwear': ['streetwear', 'street', 'urban', 'contemporary', 'modern', 'hip-hop', 'urban style', 'edgy'],
                'athleisure': ['athleisure', 'active', 'sport', 'athletic', 'performance', 'gym', 'workout', 'sporty'],
                'casual': ['casual', 'relaxed', 'everyday', 'comfortable', 'informal'],
                'formal': ['formal', 'elegant', 'sophisticated', 'business formal', 'evening formal'],
                'contemporary': ['contemporary', 'modern', 'current', 'trendy', 'fashion-forward'],
                'vintage': ['vintage', 'retro', 'classic', 'heritage', 'nostalgic'],
                'bohemian': ['bohemian', 'boho', 'free-spirited', 'artistic', 'eclectic'],
                'minimalist': ['minimalist', 'minimal', 'simple', 'clean', 'essential']
            }
            
            # Get keywords for the target style
            style_keywords = style_mappings.get(target_style.lower(), [target_style.lower()])
            
            # ✅ FIX: More strict filtering using multiple fields
            filtered_products = products_df[
                # Check primary_style field
                products_df['primary_style'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check full_caption field
                products_df['full_caption'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check title field for style keywords
                products_df['title'].str.contains('|'.join(style_keywords), case=False, na=False) |
                # Check category field
                products_df['category'].str.contains('|'.join(style_keywords), case=False, na=False)
            ].copy()
            
            # ✅ FIX: Additional validation - ensure wear type is correct
            if target_style.lower() == 'business casual':
                # Business casual should not include very casual items like t-shirts for tops
                filtered_products = filtered_products[
                    ~((filtered_products['wear_type'] == 'Upperwear') & 
                      (filtered_products['title'].str.contains('t-shirt|tshirt|tee', case=False, na=False)))
                ]
            
            elif target_style.lower() == 'streetwear':
                # Streetwear should be more casual and urban
                filtered_products = filtered_products[
                    ~((filtered_products['wear_type'] == 'Upperwear') & 
                      (filtered_products['title'].str.contains('formal|business|office', case=False, na=False)))
                ]
            
            elif target_style.lower() == 'athleisure':
                # ✅ FIX: Much more inclusive athleisure filtering based on product analysis
                # Found 679 joggers, 548 joggers in product_type, 434 leggings, etc.
                athleisure_keywords = [
                    'sport', 'athletic', 'active', 'gym', 'workout', 'performance', 
                    'training', 'fitness', 'exercise', 'jogger', 'track', 'sweat', 
                    'hoodie', 'legging', 'sports', 'athleisure', 'joggers', 'leggings',
                    'training', 'fitness', 'exercise', 'performance', 'gym', 'workout',
                    'sporty', 'athletic', 'active', 'casual sport', 'sport casual'
                ]
                
                # Check multiple fields with broader matching
                filtered_products = products_df[
                    # Check primary_style field
                    products_df['primary_style'].str.contains('|'.join(athleisure_keywords), case=False, na=False) |
                    # Check full_caption field (found 2659 products here)
                    products_df['full_caption'].str.contains('|'.join(athleisure_keywords), case=False, na=False) |
                    # Check title field (found 1752 products here)
                    products_df['title'].str.contains('|'.join(athleisure_keywords), case=False, na=False) |
                    # Check category field (found 3115 products here)
                    products_df['category'].str.contains('|'.join(athleisure_keywords), case=False, na=False) |
                    # Check product_type field (found 548 joggers + 434 leggings)
                    products_df['product_type'].str.contains('jogger|legging|sport|athletic', case=False, na=False) |
                    # Check style_subcategory field
                    products_df['style_subcategory'].str.contains('|'.join(athleisure_keywords), case=False, na=False) |
                    # Check primary_occasion field for active/sport occasions
                    products_df['primary_occasion'].str.contains('sport|active|gym|workout|training|fitness', case=False, na=False) |
                    # Check activity_level field
                    products_df['activity_level'].str.contains('high|moderate', case=False, na=False) |
                    # Check venue_type field for gym/sport venues
                    products_df['venue_type'].str.contains('gym|sport|fitness|workout', case=False, na=False)
                ].copy()
                
                # Additional filtering for specific product types that are clearly athleisure
                athleisure_product_types = ['joggers', 'leggings', 'sports bra', 'athletic shorts', 'track pants']
                athleisure_by_type = products_df[
                    products_df['product_type'].isin(athleisure_product_types)
                ].copy()
                
                # Combine both filtered results
                filtered_products = pd.concat([filtered_products, athleisure_by_type]).drop_duplicates(subset=['id'])
                
                # Remove any products that are clearly NOT athleisure (formal wear)
                filtered_products = filtered_products[
                    ~((filtered_products['wear_type'] == 'Upperwear') & 
                      (filtered_products['title'].str.contains('formal|business|office|dress|blazer|suit', case=False, na=False)))
                ]
                
                # Remove any products that are clearly NOT athleisure (formal bottoms)
                filtered_products = filtered_products[
                    ~((filtered_products['wear_type'] == 'Bottomwear') & 
                      (filtered_products['title'].str.contains('formal|business|office|dress|skirt|trousers', case=False, na=False)))
                ]
            
            logger.info(f"🎨 Found {len(filtered_products)} products for style '{target_style}' using strict filtering")
            
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
            tops = tops[~tops['id'].isin(used_top_ids)]
            bottoms = bottoms[~bottoms['id'].isin(used_bottom_ids)]
            
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
                tops = tops[tops['id'] != top['id']]
                bottoms = bottoms[bottoms['id'] != bottom['id']]
                
                # Add to used sets
                used_top_ids.add(top['id'])
                used_bottom_ids.add(bottom['id'])
                
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
                top_id = top.get('id', '')
                bottom_id = bottom.get('id', '')
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
                'top_style': str(top.get('primary_style', '')),
                'bottom_style': str(bottom.get('primary_style', '')),
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

            # 🚫 FIRST: Completely clear ALL existing outfits for this user (including retry outfits)
            cleanup_success = self._clear_user_outfits_completely(user_id)
            if not cleanup_success:
                logger.error("❌ Failed to clear existing outfits, aborting save")
                return False
            
            logger.info("✅ Successfully cleared all existing outfits for user")
            
            # 🎯 SECOND: Use simple, clean IDs without retry suffixes
            processed_outfits = []
            for i, outfit in enumerate(outfits_data):
                # Use simple, clean main_outfit_id format
                clean_id = f"main_{user_id}_{i+1}"
                
                outfit_copy = outfit.copy()
                outfit_copy['main_outfit_id'] = clean_id
                # Ensure rank starts from 1 and is sequential
                outfit_copy['rank'] = i + 1
                processed_outfits.append(outfit_copy)

            # 🚀 THIRD: Use batch insert with proper error handling
            batch_size = 10  # Insert in smaller batches to avoid timeouts
            total_inserted = 0
            
            for i in range(0, len(processed_outfits), batch_size):
                batch = processed_outfits[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    logger.info(f"📦 Inserting batch {batch_num}: {len(batch)} outfits")
                    
                    # Convert batch to DataFrame for easier handling
                    batch_df = pd.DataFrame(batch)
                    
                    # Insert batch using database method
                    # Skip deletion for all batches since we already cleared outfits above
                    success = self.db.save_user_outfits(batch_df, user_id, skip_deletion=True)
                    
                    if success:
                        total_inserted += len(batch)
                        logger.info(f"✅ Batch {batch_num} inserted successfully")
                    else:
                        logger.error(f"❌ Batch {batch_num} failed to insert")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Error inserting batch {batch_num}: {e}")
                    return False

            logger.info(f"✅ Successfully saved {total_inserted}/{len(processed_outfits)} outfits to Supabase")
            return total_inserted == len(processed_outfits)

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
            import time
            
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

def main():
    """Main function to test the enhanced Supabase outfit generator."""
    try:
        # Initialize enhanced generator
        generator = SupabaseMainOutfitsGenerator()
        
        # Test with a user
        user_id = 2
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