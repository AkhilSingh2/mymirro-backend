# Phase 3 ENHANCED: Same-Category Similar Products with Supabase Integration
# Focus: Same product type with color/design diversity + user preference integration + Supabase DB

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
from database import get_db
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseEnhancedSimilarProductsGenerator:
    """
    Phase 3 ENHANCED: Same-category similar products with diversity, user preferences, and Supabase integration
    ✅ ENHANCED: Now with Supabase database integration + Advanced Fashion Intelligence
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the Supabase-enabled enhanced similar products generator."""
        self.config = config or self._default_config()
        
        # Railway CPU optimization - delay until needed
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
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
        self.db = get_db()
        if not self.db.test_connection():
            logger.error("❌ Database connection failed. Please check your Supabase configuration.")
            raise ConnectionError("Failed to connect to Supabase database")
        
        # Load model with CPU optimization
        try:
            if self.is_railway:
                logger.info("🔧 Loading model with Railway CPU optimizations")
            self.model = SentenceTransformer(self.config['model_name'])
            logger.info(f"✅ Model loaded: {self.config['model_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
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
            'outfit_context_weight': 2.8,    # NEW: Outfit context intelligence (Phase 1 derived)
            'brand_weight': 1.2,             # Brand consistency (lower for diversity)
            'price_weight': 2.0,             # Price similarity importance
            'user_preference_weight': 2.5,   # User preference boost
            'wear_caption_weight': 2.2,      # Lower/upper wear caption matching
            'diversity_bonus': 0.3,          # Bonus for diverse but relevant products
            'min_similarity_score': 0.35,    # Lowered for more outfit context candidates
            'max_similar_products': 20,      # Maximum products to return
            'color_diversity_threshold': 0.7 # Threshold for color diversity bonus
        }
        
        # Initialize enhanced mappings (focused on same-category)
        self.color_diversity_matrix = self._initialize_color_diversity_matrix()
        self.style_compatibility = self._initialize_same_category_style_compatibility()
        self.design_variation_keywords = self._initialize_design_variation_keywords()
        
        # Load context mappings (outfit intelligence) if available
        self.context_mappings = {}
        
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
            'diversity_threshold': 0.75,      # Similarity threshold for diversity filtering
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
    
    def load_products_from_supabase(self) -> pd.DataFrame:
        """Load products data from Supabase database."""
        try:
            logger.info("Loading products from Supabase for Phase 3...")
            
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
    
    def validate_products_data(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation with price integration."""
        required_columns = ["title", "wear_type"]
        missing_columns = [col for col in required_columns if col not in products_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        products_df = products_df.dropna(subset=["title", "wear_type"])
        
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
        # Apply Railway CPU optimization before heavy computation
        if self.is_railway:
            for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                os.environ[var] = '2'
            logger.info("🔧 Applied CPU limits for similar products computation")
        """Enhanced same-category similar products with diversity using Supabase."""
        
        try:
            # Load and validate products from Supabase
            products_df = self.load_products_from_supabase()
            
            if products_df.empty:
                logger.error("No products available from Supabase")
                return []
            
            products_df = self.validate_products_data(products_df)
            
            # Get source product
            source_product = self.get_product_by_id(product_id, products_df)
            logger.info(f"Finding similar products for: {source_product.get('title', 'Unknown')}")
            
            # Build FAISS indexes
            self.build_faiss_indexes(products_df)
            
            # Generate same-category candidates only
            candidates = self._generate_same_category_candidates(source_product, products_df, user_preferences)
            
            if not candidates:
                logger.warning("No candidate products found")
                return []
            
            # Apply enhanced scoring and filtering for diversity
            similar_products = self._score_and_filter_same_category_candidates(
                source_product, candidates, user_preferences, filters, num_similar
            )
            
            logger.info(f"Found {len(similar_products)} diverse same-category products")
            return similar_products
            
        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            raise
    
    def _generate_same_category_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                         user_preferences: Dict = None) -> List[Dict]:
        """Generate candidates from same category only with diversity focus."""
        
        source_wear_type = source_product.get('wear_type', '')
        source_text = source_product.get('final_caption', '') or source_product.get('title', '')
        
        all_candidates = []
        
        # 1. Core similar products (same category, similar features)
        core_candidates = self.search_similar_products_faiss(
            source_text, source_wear_type, k=40  # More candidates for better diversity
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
        source_wear_type = source_product.get('wear_type', '')
        
        if source_color in self.color_diversity_matrix:
            diverse_colors = self.color_diversity_matrix[source_color]['diverse_colors']
            
            for color in diverse_colors:
                # Search for products with diverse colors in same category
                color_query = f"{color} {source_wear_type}"
                color_candidates = self.search_similar_products_faiss(
                    color_query, source_wear_type, k=8
                )
                for candidate in color_candidates:
                    candidate['candidate_type'] = 'color_diverse'
                    candidate['boost_factor'] = 1.1  # Slight boost for color diversity
                candidates.extend(color_candidates)
        
        return candidates[:15]  # Limit color diverse candidates
    
    def _get_design_diverse_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                     user_preferences: Dict = None) -> List[Dict]:
        """Get design-diverse candidates within same category."""
        
        candidates = []
        source_title = source_product.get('title', '').lower()
        source_wear_type = source_product.get('wear_type', '')
        
        # Identify source design characteristics
        source_patterns = self._identify_design_patterns(source_title)
        
        # Search for different design patterns in same category
        for pattern_type, patterns in self.design_variation_keywords['patterns'].items():
            if pattern_type not in source_patterns:  # Different from source
                for pattern in patterns:
                    design_query = f"{pattern} {source_wear_type}"
                    design_candidates = self.search_similar_products_faiss(
                        design_query, source_wear_type, k=5
                    )
                    for candidate in design_candidates:
                        candidate['candidate_type'] = 'design_diverse'
                        candidate['boost_factor'] = 1.05  # Small boost for design diversity
                    candidates.extend(design_candidates)
        
        return candidates[:10]  # Limit design diverse candidates
    
    def _get_user_preference_candidates(self, source_product: pd.Series, products_df: pd.DataFrame, 
                                      user_preferences: Dict) -> List[Dict]:
        """Get candidates based on user preferences within same category."""
        
        candidates = []
        source_wear_type = source_product.get('wear_type', '')
        
        # User preferred styles within same category
        preferred_styles = user_preferences.get('preferred_styles', [])
        for style in preferred_styles:
            style_query = f"{style} {source_wear_type}"
            style_candidates = self.search_similar_products_faiss(
                style_query, source_wear_type, k=6
            )
            for candidate in style_candidates:
                candidate['candidate_type'] = 'user_preference'
                candidate['boost_factor'] = 1.15  # Boost for user preferences
            candidates.extend(style_candidates)
        
        # User preferred colors within same category
        preferred_colors = user_preferences.get('preferred_colors', [])
        for color in preferred_colors:
            color_query = f"{color} {source_wear_type}"
            color_candidates = self.search_similar_products_faiss(
                color_query, source_wear_type, k=6
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
            
            # Skip duplicates and source product
            if candidate_id in seen_products or candidate_id == source_product_id:
                continue
            
            # Apply filters
            if filters and not self._passes_filters(candidate_product, filters):
                continue
            
            # Calculate enhanced similarity score with diversity factors
            similarity_score, score_breakdown = self._calculate_enhanced_same_category_similarity(
                source_product, candidate_product, candidate, user_preferences
            )
            
            # Apply minimum threshold
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
                'wear_type': candidate_product.get('wear_type', ''),
                'occasion': candidate_product.get('enhanced_occasion', candidate_product.get('occasion', '')),
                'similarity_score': similarity_score,
                'score_breakdown': score_breakdown,
                'candidate_type': candidate.get('candidate_type', 'unknown'),
                'source_product_id': source_product_id,
                'generated_at': datetime.now().isoformat()
            })
            
            seen_products.add(candidate_id)
        
        # Apply diversity filtering to avoid too similar products
        diverse_products = self._apply_same_category_diversity_filtering(similar_products, source_product)
        
        # Sort by similarity score and return top matches
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
        
        # Price similarity
        price_score = self._calculate_price_similarity_enhanced(
            source_product.get('price', 1000),
            candidate_product.get('price', 1000)
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
            price_score * weights['price_weight'] +
            preference_score * weights['user_preference_weight']
        ) * type_boost
        
        # Normalize to 0-1 range
        total_weight = (weights['semantic_weight'] + weights['style_weight'] + 
                       weights['color_diversity_weight'] + weights['design_diversity_weight'] + 
                       weights['price_weight'] + weights['user_preference_weight'])
        
        final_score = min(final_score / total_weight, 1.0)
        
        score_breakdown = {
            'semantic_similarity': semantic_score,
            'style_compatibility': style_score,
            'color_diversity': color_diversity_score,
            'design_diversity': design_diversity_score,
            'price_similarity': price_score,
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
    
    def _calculate_price_similarity_enhanced(self, price1: float, price2: float) -> float:
        """Enhanced price similarity with dynamic tolerance."""
        if not price1 or not price2:
            return 0.5
        
        # Calculate percentage difference
        avg_price = (price1 + price2) / 2
        price_diff = abs(price1 - price2) / avg_price
        
        # Dynamic tolerance based on price range
        if avg_price < 1000:
            tolerance = 0.5  # Higher tolerance for lower prices
        elif avg_price < 2000:
            tolerance = 0.4
        else:
            tolerance = 0.3  # Stricter for higher prices
        
        if price_diff <= tolerance:
            return 1.0 - (price_diff / tolerance) * 0.3  # 0.7 to 1.0 range
        else:
            return max(0.2, 0.7 - price_diff)  # Minimum 0.2
    
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
        
        # Price range preferences
        preferred_price_range = user_preferences.get('price_range')
        if preferred_price_range:
            min_price, max_price = preferred_price_range
            product_price = product.get('price', 1000)
            if min_price <= product_price <= max_price:
                boost += 0.1
        
        return min(boost, 1.4)  # Cap at 1.4x boost
    
    def _apply_same_category_diversity_filtering(self, similar_products: List[Dict], 
                                               source_product: pd.Series) -> List[Dict]:
        """Apply diversity filtering for same-category products."""
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
        
        # Price range filter
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            product_price = product.get('price', 1000)
            if not (min_price <= product_price <= max_price):
                return False
        
        # Style filter
        if 'styles' in filters:
            product_style = product.get('enhanced_primary_style', product.get('primary_style', ''))
            if not any(style.lower() in product_style.lower() for style in filters['styles']):
                return False
        
        # Color filter
        if 'colors' in filters:
            product_color = product.get('primary_color', '')
            if product_color not in filters['colors']:
                return False
        
        return True
    
    # Include necessary methods from original implementation
    def build_faiss_indexes(self, products_df: pd.DataFrame) -> None:
        """Build FAISS indexes for different wear types."""
        logger.info("Building FAISS indexes for enhanced same-category search...")
        
        # Railway CPU optimization for FAISS indexing
        if self.is_railway:
            logger.info("🏭 Applying Railway CPU limits for FAISS indexing operations")
            for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                os.environ[var] = '1'  # Extra conservative for FAISS operations
        
        wear_types = products_df['wear_type'].unique()
        
        for wear_type in wear_types:
            wear_products = products_df[products_df['wear_type'] == wear_type].copy()
            
            if wear_products.empty:
                continue
            
            captions = []
            product_indices = []
            
            for idx, row in wear_products.iterrows():
                caption = row.get('final_caption', '') or row.get('title', '')
                if caption.strip():
                    captions.append(caption)
                    product_indices.append(idx)
            
            if not captions:
                continue
            
            logger.info(f"Generating embeddings for {len(captions)} {wear_type} products...")
            
            embeddings = []
            for caption in captions:
                embedding = self.get_embedding_cached(caption)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            
            self.faiss_indexes[wear_type] = index
            self.product_mappings[wear_type] = {
                'indices': product_indices,
                'products': wear_products.loc[product_indices]
            }
            
            logger.info(f"Built FAISS index for {wear_type}: {len(captions)} products indexed")
    
    def get_embedding_cached(self, text: str, cache_key: str = None) -> np.ndarray:
        """Get embedding with caching."""
        if not cache_key:
            cache_key = text[:100]
        
        if self.config['cache_embeddings'] and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.model.encode([text])[0]
        
        if self.config['cache_embeddings']:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_product_by_id(self, product_id: str, products_df: pd.DataFrame) -> pd.Series:
        """Get product by ID with improved column handling."""
        # Try with different column names
        for col in ['product_id', 'id']:
            if col in products_df.columns:
                # Try exact match first
                product = products_df[products_df[col].astype(str) == str(product_id)]
                if not product.empty:
                    return product.iloc[0]
                
                # Try numeric match if possible
                try:
                    numeric_id = int(product_id)
                    product = products_df[products_df[col] == numeric_id]
                    if not product.empty:
                        return product.iloc[0]
                except ValueError:
                    pass
        
        raise ValueError(f"Product with ID '{product_id}' not found")
    
    def search_similar_products_faiss(self, query_text: str, wear_type: str, k: int = 20) -> List[Dict]:
        """Search for similar products using FAISS."""
        
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

def main():
    """Test the Supabase enhanced same-category similar products generator."""
    
    # Initialize enhanced generator
    generator = SupabaseEnhancedSimilarProductsGenerator()
    
    # Sample user preferences
    user_preferences = {
        'preferred_styles': ['Business Formal', 'Formal'],
        'preferred_colors': ['Black', 'Navy', 'White'],
        'price_range': (800, 2500)
    }
    
    # Sample filters
    filters = {
        'price_range': (500, 3000),
        'styles': ['Business', 'Formal'],
    }
    
    # Test with enhanced same-category features
    test_product_id = "3790"
    
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
                print(f"   Color: {product['color']} | Wear: {product['wear_type']}")
                
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

if __name__ == "__main__":
    main() 