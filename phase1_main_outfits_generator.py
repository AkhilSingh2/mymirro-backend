# Phase 1: Main Outfits Generator (Pre-computed)
# This generates 100 main outfits per user and saves to CSV for API consumption

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MainOutfitsGenerator:
    """
    Phase 1: Generate and store 100 main outfits per user for fast API access
    ✅ ENHANCED: Now with Professional Fashion Designer Intelligence
    """

    def __init__(self, config: Dict = None):
        """Initialize the main outfits generator with fashion designer intelligence."""
        self.config = config or self._default_config()

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

        # Load color harmony from designer CSV
        self.color_harmony_map = self._load_color_harmony_from_csv()

        # ✅ ENHANCED: Professional fashion designer intelligence
        self.color_harmony = self._initialize_professional_color_harmony()
        self.quick_rules = self._initialize_professional_quick_rules()
        self.body_shape_intelligence = self._initialize_body_shape_intelligence()

        # ✅ ENHANCED: Professional scoring weights with cultural context
        self.scoring_weights = {
            'semantic_similarity': 3.0,           # Core AI matching
            # NEW: Based on fit_confidence and body_shape_compatibility
            'fit_compatibility': 2.8,
            # NEW: Based on comfort_level and movement_restriction
            'comfort_metrics': 2.5,
            'style_intelligence': 2.5,           # ENHANCED: Using new style attributes
            'color_harmony': 2.3,                # ENHANCED: Using new color attributes
            # NEW: Based on quality_indicators and durability
            'quality_metrics': 2.0,
            'occasion_context': 2.0,             # ENHANCED: Using detailed occasion attributes
            'cultural_relevance': 1.8,           # ENHANCED: Better cultural context matching
            # ENHANCED: Using style_versatility and adaptability
            'versatility_score': 1.5,
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

        # ✅ ENHANCED: Seasonal intelligence with cultural context
        self.seasonal_preferences = {
            'Spring': {
                'colors': ['Pink', 'Light Blue', 'Yellow', 'Green', 'White'],
                'fabrics': ['Cotton', 'Linen', 'Silk'],
                'patterns': ['Floral', 'Stripe', 'Geometric'],
                # ✅ NEW
                'cultural_favorites': ['Sage Green', 'Light Pink', 'White']
            },
            'Summer': {
                'colors': ['White', 'Light Blue', 'Yellow', 'Pink', 'Orange'],
                'fabrics': ['Cotton', 'Linen', 'Chambray'],
                'patterns': ['Solid', 'Stripe', 'Small Print'],
                # ✅ NEW
                'cultural_favorites': ['White', 'Sky Blue', 'Mint Green']
            },
            'Fall': {
                'colors': ['Brown', 'Orange', 'Navy', 'Burgundy', 'Green'],
                'fabrics': ['Wool', 'Cotton', 'Denim'],
                'patterns': ['Plaid', 'Check', 'Solid'],
                # ✅ NEW
                'cultural_favorites': ['Navy', 'Olive Green', 'Dark Brown']
            },
            'Winter': {
                'colors': ['Black', 'Gray', 'Navy', 'Burgundy', 'Purple'],
                'fabrics': ['Wool', 'Cashmere', 'Thick Cotton'],
                'patterns': ['Solid', 'Check', 'Herringbone'],
                'cultural_favorites': ['Black', 'Navy', 'Maroon']  # ✅ NEW
            }
        }

    def _default_config(self) -> Dict:
        """Default configuration for the main outfits generator."""
        return {
            'model_name': "all-MiniLM-L6-v2",
            'query_expansion': True,
            'reranking_enabled': True,
            'cache_embeddings': True,
            'user_file': "data/User_Data.xlsx",
            'products_file': "runpod_captions_combined_tagged_cleaned.csv",
            'output_dir': "data/user_recommendations",
            'main_outfits_count': 50,  # Changed from 100 to 50
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
                f"✅ Loaded {
                    len(color_harmony_map)} color harmony rules from designer CSV")
        except Exception as e:
            logger.warning(f"Could not load color harmony CSV: {e}")
        return color_harmony_map

    def _initialize_professional_color_harmony(self) -> Dict:
        """✅ NEW: Initialize professional color harmony with expert ratings and cultural context."""
        return {
            # ✅ PROFESSIONAL: Based on fashion designer ratings (1-10 scale)
            'Black': {
                'perfect': [
                    {'color': 'Deep Red', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Maroon', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Navy Blue', 'rating': 9, 'context': 'Elegant'},
                    {'color': 'Emerald Green', 'rating': 9, 'context': 'Rich look'},
                    {'color': 'Olive Green', 'rating': 9,
                        'context': 'New n fresh, very flattering on Indian skin'}
                ],
                'excellent': [
                    {'color': 'Light Pink', 'rating': 8,
                        'context': 'Soft, delicate'},
                    {'color': 'Mint Green', 'rating': 8,
                        'context': 'Very clean look, breezy'},
                    {'color': 'Sage Green', 'rating': 8,
                        'context': 'Very clean look, breezy'},
                    {'color': 'Sky Blue', 'rating': 8, 'context': 'Casual formal'},
                    {'color': 'Cobalt Blue', 'rating': 8,
                        'context': 'Bold with a pop of color yet not too bright'},
                    {'color': 'Baby Blue', 'rating': 8,
                        'context': 'Summer, cool tones, breezy'},
                    {'color': 'Charcoal Gray', 'rating': 8,
                        'context': 'Close to monochrome'},
                    {'color': 'Dark Brown', 'rating': 8,
                        'context': 'New look, very rich'}
                ],
                'good': [
                    {'color': 'Medium Pink', 'rating': 7,
                        'context': 'Pop of color'},
                    {'color': 'Mustard Yellow', 'rating': 7,
                        'context': 'Occasionally'},
                    {'color': 'Light Brown', 'rating': 7,
                        'context': 'Not in formals, only in casuals'},
                    {'color': 'Tan', 'rating': 7,
                        'context': 'Not in formals, only in casuals'},
                    {'color': 'Light Gray', 'rating': 6,
                        'context': 'Not in formals, only in casuals'}
                ],
                'avoid': [
                    {'color': 'Hot Pink', 'rating': 6,
                        'context': 'Bold choice, not accepted by majority of audience'},
                    {'color': 'Bright Red', 'rating': 5,
                        'context': 'Bold, may not work for Indian skin tones'},
                    {'color': 'Light Red', 'rating': 4, 'context': 'Too dull'},
                    {'color': 'Bright Yellow', 'rating': 3, 'context': 'Too bold'},
                    {'color': 'White', 'rating': 1, 'context': 'Uniform look'}
                ]
            },
            'White': {
                'perfect': [
                    {'color': 'Light Pink', 'rating': 9, 'context': 'Dreamy'},
                    {'color': 'Bright Red', 'rating': 9,
                        'context': 'Classic contrast'},
                    {'color': 'Mint Green', 'rating': 9,
                        'context': 'Spring, soft, light, calm'},
                    {'color': 'Emerald Green', 'rating': 9, 'context': 'Statement'},
                    {'color': 'Sky Blue', 'rating': 9,
                        'context': 'Best summer combo'},
                    {'color': 'Navy Blue', 'rating': 9, 'context': 'Classy'},
                    {'color': 'Baby Blue', 'rating': 9, 'context': 'Summer'}
                ],
                'excellent': [
                    {'color': 'Maroon', 'rating': 8,
                        'context': 'Elegant n formal'},
                    {'color': 'Lemon Yellow', 'rating': 8, 'context': 'Summer'},
                    {'color': 'Sage Green', 'rating': 8, 'context': 'Classy'},
                    {'color': 'Cobalt Blue', 'rating': 8,
                        'context': 'Bold contrast'},
                    {'color': 'Light Gray', 'rating': 8, 'context': 'New look'},
                    {'color': 'Dark Brown', 'rating': 8, 'context': 'New look'}
                ],
                'good': [
                    {'color': 'Medium Pink', 'rating': 7, 'context': 'New'},
                    {'color': 'Hot Pink', 'rating': 7, 'context': 'Bold contrast'},
                    {'color': 'Bright Yellow', 'rating': 7,
                        'context': 'Bold summer look'},
                    {'color': 'Olive Green', 'rating': 7, 'context': 'Natural'},
                    {'color': 'Charcoal Gray', 'rating': 7,
                        'context': 'Might be dull'},
                    {'color': 'Camel', 'rating': 7, 'context': 'Subjective to use'}
                ],
                'avoid': [
                    {'color': 'Mustard Yellow', 'rating': 6,
                        'context': 'Might be dull'},
                    {'color': 'Tan', 'rating': 6, 'context': 'Subjective to use'},
                    {'color': 'Light Red', 'rating': 5, 'context': 'Subtle'}
                ]
            },
            'Navy': {
                'perfect': [
                    {'color': 'Emerald Green', 'rating': 9,
                        'context': 'Bold yet classy'},
                    {'color': 'White', 'rating': 8, 'context': 'Rich look'},
                    {'color': 'Gray', 'rating': 8, 'context': 'Elegant n new'},
                    {'color': 'Black', 'rating': 8, 'context': 'Classy'},
                    {'color': 'Burgundy', 'rating': 8,
                        'context': 'New, rich, colorful'},
                    {'color': 'Olive Green', 'rating': 8, 'context': 'New look'}
                ],
                'excellent': [
                    {'color': 'Light Pink', 'rating': 7,
                        'context': 'Rare, if put together right then good'},
                    {'color': 'Lemon Yellow', 'rating': 7,
                        'context': 'Fresh contrast'},
                    {'color': 'Mustard Yellow', 'rating': 7,
                        'context': 'Vintage look'},
                    {'color': 'Light Blue', 'rating': 7,
                        'context': 'Not in formals'},
                    {'color': 'Sky Blue', 'rating': 7,
                        'context': 'Not in formals'},
                    {'color': 'Cobalt Blue', 'rating': 7,
                        'context': 'Not in formals'},
                    {'color': 'Camel', 'rating': 7, 'context': 'Not in formals'},
                    {'color': 'Tan', 'rating': 7, 'context': 'Not in formals'}
                ],
                'good': [
                    {'color': 'Medium Pink', 'rating': 6,
                        'context': 'Bold, for highstreet'},
                    {'color': 'Hot Pink', 'rating': 6,
                        'context': 'Bold, for highstreet'}
                ],
                'avoid': [
                    {'color': 'Bright Red', 'rating': 4,
                        'context': 'Too much contrast'}
                ]
            }
        }

    def _initialize_professional_quick_rules(self) -> Dict:
        """✅ NEW: Initialize professional quick rules from fashion designer."""
        # Parse quick rules from designer CSV
        quick_rules = {
            'validated_rules': {
                'all_black_everything': {
                    'allowed': True,
                    'rating': 8,
                    'condition': 'Safe play for majority. Elevate with textures, patterns, shapes.',
                    'enhancement': 'Play with textures, patterns, shape to make it more fashionable'
                },
                'all_white_everything': {
                    'allowed': False,
                    'rating': 2,
                    'condition': 'Uniforms are all white, not a fashion choice',
                    'alternative': 'Add color accents or textural elements'
                },
                'brown_and_black': {
                    'allowed': True,
                    'rating': 8,
                    'condition': 'High end fashion combination',
                    'context': 'sophisticated, modern'
                },
                'pattern_mixing': {
                    'allowed': False,
                    'rating': 3,
                    'condition': 'Too many colors in small space, makes individual look heavier',
                    'cultural_context': 'Not ideal for Indian audience from western wear pov'
                },
                'fit_balance': {
                    'rule': 'tight_top_loose_bottom',
                    'allowed': True,
                    'rating': 9,
                    'condition': 'Balances body shape. Tight fitted top n bottom only works for females bodycon/athleisure, not for men'
                }
            },
            'never_break_rules': [
                'Color balance',
                'Mindfully accessorised',
                'Top wear and bottom wear should never have same silhouette',
                'Every season has its own fashion, dont mix seasons',
                'Right sizing - avoid too tight or too loose'
            ],
            'biggest_mistakes': [
                'Skinny fit everywhere',
                'White shirt and black pants combination',
                'Knee high boots in summer',
                'Animal prints overuse',
                'Clashing colors or too many colors together',
                'Wrong sizing clothing',
                'Wearing Oversized Everything',
                'Too Many Statement Pieces Together',
                'Activewear as All-Daywear',
                'Formals for casual outings like brunch, shopping, etc',
                'no to oversized topwear n  oversized bottomwear at the same time, have sone definition or division',
                'Accessories should not be in the exact color as your outfit'
            ],
            'quick_fixes': {
                'instant_improvement': 'Add accessories',
                'universal_elevator': 'Good handbag or jewellery',
                'boring_outfit_fix': 'Add pop of color in minimal way - belt, shoes, handbag, earrings',
                'rushed_formal': 'Statement/loose shirt and pants OR tailored shirt and loose pants with good belt',
                'most_versatile': 'Blue denims',
                'emergency_combo': {
                    'casual': 'Graphic tee + shorts',
                    'formal': 'Shirt (not white) + black/dark blue pants (not denims)'
                }
            }
        }
        return quick_rules

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
                f"Invalid gender: {
                    user_data['Gender']}. Must be one of {valid_genders}")

        # Add confidence scores for user preferences
        confidence_scores = {
            'body_shape': 1.0 if 'Body Shape' in user_data else 0.7,
            'style_preference': 1.0 if 'Style Preference' in user_data else 0.7,
            'occasion': 1.0 if 'Occasion' in user_data else 0.7,
            'budget': 1.0 if 'Budget Preference' in user_data else 0.7}

        user_data['confidence_scores'] = confidence_scores
        logger.info(f"User data validation complete. Average confidence: {
                    sum(confidence_scores.values()) / len(confidence_scores):.2f}")

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

    def validate_products_data(
            self, products_df: pd.DataFrame) -> pd.DataFrame:
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
            'missing_category': products_df['category'].isna().sum()
        }

        # Drop rows with missing critical fields
        products_df = products_df.dropna(subset=['title', 'category'])

        # Infer wear_type from category and other information
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

        # Try to load price data
        try:
            price_df = pd.read_csv('data/products_scraped.csv')
            if 'id' in price_df.columns and 'price' in price_df.columns:
                products_df = products_df.merge(
                    price_df[['id', 'price']],
                    on='id',
                    how='left'
                )
                products_df['actual_price'] = products_df['price']
            else:
                logger.warning("Price data file missing required columns")
                products_df['actual_price'] = None
        except Exception as e:
            logger.warning(f"Could not load price data: {str(e)}")
            products_df['actual_price'] = None

        # Smart default pricing based on product context
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
            if row['wear_type'] == 'top':
                base_price *= 1.2  # Tops generally cost more than bottoms

            return round(base_price)

        # Apply smart pricing
        products_df['price'] = products_df.apply(
            get_smart_default_price, axis=1)

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
            products_df['product_id'] = [
                f"PROD_{
                    i:06d}" for i in range(
                    len(products_df))]

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

        logger.info(f"Product validation complete. Average quality score: {
                    products_df['data_quality_score'].mean():.2f}")
        logger.info(f"Price range: ₹{products_df['price'].min(
        ):.0f} - ₹{products_df['price'].max():.0f}")

        return products_df

    def get_embedding_cached(
            self,
            text: str,
            cache_key: str = None) -> np.ndarray:
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
        """ENHANCED: Calculate color harmony score using designer's CSV ratings if available."""
        if not color1 or not color2:
            return 0.5
        color1 = color1.strip().title()
        color2 = color2.strip().title()
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
        shape = user_body_shape.strip().lower()
        gender = user_gender.strip().lower()
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

    def filter_products_enhanced(
            self,
            products_df: pd.DataFrame,
            user: Dict,
            wear_type: str = None) -> pd.DataFrame:
        """Enhanced manual filtering on Gender, Fashion Style, and Body Shape."""
        logger.info(
            f"Starting filtering - Initial products: {len(products_df)}")

        def match(row):
            # 1. GENDER FILTERING (Strict)
            user_gender = user.get('Gender', user.get('gender', ''))
            product_gender = row.get('gender', 'Unisex')

            if user_gender and product_gender:
                if user_gender in ['Male', 'Men']:
                    acceptable_genders = ['Men', 'Male', 'Unisex']
                elif user_gender in ['Female', 'Women']:
                    acceptable_genders = ['Women', 'Female', 'Unisex']
                else:
                    acceptable_genders = [user_gender, 'Unisex']

                if product_gender not in acceptable_genders:
                    return False

            # 2. FASHION STYLE FILTERING (Flexible)
            user_style = user.get('Fashion Style', '').strip()
            product_style = row.get(
                'enhanced_primary_style', row.get(
                    'primary_style', '')).strip()

            if user_style and product_style:
                style_compatibility = {
                    'Streetwear': [
                        'Streetwear', 'Casual', 'Contemporary', 'Activewear', 'Athleisure'], 'Athleisure': [
                        'Athleisure', 'Activewear', 'Streetwear', 'Casual', 'Contemporary'], 'Contemporary': [
                        'Contemporary', 'Business Casual', 'Smart Casual', 'Casual'], 'Business': [
                        'Business', 'Business Formal', 'Business Casual', 'Professional'], 'Formal': [
                        'Formal', 'Business Formal', 'Evening Formal', 'Ultra Formal']}

                compatible_styles = style_compatibility.get(
                    user_style, [user_style])
                style_match = any(compatible_style.lower() in product_style.lower(
                ) for compatible_style in compatible_styles)

                if not style_match:
                    return False

            # 3. ✅ ENHANCED: Professional Body Shape Filtering
            user_body_shape = self.map_user_body_shape_to_designer(
                user.get('Body Shape', ''), user.get('Gender', '')
            )
            user_gender = user.get('Gender', '').strip().lower()

            if user_body_shape:
                product_fit = row.get('fit_analysis', '')
                llava_fit = row.get('llava_fit_type', '')
                parsed_specs = str(row.get('parsed_specs', ''))

                body_shape_fits = {
                    'Rectangle': ['Regular Fit', 'Tailored', 'Fitted', 'Straight'],
                    'Hourglass': ['Fitted', 'Bodycon', 'Wrap', 'Tailored'],
                    'Inverted Triangle': ['Regular Fit', 'Relaxed', 'Straight', 'A-line'],
                    'Pear': ['A-line', 'Straight', 'Regular Fit', 'Flared'],
                    'Apple': ['A-line', 'Wrap', 'Regular Fit', 'Empire']
                }

                suitable_fits = body_shape_fits.get(user_body_shape, [])

                if suitable_fits:
                    fit_text = f"{product_fit} {llava_fit} {parsed_specs}".lower()
                    fit_match = any(fit.lower() in fit_text for fit in suitable_fits)

                    if not fit_text.strip() or len(fit_text.strip()) < 10:
                        fit_match = True

                    if not fit_match:
                        return False

            # 4. WEAR TYPE FILTERING (if specified)
            if wear_type:
                if row.get('wear_type', '') != wear_type:
                    return False
                
                # 5. WINTER UPPERWEAR FILTERING (for upperwear only)
                if wear_type == 'Upperwear':
                    title = row.get('title', '').lower()
                    style = row.get('enhanced_primary_style', row.get('primary_style', '')).lower()
                    
                    # List of winter upperwear keywords (excluding cardigans and pullovers)
                    winter_keywords = [
                        'jacket', 'sweater', 'hoodie', 'sweatshirt',
                        'jumper', 'fleece', 'thermal', 'winter',
                        'wool', 'knit', 'quilted', 'padded', 'insulated'
                    ]
                    
                    # Check if any winter keyword is in title or style
                    if any(keyword in title or keyword in style for keyword in winter_keywords):
                        return False

            return True

        filtered_df = products_df[products_df.apply(match, axis=1)]

        logger.info(
            f"Filtering complete - Remaining products: {
                len(filtered_df)} " f"(Reduction: {
                len(products_df) -
                len(filtered_df)} products)")

        return filtered_df

    def build_faiss_indexes(self, products_df: pd.DataFrame) -> None:
        """Build FAISS indexes for different wear types."""
        logger.info("Building FAISS indexes for product recommendations...")

        wear_types = ['Upperwear', 'Bottomwear']

        for wear_type in wear_types:
            wear_products = products_df[products_df['wear_type'] == wear_type].copy(
            )

            if wear_products.empty:
                logger.warning(f"No products found for wear_type: {wear_type}")
                continue

            captions = []
            product_indices = []

            for idx, row in wear_products.iterrows():
                caption = row.get(
                    'final_caption',
                    '') or row.get(
                    'full_caption',
                    '') or row.get(
                    'title',
                    '')
                if caption.strip():
                    captions.append(caption)
                    product_indices.append(idx)

            if not captions:
                logger.warning(
                    f"No valid captions found for wear_type: {wear_type}")
                continue

            logger.info(
                f"Generating embeddings for {
                    len(captions)} {wear_type} products...")

            embeddings = []
            for caption in captions:
                embedding = self.get_embedding_cached(caption)
                embeddings.append(embedding)
            embeddings = np.array(embeddings)

            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)

            # Normalize embeddings for cosine similarity
            embeddings = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)
            index.add(embeddings.astype('float32'))

            # Store index and mapping
            self.faiss_indexes[wear_type] = index
            self.product_mappings[wear_type] = {'indices': product_indices, 'products': wear_products.iloc[[
                wear_products.index.get_loc(idx) for idx in product_indices]].copy()}

            logger.info(
                f"Built FAISS index for {wear_type}: {
                    len(captions)} products indexed")

    def get_semantic_recommendations(
            self,
            user_profile: str,
            wear_type: str,
            gender_filter: str = None,
            k: int = 20,
            user: Dict = None) -> List[Dict]:
        """✅ ENHANCED: Get semantic recommendations using FAISS with improved diversity and context awareness."""

        if wear_type not in self.faiss_indexes:
            logger.warning(
                f"No FAISS index available for wear_type: {wear_type}")
            return []

        # Enhanced query expansion with more context
        if user and self.config.get('query_expansion', False):
            expanded_profile = self.expand_user_query(
                user_profile, user, wear_type)
            query_embedding = self.get_embedding_cached(expanded_profile)
        else:
            query_embedding = self.get_embedding_cached(user_profile)

        # Search FAISS index with increased k for diversity
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding / \
            np.linalg.norm(query_embedding, axis=1, keepdims=True)

        search_k = k * 3  # Get more candidates for diversity
        index = self.faiss_indexes[wear_type]
        scores, indices = index.search(
            query_embedding.astype('float32'), search_k)

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
                product_gender = product.get('gender', 'Unisex')

                if gender_filter in ['Male', 'Men']:
                    acceptable_genders = ['Men', 'Male', 'Unisex']
                elif gender_filter in ['Female', 'Women']:
                    acceptable_genders = ['Women', 'Female', 'Unisex']
                else:
                    acceptable_genders = [gender_filter, 'Unisex']

                if product_gender not in acceptable_genders:
                    continue

            # Check style diversity
            style = product.get(
                'enhanced_primary_style', product.get(
                    'primary_style', ''))
            if style in seen_styles and len(seen_styles) >= k // 2:
                continue

            # Check color diversity
            color = product.get('primary_color', '')
            if color in seen_colors and len(seen_colors) >= k // 2:
                continue

            # Calculate diversity-adjusted score
            diversity_score = self._calculate_diversity_score(
                product, seen_styles, seen_colors, wear_type
            )
            adjusted_score = float(score) * diversity_score

            candidates.append({
                'product_idx': product_idx,
                'product': product,
                'semantic_score': adjusted_score,
                'faiss_rank': i + 1,
                'diversity_score': diversity_score
            })

            seen_styles.add(style)
            seen_colors.add(color)

            if len(candidates) >= k:
                break

        # Sort by adjusted score
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
        return candidates[:k]

    def expand_user_query(
            self,
            user_profile: str,
            user: Dict,
            wear_type: str) -> str:
        """✅ ENHANCED: Expand user query with comprehensive fashion context."""
        expanded_segments = [user_profile]

        # Style context expansion with enhanced mappings
        style = user.get('Fashion Style', '')
        if style:
            style_expansions = {
                'Streetwear': 'urban fashion street style casual contemporary edgy youthful trendy modern',
                'Athleisure': 'sporty casual athletic wear active lifestyle performance casual comfort versatile',
                'Contemporary': 'modern style current fashion updated classic fresh aesthetic sophisticated trendy',
                'Business': 'professional attire work appropriate office wear corporate fashion polished formal',
                'Formal': 'elegant sophisticated professional business formal evening wear polished refined'}

            if style in style_expansions:
                expanded_segments.append(
                    f"Style context: {
                        style_expansions[style]}")

        # Enhanced color intelligence expansion
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
                expanded_segments.append(
                    f"Color preferences: {
                        color_expansions[colors]}")

        # Body shape context
        body_shape = user.get('Body Shape', '')
        if body_shape:
            body_expansions = {
                'Hourglass': 'fitted structured balanced proportions',
                'Rectangle': 'straight balanced proportions',
                'Pear': 'balanced top fitted bottom',
                'Apple': 'loose top structured bottom'
            }

            if body_shape in body_expansions:
                expanded_segments.append(
                    f"Body shape context: {
                        body_expansions[body_shape]}")

        # Occasion context
        occasion = user.get('occasion_preference', '')
        if occasion:
            occasion_expansions = {
                'Daily Activities': 'casual comfortable everyday wear',
                'Work': 'professional business appropriate formal',
                'Evening': 'elegant sophisticated evening wear',
                'Special Events': 'formal special occasion elegant'
            }

            if occasion in occasion_expansions:
                expanded_segments.append(
                    f"Occasion context: {
                        occasion_expansions[occasion]}")

        # Wear type specific context
        if wear_type == 'Upperwear':
            expanded_segments.append("top wear shirt blouse t-shirt")
        elif wear_type == 'Bottomwear':
            expanded_segments.append("bottom wear pants trousers jeans")

        return ". ".join(expanded_segments)

    def _calculate_diversity_score(self, product: pd.Series, seen_styles: set,
                                   seen_colors: set, wear_type: str) -> float:
        """Calculate diversity score for a product based on seen styles and colors."""
        style = product.get(
            'enhanced_primary_style', product.get(
                'primary_style', ''))
        color = product.get('primary_color', '')

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
        # Get categories and colors
        top_category = top.get('category', '').lower()
        bottom_category = bottom.get('category', '').lower()
        top_color = top.get('color', '').lower()
        bottom_color = bottom.get('color', '').lower()

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
            'gray': [
                'black',
                'white',
                'navy',
                'burgundy',
                'emerald green',
                'olive green',
                'camel',
                'tan',
                'light pink',
                'medium pink'],
            'burgundy': [
                'black',
                'white',
                'navy',
                'gray',
                'camel',
                'tan',
                'olive green',
                'emerald green',
                'mustard yellow'],
            'olive green': [
                'black',
                'white',
                'navy',
                'burgundy',
                'camel',
                'tan',
                'gray',
                'mustard yellow',
                'emerald green'],
            'camel': [
                'black',
                'white',
                'navy',
                'burgundy',
                'olive green',
                'emerald green',
                'gray',
                'mustard yellow',
                'tan']}

        # Seasonal color rules
        seasonal_colors = {
            'spring_summer': [
                'white',
                'blue',
                'gray',
                'pink',
                'butter yellow',
                'sage',
                'beige',
                'pastels',
                'mint green',
                'sky blue',
                'baby blue',
                'light pink',
                'medium pink'],
            'fall_winter': [
                'black',
                'navy',
                'burgundy',
                'olive',
                'camel',
                'charcoal',
                'deep red',
                'maroon',
                'emerald green',
                'mustard yellow',
                'tan',
                'dark brown']}

        # Pattern compatibility rules
        pattern_compatibility = {
            'Geometric': {
                'compatible': ['Solid', 'Abstract', 'Ethnic'],
                'incompatible': ['Geometric', 'Animal']
            },
            'Floral': {
                'compatible': ['Solid', 'Abstract', 'Ethnic'],
                'incompatible': ['Floral', 'Animal']
            },
            'Abstract': {
                'compatible': ['Solid', 'Geometric', 'Ethnic'],
                'incompatible': ['Abstract', 'Animal']
            },
            'Ethnic': {
                'compatible': ['Solid', 'Geometric', 'Abstract'],
                'incompatible': ['Ethnic', 'Animal']
            },
            'Animal': {
                'compatible': ['Solid'],
                'incompatible': ['Geometric', 'Floral', 'Abstract', 'Ethnic', 'Animal']
            },
            'Solid': {
                'compatible': ['Geometric', 'Floral', 'Abstract', 'Ethnic', 'Animal'],
                'incompatible': []
            }
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
                return False, f"Incompatible combination: {
                    top_type} with {bottom_category}"

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

        # Check pattern compatibility
        def check_pattern_compatibility(top_pattern, bottom_pattern):
            if not top_pattern or not bottom_pattern:
                return True

            # Get pattern types
            top_pattern_type = top_pattern.split(
            )[0] if top_pattern else 'Solid'
            bottom_pattern_type = bottom_pattern.split(
            )[0] if bottom_pattern else 'Solid'

            # Check pattern compatibility rules
            if top_pattern_type in pattern_compatibility:
                if bottom_pattern_type in pattern_compatibility[top_pattern_type]['incompatible']:
                    return False
            if bottom_pattern_type in pattern_compatibility:
                if top_pattern_type in pattern_compatibility[bottom_pattern_type]['incompatible']:
                    return False

            return True

        # Get patterns
        top_pattern = top.get('pattern', 'Solid').lower()
        bottom_pattern = bottom.get('pattern', 'Solid').lower()

        # Check color harmony
        if not check_color_harmony(top_color, bottom_color):
            return False, f"Color combination {
                top_color} + {bottom_color} may not be harmonious"

        # Check pattern compatibility
        if not check_pattern_compatibility(top_pattern, bottom_pattern):
            return False, "Pattern combination may be overwhelming"

        # Check for texture balance
        top_texture = top.get('texture', '').lower()
        bottom_texture = bottom.get('texture', '').lower()
        if top_texture and bottom_texture and top_texture == bottom_texture:
            return False, "Too much of the same texture"

        return True, "Outfit is compatible"

    def score_outfit_enhanced(self,
                              top: pd.Series,
                              bottom: pd.Series,
                              user: Dict,
                              top_semantic: float,
                              bottom_semantic: float) -> Tuple[float,
                                                               str]:
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
                        f"{confidence_indicator} Strong {
                            component.replace(
                                '_', ' ').title()}")
                elif data['score'] < 0.3:
                    confidence_indicator = "✓" if data['confidence'] > 0.9 else "~"
                    explanations.append(
                        f"{confidence_indicator} Weak {
                            component.replace(
                                '_', ' ').title()}")

            # Add overall confidence indicator
            if avg_confidence < 0.7:
                explanations.append("⚠️ Some attributes have low confidence")

            explanation = self.get_explanation_for_outfit(
                explanations, scoring_components['color_harmony']['score'])

            return final_score, explanation

        except Exception as e:
            logger.error(f"Error in outfit scoring: {str(e)}")
            return 0.5, "Error in scoring, using default score"

    def _apply_confidence_adjustments(
            self,
            score: float,
            top: pd.Series,
            bottom: pd.Series) -> float:
        """Apply confidence-based adjustments to the final score."""
        # Get confidence scores
        top_confidence = float(top.get('style_confidence_score', 0.5))
        bottom_confidence = float(bottom.get('style_confidence_score', 0.5))

        # Calculate average confidence
        avg_confidence = (top_confidence + bottom_confidence) / 2

        # Apply confidence adjustment
        if avg_confidence > 0.8:
            return score * 1.1  # Boost high confidence predictions
        elif avg_confidence < 0.4:
            return score * 0.9  # Penalize low confidence predictions

        return score

    def _calculate_style_intelligence_score(
            self, top: pd.Series, bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate style intelligence score using professional rules."""
        top_style = top.get(
            'enhanced_primary_style', top.get(
                'primary_style', ''))
        bottom_style = bottom.get(
            'enhanced_primary_style', bottom.get(
                'primary_style', ''))

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
        bottom_compatibility = style_compatibility.get(
            bottom_style, [bottom_style])

        # Calculate style match score
        if bottom_style in top_compatibility or top_style in bottom_compatibility:
            base_score = 0.8
        else:
            base_score = 0.5

        # Apply additional style rules
        if 'Formal' in top_style and 'Casual' in bottom_style:
            base_score *= 0.7
        elif 'Casual' in top_style and 'Formal' in bottom_style:
            base_score *= 0.7

        # Check for universal style rules
        if self.quick_rules['validated_rules'].get(
                'fit_balance', {}).get('allowed', True):
            top_fit = top.get('fit_analysis', '').lower()
            bottom_fit = bottom.get('fit_analysis', '').lower()

            if 'tight' in top_fit and 'loose' in bottom_fit:
                base_score *= 1.1
            elif 'loose' in top_fit and 'tight' in bottom_fit:
                base_score *= 1.1

        return min(base_score, 1.0)

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

    def _calculate_price_coherence(
            self,
            top: pd.Series,
            bottom: pd.Series) -> float:
        """✅ ENHANCED: Calculate price coherence score with smart thresholds."""
        top_price = float(top.get('price', 0))
        bottom_price = float(bottom.get('price', 0))

        if not top_price or not bottom_price:
            return 0.5

        # Calculate price ratio
        price_ratio = max(top_price, bottom_price) / \
            min(top_price, bottom_price)

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

    def diversify_outfit_recommendations(
            self,
            recommendations: list,
            top_n: int = 100) -> list:
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
        min_style_target = max(3, top_n // (len(user_styles) * 2))  # Ensure at least 3 outfits per style
        target_distribution = {style: min_style_target for style in user_styles}

        # Sort by score first
        sorted_recs = sorted(
            recommendations,
            key=lambda x: x['score'],
            reverse=True)

        # First pass: Ensure top 10 has minimum distribution
        top_10_styles = {style: 0 for style in user_styles}
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
            matching_styles = outfit_styles & user_styles

            if not matching_styles:
                continue

            # Check if we need to maintain minimum distribution for top 10
            needs_min_distribution = False
            for style in matching_styles:
                if top_10_styles[style] < 3:  # Ensure at least 3 outfits per style in top 10
                    needs_min_distribution = True
                    break

            # If we don't need minimum distribution, check if we have too many of this style
            if not needs_min_distribution:
                style_ratio = max(top_10_styles.values()) / min(top_10_styles.values())
                if style_ratio > 2.5:  # Allow up to 2.5x more of one style than another
                    continue

            # Add to top 10 recommendations
            top_10_recs.append(rec)
            seen_outfits.add(outfit_hash)
            
            # Update top 10 style distribution
            for style in matching_styles:
                top_10_styles[style] += 1

        # Second pass: Collect remaining outfits while maintaining overall distribution
        for rec in sorted_recs:
            if len(diverse_recs) >= top_n:
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
            matching_styles = outfit_styles & user_styles

            if not matching_styles:
                continue

            # Check if we need to maintain minimum distribution for any style
            needs_min_distribution = False
            for style in matching_styles:
                if style_distribution[style] < target_distribution[style]:
                    needs_min_distribution = True
                    break

            # If we don't need minimum distribution, check if we have too many of this style
            if not needs_min_distribution:
                style_ratio = max(style_distribution.values()) / min(style_distribution.values())
                if style_ratio > 2.5:  # Allow up to 2.5x more of one style than another
                    continue

            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(
                top, seen_styles, seen_colors, 'Upperwear')
            diversity_score += self._calculate_diversity_score(
                bottom, seen_styles, seen_colors, 'Bottomwear')

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

    def _create_outfit_hash(self, top: pd.Series, bottom: pd.Series) -> tuple:
        """Create a hash tuple for an outfit based on multiple attributes."""
        return (
            top.get('product_id', ''),
            bottom.get('product_id', ''),
            top.get('enhanced_primary_style', ''),
            bottom.get('enhanced_primary_style', ''),
            top.get('primary_color', ''),
            bottom.get('primary_color', ''),
            top.get('fit_analysis', ''),
            bottom.get('fit_analysis', ''),
            top.get('quality_indicator', ''),
            bottom.get('quality_indicator', ''),
            top.get('occasion', ''),
            bottom.get('occasion', '')
        )

    def generate_main_outfits(
            self,
            user_id: int,
            budget_preference: str = None) -> pd.DataFrame:
        """✅ ENHANCED: Generate 100 main outfits for a user with improved diversity and quality."""

        try:
            # Load user data
            users_df = pd.read_excel(self.config['user_file'])
            user_row = users_df[users_df['User'] == user_id]

            if user_row.empty:
                raise ValueError(f"User {user_id} not found")

            user = user_row.iloc[0].to_dict()
            if budget_preference:
                user['budget_preference'] = budget_preference

            # Validate user data
            self.validate_user_data(user)

            # Load and validate products
            products_df = pd.read_csv(self.config['products_file'])
            products_df = self.validate_products_data(products_df)

            # Enhanced filtering
            filtered_products_df = self.filter_products_enhanced(
                products_df, user)

            # Build FAISS indexes
            logger.info(
                "Building FAISS indexes on filtered product dataset...")
            self.build_faiss_indexes(filtered_products_df)

            # Get user profiles
            upperwear_profile = user.get('Upper Wear Caption', '')
            bottomwear_profile = user.get('Lower Wear Caption', '')
            user_gender = user.get('Gender', 'Unisex')

            if not upperwear_profile or not bottomwear_profile:
                logger.warning(f"Missing user profiles")
                fashion_style = user.get('Fashion Style', 'Contemporary')
                color_family = user.get('Colors family', 'Blue')
                upperwear_profile = f"Style: {
                    fashion_style} Color: {color_family}"
                bottomwear_profile = f"Style: {
                    fashion_style} Color: {color_family}"

            logger.info(f"Getting recommendations for User {user_id}")

            # Get semantic recommendations
            upperwear_recs = self.get_semantic_recommendations(
                upperwear_profile,
                'Upperwear',
                user_gender,
                k=self.config['tops_per_outfit'],
                user=user)
            bottomwear_recs = self.get_semantic_recommendations(
                bottomwear_profile,
                'Bottomwear',
                user_gender,
                k=self.config['bottoms_per_outfit'],
                user=user)

            if not upperwear_recs or not bottomwear_recs:
                logger.warning("Insufficient products for outfit generation")
                return pd.DataFrame()

            # Generate outfit combinations
            recommendations = []
            outfit_count = 0
            target_outfits = self.config['main_outfits_count']

            for i, top_rec in enumerate(upperwear_recs):
                for j, bottom_rec in enumerate(bottomwear_recs):
                    if outfit_count >= target_outfits:
                        break

                    top = top_rec['product']
                    bottom = bottom_rec['product']

                    # Enhanced outfit scoring
                    outfit_score, explanation = self.score_outfit_enhanced(
                        top, bottom, user, top_rec['semantic_score'], bottom_rec['semantic_score'])

                    recommendations.append({
                        'main_outfit_id': f"main_{user_id}_{outfit_count + 1}",
                        'user_id': user_id,
                        'rank': outfit_count + 1,
                        'score': outfit_score,
                        'explanation': explanation,

                        # Top product details
                        'top_id': top.get('product_id', top.get('id', '')),
                        'top_title': top.get('title', ''),
                        'top_image': top.get('image_url', ''),
                        'top_price': float(top.get('price', 0)),
                        'top_style': top.get('enhanced_primary_style', top.get('primary_style', '')),
                        'top_color': top.get('primary_color', ''),
                        'top_occasion': top.get('enhanced_occasion', top.get('occasion', '')),
                        'top_semantic_score': top_rec['semantic_score'],
                        'top_faiss_rank': top_rec['faiss_rank'],
                        'top_versatility': top.get('versatility_analysis', ''),
                        'top_confidence': top.get('style_confidence_score', ''),

                        # Bottom product details
                        'bottom_id': bottom.get('product_id', bottom.get('id', '')),
                        'bottom_title': bottom.get('title', ''),
                        'bottom_image': bottom.get('image_url', ''),
                        'bottom_price': float(bottom.get('price', 0)),
                        'bottom_style': bottom.get('enhanced_primary_style', bottom.get('primary_style', '')),
                        'bottom_color': bottom.get('primary_color', ''),
                        'bottom_occasion': bottom.get('enhanced_occasion', bottom.get('occasion', '')),
                        'bottom_semantic_score': bottom_rec['semantic_score'],
                        'bottom_faiss_rank': bottom_rec['faiss_rank'],
                        'bottom_versatility': bottom.get('versatility_analysis', ''),
                        'bottom_confidence': bottom.get('style_confidence_score', ''),

                        # Combined outfit details
                        'total_price': float(top.get('price', 0)) + float(bottom.get('price', 0)),
                        'combined_semantic_score': (top_rec['semantic_score'] + bottom_rec['semantic_score']) / 2,
                        'color_harmony_analysis': f"{top.get('advanced_color_analysis', '')} + {bottom.get('advanced_color_analysis', '')}",

                        # Metadata
                        'generated_at': datetime.now().isoformat(),
                        'generation_method': 'faiss_semantic_enhanced'
                    })

                    outfit_count += 1

                if outfit_count >= target_outfits:
                    break

            # Convert to DataFrame and sort by score
            recommendations_df = pd.DataFrame(recommendations)

            if recommendations_df.empty:
                return pd.DataFrame()

            recommendations_df = recommendations_df.sort_values(
                'score', ascending=False)
            # Diversity logic: re-rank for diversity in top-N
            diverse_recs = self.diversify_outfit_recommendations(
                recommendations_df.to_dict('records'), top_n=target_outfits)
            recommendations_df = pd.DataFrame(diverse_recs)
            recommendations_df['rank'] = range(1, len(recommendations_df) + 1)

            logger.info(
                f"Generated {
                    len(recommendations_df)} main outfit recommendations for user {user_id}")

            return recommendations_df

        except Exception as e:
            logger.error(f"Error generating main outfits: {e}")
            raise

    def save_main_outfits_to_csv(
            self,
            user_id: int,
            budget_preference: str = None) -> str:
        """Generate and save main outfits to CSV file."""

        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Generate main outfits
        recommendations_df = self.generate_main_outfits(
            user_id, budget_preference)

        if recommendations_df.empty:
            logger.warning(f"No recommendations generated for user {user_id}")
            return None

        # Save to CSV
        output_file = os.path.join(
            self.config['output_dir'],
            f"user_{user_id}_main_outfits.csv")
        recommendations_df.to_csv(output_file, index=False)

        logger.info(f"✅ Main outfits saved to {output_file}")
        logger.info(f"📊 Generated {len(recommendations_df)} outfits")

        return output_file

    def generate_product_context_map(
            self, all_outfits_data: List[pd.DataFrame]) -> str:
        """Generate product-outfit context map for fast Phase 3 lookups."""
        logger.info(
            "🔄 Generating product-outfit context map for enhanced Phase 3...")

        # Combine all outfit data
        all_outfits = pd.concat(all_outfits_data, ignore_index=True)

        # Create context map
        context_map = {}

        # Get all unique products from all outfits
        all_top_ids = all_outfits['top_id'].unique()
        all_bottom_ids = all_outfits['bottom_id'].unique()
        all_product_ids = list(set(list(all_top_ids) + list(all_bottom_ids)))

        logger.info(
            f"Processing context for {
                len(all_product_ids)} products...")

        for product_id in all_product_ids:
            # Find outfits containing this product
            outfits_with_product = all_outfits[
                (all_outfits['top_id'] == product_id) |
                (all_outfits['bottom_id'] == product_id)
            ]

            if len(outfits_with_product) == 0:
                continue

            # Analyze context
            context_info = self._analyze_product_context(
                product_id, outfits_with_product, all_outfits)
            context_map[product_id] = context_info

        # Convert to DataFrame and save
        context_df = pd.DataFrame([
            {
                'product_id': pid,
                # Limit to top 10
                'outfit_contexts': ','.join(ctx['outfit_ids'][:10]),
                # Top 15
                'compatible_products': ','.join(map(str, ctx['compatible_products'][:15])),
                'style_context': ctx['dominant_style'],
                'occasion_context': ctx['dominant_occasion'],
                'formality_level': ctx['avg_formality'],
                'price_range': ctx['avg_price_range'],
                'compatibility_score': ctx['avg_compatibility']
            }
            for pid, ctx in context_map.items()
        ])

        # Save context map
        context_filepath = os.path.join(
            self.config['output_dir'],
            "product_outfit_context.csv")
        context_df.to_csv(context_filepath, index=False)

        logger.info(
            f"✅ Saved product context map with {
                len(context_df)} products to {context_filepath}")
        return context_filepath

    def _analyze_product_context(
            self,
            product_id: str,
            product_outfits: pd.DataFrame,
            all_outfits: pd.DataFrame) -> Dict:
        """Analyze the context of a product within outfits."""

        # Extract outfit IDs
        outfit_ids = product_outfits['main_outfit_id'].tolist()

        # Find products that appear in similar style/score contexts
        avg_score = product_outfits['score'].mean()
        score_threshold = 0.1  # ±10% score range

        similar_context_outfits = all_outfits[
            (all_outfits['score'] >= avg_score - score_threshold) &
            (all_outfits['score'] <= avg_score + score_threshold)
        ]

        # Get compatible products (excluding the source product)
        compatible_top_ids = similar_context_outfits['top_id'].value_counts().head(
            20).index.tolist()
        compatible_bottom_ids = similar_context_outfits['bottom_id'].value_counts().head(
            20).index.tolist()
        compatible_products = list(
            set(compatible_top_ids + compatible_bottom_ids))

        # Remove the source product itself
        if product_id in compatible_products:
            compatible_products.remove(product_id)

        # Analyze style context
        if 'top_style' in product_outfits.columns:
            styles = list(product_outfits['top_style'].dropna(
            )) + list(product_outfits['bottom_style'].dropna())
        else:
            styles = ['Contemporary']  # Default
        dominant_style = max(
            set(styles),
            key=styles.count) if styles else 'Contemporary'

        # Analyze occasion context
        if 'top_occasion' in product_outfits.columns:
            occasions = list(product_outfits['top_occasion'].dropna(
            )) + list(product_outfits['bottom_occasion'].dropna())
        else:
            occasions = ['Daily']  # Default
        dominant_occasion = max(set(occasions),
                                key=occasions.count) if occasions else 'Daily'

        # Calculate average formality level
        avg_formality = self._calculate_avg_formality(product_outfits)

        # Calculate average price range
        avg_price = (product_outfits['top_price'].mean(
        ) + product_outfits['bottom_price'].mean()) / 2
        avg_price_range = self._get_price_range_category(avg_price)

        # Calculate average compatibility score
        avg_compatibility = product_outfits['score'].mean()

        return {
            'outfit_ids': outfit_ids,
            # Top 15 most compatible
            'compatible_products': compatible_products[:15],
            'dominant_style': dominant_style,
            'dominant_occasion': dominant_occasion,
            'avg_formality': avg_formality,
            'avg_price_range': avg_price_range,
            'avg_compatibility': round(avg_compatibility, 3)
        }

    def _calculate_avg_formality(self, outfits: pd.DataFrame) -> float:
        """Calculate average formality level for outfits."""
        formality_scores = []

        for _, outfit in outfits.iterrows():
            top_style = outfit.get('top_style', 'Contemporary')
            bottom_style = outfit.get('bottom_style', 'Contemporary')

            top_formality = self.style_formality.get(top_style, 5)
            bottom_formality = self.style_formality.get(bottom_style, 5)

            formality_scores.append((top_formality + bottom_formality) / 2)

        return round(
            sum(formality_scores) /
            len(formality_scores),
            1) if formality_scores else 5.0

    def _get_price_range_category(self, price: float) -> str:
        """Categorize price into ranges."""
        if price < 1000:
            return "Budget"
        elif price < 2000:
            return "Mid-Range"
        elif price < 3500:
            return "Premium"
        else:
            return "Luxury"

    def batch_generate_for_all_users(self) -> List[str]:
        """Generate main outfits for all users AND create product context map."""

        try:
            users_df = pd.read_excel(self.config['user_file'])
            user_ids = users_df['User'].unique()

            generated_files = []
            all_outfits_data = []

            # Generate outfits for all users
            for user_id in user_ids:
                try:
                    logger.info(f"Processing user {user_id}...")
                    output_file = self.save_main_outfits_to_csv(user_id)
                    if output_file:
                        generated_files.append(output_file)
                        # Load the generated data for context map creation
                        user_outfits = pd.read_csv(output_file)
                        all_outfits_data.append(user_outfits)
                except Exception as e:
                    logger.error(f"Failed to process user {user_id}: {e}")
                    continue

            # Generate product context map from all outfit data
            if all_outfits_data:
                context_file = self.generate_product_context_map(
                    all_outfits_data)
                logger.info(f"🎯 Product context map generated: {context_file}")

            logger.info(
                f"✅ Batch processing complete. Generated files for {
                    len(generated_files)} users.")
            return generated_files

        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise


def main():
    """Main function to generate main outfits."""

    # Initialize generator
    generator = MainOutfitsGenerator()

    # Generate for a single user
    user_id = 1
    logger.info(f"🎯 Generating 50 main outfits for User {user_id}")

    output_file = generator.save_main_outfits_to_csv(
        user_id, budget_preference="Mid-Range")

    if output_file:
        print(
            f"\n✅ SUCCESS: Main outfits generated and saved to {output_file}")
        print(f"📁 This file is ready for API consumption")
        print(f"🔗 API Endpoint: GET /api/user/{user_id}/outfits")
    else:
        print(f"\n❌ FAILED: Could not generate outfits for user {user_id}")


if __name__ == "__main__":
    main()
