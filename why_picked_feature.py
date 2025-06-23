"""
Why Picked Feature - Generates personalized explanations for outfit recommendations
"""

import pandas as pd
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WhyPickedFeature:
    """
    Generates personalized explanations for why outfits were picked for users.
    Uses style vibe and occasion tags mapped to descriptions.
    """
    
    def __init__(self):
        """Initialize the WhyPickedFeature with style and occasion mappings."""
        self.style_vibe_mappings = {}
        self.occasion_mappings = {}
        self._load_mappings()
    
    def _load_mappings(self):
        """Load style vibe and occasion mappings from Excel file."""
        try:
            # Load style vibe mappings
            style_df = pd.read_excel('user tags (style vibe).xlsx', sheet_name='Style Vibe')
            self.style_vibe_mappings = dict(zip(style_df['Tag'], style_df['Description']))
            
            # Load occasion mappings
            occasion_df = pd.read_excel('user tags (style vibe).xlsx', sheet_name='Ocassion')
            self.occasion_mappings = dict(zip(occasion_df['Tag'], occasion_df['Description']))
            
            logger.info(f"✅ Loaded {len(self.style_vibe_mappings)} style vibe mappings and {len(self.occasion_mappings)} occasion mappings")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load mappings from Excel file: {e}")
            # Fallback mappings
            self.style_vibe_mappings = {
                'streetwear': 'This outfit embodies a bold, urban aesthetic that matches your streetwear preference.',
                'business casual': 'This outfit strikes the perfect balance between professional and comfortable.',
                'athleisure': 'This outfit combines athletic comfort with everyday style.',
                'minimalist': 'This outfit features clean lines and simple elegance.',
                'bohemian': 'This outfit has a free-spirited, artistic vibe.'
            }
            self.occasion_mappings = {
                'work': 'Perfect for your professional environment.',
                'weekend': 'Great for casual weekend activities.',
                'party': 'Ideal for social gatherings and celebrations.',
                'date': 'Perfect for romantic occasions.',
                'travel': 'Comfortable and stylish for your travels.'
            }
    
    def _get_style_vibe_tag(self, outfit_data: Dict[str, Any]) -> str:
        """Extract style vibe tag from outfit data."""
        # Combine all text fields for analysis
        text_fields = [
            outfit_data.get('top_title', ''),
            outfit_data.get('top_primary_style', ''),
            outfit_data.get('top_style_tags', ''),
            outfit_data.get('top_persona_tag', ''),
            outfit_data.get('bottom_title', ''),
            outfit_data.get('bottom_primary_style', ''),
            outfit_data.get('bottom_style_tags', ''),
            outfit_data.get('bottom_persona_tag', '')
        ]
        
        combined_text = ' '.join(text_fields).lower()
        
        # Simple keyword matching
        if 'street' in combined_text or 'urban' in combined_text:
            return 'streetwear'
        elif 'business' in combined_text or 'casual' in combined_text:
            return 'business casual'
        elif 'athletic' in combined_text or 'sport' in combined_text:
            return 'athleisure'
        elif 'minimal' in combined_text or 'simple' in combined_text:
            return 'minimalist'
        elif 'bohemian' in combined_text or 'artistic' in combined_text:
            return 'bohemian'
        else:
            # Default based on user preferences
            return 'business casual'
    
    def _get_occasion_tag(self, outfit_data: Dict[str, Any]) -> str:
        """Extract occasion tag from outfit data."""
        # Combine all text fields for analysis
        text_fields = [
            outfit_data.get('top_title', ''),
            outfit_data.get('top_primary_style', ''),
            outfit_data.get('top_style_tags', ''),
            outfit_data.get('bottom_title', ''),
            outfit_data.get('bottom_primary_style', ''),
            outfit_data.get('bottom_style_tags', '')
        ]
        
        combined_text = ' '.join(text_fields).lower()
        
        # Simple keyword matching
        if 'work' in combined_text or 'office' in combined_text:
            return 'work'
        elif 'party' in combined_text or 'celebration' in combined_text:
            return 'party'
        elif 'date' in combined_text or 'romantic' in combined_text:
            return 'date'
        elif 'travel' in combined_text or 'comfort' in combined_text:
            return 'travel'
        else:
            # Default based on user preferences
            return 'weekend'
    
    def get_style_vibe_explanation(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Get style vibe explanation based on outfit and user data.
        Returns format: "keyword - explanation"
        """
        try:
            # Extract keywords from outfit data with more product-specific fields
            text_to_analyze = " ".join([
                str(outfit_data.get('top_title', '')),
                str(outfit_data.get('top_primary_style', '')),
                str(outfit_data.get('top_style_tags', '')),
                str(outfit_data.get('top_persona_tag', '')),
                str(outfit_data.get('top_full_caption', '')),
                str(outfit_data.get('top_color', '')),
                str(outfit_data.get('top_category', '')),
                str(outfit_data.get('top_scraped_category', '')),
                str(outfit_data.get('top_brand', '')),
                str(outfit_data.get('top_material', '')),
                str(outfit_data.get('top_fit', '')),
                str(outfit_data.get('bottom_title', '')),
                str(outfit_data.get('bottom_primary_style', '')),
                str(outfit_data.get('bottom_style_tags', '')),
                str(outfit_data.get('bottom_persona_tag', '')),
                str(outfit_data.get('bottom_full_caption', '')),
                str(outfit_data.get('bottom_color', '')),
                str(outfit_data.get('bottom_category', '')),
                str(outfit_data.get('bottom_scraped_category', '')),
                str(outfit_data.get('bottom_brand', '')),
                str(outfit_data.get('bottom_material', '')),
                str(outfit_data.get('bottom_fit', ''))
            ]).lower()
            
            # Find best matching style vibe
            best_match = None
            best_score = 0
            
            for keyword, explanation in self.style_vibe_mappings.items():
                if keyword.lower() in text_to_analyze:
                    score = text_to_analyze.count(keyword.lower())
                    if score > best_score:
                        best_score = score
                        best_match = (keyword, explanation)
            
            if best_match:
                keyword, explanation = best_match
                return f"{keyword} - {explanation}"
            else:
                # Enhanced fallback based on product attributes
                return self._get_product_specific_style_vibe(outfit_data, user_data)
                    
        except Exception as e:
            logger.error(f"Error in get_style_vibe_explanation: {e}")
            return "casual - This outfit offers a versatile casual look."
    
    def _get_product_specific_style_vibe(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """Generate product-specific style vibe based on actual product attributes."""
        try:
            # Analyze product attributes for style determination
            top_color = outfit_data.get('top_color', '').lower()
            bottom_color = outfit_data.get('bottom_color', '').lower()
            top_material = outfit_data.get('top_material', '').lower()
            bottom_material = outfit_data.get('bottom_material', '').lower()
            top_fit = outfit_data.get('top_fit', '').lower()
            bottom_fit = outfit_data.get('bottom_fit', '').lower()
            top_price = outfit_data.get('top_price', 0)
            bottom_price = outfit_data.get('bottom_price', 0)
            total_price = top_price + bottom_price
            
            # Style determination logic based on product attributes
            style_keywords = []
            
            # Color-based style
            if any(color in top_color + bottom_color for color in ['black', 'navy', 'gray', 'white']):
                style_keywords.append('minimalist')
            elif any(color in top_color + bottom_color for color in ['bright', 'neon', 'vibrant']):
                style_keywords.append('bold')
            elif any(color in top_color + bottom_color for color in ['pastel', 'soft', 'light']):
                style_keywords.append('gentle')
            
            # Material-based style
            if any(material in top_material + bottom_material for material in ['denim', 'leather', 'canvas']):
                style_keywords.append('rugged')
            elif any(material in top_material + bottom_material for material in ['silk', 'satin', 'velvet']):
                style_keywords.append('luxurious')
            elif any(material in top_material + bottom_material for material in ['cotton', 'linen']):
                style_keywords.append('comfortable')
            
            # Fit-based style
            if any(fit in top_fit + bottom_fit for fit in ['oversized', 'loose', 'relaxed']):
                style_keywords.append('streetwear')
            elif any(fit in top_fit + bottom_fit for fit in ['slim', 'fitted', 'tailored']):
                style_keywords.append('business casual')
            
            # Price-based style
            if total_price > 2000:
                style_keywords.append('premium')
            elif total_price < 800:
                style_keywords.append('affordable')
            
            # Select the most appropriate style
            if style_keywords:
                # Use the first style keyword found
                selected_style = style_keywords[0]
                
                # Generate specific explanation based on the style
                explanations = {
                    'minimalist': 'This outfit embodies a clean, minimalist aesthetic with its neutral color palette.',
                    'bold': 'This outfit makes a statement with its vibrant colors and bold styling.',
                    'gentle': 'This outfit offers a soft, gentle approach with its pastel and light tones.',
                    'rugged': 'This outfit has a rugged, durable feel with its sturdy materials.',
                    'luxurious': 'This outfit provides a luxurious touch with its premium materials.',
                    'comfortable': 'This outfit prioritizes comfort with its breathable, soft materials.',
                    'streetwear': 'This outfit captures the streetwear vibe with its relaxed, urban fit.',
                    'business casual': 'This outfit strikes the perfect balance between professional and casual.',
                    'premium': 'This outfit reflects premium quality with its higher-end materials and construction.',
                    'affordable': 'This outfit offers great value with its budget-friendly price point.'
                }
                
                explanation = explanations.get(selected_style, f'This outfit reflects a {selected_style} style.')
                return f"{selected_style} - {explanation}"
            else:
                # Fallback to user preference with randomization
                user_style = user_data.get('Fashion Style', '').lower()
                fallback_styles = ['casual', 'versatile', 'comfortable', 'stylish']
                selected_fallback = random.choice(fallback_styles)
                
                fallback_explanations = {
                    'casual': 'This outfit offers a relaxed, casual look perfect for everyday wear.',
                    'versatile': 'This outfit provides versatility for various occasions and settings.',
                    'comfortable': 'This outfit prioritizes comfort while maintaining style.',
                    'stylish': 'This outfit combines style and functionality for a polished look.'
                }
                
                explanation = fallback_explanations.get(selected_fallback, 'This outfit offers a versatile casual look.')
                return f"{selected_fallback} - {explanation}"
                
        except Exception as e:
            logger.error(f"Error in _get_product_specific_style_vibe: {e}")
            return "casual - This outfit offers a versatile casual look."
    
    def get_occasion_explanation(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Get occasion explanation based on outfit and user data.
        Returns format: "keyword - explanation"
        """
        try:
            # Extract keywords from outfit data with more product-specific fields
            text_to_analyze = " ".join([
                str(outfit_data.get('top_title', '')),
                str(outfit_data.get('top_primary_style', '')),
                str(outfit_data.get('top_style_tags', '')),
                str(outfit_data.get('top_persona_tag', '')),
                str(outfit_data.get('top_full_caption', '')),
                str(outfit_data.get('top_color', '')),
                str(outfit_data.get('top_category', '')),
                str(outfit_data.get('top_scraped_category', '')),
                str(outfit_data.get('top_brand', '')),
                str(outfit_data.get('top_material', '')),
                str(outfit_data.get('top_fit', '')),
                str(outfit_data.get('bottom_title', '')),
                str(outfit_data.get('bottom_primary_style', '')),
                str(outfit_data.get('bottom_style_tags', '')),
                str(outfit_data.get('bottom_persona_tag', '')),
                str(outfit_data.get('bottom_full_caption', '')),
                str(outfit_data.get('bottom_color', '')),
                str(outfit_data.get('bottom_category', '')),
                str(outfit_data.get('bottom_scraped_category', '')),
                str(outfit_data.get('bottom_brand', '')),
                str(outfit_data.get('bottom_material', '')),
                str(outfit_data.get('bottom_fit', ''))
            ]).lower()
            
            # Find best matching occasion
            best_match = None
            best_score = 0
            
            for keyword, explanation in self.occasion_mappings.items():
                if keyword.lower() in text_to_analyze:
                    score = text_to_analyze.count(keyword.lower())
                    if score > best_score:
                        best_score = score
                        best_match = (keyword, explanation)
            
            if best_match:
                keyword, explanation = best_match
                return f"{keyword} - {explanation}"
            else:
                # Enhanced fallback based on product attributes
                return self._get_product_specific_occasion(outfit_data, user_data)
                    
        except Exception as e:
            logger.error(f"Error in get_occasion_explanation: {e}")
            return "casual - Perfect for casual weekend outings and social gatherings."
    
    def _get_product_specific_occasion(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """Generate product-specific occasion based on actual product attributes."""
        try:
            # Analyze product attributes for occasion determination
            top_color = outfit_data.get('top_color', '').lower()
            bottom_color = outfit_data.get('bottom_color', '').lower()
            top_material = outfit_data.get('top_material', '').lower()
            bottom_material = outfit_data.get('bottom_material', '').lower()
            top_fit = outfit_data.get('top_fit', '').lower()
            bottom_fit = outfit_data.get('bottom_fit', '').lower()
            top_price = outfit_data.get('top_price', 0)
            bottom_price = outfit_data.get('bottom_price', 0)
            total_price = top_price + bottom_price
            
            # Occasion determination logic based on product attributes
            occasion_keywords = []
            
            # Color-based occasion
            if any(color in top_color + bottom_color for color in ['black', 'navy', 'gray']):
                occasion_keywords.append('work')
            elif any(color in top_color + bottom_color for color in ['bright', 'vibrant']):
                occasion_keywords.append('party')
            elif any(color in top_color + bottom_color for color in ['pastel', 'soft']):
                occasion_keywords.append('date')
            
            # Material-based occasion
            if any(material in top_material + bottom_material for material in ['silk', 'satin', 'velvet']):
                occasion_keywords.append('formal')
            elif any(material in top_material + bottom_material for material in ['cotton', 'linen']):
                occasion_keywords.append('casual')
            elif any(material in top_material + bottom_material for material in ['denim', 'canvas']):
                occasion_keywords.append('weekend')
            
            # Fit-based occasion
            if any(fit in top_fit + bottom_fit for fit in ['slim', 'fitted', 'tailored']):
                occasion_keywords.append('work')
            elif any(fit in top_fit + bottom_fit for fit in ['oversized', 'loose']):
                occasion_keywords.append('casual')
            
            # Price-based occasion
            if total_price > 2000:
                occasion_keywords.append('special')
            elif total_price < 800:
                occasion_keywords.append('everyday')
            
            # Select the most appropriate occasion
            if occasion_keywords:
                # Use the first occasion keyword found
                selected_occasion = occasion_keywords[0]
                
                # Generate specific explanation based on the occasion
                explanations = {
                    'work': 'Perfect for professional settings and office environments.',
                    'party': 'Ideal for social gatherings and celebratory events.',
                    'date': 'Great for romantic outings and intimate occasions.',
                    'formal': 'Suitable for formal events and special ceremonies.',
                    'casual': 'Perfect for relaxed, everyday activities.',
                    'weekend': 'Ideal for weekend outings and leisure activities.',
                    'special': 'Perfect for special occasions and important events.',
                    'everyday': 'Great for daily wear and routine activities.'
                }
                
                explanation = explanations.get(selected_occasion, f'Perfect for {selected_occasion} occasions.')
                return f"{selected_occasion} - {explanation}"
            else:
                # Fallback to user preference with randomization
                weekend_pref = user_data.get('Weekend Preference', '').lower()
                fallback_occasions = ['casual', 'weekend', 'everyday', 'versatile']
                selected_fallback = random.choice(fallback_occasions)
                
                fallback_explanations = {
                    'casual': 'Perfect for casual weekend outings and social gatherings.',
                    'weekend': 'Ideal for weekend activities and relaxed settings.',
                    'everyday': 'Great for daily wear and routine activities.',
                    'versatile': 'Suitable for various occasions and settings.'
                }
                
                explanation = fallback_explanations.get(selected_fallback, 'Perfect for casual weekend outings.')
                return f"{selected_fallback} - {explanation}"
                
        except Exception as e:
            logger.error(f"Error in _get_product_specific_occasion: {e}")
            return "casual - Perfect for casual weekend outings and social gatherings."
    
    def get_body_type_explanation(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Get body type explanation based on user data and product attributes.
        Returns format: "keyword - explanation"
        """
        try:
            body_shape = user_data.get('Body Shape', '').lower()
            
            # Get product-specific attributes
            top_fit = outfit_data.get('top_fit', '').lower()
            bottom_fit = outfit_data.get('bottom_fit', '').lower()
            top_material = outfit_data.get('top_material', '').lower()
            bottom_material = outfit_data.get('bottom_material', '').lower()
            
            # Enhanced body type explanations based on product attributes
            if 'inverted triangle' in body_shape:
                if 'relaxed' in top_fit or 'oversized' in top_fit:
                    return "inverted triangle - The relaxed top fit balances your broader shoulders perfectly."
                elif 'slim' in bottom_fit or 'fitted' in bottom_fit:
                    return "inverted triangle - The fitted bottom creates definition for your inverted triangle shape."
                else:
                    return "inverted triangle - The proportions of this outfit complement your inverted triangle body shape."
            elif 'rectangle' in body_shape:
                if 'fitted' in top_fit or 'slim' in top_fit:
                    return "rectangle - The fitted top creates definition for your rectangular body shape."
                elif 'structured' in top_material or 'denim' in bottom_material:
                    return "rectangle - The structured materials add shape to your rectangular frame."
                else:
                    return "rectangle - This outfit creates definition for your rectangular body shape."
            elif 'triangle' in body_shape:
                if 'fitted' in top_fit or 'slim' in top_fit:
                    return "triangle - The fitted top balances your triangular body shape."
                elif 'relaxed' in bottom_fit or 'loose' in bottom_fit:
                    return "triangle - The relaxed bottom proportions balance your triangular shape."
                else:
                    return "triangle - The proportions of this outfit balance your triangular body shape."
            elif 'hourglass' in body_shape:
                if 'fitted' in top_fit and 'fitted' in bottom_fit:
                    return "hourglass - The fitted pieces accentuate your natural hourglass curves."
                elif 'stretchy' in top_material or 'stretchy' in bottom_material:
                    return "hourglass - The stretchy materials highlight your hourglass figure."
                else:
                    return "hourglass - This outfit accentuates your natural hourglass curves."
            elif 'oval' in body_shape:
                if 'structured' in top_material or 'tailored' in top_fit:
                    return "oval - The structured top provides definition for your oval body shape."
                elif 'fitted' in bottom_fit:
                    return "oval - The fitted bottom creates structure for your oval shape."
                else:
                    return "oval - This outfit provides structure that flatters your oval body shape."
            else:
                # Fallback with randomization
                fallback_fits = ['versatile fit', 'comfortable fit', 'flattering fit', 'balanced fit']
                selected_fit = random.choice(fallback_fits)
                return f"{selected_fit} - This outfit is designed to flatter your body type."
                
        except Exception as e:
            logger.error(f"Error in get_body_type_explanation: {e}")
            return "versatile fit - This outfit is designed to flatter your body type."
    
    def get_skin_undertone_explanation(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Get skin undertone explanation based on user data and product colors.
        Returns format: "keyword - explanation"
        """
        try:
            color_analysis = user_data.get('Color Analysis', '').lower()
            
            # Get product-specific colors
            top_color = outfit_data.get('top_color', '').lower()
            bottom_color = outfit_data.get('bottom_color', '').lower()
            combined_colors = top_color + ' ' + bottom_color
            
            # Enhanced undertone explanations based on actual product colors
            if 'warm' in color_analysis:
                if any(color in combined_colors for color in ['brown', 'beige', 'cream', 'gold', 'orange', 'yellow']):
                    return "warm undertone - The warm colors in this outfit complement your warm undertone beautifully."
                elif any(color in combined_colors for color in ['navy', 'olive', 'burgundy']):
                    return "warm undertone - The rich, warm-toned colors enhance your warm undertone."
                else:
                    return "warm undertone - The color palette works well with your warm undertone."
            elif 'cool' in color_analysis:
                if any(color in combined_colors for color in ['blue', 'purple', 'pink', 'silver', 'gray']):
                    return "cool undertone - The cool colors in this outfit complement your cool undertone perfectly."
                elif any(color in combined_colors for color in ['navy', 'charcoal', 'lavender']):
                    return "cool undertone - The cool-toned colors enhance your cool undertone."
                else:
                    return "cool undertone - The colors complement your cool undertone beautifully."
            elif 'neutral' in color_analysis:
                if any(color in combined_colors for color in ['black', 'white', 'gray', 'navy']):
                    return "neutral undertone - The neutral colors work harmoniously with your neutral undertone."
                elif any(color in combined_colors for color in ['beige', 'cream', 'taupe']):
                    return "neutral undertone - The balanced colors complement your neutral undertone."
                else:
                    return "neutral undertone - These colors work harmoniously with your neutral undertone."
            else:
                # Fallback with randomization based on actual colors
                if any(color in combined_colors for color in ['black', 'white', 'gray']):
                    return "versatile colors - The neutral colors work well with any undertone."
                elif any(color in combined_colors for color in ['blue', 'navy']):
                    return "universal blue - The blue tones are universally flattering."
                elif any(color in combined_colors for color in ['brown', 'beige']):
                    return "earth tones - The earth tones provide a natural, flattering look."
                else:
                    fallback_undertones = ['complementary colors', 'flattering palette', 'versatile tones']
                    selected_undertone = random.choice(fallback_undertones)
                    return f"{selected_undertone} - The colors work well with your skin tone."
                
        except Exception as e:
            logger.error(f"Error in get_skin_undertone_explanation: {e}")
            return "versatile colors - The colors work well with your skin tone."
    
    def generate_why_picked_explanation(self, outfit_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Generate personalized explanation for why this outfit was picked.
        Returns a formatted string for frontend display.
        """
        try:
            # Get style vibe explanation
            style_vibe_explanation = self.get_style_vibe_explanation(outfit_data, user_data)
            
            # Get occasion explanation  
            occasion_explanation = self.get_occasion_explanation(outfit_data, user_data)
            
            # Get body type explanation
            body_type_explanation = self.get_body_type_explanation(outfit_data, user_data)
            
            # Get skin undertone explanation
            skin_undertone_explanation = self.get_skin_undertone_explanation(outfit_data, user_data)
            
            # Format for frontend display
            explanation_parts = []
            
            if style_vibe_explanation:
                explanation_parts.append(f"Style Vibe: {style_vibe_explanation}")
            
            if occasion_explanation:
                explanation_parts.append(f"Occasion: {occasion_explanation}")
                
            if body_type_explanation:
                explanation_parts.append(f"Body Type: {body_type_explanation}")
                
            if skin_undertone_explanation:
                explanation_parts.append(f"Skin Undertone: {skin_undertone_explanation}")
            
            # Join with double newlines for clear separation
            final_explanation = "\n\n".join(explanation_parts)
            
            logger.info(f"Generated why picked explanation: {final_explanation[:100]}...")
            return final_explanation
            
        except Exception as e:
            logger.error(f"Error generating why picked explanation: {e}")
            return "Personalized explanation could not be generated."


# Test the feature if run directly
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_outfit_data = {
        'top_title': 'Urban Streetwear T-Shirt',
        'top_primary_style': 'streetwear',
        'top_style_tags': 'casual, urban',
        'bottom_title': 'Slim Fit Jeans',
        'bottom_primary_style': 'casual',
        'bottom_style_tags': 'denim, comfortable'
    }
    
    test_user_data = {
        'Body Shape': 'Rectangle',
        'Color Analysis': 'Warm undertones',
        'Fashion Style': 'Streetwear'
    }
    
    # Test the feature
    why_picked = WhyPickedFeature()
    explanation = why_picked.generate_why_picked_explanation(test_outfit_data, test_user_data)
    
    print("🎯 Test Why Picked Explanation:")
    print(explanation) 