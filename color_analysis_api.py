"""
Color Analysis API Module

This module provides the enhanced color analysis functionality using the SkinToneAnalyzer.
It handles all color analysis operations and integrates with the main Flask application.
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple
from skin_tone_analyzer import SkinToneAnalyzer

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedColorAnalysisAPI:
    """
    Enhanced Color Analysis API using the new SkinToneAnalyzer
    
    This class provides a clean interface for color analysis operations
    that can be easily integrated into Flask applications or used standalone.
    
    Supports two modes:
    1. Photo analysis - automatic skin tone detection from uploaded images
    2. Manual selection - user provides hex code for direct analysis
    """
    
    def __init__(self, color_map_path: str = None):
        """
        Initialize the Enhanced Color Analysis API.
        
        Args:
            color_map_path: Optional path to Excel color map file.
                          If None, will look for "Colour map.xlsx" in current directory.
        """
        # Initialize the new skin tone analyzer
        if color_map_path is None:
            color_map_path = "Colour map.xlsx" if os.path.exists("Colour map.xlsx") else None
        
        self.analyzer = SkinToneAnalyzer(color_map_path)
        
        if color_map_path:
            logger.info(f"✅ Enhanced Color Analysis initialized with color map: {color_map_path}")
        else:
            logger.info("✅ Enhanced Color Analysis initialized with default color recommendations")
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color to RGB values.
        
        Args:
            hex_color: Hex color string (e.g., "#FDB4A6" or "FDB4A6")
            
        Returns:
            Tuple of RGB values (r, g, b)
        """
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Validate hex format
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color format: {hex_color}. Expected 6 characters.")
        
        try:
            # Convert hex to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except ValueError as e:
            raise ValueError(f"Invalid hex color: {hex_color}. {str(e)}")
    
    def analyze_from_hex(self, hex_color: str) -> Dict:
        """
        Analyze skin tone from a hex color code and return results in the same format
        as photo analysis.
        
        Args:
            hex_color: Hex color string representing the skin tone (e.g., "#FDB4A6")
            
        Returns:
            Dictionary containing comprehensive skin tone analysis results
        """
        try:
            # Convert hex to RGB
            rgb = self.hex_to_rgb(hex_color)
            avg_skin_tone = np.array(rgb, dtype=float)
            
            logger.info(f"🎨 Analyzing manual skin tone: {hex_color} -> RGB{rgb}")
            
            # Convert to LAB color space for analysis (same as photo analysis)
            lab_skin_tone = self.analyzer.rgb_to_lab(avg_skin_tone)
            
            # Determine undertone based on a and b values (same logic as photo analysis)
            a_value = lab_skin_tone[1]  # Green-Red axis
            b_value = lab_skin_tone[2]  # Blue-Yellow axis
            
            logger.debug(f"LAB values - L:{lab_skin_tone[0]:.2f}, a:{a_value:.2f}, b:{b_value:.2f}")
            
            # Define thresholds for undertone determination (same as photo analysis)
            THRESHOLD = 2.0
            a_magnitude = abs(a_value)
            b_magnitude = abs(b_value)
            
            # Determine undertone
            if a_magnitude < THRESHOLD and b_magnitude < THRESHOLD:
                undertone = "Neutral"
            else:
                if a_magnitude > b_magnitude:
                    undertone = "Cool" if a_value < 0 else "Warm"
                else:
                    undertone = "Cool" if b_value < 0 else "Warm"
            
            # Get lightness value (L component in LAB)
            lightness = lab_skin_tone[0]
            
            # Determine Fitzpatrick scale based on lightness (same logic as photo analysis)
            if lightness > 74:
                fitzpatrick = "I"
            elif lightness > 64:
                fitzpatrick = "II"
            elif lightness > 54:
                fitzpatrick = "III"
            elif lightness > 49:
                fitzpatrick = "IV"
            elif lightness > 44:
                fitzpatrick = "V"
            else:
                fitzpatrick = "VI"
            
            # For manual input, dominant colors is just the provided color
            dominant_colors = [list(rgb)]
            
            # Get color recommendations using the same method as photo analysis
            recommended_colours = self.analyzer.get_recommended_colours_from_excel(undertone, fitzpatrick)
            
            # Prepare results in the EXACT same format as photo analysis
            results = {
                'success': True,
                'average_skin_tone': avg_skin_tone.tolist(),
                'undertone': undertone,
                'fitzpatrick_scale': fitzpatrick,
                'lightness': float(lightness),
                'a_value': float(a_value),
                'b_value': float(b_value),
                'dominant_colors': dominant_colors,
                'recommended_colours': recommended_colours,
                'skin_regions_detected': True,  # Set to True since user manually provided the color
                'analysis_metadata': {
                    'lab_values': lab_skin_tone.tolist(),
                    'skin_pixel_count': 1,  # Symbolic since it's manual input
                    'total_pixels': 1,
                    'input_method': 'manual_hex',
                    'input_hex': hex_color
                }
            }
            
            logger.info(f"✅ Manual skin tone analysis completed: {undertone} undertone, Fitzpatrick {fitzpatrick}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hex color analysis: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def analyze_image(self, image_data: str) -> Dict:
        """
        Analyze image using the SkinToneAnalyzer.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Dictionary containing comprehensive skin tone analysis results
        """
        try:
            # Use the new skin tone analyzer
            result = self.analyzer.analyze_skin_tone_from_base64(image_data)
            
            # Add success flag and metadata about input method
            result['success'] = True
            if 'analysis_metadata' not in result:
                result['analysis_metadata'] = {}
            result['analysis_metadata']['input_method'] = 'photo_upload'
            
            logger.info(f"✅ Photo skin tone analysis completed: {result['undertone']} undertone, Fitzpatrick {result['fitzpatrick_scale']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced image analysis: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def analyze_image_from_path(self, image_path: str) -> Dict:
        """
        Analyze image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing comprehensive skin tone analysis results
        """
        try:
            result = self.analyzer.analyze_skin_tone_from_path(image_path)
            result['success'] = True
            
            # Add metadata about input method
            if 'analysis_metadata' not in result:
                result['analysis_metadata'] = {}
            result['analysis_metadata']['input_method'] = 'photo_file'
            
            logger.info(f"✅ Photo skin tone analysis completed: {result['undertone']} undertone, Fitzpatrick {result['fitzpatrick_scale']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced image analysis from path: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def analyze_image_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Analyze image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing comprehensive skin tone analysis results
        """
        try:
            result = self.analyzer.analyze_skin_tone_from_bytes(image_bytes)
            result['success'] = True
            
            # Add metadata about input method
            if 'analysis_metadata' not in result:
                result['analysis_metadata'] = {}
            result['analysis_metadata']['input_method'] = 'photo_bytes'
            
            logger.info(f"✅ Photo skin tone analysis completed: {result['undertone']} undertone, Fitzpatrick {result['fitzpatrick_scale']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced image analysis from bytes: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def get_color_recommendations(self, undertone: str, fitzpatrick_type: str = None) -> Dict:
        """
        Get color recommendations for a specific undertone and Fitzpatrick type.
        
        Args:
            undertone: Skin undertone (warm/cool/neutral)
            fitzpatrick_type: Optional Fitzpatrick skin type
            
        Returns:
            Dictionary containing color recommendations by category
        """
        try:
            if fitzpatrick_type:
                recommendations = self.analyzer.get_recommended_colours_from_excel(undertone, fitzpatrick_type)
            else:
                recommendations = self.analyzer._get_default_recommendations(undertone)
            
            return {
                "success": True,
                "undertone": undertone,
                "fitzpatrick_type": fitzpatrick_type,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting color recommendations: {e}")
            return {
                "success": False,
                "error": f"Failed to get recommendations: {str(e)}"
            }

# Convenience function for direct usage
def create_color_analysis_api(color_map_path: str = None) -> EnhancedColorAnalysisAPI:
    """
    Factory function to create an Enhanced Color Analysis API instance.
    
    Args:
        color_map_path: Optional path to Excel color map file
        
    Returns:
        EnhancedColorAnalysisAPI instance
    """
    return EnhancedColorAnalysisAPI(color_map_path)

# Example usage
if __name__ == "__main__":
    # Example of how to use the Color Analysis API
    api = EnhancedColorAnalysisAPI()
    
    # You can now call:
    # result = api.analyze_image(base64_image)
    # result = api.analyze_image_from_path("image.jpg") 
    # result = api.analyze_image_from_bytes(image_bytes)
    # recommendations = api.get_color_recommendations("warm", "III")
    
    print("✅ Color Analysis API module loaded successfully") 