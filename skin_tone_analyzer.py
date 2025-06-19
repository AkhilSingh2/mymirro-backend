"""
Skin Tone Analyzer - Single File Version

This file consolidates all the skin tone analysis functionality from the original Flask app
into a single, self-contained module that can be easily integrated into existing projects.

Usage:
    from skin_tone_analyzer import analyze_skin_tone_from_file, SkinToneAnalyzer
    
    # Simple usage
    results = analyze_skin_tone_from_file("image.jpg")
    
    # Advanced usage with custom color map
    analyzer = SkinToneAnalyzer("path/to/color_map.xlsx")
    results = analyzer.analyze_skin_tone_from_bytes(image_bytes)
"""

import cv2
import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Any
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class SkinToneAnalyzer:
    """
    Comprehensive skin tone analysis and color recommendation system.
    
    This class provides functionality to:
    - Analyze skin tone from images
    - Determine undertones (warm/cool/neutral)
    - Classify according to Fitzpatrick scale
    - Provide personalized color recommendations
    """
    
    def __init__(self, color_map_path: str = None):
        """
        Initialize the analyzer with optional color map.
        
        Args:
            color_map_path: Path to Excel file with color recommendations.
                          If None, uses built-in color mapping.
        """
        self.color_map_path = color_map_path
        self.color_map_data = None
        self._load_color_map()
        
        # Color mapping dictionary for converting descriptive names to CSS colors
        self.COLOR_MAP = {
            # Blues
            'Navy Blue': '#000080',
            'Powder Blue': '#B0E0E6',
            'Ice Blue': '#B5E0E6',
            'Cobalt Blue': '#0047AB',
            'Steel Blue': '#4682B4',
            'Teal': '#008080',
            'Mint': '#98FF98',
            
            # Reds and Pinks
            'Coral': '#FF7F50',
            'Burgundy': '#800020',
            'Plum': '#8E4585',
            'Soft Pink': '#FFB6C1',
            'Pale Pink': '#FFC0CB',
            'Lavender': '#E6E6FA',
            
            # Greens
            'Olive Green': '#808000',
            'Emerald': '#50C878',
            'Emerald Green': '#50C878',
            
            # Yellows and Oranges
            'Golden Yellow': '#FFD700',
            'Mustard': '#FFDB58',
            'Mustard Yellow': '#FFDB58',
            'Burnt Orange': '#CC5500',
            'Peach': '#FFE5B4',
            
            # Grays and Neutrals
            'Charcoal': '#36454F',
            'Charcoal Gray': '#36454F',
            'Heather Gray': '#B6B6B6',
            'Light Gray': '#D3D3D3',
            'Warm Gray': '#808080',
            'Taupe': '#483C32',
            'Khaki': '#F0E68C',
            
            # Browns and Beiges
            'Cream': '#FFFDD0',
            'Beige': '#F5F5DC',
            'Cream/Beige': '#F5F5DC',
            'Camel': '#C19A6B',
            'Camel (Warm Beige)': '#C19A6B',
            
            # Turquoise
            'Turquoise': '#40E0D0'
        }
        
        # Default color recommendations when Excel file is not available
        self.DEFAULT_RECOMMENDATIONS = {
            'warm': {
                'Formal': ['Navy Blue', 'Burgundy', 'Golden Yellow', 'Cream'],
                'Streetwear': ['Burnt Orange', 'Mustard', 'Olive Green', 'Warm Gray'],
                'Athleisure': ['Coral', 'Peach', 'Khaki', 'Camel']
            },
            'cool': {
                'Formal': ['Powder Blue', 'Plum', 'Charcoal Gray', 'Ice Blue'],
                'Streetwear': ['Teal', 'Soft Pink', 'Light Gray', 'Mint'],
                'Athleisure': ['Steel Blue', 'Lavender', 'Emerald', 'Pale Pink']
            },
            'neutral': {
                'Formal': ['Charcoal', 'Beige', 'Navy Blue', 'Heather Gray'],
                'Streetwear': ['Taupe', 'Emerald Green', 'Turquoise', 'Light Gray'],
                'Athleisure': ['Mint', 'Coral', 'Khaki', 'Steel Blue']
            }
        }
    
    def _load_color_map(self):
        """Load color recommendations from Excel file if available."""
        if self.color_map_path and os.path.exists(self.color_map_path):
            try:
                self.color_map_data = pd.read_excel(self.color_map_path)
                logging.info(f"Loaded color map from {self.color_map_path}")
            except Exception as e:
                logging.warning(f"Could not load color map from {self.color_map_path}: {e}")
                self.color_map_data = None
        else:
            logging.info("Using default color recommendations")
    
    def get_css_color(self, color_name: str) -> str:
        """Convert a descriptive color name to a CSS color value."""
        if color_name in self.COLOR_MAP:
            return self.COLOR_MAP[color_name]
        
        # Try to find a partial match
        for key, value in self.COLOR_MAP.items():
            if color_name.lower() in key.lower() or key.lower() in color_name.lower():
                return value
        
        # If no match found, return a default color
        return '#CCCCCC'
    
    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB values to LAB color space using OpenCV."""
        logging.debug(f"Input RGB values: {rgb}")
        
        # Ensure RGB is in the correct format for OpenCV
        rgb = np.asarray(rgb, dtype=np.uint8)
        
        # Reshape to (1, 1, 3) for OpenCV color conversion
        rgb_reshaped = rgb.reshape(1, 1, 3)
        
        # Convert RGB to LAB using OpenCV
        lab_reshaped = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2LAB)
        
        # Extract the LAB values
        lab = lab_reshaped[0, 0].astype(float)
        
        logging.debug(f"LAB values: {lab}")
        
        return lab
    
    def detect_skin(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect skin regions in an RGB image using multiple methods.
        
        Args:
            image_rgb: Input image in RGB format
            
        Returns:
            Tuple of (skin_regions, skin_mask)
        """
        height, width = image_rgb.shape[:2]
        
        # Method 1: YCrCb color space
        image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCR_CB)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        skin_mask_ycrcb = cv2.inRange(image_ycrcb, min_YCrCb, max_YCrCb)
        
        # Method 2: HSV color space for additional skin detection
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        min_HSV = np.array([0, 40, 60], np.uint8)
        max_HSV = np.array([25, 255, 255], np.uint8)
        skin_mask_hsv = cv2.inRange(image_hsv, min_HSV, max_HSV)
        
        # Method 3: RGB thresholding for broader skin tones
        r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
        skin_mask_rgb = ((r > 95) & (g > 40) & (b > 20) & 
                        (r > g) & (r > b) & 
                        (np.abs(r.astype(int) - g.astype(int)) > 15))
        
        # Combine all methods
        skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask_rgb.astype(np.uint8) * 255)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to get skin regions
        skin_regions = cv2.bitwise_and(image_rgb, image_rgb, mask=skin_mask)
        
        return skin_regions, skin_mask
    
    def convert_fitzpatrick_scale(self, scale: str) -> str:
        """Convert Fitzpatrick scale to Excel format."""
        if '-' in scale:
            scale = scale.split('-')[0]  # Take the first number
        return f"Type {scale}"
    
    def get_recommended_colours_from_excel(self, undertone: str, fitzpatrick_type: str) -> Dict[str, List[Tuple[str, str]]]:
        """Get color recommendations from Excel file."""
        if self.color_map_data is None:
            return self._get_default_recommendations(undertone)
        
        logging.debug(f"Looking for recommendations for undertone: {undertone}, Fitzpatrick Type: {fitzpatrick_type}")
        
        # Convert Fitzpatrick type to Excel format
        excel_fitzpatrick = self.convert_fitzpatrick_scale(fitzpatrick_type)
        logging.debug(f"Converted Fitzpatrick type to Excel format: {excel_fitzpatrick}")
        
        # Clean up the data
        color_map = self.color_map_data.copy()
        color_map['Undertone'] = color_map['Undertone'].str.strip()
        color_map['Fitzpatrick Type'] = color_map['Fitzpatrick Type'].str.strip()
        
        # Try exact match
        row = color_map[(color_map['Undertone'].str.lower() == undertone.lower()) & 
                       (color_map['Fitzpatrick Type'] == excel_fitzpatrick)]
        
        if row.empty:
            logging.debug("No matching row found in colour map, using defaults")
            return self._get_default_recommendations(undertone)
        
        def clean_colors(color_str):
            if pd.isna(color_str):
                return []
            colors = [c.strip() for c in str(color_str).split(',')]
            return [(c, self.get_css_color(c)) for c in colors if c]
        
        recommendations = {
            'Formal': clean_colors(row['Formal'].iloc[0]) if 'Formal' in row.columns else [],
            'Streetwear': clean_colors(row['Streetwear'].iloc[0]) if 'Streetwear' in row.columns else [],
            'Athleisure': clean_colors(row['Athleisure'].iloc[0]) if 'Athleisure' in row.columns else []
        }
        
        return recommendations
    
    def _get_default_recommendations(self, undertone: str) -> Dict[str, List[Tuple[str, str]]]:
        """Get default color recommendations when Excel file is not available."""
        undertone_key = undertone.lower()
        if undertone_key not in self.DEFAULT_RECOMMENDATIONS:
            undertone_key = 'neutral'  # fallback
        
        recommendations = {}
        for category, colors in self.DEFAULT_RECOMMENDATIONS[undertone_key].items():
            recommendations[category] = [(color, self.get_css_color(color)) for color in colors]
        
        return recommendations
    
    def analyze_skin_tone_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze skin tone from image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self._analyze_skin_tone(image_rgb)
    
    def analyze_skin_tone_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze skin tone from image bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image from bytes")
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self._analyze_skin_tone(image_rgb)
    
    def analyze_skin_tone_from_base64(self, base64_string: str) -> Dict[str, Any]:
        """
        Analyze skin tone from base64 encoded image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            return self.analyze_skin_tone_from_bytes(image_bytes)
            
        except Exception as e:
            raise ValueError(f"Could not decode base64 image: {e}")
    
    def _analyze_skin_tone(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Core skin tone analysis logic.
        
        Args:
            image_rgb: Input image in RGB format
            
        Returns:
            Dictionary containing analysis results
        """
        logging.debug("Starting skin tone analysis")
        
        # Detect skin regions
        skin_regions, skin_mask = self.detect_skin(image_rgb)
        
        logging.debug(f"Image shape: {image_rgb.shape}")
        logging.debug(f"Skin mask shape: {skin_mask.shape if skin_mask is not None else None}")
        logging.debug(f"Skin pixels detected: {np.sum(skin_mask) if skin_mask is not None else 0}")
        
        if skin_mask is None or np.sum(skin_mask) == 0:
            # If no skin detected, fall back to using the center region of the image
            logging.warning("No skin regions detected, using center region as fallback")
            height, width = image_rgb.shape[:2]
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            
            # Create a circular mask in the center
            y, x = np.ogrid[:height, :width]
            center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            skin_mask = center_mask.astype(np.uint8) * 255
            skin_regions = cv2.bitwise_and(image_rgb, image_rgb, mask=skin_mask)
            
            logging.debug(f"Fallback skin pixels: {np.sum(skin_mask)}")
            
            if np.sum(skin_mask) == 0:
                raise ValueError("No skin regions detected in the image")
        
        # Get average skin tone from detected regions
        skin_pixels = image_rgb[skin_mask > 0]
        avg_skin_tone = np.mean(skin_pixels, axis=0)
        
        logging.debug(f"Raw average skin tone (RGB): {avg_skin_tone}")
        
        # Convert to LAB color space for better analysis
        lab_skin_tone = self.rgb_to_lab(avg_skin_tone)
        
        # Determine undertone based on a and b values
        a_value = lab_skin_tone[1]  # Green-Red axis
        b_value = lab_skin_tone[2]  # Blue-Yellow axis
        
        logging.debug(f"LAB a value (green-red): {a_value}")
        logging.debug(f"LAB b value (blue-yellow): {b_value}")
        
        # Define thresholds for undertone determination
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
        
        # Determine Fitzpatrick scale based on lightness
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
        
        # Get dominant colors from skin regions
        unique_colors, counts = np.unique(skin_pixels, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[-5:][::-1]
        dominant_colors = unique_colors[top_indices].tolist()
        
        # Get color recommendations
        recommended_colours = self.get_recommended_colours_from_excel(undertone, fitzpatrick)
        
        # Prepare results
        results = {
            'average_skin_tone': avg_skin_tone.tolist(),
            'undertone': undertone,
            'fitzpatrick_scale': fitzpatrick,
            'lightness': float(lightness),
            'a_value': float(a_value),
            'b_value': float(b_value),
            'dominant_colors': dominant_colors,
            'recommended_colours': recommended_colours,
            'skin_regions_detected': bool(np.sum(skin_mask) > 0),
            'analysis_metadata': {
                'lab_values': lab_skin_tone.tolist(),
                'skin_pixel_count': int(np.sum(skin_mask > 0)),
                'total_pixels': int(skin_mask.size)
            }
        }
        
        logging.debug("Skin tone analysis completed successfully")
        return results


# Convenience functions for direct usage
def analyze_skin_tone_from_file(image_path: str, color_map_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze skin tone from image file.
    
    Args:
        image_path: Path to the image file
        color_map_path: Optional path to Excel color map file
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = SkinToneAnalyzer(color_map_path)
    return analyzer.analyze_skin_tone_from_path(image_path)


def analyze_skin_tone_from_bytes(image_bytes: bytes, color_map_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze skin tone from image bytes.
    
    Args:
        image_bytes: Image data as bytes
        color_map_path: Optional path to Excel color map file
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = SkinToneAnalyzer(color_map_path)
    return analyzer.analyze_skin_tone_from_bytes(image_bytes)


def analyze_skin_tone_from_base64(base64_string: str, color_map_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze skin tone from base64 encoded image.
    
    Args:
        base64_string: Base64 encoded image string
        color_map_path: Optional path to Excel color map file
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = SkinToneAnalyzer(color_map_path)
    return analyzer.analyze_skin_tone_from_base64(base64_string)


# Example usage
if __name__ == "__main__":
    # Example 1: Analyze from file path
    try:
        results = analyze_skin_tone_from_file("test_image.jpg")
        print("Analysis Results:")
        print(f"Undertone: {results['undertone']}")
        print(f"Fitzpatrick Scale: {results['fitzpatrick_scale']}")
        print(f"Lightness: {results['lightness']:.2f}")
        print(f"Recommended colors: {results['recommended_colours']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using the class directly
    analyzer = SkinToneAnalyzer("Colour map.xlsx")  # Optional Excel file
    
    # You can now call:
    # results = analyzer.analyze_skin_tone_from_path("image.jpg")
    # results = analyzer.analyze_skin_tone_from_bytes(image_bytes)
    # results = analyzer.analyze_skin_tone_from_base64(base64_string) 