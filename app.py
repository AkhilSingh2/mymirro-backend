import os
from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, List, Tuple, Optional

# Import the enhanced color analysis API
from color_analysis_api import EnhancedColorAnalysisAPI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Swagger UI
api = Api(
    app,
    version='1.0.0',
    title='MyMirror Backend API',
    description='A comprehensive backend API for fashion recommendation system with color analysis, outfit generation, and personalized styling.',
    doc='/swagger/',  # Swagger UI will be available at /swagger/
    prefix='/api/v1'
)

# Create namespaces for better organization
color_ns = api.namespace('color', description='Color Analysis Operations')
outfit_ns = api.namespace('outfits', description='Outfit Generation and Recommendations')
products_ns = api.namespace('products', description='Product Similarity and Recommendations (Phase 3)')
health_ns = api.namespace('health', description='Health Check Operations')

# Define API models for request/response documentation
color_request_model = api.model('ColorAnalysisRequest', {
    'image': fields.String(required=True, description='Base64 encoded image string', example='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...')
})

# New model for manual hex color analysis
hex_color_request_model = api.model('HexColorAnalysisRequest', {
    'hex_color': fields.String(required=True, description='Hex color code representing skin tone', example='#FDB4A6')
})

# Unified model that supports both photo and hex input
unified_color_request_model = api.model('UnifiedColorAnalysisRequest', {
    'image': fields.String(required=False, description='Base64 encoded image string (for photo analysis)', example='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...'),
    'hex_color': fields.String(required=False, description='Hex color code representing skin tone (for manual selection)', example='#FDB4A6')
})

# Updated models to match the new SkinToneAnalyzer output
dominant_color_model = api.model('DominantColor', {
    'rgb': fields.List(fields.Integer, description='RGB color values'),
    'hex': fields.String(description='Hex color code', example='#9C8677')
})

recommended_color_category_model = api.model('RecommendedColorCategory', {
    'color_name': fields.String(description='Color name', example='Navy Blue'),
    'css_color': fields.String(description='CSS color code', example='#000080')
})

category_recommendations_model = api.model('CategoryRecommendations', {
    'Formal': fields.List(fields.List(fields.String), description='Formal wear color recommendations'),
    'Streetwear': fields.List(fields.List(fields.String), description='Streetwear color recommendations'), 
    'Athleisure': fields.List(fields.List(fields.String), description='Athleisure color recommendations')
})

analysis_metadata_model = api.model('AnalysisMetadata', {
    'lab_values': fields.List(fields.Float, description='LAB color space values'),
    'skin_pixel_count': fields.Integer(description='Number of skin pixels detected'),
    'total_pixels': fields.Integer(description='Total image pixels'),
    'input_method': fields.String(description='Method used for analysis', example='photo_upload'),
    'input_hex': fields.String(description='Original hex color if manual input', example='#FDB4A6')
})

analysis_model = api.model('Analysis', {
    'average_skin_tone': fields.List(fields.Float, description='Average skin tone RGB values'),
    'undertone': fields.String(description='Skin undertone', example='Warm'),
    'fitzpatrick_scale': fields.String(description='Fitzpatrick skin type', example='III'),
    'lightness': fields.Float(description='Skin lightness value', example=65.2),
    'a_value': fields.Float(description='LAB a-value (green-red axis)', example=5.1),
    'b_value': fields.Float(description='LAB b-value (blue-yellow axis)', example=12.3),
    'dominant_colors': fields.List(fields.List(fields.Integer), description='Dominant skin colors'),
    'recommended_colours': fields.Nested(category_recommendations_model),
    'skin_regions_detected': fields.Boolean(description='Whether skin regions were detected'),
    'analysis_metadata': fields.Nested(analysis_metadata_model)
})

color_response_model = api.model('ColorAnalysisResponse', {
    'success': fields.Boolean(description='Analysis success status'),
    'average_skin_tone': fields.List(fields.Float, description='Average skin tone RGB values'),
    'undertone': fields.String(description='Skin undertone', example='Warm'),
    'fitzpatrick_scale': fields.String(description='Fitzpatrick skin type', example='III'),
    'lightness': fields.Float(description='Skin lightness value', example=65.2),
    'a_value': fields.Float(description='LAB a-value (green-red axis)', example=5.1),
    'b_value': fields.Float(description='LAB b-value (blue-yellow axis)', example=12.3),
    'dominant_colors': fields.List(fields.List(fields.Integer), description='Dominant skin colors'),
    'recommended_colours': fields.Nested(category_recommendations_model),
    'skin_regions_detected': fields.Boolean(description='Whether skin regions were detected'),
    'analysis_metadata': fields.Nested(analysis_metadata_model)
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(description='API status', example='healthy'),
    'service': fields.String(description='Service name', example='MyMirror Backend API'),
    'version': fields.String(description='API version', example='1.0.0')
})

# Outfit Generation Models
outfit_request_model = api.model('OutfitGenerationRequest', {
    'user_id': fields.Integer(required=True, description='User ID for outfit generation', example=2),
    'regenerate': fields.Boolean(required=False, description='Force regenerate outfits even if they exist', example=False, default=False)
})

product_model = api.model('Product', {
    'id': fields.String(description='Product ID', example='ethnic_main_2_1'),
    'title': fields.String(description='Product title', example='Men Solid Casual Shirt'),
    'image': fields.String(description='Product image URL', example='https://example.com/image.jpg'),
    'price': fields.Float(description='Product price', example=1299.0),
    'style': fields.String(description='Product style', example='Contemporary'),
    'color': fields.String(description='Product color', example='Blue'),
    'semantic_score': fields.Float(description='AI matching score', example=0.85)
})

outfit_model = api.model('Outfit', {
    'main_outfit_id': fields.String(description='Unique outfit ID', example='ethnic_main_2_1'),
    'rank': fields.Integer(description='Outfit ranking', example=1),
    'score': fields.Float(description='Overall outfit score', example=0.87),
    'explanation': fields.String(description='Outfit explanation', example='Ethnic wear recommendation | Men Solid Casual Shirt | Matches your contemporary style preference'),
    'top': fields.Nested(product_model, description='Top product details'),
    'bottom': fields.Nested(product_model, description='Bottom product details'),
    'total_price': fields.Float(description='Combined price of outfit', example=2099.0),
    'generated_at': fields.String(description='Generation timestamp', example='2024-01-15T10:30:00'),
    'generation_method': fields.String(description='Method used for generation', example='ethnic_wear_specialized')
})

outfit_generation_response_model = api.model('OutfitGenerationResponse', {
    'success': fields.Boolean(description='Generation success status'),
    'message': fields.String(description='Response message', example='Successfully generated 50 outfits for user 2'),
    'user_id': fields.Integer(description='User ID', example=2),
    'outfits_count': fields.Integer(description='Number of outfits generated', example=50),
    'generation_time_seconds': fields.Float(description='Time taken to generate', example=12.5),
    'data_source': fields.String(description='Where data is stored', example='supabase')
})

outfit_list_response_model = api.model('OutfitListResponse', {
    'success': fields.Boolean(description='Request success status'),
    'user_id': fields.Integer(description='User ID', example=2),
    'outfits': fields.List(fields.Nested(outfit_model), description='List of outfits'),
    'total_count': fields.Integer(description='Total number of outfits', example=50),
    'filters_applied': fields.Raw(description='Any filters that were applied')
})

# Phase 2: Similar Outfits Models
similar_outfit_request_model = api.model('SimilarOutfitRequest', {
    'outfit_id': fields.String(required=True, description='Main outfit ID to find similar outfits for', example='main_2_1'),
    'count': fields.Integer(required=False, description='Number of similar outfits to return (default: 10)', example=10, default=10)
})

score_breakdown_model = api.model('ScoreBreakdown', {
    'semantic_similarity': fields.Float(description='Semantic matching score', example=0.85),
    'style_harmony': fields.Float(description='Style compatibility score', example=0.92),
    'color_harmony': fields.Float(description='Color harmony score', example=0.78),
    'formality_matching': fields.Float(description='Formality level matching', example=0.88),
    'pattern_compatibility': fields.Float(description='Pattern mixing compatibility', example=0.80),
    'price_similarity': fields.Float(description='Price range compatibility', example=0.75),
    'occasion_matching': fields.Float(description='Occasion appropriateness', example=0.90),
    'seasonal_appropriateness': fields.Float(description='Seasonal compatibility', example=0.80),
    'final_score': fields.Float(description='Overall similarity score', example=0.83),
    'explanation': fields.String(description='Detailed score breakdown', example='Semantic match: 0.85 | Style compatibility: 0.92 | Color harmony: 0.78')
})

similar_outfit_model = api.model('SimilarOutfit', {
    'outfit_data': fields.Nested(outfit_model, description='Similar outfit details'),
    'similarity_score': fields.Float(description='Overall similarity score (0.0-1.0)', example=0.83),
    'score_breakdown': fields.Nested(score_breakdown_model, description='Detailed scoring breakdown'),
    'source_outfit_id': fields.String(description='Original outfit ID used for comparison', example='main_2_1'),
    'generated_at': fields.String(description='When the similar outfit was found', example='2024-01-15T14:30:00')
})

similar_outfits_response_model = api.model('SimilarOutfitsResponse', {
    'success': fields.Boolean(description='Request success status'),
    'source_outfit_id': fields.String(description='Original outfit ID', example='main_2_1'),
    'similar_outfits': fields.List(fields.Nested(similar_outfit_model), description='List of similar outfits'),
    'total_found': fields.Integer(description='Number of similar outfits found', example=8),
    'search_time_seconds': fields.Float(description='Time taken to find similar outfits', example=2.3),
    'algorithm_info': fields.String(description='Algorithm and weights used', example='FAISS semantic search + 8-factor fashion intelligence')
})

# Phase 3: Similar Products Models
user_preferences_model = api.model('UserPreferences', {
    'preferred_styles': fields.List(fields.String, description='User preferred styles', example=['Business Formal', 'Casual']),
    'preferred_colors': fields.List(fields.String, description='User preferred colors', example=['Black', 'Navy', 'White']),
    'price_range': fields.List(fields.Float, description='Price range preference [min, max]', example=[800, 2500])
})

filters_model = api.model('Filters', {
    'price_range': fields.List(fields.Float, description='Price range filter [min, max]', example=[500, 3000]),
    'styles': fields.List(fields.String, description='Style filters', example=['Business', 'Formal']),
    'colors': fields.List(fields.String, description='Color filters', example=['Black', 'Navy'])
})

similar_products_request_model = api.model('SimilarProductsRequest', {
    'count': fields.Integer(required=False, description='Number of similar products to return (default: 10)', example=8, default=10),
    'user_preferences': fields.Nested(user_preferences_model, required=False, description='User style/color/price preferences'),
    'filters': fields.Nested(filters_model, required=False, description='Additional filters to apply'),
    'diverse': fields.Boolean(required=False, description='Enable diversity filtering for more varied results', example=True, default=True),
    'personalized': fields.Boolean(required=False, description='Enable personalized recommendations', example=True, default=True)
})

product_score_breakdown_model = api.model('ProductScoreBreakdown', {
    'semantic_similarity': fields.Float(description='Core semantic matching score', example=0.85),
    'style_compatibility': fields.Float(description='Style compatibility within same category', example=0.78),
    'color_diversity': fields.Float(description='Color diversity bonus score', example=0.90),
    'design_diversity': fields.Float(description='Design diversity bonus score', example=0.82),
    'price_similarity': fields.Float(description='Price similarity score', example=0.75),
    'user_preference_boost': fields.Float(description='User preference boost factor', example=1.15),
    'type_boost': fields.Float(description='Candidate type boost factor', example=1.1),
    'final_score': fields.Float(description='Overall similarity score', example=0.83)
})

similar_product_model = api.model('SimilarProduct', {
    'product_id': fields.String(description='Product identifier', example='3790'),
    'title': fields.String(description='Product title', example='Men Solid Formal Shirt'),
    'image_url': fields.String(description='Product image URL', example='https://images.example.com/shirt.jpg'),
    'price': fields.Float(description='Product price', example=1299.0),
    'brand': fields.String(description='Product brand', example='Allen Solly'),
    'style': fields.String(description='Product style', example='Business Formal'),
    'color': fields.String(description='Product color', example='Navy'),
    'wear_type': fields.String(description='Product category', example='Upperwear'),
    'occasion': fields.String(description='Suitable occasion', example='Formal'),
    'similarity_score': fields.Float(description='Overall similarity score (0.0-1.0)', example=0.83),
    'score_breakdown': fields.Nested(product_score_breakdown_model, description='Detailed scoring breakdown'),
    'candidate_type': fields.String(description='Type of recommendation', example='color_diverse'),
    'source_product_id': fields.String(description='Original product ID used for comparison', example='5438'),
    'generated_at': fields.String(description='When the similar product was found', example='2024-01-15T14:30:00')
})

similar_products_response_model = api.model('SimilarProductsResponse', {
    'success': fields.Boolean(description='Request success status'),
    'source_product_id': fields.String(description='Original product ID', example='5438'),
    'source_product_title': fields.String(description='Original product title', example='Men Formal Business Blazer'),
    'similar_products': fields.List(fields.Nested(similar_product_model), description='List of similar products'),
    'total_found': fields.Integer(description='Number of similar products found', example=8),
    'search_time_seconds': fields.Float(description='Time taken to find similar products', example=1.8),
    'algorithm_info': fields.String(description='Algorithm and features used', example='Enhanced same-category similarity with color/design diversity + user personalization'),
    'personalization_applied': fields.Boolean(description='Whether user preferences were applied', example=True),
    'diversity_filtering_applied': fields.Boolean(description='Whether diversity filtering was applied', example=True)
})



# Initialize the enhanced color analysis API
color_api = EnhancedColorAnalysisAPI()

# Import outfit generation functionality
try:
    from phase1_supabase_outfits_generator import SupabaseMainOutfitsGenerator
    from database import SupabaseDB
    outfit_generator_available = True
except ImportError as e:
    logger.warning(f"Outfit generation modules not available: {e}")
    outfit_generator_available = False

# Import Phase 2 similar outfits functionality
try:
    from phase2_supabase_similar_outfits_api import SupabaseSimilarOutfitsGenerator
    similar_outfits_available = True
except ImportError as e:
    logger.warning(f"Similar outfits modules not available: {e}")
    similar_outfits_available = False

# Import Phase 3 similar products functionality
try:
    from phase3_supabase_similar_products_api import SupabaseEnhancedSimilarProductsGenerator
    similar_products_available = True
    logger.info("✅ Phase 3 Similar Products Generator loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Phase 3 Similar Products Generator not available: {e}")
    similar_products_available = False

@outfit_ns.route('/generate')
class OutfitGeneration(Resource):
    """Generate personalized outfits for a user"""
    
    @api.doc('generate_outfits')
    @api.expect(outfit_request_model)
    @api.marshal_with(outfit_generation_response_model)
    @api.response(200, 'Success', outfit_generation_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Generate personalized outfit recommendations for a user
        
        This endpoint generates 50+ personalized outfit recommendations using AI-powered
        semantic search and fashion intelligence. The outfits are saved to the database
        and can be retrieved via the /outfits/{user_id} endpoint.
        """
        try:
            if not outfit_generator_available:
                return {
                    'success': False,
                    'message': 'Outfit generation service is not available'
                }, 500
            
            data = request.get_json()
            if not data:
                return {
                    'success': False,
                    'message': 'No JSON data provided'
                }, 400
            
            user_id = data.get('user_id')
            regenerate = data.get('regenerate', False)
            
            if not user_id:
                return {
                    'success': False,
                    'message': 'user_id is required'
                }, 400
            
            # Initialize generator
            import time
            start_time = time.time()
            
            generator = SupabaseMainOutfitsGenerator()
            
            # Check if outfits already exist (unless regenerate is True)
            if not regenerate:
                db = SupabaseDB()
                existing_outfits = db.get_user_outfits(user_id)
                if existing_outfits is not None and len(existing_outfits) > 0:
                    return {
                        'success': True,
                        'message': f'User {user_id} already has {len(existing_outfits)} outfits. Use regenerate=true to force regeneration.',
                        'user_id': user_id,
                        'outfits_count': len(existing_outfits),
                        'generation_time_seconds': 0.0,
                        'data_source': 'existing_database'
                    }, 200
            
            # Generate outfits
            success = generator.generate_and_save_outfits(user_id)
            
            generation_time = time.time() - start_time
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully generated 50 outfits for user {user_id}',
                    'user_id': user_id,
                    'outfits_count': 50,
                    'generation_time_seconds': round(generation_time, 2),
                    'data_source': 'supabase'
                }, 200
            else:
                return {
                    'success': False,
                    'message': f'Failed to generate outfits for user {user_id}'
                }, 500
                
        except Exception as e:
            logger.error(f"Error in outfit generation: {e}")
            return {
                'success': False,
                'message': f'Internal server error: {str(e)}'
            }, 500

@outfit_ns.route('/<int:user_id>')
class UserOutfits(Resource):
    """Get outfit recommendations for a user"""
    
    @api.doc('get_user_outfits', 
             params={
                 'limit': {'description': 'Maximum number of outfits to return', 'type': 'integer', 'default': None, 'example': 10},
                 'min_score': {'description': 'Minimum outfit score (0.0 to 1.0)', 'type': 'number', 'default': None, 'example': 0.7},
                 'style': {'description': 'Filter by style (case-insensitive)', 'type': 'string', 'default': None, 'example': 'contemporary'}
             })
    @api.marshal_with(outfit_list_response_model)
    @api.response(200, 'Success', outfit_list_response_model)
    @api.response(404, 'User not found or no outfits', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    def get(self, user_id):
        """
        Get outfit recommendations for a specific user
        
        Returns generated outfits for the user, ranked by AI score.
        If no outfits exist, use the /outfits/generate endpoint first.
        
        Query Parameters:
        - limit: Maximum number of outfits to return (e.g., 10)
        - min_score: Minimum outfit score between 0.0 and 1.0 (e.g., 0.7)  
        - style: Filter by style name, case-insensitive (e.g., "contemporary")
        
        Examples:
        - /api/v1/outfits/2?limit=5
        - /api/v1/outfits/2?limit=10&min_score=0.8
        - /api/v1/outfits/2?style=contemporary&limit=3
        """
        try:
            if not outfit_generator_available:
                return {
                    'success': False,
                    'message': 'Outfit service is not available'
                }, 500
            
            # Get query parameters
            limit = request.args.get('limit', type=int)
            min_score = request.args.get('min_score', type=float)
            style_filter = request.args.get('style')
            
            # Get outfits from database
            db = SupabaseDB()
            outfits_data = db.get_user_outfits(user_id)
            
            if outfits_data is None or len(outfits_data) == 0:
                return {
                    'success': False,
                    'message': f'No outfits found for user {user_id}. Generate outfits first using /outfits/generate'
                }, 404
            
            # Convert DataFrame to list of dictionaries
            outfits_list = outfits_data.to_dict('records')
            
            # Apply filters
            filtered_outfits = outfits_list
            filters_applied = {}
            
            if min_score is not None:
                filtered_outfits = [o for o in filtered_outfits if o.get('score', 0) >= min_score]
                filters_applied['min_score'] = min_score
                
            if style_filter:
                filtered_outfits = [o for o in filtered_outfits if 
                                  style_filter.lower() in (str(o.get('top_style', '')) + ' ' + str(o.get('bottom_style', ''))).lower()]
                filters_applied['style'] = style_filter
            
            if limit:
                filtered_outfits = filtered_outfits[:limit]
                filters_applied['limit'] = limit
            
            # Transform data for API response
            formatted_outfits = []
            for outfit in filtered_outfits:
                formatted_outfit = {
                    'main_outfit_id': outfit.get('main_outfit_id', ''),
                    'rank': outfit.get('rank', 0),
                    'score': outfit.get('score', 0.0),
                    'explanation': outfit.get('explanation', ''),
                    'top': {
                        'id': outfit.get('top_id', ''),
                        'title': outfit.get('top_title', ''),
                        'image': outfit.get('top_image', ''),
                        'price': outfit.get('top_price', 0.0),
                        'style': outfit.get('top_style', ''),
                        'color': outfit.get('top_color', ''),
                        'semantic_score': outfit.get('top_semantic_score', 0.0)
                    },
                    'bottom': {
                        'id': outfit.get('bottom_id', ''),
                        'title': outfit.get('bottom_title', ''),
                        'image': outfit.get('bottom_image', ''),
                        'price': outfit.get('bottom_price', 0.0),
                        'style': outfit.get('bottom_style', ''),
                        'color': outfit.get('bottom_color', ''),
                        'semantic_score': outfit.get('bottom_semantic_score', 0.0)
                    },
                    'total_price': outfit.get('total_price', 0.0),
                    'generated_at': outfit.get('generated_at', ''),
                    'generation_method': outfit.get('generation_method', '')
                }
                formatted_outfits.append(formatted_outfit)
            
            return {
                'success': True,
                'user_id': user_id,
                'outfits': formatted_outfits,
                'total_count': len(formatted_outfits),
                'filters_applied': filters_applied
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting user outfits: {e}")
            return {
                'success': False,
                'message': f'Internal server error: {str(e)}'
            }, 500

@outfit_ns.route('/<string:outfit_id>/similar')
class SimilarOutfits(Resource):
    """Find similar outfits for a given outfit"""
    
    @api.doc('get_similar_outfits',
             params={
                 'count': {'description': 'Number of similar outfits to return', 'type': 'integer', 'default': 10, 'example': 5}
             })
    @api.marshal_with(similar_outfits_response_model)
    @api.response(200, 'Success', similar_outfits_response_model)
    @api.response(404, 'Outfit not found', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    def get(self, outfit_id):
        """
        Find similar outfits for a given outfit using advanced AI similarity
        
        This endpoint uses Phase 2 AI-powered similarity matching to find outfits
        that are similar to a given main outfit. It employs:
        
        **Advanced Similarity Factors:**
        - 🔍 **Semantic Similarity**: AI-powered text matching of product descriptions
        - 👔 **Style Harmony**: Fashion style compatibility and formality matching
        - 🎨 **Color Theory**: Professional color harmony and coordination rules
        - 📐 **Pattern Compatibility**: Smart pattern mixing guidelines
        - 💰 **Price Range**: Similar budget/price point matching
        - 🎯 **Occasion Matching**: Event and use-case appropriateness
        - 🌟 **Quality Scoring**: Product quality and versatility factors
        
        **Input:**
        - outfit_id: Main outfit ID from Phase 1 (e.g., "main_2_1")
        - count: Number of similar outfits to return (default: 10)
        
        **Output:**
        - List of similar outfits ranked by similarity score
        - Detailed score breakdowns for each factor
        - Real-time generation (typically 2-5 seconds)
        
        **Examples:**
        - `/api/v1/outfits/main_2_1/similar?count=5`
        - `/api/v1/outfits/main_2_3/similar` (default count=10)
        
        **Note:** The source outfit must exist in the database (generated via Phase 1).
        """
        try:
            if not similar_outfits_available:
                return {
                    'success': False,
                    'message': 'Similar outfits service is not available'
                }, 500
            
            # Get query parameters
            count = request.args.get('count', 10, type=int)
            
            # Validate count parameter
            if count < 1 or count > 50:
                return {
                    'success': False,
                    'message': 'Count must be between 1 and 50'
                }, 400
            
            # Initialize similar outfits generator
            import time
            start_time = time.time()
            
            generator = SupabaseSimilarOutfitsGenerator()
            
            # Find similar outfits
            similar_outfits = generator.find_similar_outfits(outfit_id, num_similar=count)
            
            search_time = time.time() - start_time
            
            if not similar_outfits:
                return {
                    'success': False,
                    'message': f'No similar outfits found for outfit {outfit_id}. Outfit may not exist or no suitable matches available.'
                }, 404
            
            # Format response
            formatted_similar_outfits = []
            for similar_outfit in similar_outfits:
                outfit_data = similar_outfit['outfit_data']
                
                formatted_outfit = {
                    'outfit_data': {
                        'main_outfit_id': f"similar_{outfit_id}_{len(formatted_similar_outfits) + 1}",
                        'rank': len(formatted_similar_outfits) + 1,
                        'score': similar_outfit['similarity_score'],
                        'explanation': f"Similar to {outfit_id} | {outfit_data.get('top_title', '')[:50]}... + {outfit_data.get('bottom_title', '')[:50]}...",
                        'top': {
                            'id': outfit_data.get('top_id', ''),
                            'title': outfit_data.get('top_title', ''),
                            'image': outfit_data.get('top_image', ''),
                            'price': outfit_data.get('top_price', 0.0),
                            'style': outfit_data.get('top_style', ''),
                            'color': outfit_data.get('top_color', ''),
                            'semantic_score': 0.0  # Not directly available in Phase 2
                        },
                        'bottom': {
                            'id': outfit_data.get('bottom_id', ''),
                            'title': outfit_data.get('bottom_title', ''),
                            'image': outfit_data.get('bottom_image', ''),
                            'price': outfit_data.get('bottom_price', 0.0),
                            'style': outfit_data.get('bottom_style', ''),
                            'color': outfit_data.get('bottom_color', ''),
                            'semantic_score': 0.0  # Not directly available in Phase 2
                        },
                        'total_price': outfit_data.get('total_price', 0.0),
                        'generated_at': similar_outfit.get('generated_at', ''),
                        'generation_method': 'phase2_similar_outfits_ai'
                    },
                    'similarity_score': similar_outfit['similarity_score'],
                    'score_breakdown': similar_outfit['score_breakdown'],
                    'source_outfit_id': similar_outfit['source_outfit_id'],
                    'generated_at': similar_outfit['generated_at']
                }
                formatted_similar_outfits.append(formatted_outfit)
            
            return {
                'success': True,
                'source_outfit_id': outfit_id,
                'similar_outfits': formatted_similar_outfits,
                'total_found': len(formatted_similar_outfits),
                'search_time_seconds': round(search_time, 2),
                'algorithm_info': 'FAISS semantic search + 8-factor fashion intelligence (semantic, style, color, formality, pattern, price, occasion, seasonal)'
            }, 200
            
        except ValueError as e:
            # Handle outfit not found
            logger.warning(f"Outfit not found: {e}")
            return {
                'success': False,
                'message': f'Outfit {outfit_id} not found in database'
            }, 404
            
        except Exception as e:
            logger.error(f"Error finding similar outfits: {e}")
            return {
                'success': False,
                'message': f'Internal server error: {str(e)}'
            }, 500

@products_ns.route('/<string:product_id>/similar')
class SimilarProducts(Resource):
    """Find similar products within the same category (Phase 3)"""
    
    @api.doc('get_similar_products',
             params={
                 'count': {'description': 'Number of similar products to return', 'type': 'integer', 'default': 10, 'example': 8},
                 'diverse': {'description': 'Enable diversity filtering for varied results', 'type': 'boolean', 'default': True, 'example': True},
                 'personalized': {'description': 'Enable personalized recommendations', 'type': 'boolean', 'default': False, 'example': True}
             })
    @api.expect(similar_products_request_model, validate=False)
    @api.marshal_with(similar_products_response_model)
    @api.response(200, 'Success', similar_products_response_model)
    @api.response(404, 'Product not found', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    def post(self, product_id):
        """
        Find similar products within the same category with diversity and personalization
        
        **Phase 3 ENHANCED Features:**
        
        🎯 **Same-Category Focus:** Only finds products in the same category (e.g., shirts → shirts)
        
        🌈 **Color Diversity:** Intelligently recommends products in harmonious but different colors
        
        🎨 **Design Diversity:** Suggests different patterns, fits, and design variations
        
        👤 **Personalization:** Considers user style, color, and price preferences
        
        💰 **Smart Price Matching:** Dynamic price similarity based on product price range
        
        🔍 **Advanced AI:** FAISS semantic search + 6-factor fashion intelligence
        
        **Request Body (Optional):**
        ```json
        {
            "count": 8,
            "user_preferences": {
                "preferred_styles": ["Business Formal", "Casual"],
                "preferred_colors": ["Black", "Navy", "White"],
                "price_range": [800, 2500]
            },
            "filters": {
                "price_range": [500, 3000],
                "styles": ["Business", "Formal"],
                "colors": ["Black", "Navy"]
            },
            "diverse": true,
            "personalized": true
        }
        ```
        
        **Query Parameters:**
        - count: Number of products to return (default: 10)
        - diverse: Enable diversity filtering (default: true) 
        - personalized: Use user preferences (default: false)
        
        **Use Cases:**
        - "Show me similar shirts in different colors"
        - "Find formal wear like this but in my preferred style"
        - "Recommend similar products within my budget"
        """
        try:
            if not similar_products_available:
                return {
                    'success': False,
                    'message': 'Similar products service is not available'
                }, 500
            
            import time
            start_time = time.time()
            
            # Get query parameters
            count = request.args.get('count', 10, type=int)
            diverse = request.args.get('diverse', True, type=bool)
            personalized = request.args.get('personalized', False, type=bool)
            
            # Get request body (optional)
            data = request.get_json() or {}
            
            # Parse user preferences and filters from request body
            user_preferences = data.get('user_preferences')
            filters = data.get('filters')
            
            # Override with body parameters if provided
            if 'count' in data:
                count = data['count']
            if 'diverse' in data:
                diverse = data['diverse']
            if 'personalized' in data:
                personalized = data['personalized']
                
            # Disable personalization if no user preferences provided
            if personalized and not user_preferences:
                personalized = False
                logger.info("Personalization disabled - no user preferences provided")
            
            # Initialize Phase 3 generator
            generator = SupabaseEnhancedSimilarProductsGenerator()
            
            # Find similar products
            logger.info(f"🔍 Finding {count} similar products for product {product_id}")
            logger.info(f"   Diverse: {diverse}, Personalized: {personalized}")
            
            similar_products = generator.find_similar_products(
                product_id=product_id,
                num_similar=count,
                user_preferences=user_preferences if personalized else None,
                filters=filters
            )
            
            search_time = time.time() - start_time
            
            if not similar_products:
                return {
                    'success': False,
                    'message': f'No similar products found for product {product_id}. Product may not exist or no suitable matches available.'
                }, 404
            
            # Get source product info for response
            source_product_title = "Unknown Product"
            if similar_products:
                source_product_title = f"Product {product_id}"
                
            return {
                'success': True,
                'source_product_id': product_id,
                'source_product_title': source_product_title,
                'similar_products': similar_products,
                'total_found': len(similar_products),
                'search_time_seconds': round(search_time, 2),
                'algorithm_info': 'Enhanced same-category similarity with color/design diversity + user personalization',
                'personalization_applied': personalized and user_preferences is not None,
                'diversity_filtering_applied': diverse
            }, 200
            
        except ValueError as e:
            # Handle product not found
            logger.warning(f"Product not found: {e}")
            return {
                'success': False,
                'message': f'Product {product_id} not found in database'
            }, 404
            
        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            return {
                'success': False,
                'message': f'Internal server error: {str(e)}'
            }, 500

@health_ns.route('')
class Health(Resource):
    @api.doc('health_check')
    @api.marshal_with(health_response_model)
    def get(self):
        """Health check endpoint - Check if the API is running"""
        return {
            "status": "healthy",
            "service": "MyMirror Backend API",
            "version": "1.0.0"
        }

@color_ns.route('/analyze')
class ColorAnalysis(Resource):
    @api.doc('analyze_color_unified')
    @api.expect(unified_color_request_model)
    @api.marshal_with(color_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(200, 'Success', color_response_model)
    def post(self):
        """
        Enhanced skin tone analysis and color recommendations (Unified)
        
        This endpoint supports TWO modes of analysis:
        
        **Mode 1: Photo Analysis (Automatic)**
        - Provide 'image' field with base64 encoded image
        - Automatic skin tone detection from uploaded photos
        - Advanced skin undertone detection using LAB color space
        - Intelligent skin region detection and analysis
        
        **Mode 2: Manual Selection**
        - Provide 'hex_color' field with hex color code (e.g., "#FDB4A6")
        - User manually selects their skin tone color
        - Same analysis algorithm applied to the provided color
        
        **Output (Same for both modes):**
        - Fitzpatrick skin type classification (Type I-VI)
        - Precise lightness measurement and color analysis 
        - Dominant skin color information
        - Category-specific color recommendations (Formal, Streetwear, Athleisure)
        - Excel-based color mapping support
        - Professional fashion color coordination
        
        **Request Examples:**
        ```
        Photo mode: {"image": "data:image/jpeg;base64,/9j/4AAQ..."}
        Manual mode: {"hex_color": "#FDB4A6"}
        ```
        
        **Note:** Provide either 'image' OR 'hex_color', not both.
        """
        try:
            data = request.get_json()
            if not data:
                api.abort(400, "No JSON data provided")
            
            # Check which mode is being used
            has_image = 'image' in data and data.get('image')
            has_hex = 'hex_color' in data and data.get('hex_color')
            
            if not has_image and not has_hex:
                return {
                    'success': False,
                    'error': "Must provide either 'image' (for photo analysis) or 'hex_color' (for manual selection)"
                }, 400
            
            if has_image and has_hex:
                return {
                    'success': False,
                    'error': "Provide either 'image' OR 'hex_color', not both"
                }, 400
            
            # Route to appropriate analysis method
            if has_image:
                logger.info("🖼️ Processing photo analysis request")
                result = color_api.analyze_image(data['image'])
            else:  # has_hex
                logger.info(f"🎨 Processing manual hex color analysis: {data['hex_color']}")
                result = color_api.analyze_from_hex(data['hex_color'])
            
            if 'error' in result:
                return result, 400
            
            return result
            
        except Exception as e:
            logger.error(f"Error in unified analyze_color endpoint: {e}")
            return {
                'success': False,
                'error': f"Internal server error: {str(e)}"
            }, 500

@color_ns.route('/analyze-photo')
class PhotoColorAnalysis(Resource):
    @api.doc('analyze_color_photo')
    @api.expect(color_request_model)
    @api.marshal_with(color_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(200, 'Success', color_response_model)
    def post(self):
        """
        Photo-based skin tone analysis (Legacy endpoint)
        
        Upload a base64 encoded image to get comprehensive skin tone analysis.
        This is the original endpoint for photo-only analysis.
        
        **Recommendation:** Use `/analyze` endpoint which supports both photo and manual modes.
        """
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                api.abort(400, "Missing 'image' field in request")
            
            # Analyze the image
            result = color_api.analyze_image(data['image'])
            
            if 'error' in result:
                api.abort(400, result['error'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_color_photo endpoint: {e}")
            api.abort(500, "Internal server error")

@color_ns.route('/analyze-hex')
class HexColorAnalysis(Resource):
    @api.doc('analyze_color_hex')
    @api.expect(hex_color_request_model)
    @api.marshal_with(color_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(200, 'Success', color_response_model)
    def post(self):
        """
        Manual hex color skin tone analysis
        
        Provide a hex color code representing the user's skin tone to get 
        comprehensive analysis and color recommendations.
        
        **Input:** Hex color code (e.g., "#FDB4A6", "FDB4A6")
        **Output:** Same format as photo analysis
        
        **Benefits:**
        - No photo upload required
        - Privacy-friendly option
        - User has full control over their skin tone selection
        - Same professional color recommendations as photo analysis
        
        **Recommendation:** Use `/analyze` endpoint which supports both photo and manual modes.
        """
        try:
            data = request.get_json()
            if not data or 'hex_color' not in data:
                api.abort(400, "Missing 'hex_color' field in request")
            
            # Analyze the hex color
            result = color_api.analyze_from_hex(data['hex_color'])
            
            if 'error' in result:
                api.abort(400, result['error'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_color_hex endpoint: {e}")
            api.abort(500, "Internal server error")

# Add a root endpoint to redirect to Swagger UI
@app.route('/')
def index():
    """Redirect to Swagger UI"""
    return '''
    <h1>MyMirror Backend API</h1>
    <p><a href="/swagger/">🔗 Go to Swagger UI for API Documentation & Testing</a></p>
    <h3>Available Endpoints:</h3>
    <ul>
        <li><a href="/api/v1/health">❤️ Health Check</a></li>
        <li><a href="/swagger/#/color/analyze_color_unified">🎨 Enhanced Color Analysis (Unified)</a> - POST /api/v1/color/analyze</li>
        <li><a href="/swagger/#/color/analyze_color_photo">📸 Photo-based Analysis</a> - POST /api/v1/color/analyze-photo</li>
        <li><a href="/swagger/#/color/analyze_color_hex">🎨 Manual Hex Analysis</a> - POST /api/v1/color/analyze-hex</li>
        <li><a href="/swagger/#/outfits/generate_outfits">👔 Generate Outfits</a> - POST /api/v1/outfits/generate</li>
        <li><a href="/swagger/#/outfits/get_user_outfits">👕 Get User Outfits</a> - GET /api/v1/outfits/{user_id}</li>
        <li><a href="/swagger/#/outfits/get_similar_outfits">🔄 Find Similar Outfits</a> - GET /api/v1/outfits/{outfit_id}/similar</li>
        <li><a href="/swagger/#/products/get_similar_products">🛍️ Find Similar Products (Phase 3)</a> - POST /api/v1/products/{product_id}/similar</li>
    </ul>
    <p><strong>🆕 New Features:</strong></p>
    <ul>
        <li>📸 <strong>Photo Analysis:</strong> Upload selfie for automatic skin tone detection</li>
        <li>🎨 <strong>Manual Selection:</strong> Choose your skin tone using hex color codes</li>
        <li>🔄 <strong>Unified API:</strong> Single endpoint supports both modes</li>
        <li>👔 <strong>Outfit Generation:</strong> Test with user_id=2 (has sample data)</li>
        <li>🔄 <strong>Similar Outfits:</strong> Find 10 similar outfits for any existing outfit using advanced AI</li>
        <li>🛍️ <strong>Similar Products (NEW):</strong> Find similar products within same category with color/design diversity + personalization</li>
    </ul>
    '''

if __name__ == '__main__':
    import os
    from config import get_config
    
    config = get_config()
    
    logger.info("🚀 Starting MyMirror Backend API...")
    logger.info("📝 Swagger UI available at: /swagger/")
    logger.info("❤️ Health check available at: /api/v1/health")
    logger.info("🎨 Enhanced color analysis (unified) at: /api/v1/color/analyze")
    logger.info("📸 Photo-based analysis at: /api/v1/color/analyze-photo")
    logger.info("🎨 Manual hex analysis at: /api/v1/color/analyze-hex")
    logger.info("👔 Outfit generation available at: /api/v1/outfits/generate")
    logger.info("👕 User outfits available at: /api/v1/outfits/{user_id}")
    logger.info("🔄 Similar outfits available at: /api/v1/outfits/{outfit_id}/similar")
    logger.info("🛍️ Similar products (Phase 3) available at: /api/v1/products/{product_id}/similar")
    logger.info("✨ Test with user_id=2 (has sample data)")
    logger.info("🆕 New: Support for both photo upload and manual hex color selection!")
    logger.info("🎯 NEW: Phase 3 Same-category similar products with diversity & personalization!")
    
    # Production configuration
    port = int(os.environ.get('PORT', config.API_PORT))
    debug = config.DEBUG and os.environ.get('FLASK_ENV') != 'production'
    
    if os.environ.get('FLASK_ENV') == 'production':
        logger.info("🌐 Running in PRODUCTION mode")
        logger.info(f"📡 Listening on port {port}")
    else:
        logger.info("🔧 Running in DEVELOPMENT mode")
    
    app.run(debug=debug, host=config.API_HOST, port=port) 