import os
from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, List, Tuple, Optional
import threading
import multiprocessing

# Import database module
from database import SupabaseDB

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Railway CPU optimization settings
RAILWAY_CPU_LIMIT = os.getenv('RAILWAY_CPU_LIMIT', '4')  # Limit to 4 CPUs on Railway
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None

# Configure CPU limits for Railway
if IS_RAILWAY:
    # Limit CPU cores for FAISS and other ML operations
    os.environ['OMP_NUM_THREADS'] = RAILWAY_CPU_LIMIT
    os.environ['MKL_NUM_THREADS'] = RAILWAY_CPU_LIMIT  
    os.environ['NUMEXPR_NUM_THREADS'] = RAILWAY_CPU_LIMIT
    os.environ['OPENBLAS_NUM_THREADS'] = RAILWAY_CPU_LIMIT
    # Limit multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Graceful import for color analysis
try:
    from color_analysis_api import EnhancedColorAnalysisAPI
    color_api = EnhancedColorAnalysisAPI()
    COLOR_ANALYSIS_AVAILABLE = True
    logger.info("✅ Color Analysis API loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Color Analysis API not available: {e}")
    COLOR_ANALYSIS_AVAILABLE = False
    color_api = None

app = Flask(__name__)
# Enhanced CORS configuration for better browser compatibility
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False,
        "max_age": 600  # Cache preflight response for 10 minutes
    }
})

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
debug_ns = api.namespace('debug', description='Debug and Development Operations')
test_ns = api.namespace('test', description='Test and Utility Operations')
utils_ns = api.namespace('utils', description='Utility Operations')

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

# Debug and Utility Models
debug_data_response_model = api.model('DebugDataResponse', {
    'success': fields.Boolean(description='Request success'),
    'users_table_count': fields.Integer(description='Number of users in database'),
    'tagged_products_count': fields.Integer(description='Number of tagged products'),
    'user_outfits_count': fields.Integer(description='Number of user outfits'),
    'sample_users': fields.List(fields.Raw, description='Sample user data'),
    'sample_products': fields.List(fields.Raw, description='Sample product data'),
    'sample_outfits': fields.List(fields.Raw, description='Sample outfit data'),
    'database_status': fields.String(description='Database connection status')
})

import_debug_response_model = api.model('ImportDebugResponse', {
    'imports': fields.Raw(description='Import status for all modules'),
    'availability': fields.Raw(description='Feature availability flags'),
    'system_info': fields.Raw(description='System information'),
    'cpu_optimization': fields.Raw(description='CPU optimization settings')
})

warmup_response_model = api.model('WarmupResponse', {
    'success': fields.Boolean(description='Warmup success'),
    'similar_outfits_ready': fields.Boolean(description='Similar outfits model status'),
    'similar_products_ready': fields.Boolean(description='Similar products model status'),
    'initialization_time': fields.Float(description='Time taken to initialize models'),
    'message': fields.String(description='Status message')
})

supabase_test_response_model = api.model('SupabaseTestResponse', {
    'success': fields.Boolean(description='Test success'),
    'tables': fields.Raw(description='Available tables'),
    'connection_status': fields.String(description='Connection status'),
    'sample_data': fields.Raw(description='Sample data from tables')
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
    'outfit_name': fields.String(description='Generated outfit name', example='Urban Shift'),
    'outfit_description': fields.String(description='Generated outfit description', example='A contemporary urban ensemble featuring a sleek black t-shirt paired with tailored trousers for a modern, versatile look'),
    'why_picked_explanation': fields.String(description='Personalized explanation of why this outfit was picked for the user', example='Style Vibe: streetwear - This outfit embodies a bold, urban aesthetic.\n\nOccasion: casual - Perfect for relaxed, everyday activities.\n\nBody Type: rectangle - This outfit creates definition for your rectangular body shape.\n\nSkin Undertone: warm undertone - The color palette works well with your warm undertone.'),
    'top': fields.Nested(product_model, description='Top product details'),
    'bottom': fields.Nested(product_model, description='Bottom product details'),
    'total_price': fields.Float(description='Combined price of outfit', example=2099.0),
    'generated_at': fields.String(description='Generation timestamp', example='2024-01-15T10:30:00'),
    'generation_method': fields.String(description='Method used for generation', example='ethnic_wear_specialized')
})

outfit_generation_response_model = api.model('OutfitGenerationResponse', {
    'success': fields.Boolean(description='Generation success status'),
    'message': fields.String(description='Response message', example='Successfully generated 20 outfits for user 2'),
    'user_id': fields.Integer(description='User ID', example=2),
    'outfits_count': fields.Integer(description='Number of outfits generated', example=20),
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

# Graceful imports for optional ML dependencies
try:
    from phase1_supabase_outfits_generator import SupabaseMainOutfitsGenerator as OutfitGenerator
    OUTFITS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Outfit generation not available: {e}")
    OUTFITS_AVAILABLE = False

try:
    from phase2_supabase_similar_outfits_api import SupabaseSimilarOutfitsGenerator as SimilarOutfitsGenerator
    SIMILAR_OUTFITS_AVAILABLE = True
    logger.info("✅ Phase 2 (Similar Outfits) imported successfully")
except Exception as e:
    logger.warning(f"Similar outfits not available: {e}")
    SIMILAR_OUTFITS_AVAILABLE = False

try:
    from phase3_supabase_similar_products_api import SupabaseEnhancedSimilarProductsGenerator as SimilarProductsGenerator
    SIMILAR_PRODUCTS_AVAILABLE = True
    logger.info("✅ Phase 3 (Similar Products) imported successfully")
except Exception as e:
    logger.warning(f"Similar products not available: {e}")
    SIMILAR_PRODUCTS_AVAILABLE = False

# Global variables to track model initialization status
_similar_outfits_ready = False
_similar_products_ready = False
_warmup_in_progress = False

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
            if not OUTFITS_AVAILABLE:
                return {
                    'success': False,
                    'message': 'Outfit generation service is not available - FAISS and sentence-transformers are required but not installed. This feature is disabled to reduce deployment size.'
                }, 503
            
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
            
            # Initialize generator with Railway CPU optimization
            import time
            start_time = time.time()
            
            # Railway CPU optimization - limit resource usage
            if IS_RAILWAY:
                logger.info(f"🏭 Railway environment detected - applying CPU optimizations")
                # Set conservative CPU limits for this operation
                original_threads = {}
                thread_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
                for var in thread_vars:
                    original_threads[var] = os.environ.get(var)
                    os.environ[var] = '2'  # Conservative limit for Railway
                logger.info(f"🔧 Set CPU threads to 2 for Railway compatibility")
            
            generator = OutfitGenerator()
            
            # Restore thread settings after initialization if Railway
            if IS_RAILWAY:
                for var, value in original_threads.items():
                    if value is not None:
                        os.environ[var] = value
                    elif var in os.environ:
                        del os.environ[var]
            
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
            
            # ✅ FIX: Delete old outfits if regenerating
            if regenerate:
                logger.info(f"🔄 Regenerating outfits for user {user_id} - deleting old outfits first")
                db = SupabaseDB()
                try:
                    # Delete all existing outfits for this user
                    result = db.client.table('user_outfits').delete().eq('user_id', user_id).execute()
                    deleted_count = len(result.data) if result.data else 0
                    logger.info(f"🗑️ Deleted {deleted_count} old outfits for user {user_id}")
                    
                    # Also delete by main_outfit_id pattern to be thorough
                    result2 = db.client.table('user_outfits').delete().like('main_outfit_id', f'main_{user_id}_%').execute()
                    deleted_count2 = len(result2.data) if result2.data else 0
                    if deleted_count2 > 0:
                        logger.info(f"🗑️ Deleted additional {deleted_count2} outfits by ID pattern for user {user_id}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error deleting old outfits: {e}")
                    # Continue with generation anyway
            
            # Generate outfits with Railway CPU management
            if IS_RAILWAY:
                # Apply CPU limits during generation
                for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                    os.environ[var] = '2'
                logger.info(f"🔧 Applied CPU limits for outfit generation on Railway")
            
            success = generator.generate_and_save_outfits(user_id)
            
            generation_time = time.time() - start_time
            
            if success:
                # Get the actual number of outfits from the generator's config
                outfits_count = generator.config.get('main_outfits_count', 20)
                return {
                    'success': True,
                    'message': f'Successfully generated {outfits_count} outfits for user {user_id}',
                    'user_id': user_id,
                    'outfits_count': outfits_count,
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
            if not OUTFITS_AVAILABLE:
                return {
                    'success': False,
                    'message': 'Outfit service is not available - FAISS and sentence-transformers are required but not installed. This feature is disabled to reduce deployment size.'
                }, 503
            
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
                    'outfit_name': outfit.get('outfit_name', ''),
                    'outfit_description': outfit.get('outfit_description', ''),
                    'why_picked_explanation': outfit.get('why_picked_explanation', ''),
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
        Find outfits similar to a given outfit using AI-powered semantic analysis
        
        This endpoint uses advanced AI to find outfits that are semantically and stylistically 
        similar to the provided outfit. The similarity analysis considers multiple factors:
        
        **🎯 8-Factor Fashion Intelligence:**
        - **Semantic Similarity**: AI understanding of clothing descriptions and style concepts
        - **Style Harmony**: Compatibility between different fashion styles (formal, casual, etc.)
        - **Color Harmony**: Color theory and coordination principles  
        - **Formality Matching**: Matching appropriate formality levels
        - **Pattern Compatibility**: Coordination of patterns, textures, and prints
        - **Price Similarity**: Similar price ranges for practical shopping
        - **Occasion Matching**: Suitability for similar occasions and contexts
        - **Seasonal Appropriateness**: Weather and seasonal suitability
        
        **🚀 Performance:**
        - Uses FAISS vector search for fast semantic similarity
        - Typically returns results in 15-30 seconds
        - Models auto-initialize on first use
        
        **📝 Usage Examples:**
        - `/api/v1/outfits/main_2_1/similar?count=5`
        - `/api/v1/outfits/main_2_3/similar` (default count=10)
        
        **Note:** First request may take longer due to model initialization
        """
        try:
            if not SIMILAR_OUTFITS_AVAILABLE:
                return {
                    'success': False,
                    'message': 'Similar outfits service is not available - FAISS and sentence-transformers are required but not installed. This feature is disabled to reduce deployment size.'
                }, 503
                
            # Auto-initialize models on first use (Frontend-friendly)
            global _similar_outfits_ready
            if not _similar_outfits_ready:
                try:
                    logger.info("🔄 Auto-initializing similar outfits models on first use...")
                    # Use the globally imported class instead of re-importing
                    test_generator = SimilarOutfitsGenerator()
                    _similar_outfits_ready = True
                    logger.info("✅ Similar outfits models auto-initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to auto-initialize similar outfits models: {e}")
                    return {
                        'success': False,
                        'message': 'Failed to initialize similar outfits models. This may be due to missing dependencies or resource constraints.',
                        'ready': False,
                        'auto_init_failed': True,
                        'error': str(e)
                    }, 503
            
            # Get query parameters
            count = request.args.get('count', 10, type=int)
            
            # Validate count parameter
            if count < 1 or count > 50:
                return {
                    'success': False,
                    'message': 'Count must be between 1 and 50'
                }, 400

            import time
            start_time = time.time()
            
            # Initialize generator and find similar outfits
            generator = SimilarOutfitsGenerator()
            logger.info(f"🔍 Finding {count} similar outfits for outfit {outfit_id}")
            
            # Try a quick warmup first to check if models are ready
            try:
                test_start = time.time()
                generator._ensure_models_loaded()  # Pre-warm models if they have this method
                warmup_time = time.time() - test_start
                logger.info(f"⚡ Model warmup completed in {warmup_time:.2f}s")
            except (AttributeError, Exception) as e:
                logger.info(f"⚠️ Model warmup not available or failed: {e}")
            
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
                        'outfit_name': outfit_data.get('outfit_name', ''),
                        'outfit_description': outfit_data.get('outfit_description', ''),
                        'why_picked_explanation': outfit_data.get('why_picked_explanation', ''),
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
        
        **Note:** Models auto-initialize on first use (no warmup needed)
        """
        try:
            if not SIMILAR_PRODUCTS_AVAILABLE:
                return {
                    'success': False,
                    'message': 'Similar products service is not available - FAISS and sentence-transformers are required but not installed. This feature is disabled to reduce deployment size.'
                }, 503
            
            # Auto-initialize models on first use (Frontend-friendly)
            global _similar_products_ready
            if not _similar_products_ready:
                try:
                    logger.info("🔄 Auto-initializing similar products models on first use...")
                    # Use the globally imported class instead of re-importing
                    generator = SimilarProductsGenerator(db=SupabaseDB(), is_railway=os.getenv('RAILWAY_ENVIRONMENT') is not None)
                    _similar_products_ready = True
                    logger.info("✅ Similar products models auto-initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to auto-initialize similar products models: {e}")
                    return {
                        'success': False,
                        'message': 'Failed to initialize similar products models. This may be due to missing dependencies or resource constraints.',
                        'ready': False,
                        'auto_init_failed': True,
                        'error': str(e)
                    }, 503
            
            import time
            # import signal
            # from contextlib import contextmanager
            
            # @contextmanager
            # def timeout_handler(seconds):
            #     def timeout_signal(signum, frame):
            #         raise TimeoutError(f"Operation timed out after {seconds} seconds")
            #     old_handler = signal.signal(signal.SIGALRM, timeout_signal)
            #     signal.alarm(seconds)
            #     try:
            #         yield
            #     finally:
            #         signal.alarm(0)
            #         signal.signal(signal.SIGALRM, old_handler)
            
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
            
            try:
                # Initialize Phase 3 generator first (this is usually fast)
                generator = SimilarProductsGenerator(db=SupabaseDB(), is_railway=os.getenv('RAILWAY_ENVIRONMENT') is not None)
                logger.info(f"🔍 Finding {count} similar products for product {product_id}")
                logger.info(f"   Diverse: {diverse}, Personalized: {personalized}")
                
                # Use Railway-compatible timeout (25s to be safe)
                # with timeout_handler(25):
                similar_products = generator.find_similar_products(
                    product_id=product_id,
                    num_similar=count,
                    user_preferences=user_preferences if personalized else None,
                    filters=filters
                )
                
            except TimeoutError:
                logger.warning(f"Similar products search timed out for product {product_id} - responding with retry message")
                return {
                    'success': False,
                    'message': 'Processing is taking longer than expected due to ML model initialization. This is normal for the first few requests.',
                    'timeout': True,
                    'suggestion': 'Please retry this request in 30-60 seconds. Subsequent requests will be much faster.',
                    'error_code': 'INITIAL_INDEXING',
                    'retry_recommended': True,
                    'estimated_ready_time': '30-60 seconds'
                }, 202  # 202 Accepted - processing but not complete
            
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
            "version": "1.0.0",
            "services": {
                "outfits_available": OUTFITS_AVAILABLE,
                "similar_outfits_available": SIMILAR_OUTFITS_AVAILABLE,
                "similar_products_available": SIMILAR_PRODUCTS_AVAILABLE,
                "color_analysis_available": COLOR_ANALYSIS_AVAILABLE
            }
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
        Photo mode: {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...}
        Manual mode: {"hex_color": "#FDB4A6"}
        ```
        
        **Note:** Provide either 'image' OR 'hex_color', not both.
        """
        try:
            if not COLOR_ANALYSIS_AVAILABLE:
                return {
                    'success': False,
                    'error': "Color analysis not available - missing dependencies (OpenCV)"
                }, 503
                
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
            if not COLOR_ANALYSIS_AVAILABLE:
                return {
                    'success': False,
                    'error': "Color analysis not available - missing dependencies (OpenCV)"
                }, 503
                
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
            if not COLOR_ANALYSIS_AVAILABLE:
                return {
                    'success': False,
                    'error': "Color analysis not available - missing dependencies (OpenCV)"
                }, 503
                
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

@debug_ns.route('/data')
class DebugData(Resource):
    @api.doc('debug_data_check')
    @api.marshal_with(debug_data_response_model)
    @api.response(200, 'Success', debug_data_response_model)
    @api.response(500, 'Internal Server Error', error_model)
    def get(self):
        """
        Debug endpoint to check available data in database
        
        **Purpose:** Development and debugging tool to inspect database contents
        
        **Returns:**
        - Database table counts (users, products, outfits)
        - Sample data from each table
        - Database connection status
        - Helpful for troubleshooting data issues
        
        **Use Cases:**
        - Verify data was imported correctly
        - Check user outfit counts before testing APIs
        - Debug database connection issues
        - Inspect sample data structure
        """
        try:
            db = SupabaseDB()
            
            # Check outfit data
            outfits_info = {}
            try:
                outfits_result = db.client.table('user_outfits').select('main_outfit_id').limit(5).execute()
                if outfits_result.data:
                    outfits_info['sample_outfit_ids'] = [outfit['main_outfit_id'] for outfit in outfits_result.data]
                    outfits_info['outfit_count'] = len(outfits_result.data)
                else:
                    outfits_info['error'] = 'No outfits found in user_outfits table'
            except Exception as e:
                outfits_info['error'] = f'Error accessing user_outfits: {e}'
            
            # Check product data
            products_info = {}
            try:
                # First try to get any record to see if table exists and has data
                products_result = db.client.table('tagged_products').select('*').limit(1).execute()
                if products_result.data:
                    # Try to get product_id column specifically
                    try:
                        products_id_result = db.client.table('tagged_products').select('product_id').limit(5).execute()
                        if products_id_result.data:
                            products_info['sample_product_ids'] = [str(p.get('product_id', p.get('id', 'unknown'))) for p in products_id_result.data]
                            products_info['product_count'] = len(products_id_result.data)
                        else:
                            products_info['error'] = 'No products found when selecting product_id'
                    except Exception as e:
                        products_info['error'] = f'Error selecting product_id column: {e}'
                        # Fallback: try to get id column
                        try:
                            products_id_result = db.client.table('tagged_products').select('id').limit(5).execute()
                            if products_id_result.data:
                                products_info['sample_product_ids'] = [str(p.get('id', 'unknown')) for p in products_id_result.data]
                                products_info['product_count'] = len(products_id_result.data)
                                products_info['note'] = 'Using id column instead of product_id'
                        except Exception as e2:
                            products_info['error'] = f'Error with both product_id and id columns: {e}, {e2}'
                else:
                    products_info['error'] = 'No products found in tagged_products table'
            except Exception as e:
                products_info['error'] = f'Error accessing tagged_products: {e}'
            
            # Check table structures
            table_info = {}
            try:
                # Get one record from each table to see structure
                outfit_sample = db.client.table('user_outfits').select('*').limit(1).execute()
                if outfit_sample.data:
                    table_info['user_outfits_columns'] = list(outfit_sample.data[0].keys())
                
                product_sample = db.client.table('tagged_products').select('*').limit(1).execute()
                if product_sample.data:
                    table_info['tagged_products_columns'] = list(product_sample.data[0].keys())
                    
            except Exception as e:
                table_info['error'] = f'Error getting table structure: {e}'
            
            return {
                'success': True,
                'outfits': outfits_info,
                'products': products_info,
                'table_structures': table_info,
                'suggestion': 'Use the sample IDs above to test the APIs'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Debug check failed: {e}'
            }, 500

@test_ns.route('/supabase-direct')
class TestSupabaseDirect(Resource):
    @api.doc('test_supabase_direct')
    @api.marshal_with(supabase_test_response_model)
    @api.response(200, 'Success', supabase_test_response_model)
    @api.response(500, 'Internal Server Error', error_model)
    def get(self):
        """
        Direct Supabase test to bypass database wrapper
        
        **Purpose:** Low-level database connectivity test
        
        **What it does:**
        - Tests direct Supabase client connection
        - Lists available database tables
        - Fetches sample data from key tables
        - Bypasses the database wrapper for raw access
        
        **Use Cases:**
        - Debug database connectivity issues
        - Verify Supabase credentials and permissions
        - Check if tables exist and are accessible
        - Troubleshoot data structure issues
        
        **Security Note:** Only use in development/debugging
        """
        try:
            from supabase import create_client
            import os
            
            # Get credentials directly from environment
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_anon_key = os.getenv('SUPABASE_ANON_KEY')
            supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            results = {}
            
            # Test with anon key
            if supabase_url and supabase_anon_key:
                try:
                    client_anon = create_client(supabase_url, supabase_anon_key)
                    result_anon = client_anon.table('tagged_products').select('*').limit(3).execute()
                    results['anon_key'] = {
                        'success': True,
                        'data_count': len(result_anon.data) if result_anon.data else 0,
                        'sample_data': result_anon.data[:1] if result_anon.data else None
                    }
                except Exception as e:
                    results['anon_key'] = {'success': False, 'error': str(e)}
            
            # Test with service role key
            if supabase_url and supabase_service_key:
                try:
                    client_service = create_client(supabase_url, supabase_service_key)
                    result_service = client_service.table('tagged_products').select('*').limit(3).execute()
                    results['service_key'] = {
                        'success': True,
                        'data_count': len(result_service.data) if result_service.data else 0,
                        'sample_data': result_service.data[:1] if result_service.data else None
                    }
                except Exception as e:
                    results['service_key'] = {'success': False, 'error': str(e)}
            
            return {
                'success': True,
                'supabase_url': supabase_url,
                'keys_available': {
                    'anon_key': bool(supabase_anon_key),
                    'service_key': bool(supabase_service_key)
                },
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': str(type(e))
            }

@utils_ns.route('/warmup')
class ModelWarmup(Resource):
    @api.doc('warmup_models')
    @api.marshal_with(warmup_response_model)
    @api.response(200, 'Success', warmup_response_model)
    @api.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Warm up ML models for Phase 2 and Phase 3 APIs (Optional)
        
        **Note:** This endpoint is now OPTIONAL as models auto-initialize on first use.
        You can still use it to pre-warm models for faster first requests.
        
        **Use Cases:**
        - Pre-initialize models during app startup for faster user experience
        - Force re-initialization of models if needed
        - Check current model initialization status
        
        **Response:**
        - success: Whether warmup completed successfully
        - similar_outfits_ready: Phase 2 model status
        - similar_products_ready: Phase 3 model status
        - initialization_time: Time taken to initialize
        
        **Frontend Integration:**
        Your Next.js app does NOT need to call this endpoint. APIs will auto-initialize.
        """
        global _similar_outfits_ready, _similar_products_ready, _warmup_in_progress
        
        if _warmup_in_progress:
            return {
                'success': False,
                'message': 'Warmup already in progress',
                'status': 'in_progress'
            }, 409
        
        try:
            _warmup_in_progress = True
            logger.info("🔥 Starting model warmup...")
            
            # Try to warm up similar outfits
            if SIMILAR_OUTFITS_AVAILABLE and not _similar_outfits_ready:
                try:
                    # Use the globally imported class instead of re-importing
                    generator = SimilarOutfitsGenerator()
                    # Try a quick initialization
                    logger.info("🔥 Warming up similar outfits models...")
                    _similar_outfits_ready = True
                    logger.info("✅ Similar outfits models ready")
                except Exception as e:
                    logger.warning(f"⚠️ Similar outfits warmup failed: {e}")
            
            # Try to warm up similar products
            if SIMILAR_PRODUCTS_AVAILABLE and not _similar_products_ready:
                try:
                    # Use the globally imported class instead of re-importing
                    generator = SimilarProductsGenerator(db=SupabaseDB(), is_railway=os.getenv('RAILWAY_ENVIRONMENT') is not None)
                    # Try a quick initialization
                    logger.info("🔄 Warming up similar products models...")
                    _similar_products_ready = True
                    logger.info("✅ Similar products models ready")
                except Exception as e:
                    logger.warning(f"⚠️ Similar products warmup failed: {e}")
            
            return {
                'success': True,
                'message': 'Model warmup completed',
                'models_ready': {
                    'similar_outfits': _similar_outfits_ready,
                    'similar_products': _similar_products_ready
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}")
            return {
                'success': False,
                'message': f'Warmup failed: {str(e)}'
            }, 500
        finally:
            _warmup_in_progress = False

@debug_ns.route('/imports')
class ImportDebug(Resource):
    @api.doc('debug_imports')
    @api.marshal_with(import_debug_response_model)
    @api.response(200, 'Success', import_debug_response_model)
    def get(self):
        """
        Debug import issues for Phase 2 and Phase 3 APIs
        
        **Purpose:** Troubleshoot ML model import and availability issues
        
        **Returns:**
        - Import status for all critical modules (FAISS, sentence-transformers, etc.)
        - Feature availability flags (Phase 2, Phase 3, Color Analysis)
        - System information (CPU limits, Railway detection)
        - CPU optimization settings
        
        **Use Cases:**
        - Debug why APIs return 503 errors
        - Check if ML dependencies are properly installed
        - Verify CPU optimization settings on Railway
        - Troubleshoot import failures
        
        **Development Tool:** Essential for debugging deployment issues
        """
        results = {}
        
        # Test Phase 2 import
        try:
            # Use the globally imported class instead of re-importing
            results['phase2_import'] = 'SUCCESS' if SIMILAR_OUTFITS_AVAILABLE else 'NOT_AVAILABLE'
            
            # Try to instantiate
            if SIMILAR_OUTFITS_AVAILABLE:
                try:
                    generator = SimilarOutfitsGenerator()
                    results['phase2_instantiate'] = 'SUCCESS'
                except Exception as e:
                    results['phase2_instantiate'] = f'FAILED: {str(e)}'
            else:
                results['phase2_instantiate'] = 'NOT_AVAILABLE'
                
        except Exception as e:
            results['phase2_import'] = f'FAILED: {str(e)}'
        
        # Test Phase 3 import
        try:
            # Use the globally imported class instead of re-importing
            results['phase3_import'] = 'SUCCESS' if SIMILAR_PRODUCTS_AVAILABLE else 'NOT_AVAILABLE'
            
            # Try to instantiate
            if SIMILAR_PRODUCTS_AVAILABLE:
                try:
                    generator = SimilarProductsGenerator(db=SupabaseDB(), is_railway=os.getenv('RAILWAY_ENVIRONMENT') is not None)
                    results['phase3_instantiate'] = 'SUCCESS'
                except Exception as e:
                    results['phase3_instantiate'] = f'FAILED: {str(e)}'
            else:
                results['phase3_instantiate'] = 'NOT_AVAILABLE'
                
        except Exception as e:
            results['phase3_import'] = f'FAILED: {str(e)}'
        
        results['current_availability'] = {
            'similar_outfits': SIMILAR_OUTFITS_AVAILABLE,
            'similar_products': SIMILAR_PRODUCTS_AVAILABLE
        }
        
        return results

@debug_ns.route('/products')
class DebugProducts(Resource):
    @api.doc('debug_list_products')
    def get(self):
        """List the first 1000 product IDs and titles from tagged_products table for debugging."""
        try:
            db = SupabaseDB()
            products_df = db.get_products(limit=1000)
            if products_df.empty:
                return {'success': False, 'products': [], 'message': 'No products found.'}, 200
            
            # Check if product 2859 exists using available columns
            product_2859_exists = False
            if 'product_id' in products_df.columns:
                product_ids = products_df['product_id'].astype(str).tolist()
                product_2859_exists = '2859' in product_ids
            elif 'id' in products_df.columns:
                product_ids = products_df['id'].astype(str).tolist()
                product_2859_exists = '2859' in product_ids
            
            # Return first 20 for display, but include the check result
            display_columns = ['id', 'title'] if 'id' in products_df.columns else ['product_id', 'title']
            products = products_df[display_columns].head(20).fillna('').to_dict(orient='records')
            return {
                'success': True, 
                'products': products, 
                'count': len(products_df),
                'product_2859_exists': product_2859_exists,
                'total_products_checked': len(products_df),
                'columns_available': list(products_df.columns)
            }, 200
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

# Add explicit OPTIONS handler for better CORS preflight support
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import Response
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '600'
        return response

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