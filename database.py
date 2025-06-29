"""
Database module for MyMirro Backend
Handles Supabase database operations and connections
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from config import get_config
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseDB:
    """Supabase database operations manager."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.config = get_config()
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Supabase client with proper error handling."""
        try:
            if not self.config.validate_supabase_config():
                logger.error("❌ Invalid Supabase configuration")
                return
            
            supabase_config = self.config.get_supabase_config()
            
            # Use service role key for both development and production 
            # since tagged_products table requires elevated permissions
            key = (supabase_config['service_role_key'] 
                   if supabase_config['service_role_key']
                   else supabase_config['anon_key'])
            
            self.client = create_client(
                supabase_config['url'],
                key
            )
            
            logger.info("✅ Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.client = None
    
    def test_connection(self) -> bool:
        """Test the Supabase connection."""
        try:
            if not self.client:
                return False
            
            # Try a simple query to test connection
            result = self.client.table('users_updated').select('id').limit(1).execute()
            logger.info("✅ Supabase connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False
    
    def get_users(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get user data from Supabase with enhanced style quiz support.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            pandas.DataFrame: User data with style quiz information
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            query = self.client.table('users_updated').select('*')
            
            if user_id:
                query = query.eq('id', user_id)
            
            result = query.execute()
            
            if result.data:
                users_df = pd.DataFrame(result.data)
                logger.info(f"✅ Retrieved {len(users_df)} users from users_updated table")
                return users_df
            else:
                logger.warning("⚠️ No users found in users_updated table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error retrieving users: {e}")
            return pd.DataFrame()
    
    def get_products(self, limit: int = None) -> pd.DataFrame:
        """Get all products from the database."""
        try:
            logger.info("🔍 Starting get_products...")
            # ✅ FIX: Include product_id column for proper product identification
            needed_columns = [
                'id', 'product_id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category', 'style_category'
            ]
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            if limit:
                query = query.limit(limit)
            data = query.execute()
            df = pd.DataFrame(data.data)
            logger.info(f"✅ get_products: {len(df)} products")
            return df
        except Exception as e:
            logger.error(f"❌ get_products failed: {e}")
            return pd.DataFrame()
    
    def _get_all_products_chunked(self, 
                                 wear_type: Optional[str] = None,
                                 gender: Optional[str] = None,
                                 style: Optional[str] = None,
                                 chunk_size: int = 1000) -> pd.DataFrame:
        """
        Fetch all products using chunked loading to avoid database timeouts.
        
        Args:
            wear_type: Filter by wear type
            gender: Filter by gender
            style: Filter by style
            chunk_size: Number of products to fetch per chunk
            
        Returns:
            pandas.DataFrame: All products data
        """
        logger.info(f"🔄 Starting chunked loading of all products (chunk_size: {chunk_size})")
        
        all_products = []
        total_fetched = 0
        offset = 0
        
        try:
            # Load products in chunks using pagination
            while True:
                logger.info(f"📥 Loading chunk {len(all_products) + 1} (offset: {offset}, limit: {chunk_size})")
                
                # Build query for this chunk
                needed_columns = [
                    'id', 'product_id', 'title', 'product_type', 'gender',
                    'primary_style', 'primary_color', 'image_url',
                    'full_caption', 'product_embedding', 'scraped_category', 'style_category'
                ]
                query = self.client.table('tagged_products').select(','.join(needed_columns))
                
                # Apply pagination
                query = query.range(offset, offset + chunk_size - 1)
                
                # Execute query for this chunk
                result = query.execute()
                
                if not result.data:
                    logger.info(f"✅ No more products found at offset {offset}")
                    break
                
                chunk_df = pd.DataFrame(result.data)
                all_products.append(chunk_df)
                total_fetched += len(chunk_df)
                
                logger.info(f"✅ Loaded chunk {len(all_products)}: {len(chunk_df)} products (total: {total_fetched})")
                
                # If we got fewer products than chunk_size, we've reached the end
                if len(chunk_df) < chunk_size:
                    logger.info(f"✅ Reached end of products (got {len(chunk_df)} < {chunk_size})")
                    break
                
                # Move to next chunk
                offset += chunk_size
                
                # Safety check to prevent infinite loops
                if len(all_products) > 50:  # Max 50 chunks = 50,000 products
                    logger.warning(f"⚠️ Reached maximum chunks limit (50), stopping at {total_fetched} products")
                    break
            
            # Combine all chunks
            if all_products:
                products_df = pd.concat(all_products, ignore_index=True)
                logger.info(f"✅ Successfully loaded {len(products_df)} total products in {len(all_products)} chunks")
                
                # Apply filters if needed
                if wear_type:
                    products_df = products_df[products_df['wear_type'] == wear_type]
                    logger.info(f"✅ Applied wear_type filter '{wear_type}': {len(products_df)} products remaining")
                if gender:
                    products_df = products_df[products_df['gender'].str.lower() == gender.lower()]
                    logger.info(f"✅ Applied gender filter '{gender}': {len(products_df)} products remaining")
                if style:
                    products_df = products_df[products_df['primary_style'].str.contains(style, case=False, na=False)]
                    logger.info(f"✅ Applied style filter '{style}': {len(products_df)} products remaining")
                
                # Process the DataFrame to add required columns
                products_df = self._process_products_dataframe(products_df)
                
                logger.info(f"✅ Final result: {len(products_df)} products ready for use")
                return products_df
            else:
                logger.warning("⚠️ No products loaded from any chunk")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error in chunked loading: {e}")
            return pd.DataFrame()
    
    def _categorize_wear_type(self, row) -> str:
        """Categorize products into wear types based on product_type (preferred) or scraped_category."""
        # Exhaustive lists for upperwear and bottomwear
        upperwear_types = [
            't-shirt', 'tshirts', 'tee', 'tees', 'shirt', 'shirts', 'blouse', 'blouses', 'top', 'tops', 'sweater', 'sweaters', 'hoodie', 'hoodies', 'jacket', 'jackets',
            'blazer', 'blazers', 'cardigan', 'cardigans', 'tank top', 'tank tops', 'crop top', 'crop tops', 'sleeveless', 'tunic', 'tunics', 'polo shirt', 'polo shirts',
            'henley', 'henleys', 'dress', 'dresses', 'maxi dress', 'maxi dresses', 'mini dress', 'mini dresses', 'gown', 'gowns', 'sweatshirt', 'sweatshirts', 'pullover',
            'pullovers', 'jumper', 'jumpers', 'vest', 'vests', 'camisole', 'camisoles', 'bodysuit', 'bodysuits', 'turtleneck', 'turtlenecks', 'mock neck', 'mock necks',
            'crew neck', 'crew necks', 'v-neck', 'v-necks', 'round neck', 'round necks', 'kurta', 'kurtas', 'ethnic', 'formal-top', 'casual-top', 'shrug', 'shrugs',
            'cape', 'capes', 'kaftan', 'kaftans', 'poncho', 'ponchos', 'sweatervest', 'sweater vest', 'sweater vests', 'overcoat', 'overcoats', 'anorak', 'anoraks',
            'parka', 'parkas', 'windcheater', 'windcheaters', 'bolero', 'boleros', 'peplum', 'peplums', 'wrap', 'wraps', 'kimono', 'kimonos', 'tuxedo', 'tuxedos',
            'waistcoat', 'waistcoats', 'over-shirt', 'over-shirts', 'over shirt', 'over shirts', 'shrug', 'shrugs', 'sweatervest', 'sweater vest', 'sweater vests'
        ]
        bottomwear_types = [
            'trousers', 'trouser', 'jeans', 'jean', 'pants', 'pant', 'shorts', 'short', 'skirt', 'skirts', 'pencil skirt', 'pencil skirts', 'a-line skirt', 'a-line skirts',
            'midi skirt', 'midi skirts', 'mini skirt', 'mini skirts', 'maxi skirt', 'maxi skirts', 'joggers', 'jogger', 'chinos', 'chino', 'dress pants', 'slacks', 'cargos',
            'cargo', 'denim pants', 'denim pant', 'denim shorts', 'denim short', 'leggings', 'legging', 'tights', 'tight', 'culottes', 'culotte', 'palazzos', 'palazzo',
            'wide leg pants', 'wide leg pant', 'skinny pants', 'skinny pant', 'straight leg pants', 'straight leg pant', 'bootcut pants', 'bootcut pant', 'flare pants',
            'flare pant', 'cropped pants', 'cropped pant', 'ankle pants', 'ankle pant', 'high-waisted pants', 'high-waisted pant', 'low-rise pants', 'low-rise pant',
            'dhoti', 'dhotis', 'salwar', 'salwars', 'patiala', 'patialas', 'lungi', 'lungis', 'harem pants', 'harem pant', 'track pants', 'track pant', 'capri', 'capris',
            'bermuda', 'bermudas', 'pyjama', 'pyjamas', 'pajama', 'pajamas', 'formal-bottom', 'casual-bottom', 'skort', 'skorts', 'treggings', 'tregging', 'overalls', 'overall'
        ]
        # Prefer product_type if available
        product_type = row.get('product_type', '')
        scraped_category = row.get('scraped_category', '')
        # Normalize
        pt = product_type.strip().lower() if product_type else ''
        sc = scraped_category.strip().lower() if scraped_category else ''
        # Check product_type first
        for keyword in upperwear_types:
            if keyword in pt:
                return 'Upperwear'
        for keyword in bottomwear_types:
            if keyword in pt:
                return 'Bottomwear'
        # Fallback to scraped_category
        for keyword in upperwear_types:
            if keyword in sc:
                return 'Upperwear'
        for keyword in bottomwear_types:
            if keyword in sc:
                return 'Bottomwear'
        # Fallback: try to guess
        if any(word in pt for word in ['shirt', 'top', 'blouse', 'sweater', 'jacket', 'dress', 't-shirt', 'polo', 'henley', 'vest', 'tank', 'crop']):
            return 'Upperwear'
        if any(word in pt for word in ['pants', 'trousers', 'jeans', 'shorts', 'skirt', 'chinos', 'joggers', 'leggings', 'tights', 'culottes', 'palazzos']):
            return 'Bottomwear'
        if any(word in sc for word in ['shirt', 'top', 'blouse', 'sweater', 'jacket', 'dress', 't-shirt', 'polo', 'henley', 'vest', 'tank', 'crop']):
            return 'Upperwear'
        if any(word in sc for word in ['pants', 'trousers', 'jeans', 'shorts', 'skirt', 'chinos', 'joggers', 'leggings', 'tights', 'culottes', 'palazzos']):
            return 'Bottomwear'
        # Log unknown types for debugging
        logger.warning(f"Unknown wear type for product_type='{product_type}', scraped_category='{scraped_category}' - defaulting to Upperwear")
        return 'Upperwear'

    def _process_products_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Always use product_id as the main identifier
        if 'product_id' in df.columns:
            df['id'] = df['product_id']
        # Add category field mapping from scraped_category
        if 'scraped_category' in df.columns and 'category' not in df.columns:
            df['category'] = df['scraped_category']
        # Add missing columns that the outfit generator expects
        if 'wear_type' not in df.columns:
            # Use new robust categorization
            df['wear_type'] = df.apply(self._categorize_wear_type, axis=1)
        # Gender is already available in the tagged_products table, no need to infer
        if 'gender' not in df.columns:
            df['gender'] = df['scraped_category'].apply(self._infer_gender_from_category)
        if 'final_caption' not in df.columns:
            df['final_caption'] = df.apply(lambda row: row.get('full_caption', '') or f"{row.get('title', '')} - {row.get('primary_style', '')} {row.get('primary_color', '')} {row.get('primary_fabric', '')}".strip(), axis=1)
        return df
    
    def _infer_gender_from_category(self, scraped_category: str) -> str:
        """Infer gender from product scraped_category."""
        if not scraped_category:
            return 'Unisex'
        
        category_lower = scraped_category.lower()
        
        if any(word in category_lower for word in ['women', 'female', 'ladies']):
            return 'Women'
        elif any(word in category_lower for word in ['men', 'male', 'boys']):
            return 'Men'
        else:
            return 'Unisex'  # Default
    
    def get_user_outfits(self, user_id: int, limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Get pre-computed outfit recommendations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of outfits to return
            
        Returns:
            pandas.DataFrame: Outfit recommendations
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            query = self.client.table('user_outfits').select('*').eq('user_id', user_id)
            
            if limit:
                query = query.limit(limit)
            
            # Order by rank/score
            query = query.order('rank', desc=False)
            
            result = query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                logger.info(f"✅ Retrieved {len(df)} outfits for user {user_id}")
                return df
            else:
                logger.warning(f"⚠️ No outfits found for user {user_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error fetching user outfits: {e}")
            return pd.DataFrame()
    
    def save_user_outfits(self, outfits_df: pd.DataFrame, user_id: int, skip_deletion: bool = False) -> bool:
        """
        Save outfit recommendations to Supabase.
        
        Args:
            outfits_df: DataFrame containing outfit recommendations
            user_id: User ID
            skip_deletion: If True, skip deleting existing outfits (for batch operations)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return False
            
            # Convert DataFrame to list of dictionaries
            outfits_data = outfits_df.to_dict('records')
            
            # Add user_id to each record if not present
            for outfit in outfits_data:
                outfit['user_id'] = user_id
            
            # Delete existing outfits for this user (unless skipping for batch operations)
            if not skip_deletion:
                self.client.table('user_outfits').delete().eq('user_id', user_id).execute()
            
            # Insert new outfits
            result = self.client.table('user_outfits').insert(outfits_data).execute()
            
            if result.data:
                logger.info(f"✅ Saved {len(outfits_data)} outfits for user {user_id}")
                return True
            else:
                logger.error("❌ Failed to save outfits")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving user outfits: {e}")
            return False
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Get a single product by ID from tagged_products table.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dict: Product data with enhanced tags or None
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return None
            
            # Use tagged_products table and search by product_id column
            result = self.client.table('tagged_products').select('*').eq('product_id', product_id).execute()
            
            if result.data and len(result.data) > 0:
                product_data = result.data[0]
                # Ensure id field matches product_id for compatibility
                product_data['id'] = product_data.get('product_id', product_id)
                return product_data
            else:
                logger.warning(f"⚠️ Product {product_id} not found in tagged_products")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching product {product_id}: {e}")
            return None
    
    def search_products(self, 
                       query: str,
                       wear_type: Optional[str] = None,
                       limit: int = 20) -> pd.DataFrame:
        """
        Search products by text query.
        
        Args:
            query: Search query
            wear_type: Optional wear type filter
            limit: Maximum results
            
        Returns:
            pandas.DataFrame: Search results
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            # Use text search on title and description
            supabase_query = self.client.table('products').select('*')
            
            # Apply text search (you might need to setup full-text search in Supabase)
            supabase_query = supabase_query.or_(
                f'title.ilike.%{query}%,description.ilike.%{query}%,primary_style.ilike.%{query}%'
            )
            
            if wear_type:
                supabase_query = supabase_query.eq('wear_type', wear_type)
            
            supabase_query = supabase_query.limit(limit)
            
            result = supabase_query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                logger.info(f"✅ Found {len(df)} products matching '{query}'")
                return df
            else:
                logger.info(f"⚠️ No products found matching '{query}'")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error searching products: {e}")
            return pd.DataFrame()

    def get_style_quiz_data(self, style_quiz_id: int) -> Optional[Dict]:
        """
        Get style quiz data from Supabase.
        
        Args:
            style_quiz_id: Style quiz ID
            
        Returns:
            Dict: Style quiz data or None if not found
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return None
            
            result = self.client.table('style_quiz_updated').select('*').eq('id', style_quiz_id).execute()
            
            if result.data:
                quiz_data = result.data[0]
                logger.info(f"✅ Retrieved style quiz data for ID {style_quiz_id}")
                return quiz_data
            else:
                logger.warning(f"⚠️ No style quiz data found for ID {style_quiz_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching style quiz data: {e}")
            return None

    def get_user_with_style_quiz(self, user_id: int) -> Dict:
        """
        Get comprehensive user data including style quiz information.
        
        Args:
            user_id: The user ID to fetch data for
            
        Returns:
            Dict: Combined user and style quiz data with mapped column names
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return {}
            
            # Get user data from users_updated table
            user_result = self.client.table('users_updated').select('*').eq('id', user_id).execute()
            
            if not user_result.data:
                logger.error(f"❌ User {user_id} not found in users_updated table")
                return {}
            
            user_data = user_result.data[0]
            style_quiz_id = user_data.get('style_quiz_id')
            
            # Start with mapped user data
            mapped_user_data = self._map_user_columns(user_data)
            
            if style_quiz_id:
                # Try to get style quiz data from the correct table
                try:
                    quiz_result = self.client.table('style-quiz-updated').select('*').eq('id', style_quiz_id).execute()
                    
                    if quiz_result.data:
                        quiz_data = quiz_result.data[0]
                        mapped_quiz_data = self._map_style_quiz_columns(quiz_data)
                        
                        # Merge quiz data with user data (quiz data takes priority)
                        mapped_user_data.update(mapped_quiz_data)
                        logger.info(f"✅ Retrieved user data with style quiz for user {user_id}")
                    else:
                        logger.warning(f"⚠️ Style quiz {style_quiz_id} not found, using basic user data")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not retrieve style quiz data: {e}, using basic user data")
            else:
                logger.info(f"ℹ️ No style_quiz_id for user {user_id}, using basic user data")
            
            return mapped_user_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving user with style quiz: {e}")
            return {}

    def _map_user_columns(self, user_data: Dict) -> Dict:
        """
        Map columns from users_updated table to application format.
        
        Args:
            user_data: Raw user data from database
            
        Returns:
            Dict: Mapped user data with application column names
        """
        mapped_data = {}
        
        # Direct mappings
        direct_mappings = {
            'id': 'User',
            'gender': 'Gender',
            'age': 'Age',
            'location': 'Location',
            'created_at': 'Created At',
            'updated_at': 'Updated At'
        }
        
        for db_col, app_col in direct_mappings.items():
            if db_col in user_data and user_data[db_col] is not None:
                mapped_data[app_col] = user_data[db_col]
        
        # Set defaults for missing required fields
        if 'User' not in mapped_data:
            mapped_data['User'] = user_data.get('id', 0)
        
        if 'Gender' not in mapped_data:
            # Temporary manual mapping for specific users until style quiz is available
            user_id = user_data.get('id', 0)
            if user_id == 4:
                mapped_data['Gender'] = 'Male'
            else:
                mapped_data['Gender'] = 'Unisex'
        
        # Generate default captions if not provided by style quiz
        mapped_data['Upper Wear Caption'] = self._generate_default_upper_wear_caption(mapped_data)
        mapped_data['Lower Wear Caption'] = self._generate_default_lower_wear_caption(mapped_data)
        
        logger.debug(f"Mapped user columns: {list(mapped_data.keys())}")
        return mapped_data

    def _map_style_quiz_columns(self, quiz_data: Dict) -> Dict:
        """
        Map columns from style-quiz-updated table to application format.
        
        Args:
            quiz_data: Raw style quiz data from database
            
        Returns:
            Dict: Mapped style quiz data with application column names
        """
        mapped_data = {}
        
        # Map style quiz columns to application format based on actual style-quiz-updated table
        style_quiz_mappings = {
            # Personal info
            'name': 'Name',
            'gender': 'Gender',
            'body_shape': 'Body Shape',
            'height': 'Height',
            'weight': 'Weight',
            'skin_tone': 'Skin Tone',
            
            # Style preferences  
            'fashion_style': 'Fashion Style',
            'style_preference': 'Style Preference',
            'preferred_colors': 'Colors family',
            'color_family': 'Color Family',
            'hex_codes': 'Hex Codes',
            'undertone': 'Undertone',
            'contrast': 'Contrast',
            
            # NEW: Apparel preferences for each style category
            'apparel_pref_business_casual': 'Apparel Pref Business Casual',
            'apparel_pref_streetwear': 'Apparel Pref Streetwear', 
            'apparel_pref_athleisure': 'Apparel Pref Athleisure',
            
            # Lifestyle
            'lifestyle': 'Lifestyle',
            'occupation': 'Occupation',
            'occasion_preference': 'Occasion',
            'activity_level': 'Activity Level',
            
            # Budget and shopping
            'budget_range': 'Budget Preference',
            'shopping_style': 'Shopping Style',
            'brand_preference': 'Brand Preference',
            
            # Fit preferences
            'upper_fit': 'Upper Fit',
            'lower_fit': 'Lower Fit',
            'full_body_fit': 'Full Body Fit',
            'upper_size': 'Upper Size',
            'lower_waist_size': 'Lower Waist Size',
            'comfort_level': 'Comfort Level',
            'fabric_preference': 'Fabric Preference',
            
            # Advanced preferences
            'print_type': 'Print Type',
            'print_scale': 'Print Scale',
            'print_density': 'Print Density',
            'pattern_placement': 'Pattern Placement',
            'surface_texture': 'Surface Texture',
            'pattern_preference': 'Pattern Preference',
            'sleeve_preference': 'Sleeve Preference',
            'neckline_preference': 'Neckline Preference',
            
            # Personality and preferences
            'personality_tag_1': 'Personality Tag 1',
            'personality_tag_2': 'Personality Tag 2',
            'minimalistic': 'Minimalistic',
            'outfit_adventurous': 'Outfit Adventurous',
            'weekend_preference': 'Weekend Preference',
            'workspace_style': 'Workspace Style',
            'friend_compliments': 'Friend Compliments',
            'work_outfit': 'Work Outfit',
            'wardrobe_content': 'Wardrobe Content',
            
            # Generated captions (these take priority)
            'upper_wear_caption': 'Upper Wear Caption',
            'lower_wear_caption': 'Lower Wear Caption',
            'full_body_dress_caption': 'Full Body Dress Caption',
            'style_description': 'Style Description',
            
            # Additional data
            'user_tags': 'User Tags',
            'color_analysis': 'Color Analysis',
            'feedback': 'Feedback'
        }
        
        for db_col, app_col in style_quiz_mappings.items():
            if db_col in quiz_data and quiz_data[db_col] is not None:
                # Clean and process the data
                value = quiz_data[db_col]
                if isinstance(value, str):
                    value = value.strip()
                    if value:  # Only add non-empty strings
                        mapped_data[app_col] = value
                else:
                    mapped_data[app_col] = value
        
        # Generate enhanced captions if basic ones exist
        if 'Upper Wear Caption' in mapped_data:
            mapped_data['Upper Wear Caption'] = self._enhance_wear_caption(
                mapped_data['Upper Wear Caption'], mapped_data, 'upper'
            )
        
        if 'Lower Wear Caption' in mapped_data:
            mapped_data['Lower Wear Caption'] = self._enhance_wear_caption(
                mapped_data['Lower Wear Caption'], mapped_data, 'lower'
            )
        
        logger.debug(f"Mapped style quiz columns: {list(mapped_data.keys())}")
        return mapped_data

    def _generate_default_upper_wear_caption(self, user_data: Dict) -> str:
        """Generate default upper wear caption if not provided."""
        gender = user_data.get('Gender', 'Unisex').lower()
        
        if gender in ['male', 'men']:
            return "Casual comfortable shirt or t-shirt in neutral colors"
        elif gender in ['female', 'women']:
            return "Stylish top or blouse with good fit and versatile colors"
        else:
            return "Comfortable versatile top in neutral colors"

    def _generate_default_lower_wear_caption(self, user_data: Dict) -> str:
        """Generate default lower wear caption if not provided."""
        gender = user_data.get('Gender', 'Unisex').lower()
        
        if gender in ['male', 'men']:
            return "Well-fitted pants or jeans in classic colors"
        elif gender in ['female', 'women']:
            return "Comfortable pants, jeans or skirts with good fit"
        else:
            return "Comfortable well-fitted bottoms in versatile colors"

    def _enhance_wear_caption(self, basic_caption: str, user_data: Dict, wear_type: str) -> str:
        """
        Enhance basic wear captions with user preferences.
        
        Args:
            basic_caption: Basic caption from style quiz
            user_data: Complete user data for context
            wear_type: 'upper' or 'lower'
            
        Returns:
            str: Enhanced caption with user preferences
        """
        enhancement_parts = [basic_caption]
        
        # Add style preference context
        style = user_data.get('Fashion Style', '')
        if style:
            enhancement_parts.append(f"Style: {style}")
        
        # Add color preference
        colors = user_data.get('Colors family', '')
        if colors:
            enhancement_parts.append(f"Colors: {colors}")
        
        # Add occasion context
        occasion = user_data.get('Occasion', '')
        if occasion:
            enhancement_parts.append(f"Occasion: {occasion}")
        
        # Add fit preference
        fit = user_data.get('Fit Preference', '')
        if fit:
            enhancement_parts.append(f"Fit: {fit}")
        
        # Add body shape consideration
        body_shape = user_data.get('Body Shape', '')
        if body_shape:
            enhancement_parts.append(f"Body shape: {body_shape}")
        
        return ". ".join(enhancement_parts)

    def test_user_data_mapping(self, user_id: int) -> Dict:
        """
        Test method to verify user data mapping is working correctly.
        
        Args:
            user_id: User ID to test
            
        Returns:
            Dict: Test results with raw and mapped data
        """
        try:
            logger.info(f"🧪 Testing user data mapping for user {user_id}")
            
            # Get raw data
            raw_user = self.client.table('users_updated').select('*').eq('id', user_id).execute()
            raw_user_data = raw_user.data[0] if raw_user.data else {}
            
            # Get mapped data
            mapped_data = self.get_user_with_style_quiz(user_id)
            
            test_results = {
                'user_id': user_id,
                'raw_user_data': raw_user_data,
                'mapped_data': mapped_data,
                'mapping_success': bool(mapped_data),
                'required_fields_present': {
                    'User': 'User' in mapped_data,
                    'Gender': 'Gender' in mapped_data,
                    'Upper Wear Caption': 'Upper Wear Caption' in mapped_data,
                    'Lower Wear Caption': 'Lower Wear Caption' in mapped_data
                }
            }
            
            # Try to get style quiz data if available
            style_quiz_id = raw_user_data.get('style_quiz_id')
            if style_quiz_id:
                try:
                    raw_quiz = self.client.table('style_quiz_updated').select('*').eq('id', style_quiz_id).execute()
                    test_results['raw_quiz_data'] = raw_quiz.data[0] if raw_quiz.data else {}
                    test_results['has_style_quiz'] = True
                except:
                    test_results['has_style_quiz'] = False
            else:
                test_results['has_style_quiz'] = False
            
            logger.info(f"✅ User data mapping test completed for user {user_id}")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Error in user data mapping test: {e}")
            return {'error': str(e)}

    def get_products_with_user_filters(self, user_data: Dict) -> pd.DataFrame:
        """Get products with user-specific filters (gender, style preferences, etc.)."""
        try:
            logger.info("🔍 Starting get_products_with_user_filters...")
            
            # Extract user preferences
            gender = user_data.get('Gender', '').lower()
            style_preferences = user_data.get('style_preferences', [])
            color_preferences = user_data.get('color_preferences', [])
            
            logger.info(f"User filters - Gender: {gender}, Styles: {len(style_preferences)}, Colors: {len(color_preferences)}")
            
            # ✅ FIX: Only select columns that actually exist in the database
            needed_columns = [
                'id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category'
            ]
            
            # Build base query with specific columns
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            
            # ✅ FIX: Reduce limit to prevent timeouts - use a more conservative limit
            query = query.limit(5000)  # Reduced from 10000 to 5000 for faster response
            
            # Apply gender filter first (most restrictive filter)
            if gender in ['male', 'female']:
                query = query.eq('gender', gender.capitalize())
                logger.info(f"Applied gender filter: {gender.capitalize()}")
            
            # ✅ OPTIMIZATION: Apply style filters only if we have a reasonable number
            if style_preferences and len(style_preferences) <= 5:  # Limit to 5 styles max
                style_conditions = []
                for style in style_preferences[:5]:  # Take only first 5 styles
                    style_conditions.append(f"primary_style.ilike.%{style}%")
                
                if style_conditions:
                    query = query.or_(','.join(style_conditions))
                    logger.info(f"Applied style filters: {len(style_preferences[:5])} styles")
            
            # ✅ OPTIMIZATION: Apply color filters only if we have a reasonable number
            if color_preferences and len(color_preferences) <= 5:  # Limit to 5 colors max
                color_conditions = []
                for color in color_preferences[:5]:  # Take only first 5 colors
                    color_conditions.append(f"primary_color.ilike.%{color}%")
                
                if color_conditions:
                    query = query.or_(','.join(color_conditions))
                    logger.info(f"Applied color filters: {len(color_preferences[:5])} colors")
            
            # Execute query with timeout handling
            try:
                result = query.execute()
            except Exception as query_error:
                logger.warning(f"⚠️ Query failed with filters, trying without style/color filters: {query_error}")
                # Fallback: try without style and color filters
                query = self.client.table('tagged_products').select(','.join(needed_columns)).limit(5000)
                if gender in ['male', 'female']:
                    query = query.eq('gender', gender.capitalize())
                result = query.execute()
            
            if result.data:
                products_df = pd.DataFrame(result.data)
                logger.info(f"✅ get_products_with_user_filters: {len(products_df)} products")
                return self._process_products_dataframe(products_df)
            else:
                logger.warning("No products found with user filters")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ get_products_with_user_filters failed: {e}")
            # Return empty DataFrame instead of failing completely
            return pd.DataFrame()
    
    def create_similar_products_table(self) -> bool:
        """
        Create the similar_products table if it doesn't exist.
        
        Returns:
            bool: True if table was created successfully or already exists
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return False
            
            # SQL to create the similar_products table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS similar_products (
                id SERIAL PRIMARY KEY,
                
                -- Main product identification
                main_product_id VARCHAR(50) NOT NULL,
                main_product_title TEXT,
                main_product_wear_type VARCHAR(50),
                main_product_gender VARCHAR(20),
                main_product_primary_style VARCHAR(100),
                main_product_primary_style_multi TEXT[],
                main_product_primary_color VARCHAR(50),
                main_product_price DECIMAL(10,2),
                
                -- Similar product details
                similar_product_id VARCHAR(50) NOT NULL,
                similar_product_title TEXT,
                similar_product_wear_type VARCHAR(50),
                similar_product_gender VARCHAR(20),
                similar_product_primary_style VARCHAR(100),
                similar_product_primary_style_multi TEXT[],
                similar_product_primary_color VARCHAR(50),
                similar_product_price DECIMAL(10,2),
                similar_product_brand VARCHAR(100),
                similar_product_image_url TEXT,
                
                -- Similarity scoring and metadata
                similarity_score DECIMAL(5,4) NOT NULL,
                semantic_similarity DECIMAL(5,4),
                style_compatibility DECIMAL(5,4),
                color_diversity DECIMAL(5,4),
                design_diversity DECIMAL(5,4),
                price_similarity DECIMAL(5,4),
                user_preference_boost DECIMAL(5,4),
                
                -- Candidate type and diversity features
                candidate_type VARCHAR(50),
                diversity_features TEXT[],
                
                -- Filtering and user context
                user_gender VARCHAR(20),
                user_preferred_styles TEXT[],
                user_preferred_colors TEXT[],
                user_price_range_min DECIMAL(10,2),
                user_price_range_max DECIMAL(10,2),
                
                -- Applied filters
                applied_filters JSONB,
                
                -- Caching and performance
                is_cached BOOLEAN DEFAULT FALSE,
                cache_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cache_expiry TIMESTAMP,
                processing_time_ms INTEGER,
                
                -- Ranking and selection
                faiss_rank INTEGER,
                final_rank INTEGER,
                is_selected BOOLEAN DEFAULT FALSE,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Constraints
                CONSTRAINT unique_product_pair UNIQUE (main_product_id, similar_product_id)
            );
            """
            
            # Execute the SQL using Supabase's rpc method
            result = self.client.rpc('exec_sql', {'sql': create_table_sql}).execute()
            
            logger.info("✅ Similar products table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating similar products table: {e}")
            return False

    def store_similar_products(self, main_product_id: str, similar_products: List[Dict], 
                              user_preferences: Dict = None, filters: Dict = None,
                              processing_time_ms: int = None) -> bool:
        """
        Store similar products results in the database with caching.
        
        Args:
            main_product_id: ID of the main product
            similar_products: List of similar products with scores
            user_preferences: User preferences used for filtering
            filters: Applied filters
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            bool: True if stored successfully
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return False
            
            # Helper function to convert numpy types to Python native types
            def convert_numpy_types(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Prepare data for insertion
            records = []
            cache_expiry = pd.Timestamp.now() + pd.Timedelta(hours=24)  # 24 hour cache
            
            for i, product in enumerate(similar_products):
                # Convert numpy types to Python native types
                product = convert_numpy_types(product)
                
                record = {
                    'main_product_id': main_product_id,
                    'main_product_title': product.get('main_product_title', ''),
                    'main_product_wear_type': product.get('main_product_wear_type', ''),
                    'main_product_gender': product.get('main_product_gender', ''),
                    'main_product_primary_style': product.get('main_product_primary_style', ''),
                    'main_product_primary_style_multi': product.get('main_product_primary_style_multi', []),
                    'main_product_primary_color': product.get('main_product_primary_color', ''),
                    'main_product_price': float(product.get('main_product_price', 0)),
                    
                    'similar_product_id': product.get('product_id', ''),
                    'similar_product_title': product.get('title', ''),
                    'similar_product_wear_type': product.get('wear_type', ''),
                    'similar_product_gender': product.get('gender', ''),
                    'similar_product_primary_style': product.get('primary_style', ''),
                    'similar_product_primary_style_multi': product.get('primary_style_multi', []),
                    'similar_product_primary_color': product.get('primary_color', ''),
                    'similar_product_price': float(product.get('price', 0)),
                    'similar_product_brand': product.get('brand', ''),
                    'similar_product_image_url': product.get('image_url', ''),
                    
                    'similarity_score': float(product.get('similarity_score', 0)),
                    'semantic_similarity': float(product.get('score_breakdown', {}).get('semantic_similarity', 0)),
                    'style_compatibility': float(product.get('score_breakdown', {}).get('style_compatibility', 0)),
                    'color_diversity': float(product.get('score_breakdown', {}).get('color_diversity', 0)),
                    'design_diversity': float(product.get('score_breakdown', {}).get('design_diversity', 0)),
                    'price_similarity': float(product.get('score_breakdown', {}).get('price_similarity', 0)),
                    'user_preference_boost': float(product.get('score_breakdown', {}).get('user_preference_boost', 0)),
                    
                    'candidate_type': product.get('candidate_type', 'core_similar'),
                    'diversity_features': product.get('diversity_features', []),
                    
                    'user_gender': user_preferences.get('gender', '') if user_preferences else '',
                    'user_preferred_styles': user_preferences.get('preferred_styles', []) if user_preferences else [],
                    'user_preferred_colors': user_preferences.get('preferred_colors', []) if user_preferences else [],
                    'user_price_range_min': float(user_preferences.get('price_range', [0, 0])[0]) if user_preferences else 0,
                    'user_price_range_max': float(user_preferences.get('price_range', [0, 0])[1]) if user_preferences else 0,
                    
                    'applied_filters': convert_numpy_types(filters) if filters else None,
                    'is_cached': True,
                    'cache_expiry': cache_expiry.isoformat(),
                    'processing_time_ms': int(processing_time_ms) if processing_time_ms else None,
                    'faiss_rank': int(product.get('faiss_rank', 0)),
                    'final_rank': i + 1,
                    'is_selected': True
                }
                records.append(record)
            
            # Insert records using upsert to handle duplicates
            # First, delete existing records for this main_product_id to avoid constraint violations
            try:
                delete_result = self.client.table('similar_products').delete().eq('main_product_id', main_product_id).execute()
                logger.info(f"🗑️ Deleted {len(delete_result.data) if delete_result.data else 0} existing records for product {main_product_id}")
            except Exception as delete_e:
                logger.warning(f"⚠️ Could not delete existing records: {delete_e}")
            
            # Now insert the new records
            result = self.client.table('similar_products').insert(records).execute()
            
            logger.info(f"✅ Stored {len(records)} similar products for product {main_product_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing similar products: {e}")
            return False

    def get_cached_similar_products(self, main_product_id: str, user_preferences: Dict = None, 
                                   filters: Dict = None) -> List[Dict]:
        """
        Retrieve cached similar products from database.
        
        Args:
            main_product_id: ID of the main product
            user_preferences: User preferences for cache matching
            filters: Applied filters for cache matching
            
        Returns:
            List[Dict]: Cached similar products or empty list if not found
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return []
            
            # Build query for cached results
            query = self.client.table('similar_products').select('*').eq('main_product_id', main_product_id)
            
            # Check if cache is still valid
            query = query.eq('is_cached', True).gte('cache_expiry', pd.Timestamp.now().isoformat())
            
            # Add user preference matching if provided
            if user_preferences:
                user_gender = user_preferences.get('gender', '')
                if user_gender:
                    query = query.eq('user_gender', user_gender)
            
            # Add filter matching if provided
            if filters:
                # For now, we'll do basic filter matching
                # In a more sophisticated implementation, we could compare JSON filters
                pass
            
            # Get selected results only
            query = query.eq('is_selected', True).order('final_rank')
            
            result = query.execute()
            
            if result.data:
                logger.info(f"✅ Retrieved {len(result.data)} cached similar products for product {main_product_id}")
                return result.data
            else:
                logger.info(f"ℹ️ No cached results found for product {main_product_id}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error retrieving cached similar products: {e}")
            return []

    def clean_expired_cache(self) -> int:
        """
        Clean expired cache entries from the similar_products table.
        
        Returns:
            int: Number of deleted records
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return 0
            
            # Delete expired cache entries
            result = self.client.table('similar_products').delete().eq('is_cached', True).lt('cache_expiry', pd.Timestamp.now().isoformat()).execute()
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"🧹 Cleaned {deleted_count} expired cache entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning expired cache: {e}")
            return 0

    def get_products_by_type_with_filters(self, user_data: Dict, product_types: List[str], wear_category: str = None) -> pd.DataFrame:
        """Get products filtered by specific product types and user preferences for FAISS processing."""
        try:
            logger.info(f"🔍 Starting get_products_by_type_with_filters for {wear_category}...")
            
            # Extract user preferences
            gender = user_data.get('Gender', '').lower()
            style_preferences = user_data.get('style_preferences', [])
            color_preferences = user_data.get('color_preferences', [])
            
            logger.info(f"Product type filters: {product_types}, Gender: {gender}, Styles: {len(style_preferences)}, Colors: {len(color_preferences)}")
            
            # ✅ OPTIMIZATION: Only select essential columns
            needed_columns = [
                'id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category'
            ]
            
            # Build base query with specific columns
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            
            # Apply gender filter
            if gender in ['male', 'female']:
                query = query.eq('gender', gender.capitalize())
                logger.info(f"Applied gender filter: {gender.capitalize()}")
            
            # Apply product type filter
            if product_types:
                type_conditions = []
                for product_type in product_types:
                    type_conditions.append(f"product_type.ilike.%{product_type}%")
                
                if type_conditions:
                    query = query.or_(','.join(type_conditions))
                    logger.info(f"Applied product type filters: {len(product_types)} types")
            
            # Apply limit to avoid timeouts
            query = query.limit(2000)  # Conservative limit for type-specific queries
            
            # Execute query
            result = query.execute()
            
            if result.data:
                products_df = pd.DataFrame(result.data)
                logger.info(f"✅ get_products_by_type_with_filters: {len(products_df)} products for {wear_category}")
                return self._process_products_dataframe(products_df)
            else:
                logger.warning(f"No products found for {wear_category} with type filters")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ get_products_by_type_with_filters failed for {wear_category}: {e}")
            return pd.DataFrame()

    def get_products_simple(self, gender: str = None, limit: int = 50000) -> pd.DataFrame:
        """Get ALL products with minimal filtering (large limit to avoid timeouts)."""
        try:
            logger.info(f"🔍 Starting get_products_simple (gender: {gender}, limit: {limit})...")
            # Rollback: Only load columns that worked previously
            needed_columns = [
                'product_id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category', 'style_category'
            ]
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            if gender:
                # ✅ FIX: Capitalize gender to match database values (Male/Female)
                gender_capitalized = gender.capitalize()
                query = query.eq('gender', gender_capitalized)
                logger.info(f"Applied gender filter: {gender_capitalized}")
            if limit:
                query = query.limit(limit)
            data = query.execute()
            df = pd.DataFrame(data.data)
            logger.info(f"✅ get_products_simple: {len(df)} products")
            return df
        except Exception as e:
            logger.error(f"❌ get_products_simple failed: {e}")
            return pd.DataFrame()

    # ✅ NEW: Phase-specific data loading functions to prevent cross-phase interference
    
    def get_products_phase1(self, gender: str = None, limit: int = 500) -> pd.DataFrame:
        """Get products specifically for Phase 1 (outfit generation) using product_id column."""
        try:
            logger.info(f"🔍 Starting get_products_phase1 (gender: {gender}, limit: {limit})...")
            # Phase 1 specific columns - uses product_id and style_category
            needed_columns = [
                'product_id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category', 'style_category'
            ]
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            if gender:
                gender_capitalized = gender.capitalize()
                query = query.eq('gender', gender_capitalized)
                logger.info(f"Applied gender filter: {gender_capitalized}")
            if limit:
                query = query.limit(limit)
            data = query.execute()
            df = pd.DataFrame(data.data)
            logger.info(f"✅ get_products_phase1: {len(df)} products")
            return df
        except Exception as e:
            logger.error(f"❌ get_products_phase1 failed: {e}")
            return pd.DataFrame()

    def get_products_phase2(self, gender: str = None, limit: int = 500):
        """
        Efficiently load only the columns needed for Phase 2 (Similar Outfits), with a reasonable limit but only essential fields to avoid timeouts.
        """
        import pandas as pd
        from supabase import create_client, Client
        import logging
        logger = logging.getLogger("database")
        columns = [
            'product_id', 'title', 'product_type', 'gender', 'primary_style',
            'primary_color', 'image_url', 'product_embedding', 'comprehensive_style_categories'
        ]
        select_str = ','.join(columns)
        filters = {}
        # Temporarily disable gender filter to test if that's causing the timeout
        # if gender:
        #     filters['gender'] = gender.capitalize()
        
        logger.info(f"🔍 Starting get_products_phase2 (gender: {gender}, limit: {limit})...")
        
        try:
            # Build the query
            query = self.client.table('tagged_products').select(select_str)
            
            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)
            
            # Apply limit
            query = query.limit(limit)
            
            # Execute query
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                
                # ✅ ADD: Create wear_type column from product_type
                def map_to_wear_type(product_type):
                    if not product_type:
                        return 'Upperwear'  # Default fallback
                    
                    # Normalize product type for better matching
                    product_type_normalized = product_type.strip().lower()
                    
                    # Define upperwear product types (comprehensive list)
                    upperwear_types = [
                        't-shirt', 'shirt', 'blouse', 'top', 'sweater', 'hoodie', 'jacket', 
                        'blazer', 'cardigan', 'tank top', 'crop top', 'sleeveless', 'tunic',
                        'polo shirt', 'henley', 'dress', 'maxi dress', 'mini dress', 'gown',
                        'sweatshirt', 'pullover', 'jumper', 'vest', 'camisole', 'bodysuit',
                        'turtleneck', 'mock neck', 'crew neck', 'v-neck', 'round neck'
                    ]
                    
                    # Define bottomwear product types (comprehensive list)
                    bottomwear_types = [
                        'trousers', 'jeans', 'pants', 'shorts', 'skirt', 'pencil skirt',
                        'a-line skirt', 'midi skirt', 'mini skirt', 'maxi skirt', 'joggers',
                        'chinos', 'dress pants', 'slacks', 'cargos', 'denim pants', 'denim shorts',
                        'leggings', 'tights', 'culottes', 'palazzos', 'wide leg pants',
                        'skinny pants', 'straight leg pants', 'bootcut pants', 'flare pants',
                        'cropped pants', 'ankle pants', 'high-waisted pants', 'low-rise pants'
                    ]
                    
                    # Check if normalized product type is in categories
                    if product_type_normalized in upperwear_types:
                        return 'Upperwear'
                    elif product_type_normalized in bottomwear_types:
                        return 'Bottomwear'
                    else:
                        # If unknown, try to guess based on common patterns
                        if any(word in product_type_normalized for word in ['shirt', 'top', 'blouse', 'sweater', 'jacket', 'dress', 't-shirt', 'polo', 'henley', 'vest', 'tank', 'crop']):
                            return 'Upperwear'
                        elif any(word in product_type_normalized for word in ['pants', 'trousers', 'jeans', 'shorts', 'skirt', 'chinos', 'joggers', 'leggings', 'tights', 'culottes', 'palazzos']):
                            return 'Bottomwear'
                        else:
                            # Log unknown product types for debugging
                            logger.warning(f"Unknown product type for wear_type mapping: '{product_type}' - defaulting to Upperwear")
                            return 'Upperwear'  # Default fallback
                
                # Apply the mapping to create wear_type column
                df['wear_type'] = df['product_type'].apply(map_to_wear_type)
                
                logger.info(f"✅ get_products_phase2: {len(df)} products")
                return df
            else:
                logger.warning(f"⚠️ get_products_phase2: No products found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ get_products_phase2 failed: {e}")
            return pd.DataFrame()

    def get_products_phase3(self, gender: str = None, limit: int = 1000) -> pd.DataFrame:
        """Get products specifically for Phase 3 (similar products) using id column."""
        try:
            logger.info(f"🔍 Starting get_products_phase3 (gender: {gender}, limit: {limit})...")
            # Phase 3 specific columns - uses id and style_category
            needed_columns = [
                'id', 'title', 'product_type', 'gender',
                'primary_style', 'primary_color', 'image_url',
                'full_caption', 'product_embedding', 'scraped_category', 'style_category'
            ]
            query = self.client.table('tagged_products').select(','.join(needed_columns))
            if gender:
                gender_capitalized = gender.capitalize()
                query = query.eq('gender', gender_capitalized)
                logger.info(f"Applied gender filter: {gender_capitalized}")
            if limit:
                query = query.limit(limit)
            data = query.execute()
            df = pd.DataFrame(data.data)
            logger.info(f"✅ get_products_phase3: {len(df)} products")
            return df
        except Exception as e:
            logger.error(f"❌ get_products_phase3 failed: {e}")
            return pd.DataFrame()

# Global database instance
db = SupabaseDB()

def get_db() -> SupabaseDB:
    """Get the global database instance."""
    return db 