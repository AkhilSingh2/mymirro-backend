"""
Database module for MyMirro Backend
Handles Supabase database operations and connections
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from config import get_config

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
    
    def get_products(self, 
                    wear_type: Optional[str] = None,
                    gender: Optional[str] = None,
                    style: Optional[str] = None,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    chunk_size: int = 1000) -> pd.DataFrame:
        """
        Get product data from Supabase tagged_products table with enhanced tags and pagination support.
        
        Args:
            wear_type: Filter by wear type (Upperwear, Bottomwear, etc.)
            gender: Filter by gender
            style: Filter by style
            limit: Total limit of products to fetch (None for all)
            offset: Offset for pagination
            chunk_size: Number of products to fetch per chunk
            
        Returns:
            pandas.DataFrame: Product data with rich tags (color, style, fabric, etc.)
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            # If no limit specified, fetch all products using chunked loading
            if limit is None:
                return self._get_all_products_chunked(wear_type, gender, style, chunk_size)
            
            # Otherwise, fetch with pagination
            query = self.client.table('tagged_products').select('*')
            
            # Apply filters based on tagged_products table structure
            if wear_type:
                query = query.ilike('scraped_category', f'%{wear_type}%')
            if gender:
                # Filter based on gender field (already available)
                query = query.ilike('gender', f'%{gender}%')
            if style:
                query = query.ilike('primary_style', f'%{style}%')
            
            # Apply pagination
            query = query.range(offset, offset + limit - 1)
            
            result = query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df = self._process_products_dataframe(df)
                logger.info(f"✅ Retrieved {len(df)} products from tagged_products table (offset: {offset}, limit: {limit})")
                return df
            else:
                logger.warning("⚠️ No products found in tagged_products table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error fetching products: {e}")
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
        offset = 0
        total_fetched = 0
        
        while True:
            try:
                # Fetch chunk
                chunk_df = self.get_products(
                    wear_type=wear_type,
                    gender=gender,
                    style=style,
                    limit=chunk_size,
                    offset=offset
                )
                
                if chunk_df.empty:
                    logger.info(f"✅ Completed chunked loading. Total products fetched: {total_fetched}")
                    break
                
                all_products.append(chunk_df)
                total_fetched += len(chunk_df)
                offset += chunk_size
                
                logger.info(f"📦 Fetched chunk {len(all_products)}: {len(chunk_df)} products (Total: {total_fetched})")
                
                # If we got fewer products than chunk_size, we've reached the end
                if len(chunk_df) < chunk_size:
                    logger.info(f"✅ Reached end of products. Total products fetched: {total_fetched}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error fetching chunk at offset {offset}: {e}")
                break
        
        if all_products:
            # Combine all chunks
            combined_df = pd.concat(all_products, ignore_index=True)
            logger.info(f"✅ Successfully loaded {len(combined_df)} total products using chunked loading")
            return combined_df
        else:
            logger.warning("⚠️ No products loaded from chunked loading")
            return pd.DataFrame()
    
    def _process_products_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance the products DataFrame with required columns and mappings.
        
        Args:
            df: Raw products DataFrame from Supabase
            
        Returns:
            pandas.DataFrame: Processed products DataFrame
        """
        if df.empty:
            return df
        
        # Map tagged_products columns to expected outfit generator columns
        # Use product_id as the main ID (references products table)
        if 'product_id' in df.columns:
            df['id'] = df['product_id']
        
        # Add category field mapping from scraped_category
        if 'scraped_category' in df.columns and 'category' not in df.columns:
            df['category'] = df['scraped_category']
        
        # Add missing columns that the outfit generator expects
        if 'wear_type' not in df.columns:
            # Use scraped_category instead of category
            df['wear_type'] = df['scraped_category'].apply(self._categorize_wear_type)
        
        # Gender is already available in the tagged_products table, no need to infer
        if 'gender' not in df.columns:
            # Infer gender from scraped_category if needed
            df['gender'] = df['scraped_category'].apply(self._infer_gender_from_category)
        
        # Use enhanced tags from tagged_products - these are already available!
        if 'final_caption' not in df.columns:
            # Use the rich full_caption or create from enhanced fields
            df['final_caption'] = df.apply(lambda row: 
                row.get('full_caption', '') or 
                f"{row.get('title', '')} - {row.get('primary_style', '')} {row.get('primary_color', '')} {row.get('primary_fabric', '')}".strip(), axis=1)
        
        return df
    
    def _categorize_wear_type(self, scraped_category: str) -> str:
        """Categorize products into wear types based on scraped_category."""
        if not scraped_category:
            return 'Upperwear'
        
        category_lower = scraped_category.lower()
        
        # Classify based on your actual product categories from scraped_category
        if any(word in category_lower for word in [
            'kurta', 'shirt', 'top', 'blouse', 'jacket', 'sweater', 'tee', 'tank',
            'ethnic', 'print', 'cotton', 'motifs', 'formal-top', 'casual-top'
        ]):
            return 'Upperwear'
        elif any(word in category_lower for word in [
            'pant', 'jean', 'trouser', 'short', 'skirt', 'bottom', 'pajama', 'salwar',
            'formal-bottom', 'casual-bottom'
        ]):
            return 'Bottomwear'
        elif any(word in category_lower for word in [
            'dress', 'gown', 'frock', 'sheath'
        ]):
            return 'Dresses'
        else:
            # For your data, most items seem to be ethnic wear (kurtas)
            return 'Upperwear'  # Default
    
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
        """
        Get products from Supabase with user-specific filters applied at database level.
        This dramatically reduces data loading time by only fetching relevant products.
        
        Args:
            user_data: User data dictionary containing gender, style preferences, etc.
            
        Returns:
            pandas.DataFrame: Pre-filtered product data
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            logger.info("🎯 Applying user filters at database level for optimized loading...")
            
            # Extract user filters
            user_gender = user_data.get('Gender', '').lower()
            
            # Start with a simple query and build up
            query = self.client.table('tagged_products').select('*')
            
            # ✅ FIX: Use user's apparel preferences instead of hardcoded categories
            apparel_filters = []
            
            # Check user's apparel preferences
            if user_data.get('Apparel Pref Business Casual', False):
                apparel_filters.extend(['shirt', 'pant', 'trouser', 'jean', 'blouse', 'top'])
            
            if user_data.get('Apparel Pref Streetwear', False):
                apparel_filters.extend(['tshirt', 't-shirt', 'hoodie', 'sweatshirt', 'jean', 'pant', 'jogger', 'track'])
            
            if user_data.get('Apparel Pref Athleisure', False):
                apparel_filters.extend(['jogger', 'legging', 'track', 'sport', 'athletic', 'hoodie', 'sweatshirt', 'tshirt', 't-shirt'])
            
            # If no specific apparel preferences, use fallback based on gender
            if not apparel_filters:
                if user_gender in ['male', 'men']:
                    apparel_filters = ['shirt', 'pant', 'jean', 'trouser', 'tshirt', 't-shirt']
                elif user_gender in ['female', 'women']:
                    apparel_filters = ['top', 'blouse', 'shirt', 'pant', 'jean', 'trouser', 'skirt', 'dress']
                else:
                    apparel_filters = ['shirt', 'pant', 'jean', 'trouser', 'top', 'blouse']
            
            # Remove duplicates and create category filter
            apparel_filters = list(set(apparel_filters))
            category_filter = ','.join([f"scraped_category.ilike.%{filter}%" for filter in apparel_filters])
            
            # Apply category filter
            query = query.or_(category_filter)
            
            # Apply gender filtering at database level
            if user_gender:
                if user_gender in ['male', 'men']:
                    # For male users: allow male and unisex products
                    query = query.in_('gender', ['Male', 'Unisex'])
                elif user_gender in ['female', 'women']:
                    # For female users: allow female and unisex products
                    query = query.in_('gender', ['Female', 'Unisex'])
                elif user_gender in ['unisex']:
                    # For unisex users: allow all genders
                    pass  # No additional filtering needed
                else:
                    # For other genders: allow same gender and unisex
                    query = query.in_('gender', [user_gender.title(), 'Unisex'])
            
            # Execute the filtered query
            logger.info(f"📥 Executing pre-filtered database query with categories: {apparel_filters}")
            result = query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df = self._process_products_dataframe(df)
                
                logger.info(f"✅ Pre-filtered query returned {len(df)} products (vs loading all products)")
                logger.info(f"📊 Products by wear type: {df['wear_type'].value_counts().to_dict()}")
                
                return df
            else:
                logger.warning("⚠️ No products found with basic filters, trying fallback...")
                # Fallback to chunked loading if pre-filtering fails
                return self.get_products()
                
        except Exception as e:
            logger.error(f"❌ Error in pre-filtered product query: {e}")
            logger.info("🔄 Falling back to chunked loading...")
            return self.get_products()

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
            result = self.client.table('similar_products').upsert(records).execute()
            
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

# Global database instance
db = SupabaseDB()

def get_db() -> SupabaseDB:
    """Get the global database instance."""
    return db 