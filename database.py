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
                    limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get product data from Supabase tagged_products table with enhanced tags.
        
        Args:
            wear_type: Filter by wear type (Upperwear, Bottomwear, etc.)
            gender: Filter by gender
            style: Filter by style
            limit: Limit number of results
            
        Returns:
            pandas.DataFrame: Product data with rich tags (color, style, fabric, etc.)
        """
        try:
            if not self.client:
                logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            query = self.client.table('tagged_products').select('*')
            
            # Apply filters based on tagged_products table structure
            if wear_type:
                query = query.ilike('category', f'%{wear_type}%')
            if gender:
                # Filter based on category pattern or other available fields
                pass
            if style:
                query = query.ilike('primary_style', f'%{style}%')
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                
                # Map tagged_products columns to expected outfit generator columns
                # Use product_id as the main ID (references products table)
                if 'product_id' in df.columns:
                    df['id'] = df['product_id']
                
                # Add missing columns that the outfit generator expects
                if 'wear_type' not in df.columns:
                    df['wear_type'] = df['category'].apply(self._categorize_wear_type)
                
                if 'gender' not in df.columns:
                    # Infer gender from category
                    df['gender'] = df['category'].apply(self._infer_gender_from_category)
                
                # Use enhanced tags from tagged_products - these are already available!
                if 'final_caption' not in df.columns:
                    # Use the rich caption_display or create from enhanced fields
                    df['final_caption'] = df.apply(lambda row: 
                        row.get('caption_display', '') or 
                        f"{row.get('title', '')} - {row.get('primary_style', '')} {row.get('primary_color', '')} {row.get('primary_fabric', '')}".strip(), axis=1)
                
                # The tagged_products table already has these rich columns!
                # primary_color, primary_style, primary_fabric, primary_occasion, etc.
                
                logger.info(f"✅ Retrieved {len(df)} products from tagged_products table with enhanced tags")
                return df
            else:
                logger.warning("⚠️ No products found in tagged_products table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error fetching products: {e}")
            return pd.DataFrame()
    
    def _categorize_wear_type(self, category: str) -> str:
        """Categorize products into wear types based on category."""
        if not category:
            return 'Upperwear'
        
        category_lower = category.lower()
        
        # Classify based on your actual product categories
        if any(word in category_lower for word in [
            'kurta', 'shirt', 'top', 'blouse', 'jacket', 'sweater', 'tee', 'tank',
            'ethnic', 'print', 'cotton', 'motifs'
        ]):
            return 'Upperwear'
        elif any(word in category_lower for word in [
            'pant', 'jean', 'trouser', 'short', 'skirt', 'bottom', 'pajama', 'salwar'
        ]):
            return 'Bottomwear'
        elif any(word in category_lower for word in [
            'dress', 'gown', 'frock', 'sheath'
        ]):
            return 'Dresses'
        else:
            # For your data, most items seem to be ethnic wear (kurtas)
            return 'Upperwear'  # Default
    
    def _infer_gender_from_category(self, category: str) -> str:
        """Infer gender from product category."""
        if not category:
            return 'Unisex'
        
        category_lower = category.lower()
        
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
    
    def save_user_outfits(self, outfits_df: pd.DataFrame, user_id: int) -> bool:
        """
        Save outfit recommendations to Supabase.
        
        Args:
            outfits_df: DataFrame containing outfit recommendations
            user_id: User ID
            
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
            
            # Delete existing outfits for this user
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
                # Try to get style quiz data
                try:
                    quiz_result = self.client.table('style_quiz_updated').select('*').eq('id', style_quiz_id).execute()
                    
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
        Map columns from style_quiz_updated table to application format.
        
        Args:
            quiz_data: Raw style quiz data from database
            
        Returns:
            Dict: Mapped style quiz data with application column names
        """
        mapped_data = {}
        
        # Map style quiz columns to application format
        style_quiz_mappings = {
            # Personal info
            'body_shape': 'Body Shape',
            'height': 'Height',
            'weight': 'Weight',
            'skin_tone': 'Skin Tone',
            
            # Style preferences  
            'fashion_style': 'Fashion Style',
            'style_preference': 'Style Preference',
            'preferred_colors': 'Colors family',
            'color_preference': 'Color Preference',
            
            # Lifestyle
            'lifestyle': 'Lifestyle',
            'occupation': 'Occupation',
            'occasion_preference': 'Occasion',
            'activity_level': 'Activity Level',
            
            # Budget and shopping
            'budget_range': 'Budget Preference',
            'shopping_frequency': 'Shopping Frequency',
            'brand_preference': 'Brand Preference',
            
            # Fit preferences
            'fit_preference': 'Fit Preference',
            'comfort_level': 'Comfort Level',
            'fabric_preference': 'Fabric Preference',
            
            # Advanced preferences
            'pattern_preference': 'Pattern Preference',
            'sleeve_preference': 'Sleeve Preference',
            'neckline_preference': 'Neckline Preference',
            
            # Generated captions (these take priority)
            'upper_wear_caption': 'Upper Wear Caption',
            'lower_wear_caption': 'Lower Wear Caption',
            'style_description': 'Style Description'
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

# Global database instance
db = SupabaseDB()

def get_db() -> SupabaseDB:
    """Get the global database instance."""
    return db 