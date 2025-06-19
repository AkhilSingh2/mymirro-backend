-- Enhanced User Outfits Table Schema
-- Drop existing table if it exists
DROP TABLE IF EXISTS user_outfits;

-- Create enhanced user_outfits table with all required columns
CREATE TABLE user_outfits (
    -- Primary key and basic info
    id BIGSERIAL PRIMARY KEY,
    main_outfit_id VARCHAR(100) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    score DECIMAL(4,3) NOT NULL,
    explanation TEXT,
    
    -- Top product details
    top_id VARCHAR(100) NOT NULL,
    top_title VARCHAR(500),
    top_image VARCHAR(1000),
    top_price DECIMAL(10,2),
    top_style VARCHAR(200),
    top_color VARCHAR(100),
    top_semantic_score DECIMAL(4,3),
    
    -- Bottom product details
    bottom_id VARCHAR(100) NOT NULL,
    bottom_title VARCHAR(500),
    bottom_image VARCHAR(1000),
    bottom_price DECIMAL(10,2),
    bottom_style VARCHAR(200),
    bottom_color VARCHAR(100),
    bottom_semantic_score DECIMAL(4,3),
    
    -- Combined outfit details
    total_price DECIMAL(10,2),
    generation_method VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_user_outfits_user_id ON user_outfits(user_id);
CREATE INDEX idx_user_outfits_score ON user_outfits(score DESC);
CREATE INDEX idx_user_outfits_rank ON user_outfits(rank);
CREATE INDEX idx_user_outfits_main_outfit_id ON user_outfits(main_outfit_id);
CREATE INDEX idx_user_outfits_top_style ON user_outfits(top_style);
CREATE INDEX idx_user_outfits_total_price ON user_outfits(total_price);

-- Add foreign key constraint if users table exists
-- ALTER TABLE user_outfits ADD CONSTRAINT fk_user_outfits_user_id 
-- FOREIGN KEY (user_id) REFERENCES users_updated(id);

-- Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_outfits_updated_at 
    BEFORE UPDATE ON user_outfits 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions (adjust based on your user setup)
-- GRANT ALL PRIVILEGES ON TABLE user_outfits TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE user_outfits_id_seq TO your_app_user;

-- Add comments for documentation
COMMENT ON TABLE user_outfits IS 'Enhanced user outfit recommendations with complete product details and scoring';
COMMENT ON COLUMN user_outfits.main_outfit_id IS 'Unique identifier for the outfit recommendation';
COMMENT ON COLUMN user_outfits.score IS 'AI-calculated compatibility score (0.0-1.0)';
COMMENT ON COLUMN user_outfits.explanation IS 'Human-readable explanation of why this outfit works';
COMMENT ON COLUMN user_outfits.generation_method IS 'Method used to generate this recommendation'; 