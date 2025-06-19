-- Create user_outfits table for MyMirro Backend
-- This table stores AI-generated outfit recommendations for users

CREATE TABLE IF NOT EXISTS user_outfits (
    id SERIAL PRIMARY KEY,
    main_outfit_id TEXT NOT NULL,
    user_id BIGINT NOT NULL,
    rank INTEGER NOT NULL,
    score FLOAT NOT NULL,
    explanation TEXT,
    
    -- Top product details
    top_id TEXT,
    top_title TEXT,
    top_image TEXT,
    top_price FLOAT,
    top_style TEXT,
    top_color TEXT,
    top_semantic_score FLOAT,
    
    -- Bottom product details
    bottom_id TEXT,
    bottom_title TEXT,
    bottom_image TEXT,
    bottom_price FLOAT,
    bottom_style TEXT,
    bottom_color TEXT,
    bottom_semantic_score FLOAT,
    
    -- Combined details
    total_price FLOAT,
    generated_at TIMESTAMP DEFAULT NOW(),
    generation_method TEXT DEFAULT 'supabase_faiss_semantic',
    
    -- Constraints
    UNIQUE(user_id, rank),
    FOREIGN KEY (user_id) REFERENCES users_updated(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_outfits_user_id ON user_outfits(user_id);
CREATE INDEX IF NOT EXISTS idx_user_outfits_rank ON user_outfits(rank);
CREATE INDEX IF NOT EXISTS idx_user_outfits_score ON user_outfits(score);

-- Enable Row Level Security (optional)
ALTER TABLE user_outfits ENABLE ROW LEVEL SECURITY;

-- Create policy to allow service role to manage all outfits (for API)
CREATE POLICY "Service role can manage all outfits" ON user_outfits
    FOR ALL USING (auth.role() = 'service_role');

-- Note: User-specific policies disabled due to UUID/BIGINT mismatch
-- You can implement user access control in your application layer instead 