#!/usr/bin/env python3
"""
Designer Bonus Updater - Learning Loop Pack v1
Nightly script to compute pair bonuses from stylist ratings
"""

import os
import sys
import pandas as pd
import pathlib
import json
import datetime as dt
from supabase import create_client

# Add current directory to path to import from phase1 module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_supabase_outfits_generator import _calc_pair_bonus

def main():
    """Main function to update designer bonuses from outfit ratings."""
    try:
        # Get Supabase credentials from environment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Error: SUPABASE_URL and SUPABASE_ANON_KEY environment variables required")
            sys.exit(1)
        
        # Initialize Supabase client
        supa = create_client(supabase_url, supabase_key)
        
        # Get ratings from last 30 days
        since = (dt.datetime.utcnow() - dt.timedelta(days=30)).isoformat()
        print(f"🔄 Fetching outfit ratings since {since}")
        
        # Query outfit_rating_advance table
        result = supa.table("outfit_rating_advance").select("*").gte("created_at", since).execute()
        rows = result.data
        
        if not rows:
            print("✅ No ratings found in the last 30 days")
            sys.exit(0)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        print(f"📊 Found {len(df)} ratings")
        
        # Calculate pair bonuses
        df["pair_bonus"] = df.apply(_calc_pair_bonus, axis=1)
        
        # Group by top_id and bottom_id, calculate mean bonus
        bonus_df = df.groupby(["top_id", "bottom_id"])["pair_bonus"].mean().reset_index()
        print(f"🎯 Computed bonuses for {len(bonus_df)} unique outfit pairs")
        
        # Create cache directory
        cache_dir = pathlib.Path(__file__).parent / "faiss_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Save to cache file
        cache_file = cache_dir / "designer_pair_bonus.json"
        bonus_dict = {f"{r.top_id}:::{r.bottom_id}": r.pair_bonus for r in bonus_df.itertuples()}
        
        with open(cache_file, 'w') as f:
            json.dump(bonus_dict, f, indent=2)
        
        print(f"💾 Saved {len(bonus_dict)} bonuses to {cache_file}")
        
        # Print some statistics
        positive_bonuses = sum(1 for v in bonus_dict.values() if v > 0)
        negative_bonuses = sum(1 for v in bonus_dict.values() if v < 0)
        neutral_bonuses = sum(1 for v in bonus_dict.values() if v == 0)
        
        print(f"📈 Statistics:")
        print(f"   Positive bonuses: {positive_bonuses}")
        print(f"   Negative bonuses: {negative_bonuses}")
        print(f"   Neutral bonuses: {neutral_bonuses}")
        
        if bonus_dict:
            avg_bonus = sum(bonus_dict.values()) / len(bonus_dict)
            print(f"   Average bonus: {avg_bonus:.3f}")
        
        print("✅ Designer bonus update completed successfully!")
        
    except Exception as e:
        print(f"❌ Error updating designer bonuses: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# TODO: schedule this via crontab or Supabase Edge Function
# Example crontab entry for nightly run at 2 AM:
# 0 2 * * * cd /path/to/mymirro-backend && python designer_bonus_updater.py 