
"""
modules/flood_detection/config.py
Supabase-only configuration - no local files needed.
"""

import os

class FloodDetectionConfig:
    # ── Division Level ─────────────────────────────────────────
    DIVISION_LEVEL = 'DS'
    DIVISION_NAME_COLUMN = 'adm3_name' 

    # ── Supabase Storage ───────────────────────────────────────
    SUPABASE_BUCKET = 'flood-data'
    SHAPEFILE_KEY = 'gampaha_divisions.shp'
    MODEL_KEY = 'flood_depth_model.pkl'
    SCALER_KEY = 'flood_depth_scaler.pkl'

    # ── Adaptive Thresholding ─────────────────────────────────
    BLOCK_SIZE  = 15
    C_CONSTANT  = -0.1   # negative — lowers threshold so water pixels pass
    SIGMA_SPACE = 1.5
    SIGMA_RANGE = 0.2    # slightly wider spectral tolerance
    
    # ── Priority Thresholds (metres) ──────────────────────────
    HIGH_PRIORITY_DEPTH = 1.5
    MEDIUM_PRIORITY_DEPTH = 0.5

    # ── Rainfall ──────────────────────────────────────────────
    RAINFALL_MIN_MM = 150.0
    RAINFALL_MAX_MM = 375.0

    # ── Supabase ───────────────────────────────────────────────
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    SUPABASE_TABLE = 'flood_detection_results'


def get_supabase_client():
    """Get Supabase client from environment."""
    from supabase import create_client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    
    return create_client(url, key)


def get_file_from_supabase(bucket_name, file_key):
    """
    Get file bytes directly from Supabase Storage.
    Returns bytes, no local file created.
    """
    supabase = get_supabase_client()
    
    try:
        response = supabase.storage.from_(bucket_name).download(file_key)
        print(f"   ✓ Retrieved {file_key} from Supabase")
        return response
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot retrieve {file_key} from bucket '{bucket_name}': {str(e)}\n"
            f"Ensure the file exists in Supabase Storage"
        )