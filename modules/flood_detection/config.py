"""
modules/flood_detection/config.py

Central configuration file.
To switch from GN to DS divisions:
  1. Change SHAPEFILE_PATH to your DS shapefile
  2. Change DIVISION_NAME_COLUMN to the correct column name
  3. Change DIVISION_LEVEL to 'DS'
That's it — nothing else needs to change.
"""

import os

class FloodDetectionConfig:

    # ── Division Level ─────────────────────────────────────────
    # Change this to 'DS' when DS shapefile is available
    DIVISION_LEVEL = 'GN'

    # ── Shapefile Path ─────────────────────────────────────────
    # Change this to DS shapefile path when available
    SHAPEFILE_PATH = 'data/gampaha_divisions.shp'

    # ── Division Name Column ───────────────────────────────────
    # Change this to match the column name in DS shapefile
    # Common options: 'Name', 'NAME', 'DS_NAME', 'GN_NAME'
    DIVISION_NAME_COLUMN = 'Name'

    # ── Model Paths ────────────────────────────────────────────
    DEPTH_MODEL_PATH  = 'data/flood_depth_model.pkl'
    DEPTH_SCALER_PATH = 'data/flood_depth_scaler.pkl'

    # ── Adaptive Thresholding Parameters ──────────────────────
    BLOCK_SIZE = 11   # neighbourhood size (must be odd)
    C_CONSTANT = 2    # constant subtracted from mean

    # ── Priority Thresholds (metres) ──────────────────────────
    HIGH_PRIORITY_DEPTH   = 1.5   # > 1.5m = High
    MEDIUM_PRIORITY_DEPTH = 0.5   # 0.5-1.5m = Medium
                                  # < 0.5m = Low

    # ── Rainfall estimate for depth model ─────────────────────
    # Based on CHIRPS data + Cyclone Ditwah records for Gampaha
    RAINFALL_MIN_MM = 150.0
    RAINFALL_MAX_MM = 375.0

    # ── Output ─────────────────────────────────────────────────
    OUTPUT_DIR     = 'outputs'
    GEOJSON_OUTPUT = 'outputs/flood_results.geojson'
    MAP_OUTPUT     = 'outputs/flood_map.png'

    # ── Supabase ───────────────────────────────────────────────
    # Fill these in after setting up Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    SUPABASE_TABLE = 'flood_detection_results'

    
# Module-level Supabase credentials for the download function
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

import os

def download_from_supabase(bucket_name, file_path, local_dir='data'):
    """Download file from Supabase if not exists locally."""
    from supabase import create_client
    
    local_path = os.path.join(local_dir, file_path)
    
    # Return local path if file exists
    if os.path.exists(local_path):
        return local_path
    
    # Download from Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    os.makedirs(local_dir, exist_ok=True)
    
    with open(local_path, 'wb') as f:
        data = supabase.storage.from_(bucket_name).download(file_path)
        f.write(data)
    
    return local_path