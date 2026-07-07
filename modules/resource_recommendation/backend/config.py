"""
Module 3 Configuration
Loads all settings from .env file
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # Module 3 settings
    MODULE3_PORT = int(os.getenv('MODULE3_PORT', 5001))
    DEBUG = True if os.getenv('DEBUG', 'True') == 'True' else False
    
    # Supabase settings
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # Table names
    FLOOD_TABLE = 'flood_detection_results'  # Module 1's table
    RELIEF_TABLE = 'relief_predictions'       # Module 3's table
    
    # Module 1 settings
    MODULE1_API_URL = os.getenv('MODULE1_API_URL', 'http://localhost:5001')
    
    # Module 2 settings (for later integration)
    MODULE2_DATA_PATH = os.getenv('MODULE2_DATA_PATH', '../../data/processed/population')
    USE_MOCK_DATA = os.getenv('USE_MOCK_DATA', 'True') == 'True'
    
    # Module 3 data
    DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', '../data/Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx')
    MODEL_DIR = 'models_saved'
    
    # Feature columns
    FEATURE_COLUMNS = ['Affected_Population', 'Children_%', 'Elderly_%', 'Female %', 'Severity_Code']
    
    # Target columns
    TARGET_COLUMNS = [
        'Cooked Food Packs', 'Water Bottles', 'Milk Powder Packs',
        'Infant Milk Powder Packs', 'Biscuits Packs', 'Noodles Packs',
        'Tea Powder Packets', 'Sanitary', 'Soap', 'Toothpaste', 'Toothbrushes'
    ]
    
    # Severity mapping
    SEVERITY_MAP = {'Low': 1, 'Medium': 2, 'High': 3}
    
    # Division name columns
    DIVISION_NAME_COLUMN_MODULE1 = 'ds_division'
    DIVISION_NAME_COLUMN_MODULE2 = 'Ds_Division_Name'


def get_supabase_client():
    """Get Supabase client"""
    from supabase import create_client
    
    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)