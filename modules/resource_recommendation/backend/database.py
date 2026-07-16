"""
Module 3 - Relief Predictions Database
Integrated with Module 1 (flood) and Module 2 (population)
"""

import json
import os
from datetime import datetime
from config import Config, get_supabase_client

# Import Module 2 bridge
from module2_bridge import bridge


class ReliefDatabase:
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = get_supabase_client()
        except Exception:
            self.client = None

    # ============================================================
    # READ FROM MODULE 1 (Flood Severity)
    # ============================================================
    
    def get_flood_severity(self, division_name, event_date=None):
        if self.client is None:
            return 'Medium'
        try:
            query = self.client.table(self.config.FLOOD_TABLE).select('*')
            query = query.eq(self.config.DIVISION_NAME_COLUMN_MODULE1, division_name)
            if event_date:
                query = query.eq('event_date', event_date)
            else:
                query = query.order('created_at', desc=True).limit(1)
            response = query.execute()
            if response.data:
                return response.data[0].get('priority_label', 'Medium')
            return 'Medium'
        except Exception:
            return 'Medium'

    # ============================================================
    # READ FROM MODULE 2 (Population)
    # ============================================================
    
    def get_population_data(self, division_name, event_date=None):
        """
        Get population data - Priority:
        1. Module 2 (live predictions)
        2. CSV files (Module 2 saved output)
        3. Default fallback
        """
        
        # METHOD 1: Module 2 Live Engine
        if not self.config.USE_MOCK_DATA:
            try:
                flood_raster = self._find_flood_raster()
                rainfall = self._get_rainfall(event_date)
                
                data = bridge.get_population_data(
                    division_name=division_name,
                    flood_raster_path=flood_raster,
                    rainfall_mm=rainfall
                )
                
                if data and data.get('affected_population', 0) > 0:
                    return data
            except Exception:
                pass
        
        # METHOD 2: CSV (Module 2 saved output)
        try:
            import pandas as pd
            import glob
            files = glob.glob(f"{self.config.MODULE2_DATA_PATH}/Master_Feature_Matrix_*.csv")
            if files:
                df = pd.read_csv(files[0])
                row = df[df['Ds_Division_Name'] == division_name]
                if len(row) > 0:
                    row = row.iloc[0]
                    return {
                        'affected_population': int(row['Affected_People']),
                        'children_pct': float(row['Children_%']),
                        'elderly_pct': float(row['Elderly_%']),
                        'female_pct': float(row['Female_%'])
                    }
        except Exception:
            pass
        
        # METHOD 3: Default fallback
        return {
            'affected_population': 10000,
            'children_pct': 0.25,
            'elderly_pct': 0.15,
            'female_pct': 0.50
        }

    def _find_flood_raster(self):
        """Find flood raster from Module 1"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../interface/backend/outputs'))
        for ext in ['.tif', '.tiff']:
            path = os.path.join(base_dir, f'flood_extent_4326{ext}')
            if os.path.exists(path):
                return path
        return None

    def _get_rainfall(self, event_date):
        """Get rainfall for event"""
        return 150.0  # Default

    # ============================================================
    # SAVE TO MODULE 3 TABLE
    # ============================================================
    
    def save_prediction(self, division_name, input_data, predictions, overall_priority, explanation, event_date=None):
        if self.client is None:
            return False
        
        event_date = event_date or datetime.now().strftime('%Y-%m-%d')
        
        record = {
            'ds_division': division_name,
            'event_date': event_date,
            'affected_population': input_data['affected_population'],
            'children_percentage': input_data['children_percentage'],
            'elderly_percentage': input_data['elderly_percentage'],
            'female_percentage': input_data['female_percentage'],
            'flood_severity': input_data['flood_severity'],
            'overall_priority': overall_priority,
            'relief_predictions': json.dumps(predictions),
            'explanation': explanation,
            'created_at': datetime.now().isoformat()
        }
        
        try:
            self.client.table(self.config.RELIEF_TABLE).insert(record).execute()
            return True
        except Exception:
            return False

    def get_predictions(self, division_name=None, event_date=None):
        if self.client is None:
            return []
        try:
            query = self.client.table(self.config.RELIEF_TABLE).select('*')
            if division_name:
                query = query.eq('ds_division', division_name)
            if event_date:
                query = query.eq('event_date', event_date)
            query = query.order('created_at', desc=True)
            response = query.execute()
            return response.data
        except Exception:
            return []

    def get_division_list(self):
        return [
            "Gampaha", "Negombo", "Ja Ela", "Wattala", "Katana", "Kelaniya",
            "Biyagama", "Minuwangoda", "Mahara", "Dompe", "Attanagalla",
            "Mirigama", "Divulapitiya", "Colombo", "Kaduwela", "Moratuwa"
        ]


db = ReliefDatabase()