"""
Module 3 - Relief Predictions Database
"""

import json
from datetime import datetime
from config import Config, get_supabase_client


class ReliefDatabase:
    """Handles relief predictions in Supabase (similar to Module 1)"""
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self._connect()
        
        # Sample data for MOCK MODE (when Module 2 not ready)
        self.sample_data = {
            "Gampaha": {"affected_population": 42105, "children_pct": 0.28, "elderly_pct": 0.13, "female_pct": 0.52},
            "Negombo": {"affected_population": 30574, "children_pct": 0.28, "elderly_pct": 0.12, "female_pct": 0.48},
            "Ja Ela": {"affected_population": 38596, "children_pct": 0.29, "elderly_pct": 0.12, "female_pct": 0.50},
            "Wattala": {"affected_population": 39874, "children_pct": 0.29, "elderly_pct": 0.12, "female_pct": 0.51},
            "Katana": {"affected_population": 37110, "children_pct": 0.27, "elderly_pct": 0.14, "female_pct": 0.49},
            "Kelaniya": {"affected_population": 35842, "children_pct": 0.26, "elderly_pct": 0.13, "female_pct": 0.47},
            "Biyagama": {"affected_population": 32105, "children_pct": 0.26, "elderly_pct": 0.13, "female_pct": 0.48},
            "Minuwangoda": {"affected_population": 28432, "children_pct": 0.27, "elderly_pct": 0.15, "female_pct": 0.46},
            "Mahara": {"affected_population": 26398, "children_pct": 0.26, "elderly_pct": 0.14, "female_pct": 0.49},
            "Dompe": {"affected_population": 24285, "children_pct": 0.25, "elderly_pct": 0.16, "female_pct": 0.47},
            "Attanagalla": {"affected_population": 22596, "children_pct": 0.25, "elderly_pct": 0.15, "female_pct": 0.48},
            "Mirigama": {"affected_population": 19874, "children_pct": 0.23, "elderly_pct": 0.17, "female_pct": 0.47},
            "Divulapitiya": {"affected_population": 16873, "children_pct": 0.22, "elderly_pct": 0.18, "female_pct": 0.46},
            "Colombo": {"affected_population": 31044, "children_pct": 0.27, "elderly_pct": 0.12, "female_pct": 0.49},
        }

    def _connect(self):
        """Connect to Supabase (like Module 1)"""
        try:
            self.client = get_supabase_client()
        except Exception:
            self.client = None

    # ============================================================
    # READ FROM MODULE 1 (Flood Detection)
    # ============================================================
    
    def get_flood_severity(self, division_name, event_date=None):
        """
        Get flood severity from Module 1's table
        Reads from: flood_detection_results
        """
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
            
            if response.data and len(response.data) > 0:
                severity = response.data[0].get('priority_label', 'Medium')
                return severity
            else:
                return 'Medium'
                
        except Exception:
            return 'Medium'

    # ============================================================
    # READ FROM MODULE 2 (Population) - MOCK or REAL
    # ============================================================
    
    def get_population_data(self, division_name, event_date=None):
        """
        Get population and demographics
        Uses mock data if Module 2 not ready
        """
        if self.config.USE_MOCK_DATA:
            if division_name in self.sample_data:
                return self.sample_data[division_name]
            return {'affected_population': 10000, 'children_pct': 0.25, 'elderly_pct': 0.15, 'female_pct': 0.50}
        
        # REAL MODE - when Module 2 is ready
        try:
            import pandas as pd
            import glob
            files = glob.glob(f"{self.config.MODULE2_DATA_PATH}/Master_Feature_Matrix_*.csv")
            if files:
                df = pd.read_csv(files[0])
                row = df[df['Ds_Division_Name'] == division_name].iloc[0]
                return {
                    'affected_population': int(row['Affected_People']),
                    'children_pct': float(row['Children_%']),
                    'elderly_pct': float(row['Elderly_%']),
                    'female_pct': float(row['Female_%'])
                }
        except Exception:
            pass
        
        return {'affected_population': 10000, 'children_pct': 0.25, 'elderly_pct': 0.15, 'female_pct': 0.50}

    # ============================================================
    # SAVE TO MODULE 3 TABLE (Relief Predictions)
    # ============================================================
    
    def save_prediction(self, division_name, input_data, predictions, overall_priority, explanation, event_date=None):
        """
        Save relief predictions to Supabase (like Module 1's save_results)
        Table: relief_predictions
        """
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
        """
        Fetch relief predictions from Supabase (like Module 1's get_latest_results)
        """
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
        """Get list of all DS Divisions"""
        if self.config.USE_MOCK_DATA:
            return list(self.sample_data.keys())
        else:
            return [
                "Gampaha", "Negombo", "Ja Ela", "Wattala", "Katana", "Kelaniya",
                "Biyagama", "Minuwangoda", "Mahara", "Dompe", "Attanagalla",
                "Mirigama", "Divulapitiya", "Colombo", "Kaduwela", "Moratuwa"
            ]


# Create singleton instance (like Module 1)
db = ReliefDatabase()