

"""
modules/flood_detection/database.py
Supabase-only database operations.
"""

import json
from datetime import datetime
from .config import FloodDetectionConfig, get_supabase_client


class FloodDetectionDatabase:
    """Handles flood detection results in Supabase."""
    
    def __init__(self, config: FloodDetectionConfig = None):
        self.config = config or FloodDetectionConfig()
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Supabase."""
        try:
            self.client = get_supabase_client()
            print("   Connected to Supabase")
        except Exception as e:
            print(f"   Cannot connect to Supabase: {e}")
            raise

    def save_results(self, gdf, stats, event_date=None):
        """Save results directly to Supabase."""
        event_date = event_date or datetime.now().strftime('%Y-%m-%d')
        
        records = []
        name_col = self.config.DIVISION_NAME_COLUMN
        
        for _, row in gdf.iterrows():
            record = {
                'ds_division': str(row.get(name_col, 'Unknown')),
                'flood_area_ha': float(row.get('flood_area_ha', 0)),
                'flood_depth_mean': float(row.get('flood_depth_mean', 0)),
                'flood_depth_max': float(row.get('flood_depth_max', 0)),
                'priority': int(row.get('priority', 0)),
                'priority_label': str(row.get('priority_label', 'No Flood')),
                'geometry': row.geometry.wkt if row.geometry else None,
                'event_date': event_date,
            }
            records.append(record)
        
        # Delete old records for this date
        self.client.table(self.config.SUPABASE_TABLE)\
            .delete()\
            .eq('event_date', event_date)\
            .execute()
        
        # Insert new records in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.client.table(self.config.SUPABASE_TABLE)\
                .insert(batch)\
                .execute()
        
        print(f"   Saved {len(records)} records to Supabase")
        return True

    def get_latest_results(self, event_date=None):
        """Fetch results from Supabase."""
        query = self.client.table(self.config.SUPABASE_TABLE).select('*')
        
        if event_date:
            query = query.eq('event_date', event_date)
        else:
            query = query.order('created_at', desc=True).limit(1000)
        
        response = query.execute()
        return response.data