
# Supabase integration for saving flood detection results.


import json
import os
from datetime import datetime
from .config import FloodDetectionConfig


class FloodDetectionDatabase:
    """
    Handles saving flood detection results to Supabase.
    Other modules (population, resource) can read from the same table.
    """

    def __init__(self, config: FloodDetectionConfig = None):
        self.config = config or FloodDetectionConfig()
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Supabase."""
        try:
            from supabase import create_client
            url = self.config.SUPABASE_URL
            key = self.config.SUPABASE_KEY

            if not url or not key:
                print("   Warning: Supabase credentials not set.")
                print("   Set SUPABASE_URL and SUPABASE_KEY in .env file.")
                return

            self.client = create_client(url, key)
            print("   Connected to Supabase")

        except ImportError:
            print("   Warning: supabase package not installed.")
            print("   Run: pip install supabase")
        except Exception as e:
            print(f"   Warning: Supabase connection failed: {e}")

    def save_results(self, gdf, stats, event_date=None):
        """
        Save per-division flood results to Supabase.

        Parameters:
            gdf        : GeoDataFrame with per-division results
            stats      : summary statistics dict
            event_date : date string e.g. '2025-12-04'

        Returns:
            bool: True if saved successfully
        """
        if self.client is None:
            print("   Supabase not connected — saving to local JSON instead")
            return self._save_local(gdf, stats, event_date)

        try:
            event_date = event_date or datetime.now().strftime('%Y-%m-%d')

            # Build records for each division
            records = []
            name_col = self.config.DIVISION_NAME_COLUMN

            for _, row in gdf.iterrows():
                record = {
                    'ds_division':       str(row.get(name_col, 'Unknown')),
                    'flood_area_ha':     float(row.get('flood_area_ha', 0)),
                    'flood_depth_mean':  float(row.get('flood_depth_mean', 0)),
                    'flood_depth_max':   float(row.get('flood_depth_max', 0)),
                    'priority':          int(row.get('priority', 0)),
                    'priority_label':    str(row.get('priority_label', 'No Flood')),
                    'geometry':          row.geometry.wkt if row.geometry else None,
                    'event_date':        event_date,
                    'division_level':    self.config.DIVISION_LEVEL,
                }
                records.append(record)

            # Delete old results for same event date then insert new
            self.client.table(self.config.SUPABASE_TABLE)\
                .delete()\
                .eq('event_date', event_date)\
                .execute()

            # Insert in batches of 100
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                self.client.table(self.config.SUPABASE_TABLE)\
                    .insert(batch)\
                    .execute()

            print(f"   Saved {len(records)} division records to Supabase")
            return True

        except Exception as e:
            print(f"   Supabase save failed: {e}")
            print("   Saving to local JSON instead")
            return self._save_local(gdf, stats, event_date)

    def _save_local(self, gdf, stats, event_date):
        """Fallback — save results as local JSON when Supabase unavailable."""
        os.makedirs('outputs', exist_ok=True)

        output = {
            'event_date':     event_date or datetime.now().strftime('%Y-%m-%d'),
            'summary':        stats,
            'division_level': self.config.DIVISION_LEVEL,
            'divisions':      []
        }

        name_col = self.config.DIVISION_NAME_COLUMN

        for _, row in gdf.iterrows():
            output['divisions'].append({
                'name':             str(row.get(name_col, 'Unknown')),
                'flood_area_ha':    float(row.get('flood_area_ha', 0)),
                'flood_depth_mean': float(row.get('flood_depth_mean', 0)),
                'flood_depth_max':  float(row.get('flood_depth_max', 0)),
                'priority':         int(row.get('priority', 0)),
                'priority_label':   str(row.get('priority_label', 'No Flood')),
            })

        path = 'outputs/flood_detection_results.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"   Results saved locally to {path}")
        return True

    def get_latest_results(self, event_date=None):
        """
        Fetch latest flood results from Supabase.
        Used by Module 2 (population) and Module 3 (resources).
        """
        if self.client is None:
            return self._load_local()

        try:
            query = self.client.table(self.config.SUPABASE_TABLE).select('*')

            if event_date:
                query = query.eq('event_date', event_date)
            else:
                query = query.order('created_at', desc=True).limit(1000)

            response = query.execute()
            return response.data

        except Exception as e:
            print(f"   Failed to fetch from Supabase: {e}")
            return self._load_local()

    def _load_local(self):
        """Load from local JSON fallback."""
        path = 'outputs/flood_detection_results.json'
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None