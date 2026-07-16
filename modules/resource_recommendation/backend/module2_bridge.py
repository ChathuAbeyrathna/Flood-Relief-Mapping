"""
Module 2 Bridge - Connects Module 3 with Module 2
WITH CACHING - Runs Module 2 only once per flood event
"""

import os
import sys
import json
from datetime import datetime

# Add Module 2 path to sys.path
MODULE2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../affected_population'))
sys.path.insert(0, MODULE2_PATH)

try:
    from live_prediction_endpoint import LivePopulationRiskEndpoint
    MODULE2_AVAILABLE = True
except ImportError:
    MODULE2_AVAILABLE = False
    print("Module 2 not available - using fallback data")


class Module2Bridge:
    
    # Cache file location
    CACHE_FILE = os.path.join(os.path.dirname(__file__), 'module2_cache.json')
    
    def __init__(self):
        self.engine = None
        self.cache = {}
        self._load_cache()
        
        if MODULE2_AVAILABLE:
            try:
                self.engine = LivePopulationRiskEndpoint()
                print("Module 2 engine loaded")
            except Exception as e:
                print(f"Module 2 engine failed: {e}")
    
    def _load_cache(self):
        """Load cached data from file"""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached entries")
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Save cached data to file"""
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
            return True
        except Exception as e:
            print(f"Could not save cache: {e}")
            return False
    
    def _get_event_key(self, flood_raster_path, rainfall_mm):
        """Create unique key for this flood event"""
        if flood_raster_path and os.path.exists(flood_raster_path):
            # Use file modification time as event identifier
            mtime = os.path.getmtime(flood_raster_path)
            return f"event_{mtime}_{rainfall_mm}"
        return f"event_default_{rainfall_mm}"
    
    def get_population_data(self, division_name, flood_raster_path=None, rainfall_mm=150):
        """Get population data from Module 2 for a division - WITH CACHING"""
        
        # Find flood raster if not provided
        if flood_raster_path is None:
            flood_raster_path = self._find_flood_raster()
        
        if flood_raster_path is None:
            print("Module 2: No flood raster found")
            return None
        
        # Create event key
        event_key = self._get_event_key(flood_raster_path, rainfall_mm)
        
        # Check if this division is already cached for this event
        cache_key = f"{event_key}_{division_name}"
        if cache_key in self.cache:
            print(f"Cached result for {division_name}")
            return self.cache[cache_key]
        
        # If Module 2 not available, return None
        if self.engine is None:
            return None
        
        try:
            print(f"Running Module 2 for {division_name} (this may take a moment)...")
            
            # Call Module 2 (this is the slow part)
            result = self.engine.predict_realtime_demographics(
                input_precip_mm=rainfall_mm,
                flood_tiff_path=flood_raster_path
            )
            
            if result.get('status') != 'SUCCESS':
                print(f"Module 2 returned error: {result.get('error', 'Unknown')}")
                return None
            
            # Extract data for ALL divisions and cache them
            all_divisions = result.get('demographic_data', [])
            print(f"Module 2 returned data for {len(all_divisions)} divisions")
            
            for div_data in all_divisions:
                div_name = div_data.get('division_name')
                if not div_name:
                    continue
                    
                summary = div_data.get('summary_metrics', {})
                age = div_data.get('age_demographics', {})
                gender = div_data.get('gender_demographics', {})
                
                total = summary.get('predicted_mean_affected', 0)
                if total == 0:
                    continue
                
                # Cache each division
                key = f"{event_key}_{div_name}"
                self.cache[key] = {
                    'affected_population': total,
                    'children_pct': age.get('children_count_0_14', 0) / total,
                    'elderly_pct': age.get('elderly_count_60_plus', 0) / total,
                    'female_pct': gender.get('female_count', 0) / total
                }
            
            # Save cache
            self._save_cache()
            print(f"Cached {len(self.cache)} entries")
            
            # Return requested division
            return self.cache.get(cache_key)
            
        except Exception as e:
            print(f"Module 2 error: {e}")
            return None
    
    def _find_flood_raster(self):
        """Find flood raster from Module 1"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../interface/backend/outputs'))
        for ext in ['.tif', '.tiff']:
            path = os.path.join(base_dir, f'flood_extent_4326{ext}')
            if os.path.exists(path):
                return path
        return None


# Create instance
bridge = Module2Bridge()