"""
Module 2 Bridge - Connects Module 3 with Module 2
"""

import os
import sys

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
    
    def __init__(self):
        self.engine = None
        if MODULE2_AVAILABLE:
            try:
                self.engine = LivePopulationRiskEndpoint()
                print("Module 2 engine loaded")
            except Exception as e:
                print(f"Module 2 engine failed: {e}")
    
    def get_population_data(self, division_name, flood_raster_path=None, rainfall_mm=150):
        """Get population data from Module 2 for a division"""
        
        # If Module 2 not available, return None
        if self.engine is None:
            return None
        
        # Find flood raster if not provided
        if flood_raster_path is None:
            flood_raster_path = self._find_flood_raster()
        
        if flood_raster_path is None:
            print("Module 2: No flood raster found")
            return None
        
        try:
            # Call Module 2
            result = self.engine.predict_realtime_demographics(
                input_precip_mm=rainfall_mm,
                flood_tiff_path=flood_raster_path
            )
            
            if result.get('status') != 'SUCCESS':
                return None
            
            # Find division data
            for div_data in result.get('demographic_data', []):
                if div_data.get('division_name') == division_name:
                    summary = div_data.get('summary_metrics', {})
                    age = div_data.get('age_demographics', {})
                    gender = div_data.get('gender_demographics', {})
                    
                    total = summary.get('predicted_mean_affected', 0)
                    if total == 0:
                        return None
                    
                    return {
                        'affected_population': total,
                        'children_pct': age.get('children_count_0_14', 0) / total,
                        'elderly_pct': age.get('elderly_count_60_plus', 0) / total,
                        'female_pct': gender.get('female_count', 0) / total
                    }
            
            return None
            
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