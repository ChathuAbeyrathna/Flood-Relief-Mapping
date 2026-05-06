#data access/usage layer
import os
import rasterio
import pandas as pd

class AffectedPopulationProcessor:
    def __init__(self, processed_dir="data/processed/population/"):
        self.base_path = processed_dir # Where processed data lives

    def fetch_raster_path(self, year, data_type="POP"): # finds correct file
        # Finds the correct file based on year and type (POP, BUILT, SMOD)
        files = os.listdir(self.base_path)
        for f in files:
            if f"E{year}" in f and data_type in f and f.endswith(".tif"):
                return os.path.join(self.base_path, f)
        return None

    def get_population_grid(self, year):
        path = self.fetch_raster_path(year, "POP")
        if path:
            with rasterio.open(path) as src:
                return src.read(1), src.transform
        return None, None

    def get_calendar_data(self, year):
        path = os.path.join(self.base_path, f"sl_calendar_{year}_Gampaha.csv")
        if os.path.exists(path):
            return pd.read_csv(path) # Loads csv
        return None