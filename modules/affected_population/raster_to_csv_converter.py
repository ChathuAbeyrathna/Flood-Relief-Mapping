import os
import glob
import rasterio
import pandas as pd
import numpy as np
from rasterio.warp import transform  # Essential for coordinate alignment

def run_raster_to_csv_conversion():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    processed_dir = os.path.join(base_dir, "data", "processed", "population")
    tif_files = [f for f in glob.glob(os.path.join(processed_dir, "*")) if f.lower().endswith('.tif')]

    if not tif_files:
        print("No clipped .tif files found.")
        return

    # Target CRS is standard Latitude/Longitude (WGS84)
    dst_crs = 'EPSG:4326'

    for tif_path in tif_files:
        base_filename = os.path.splitext(os.path.basename(tif_path))[0]
        output_path = os.path.join(processed_dir, f"{base_filename}.csv")

        print(f"Aligning and Converting: {base_filename}.tif")

        with rasterio.open(tif_path) as src:
            band1 = src.read(1)
            mask = (band1 != src.nodata)

            # Get pixel indices
            height, width = band1.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))

            rows_filtered = rows[mask]
            cols_filtered = cols[mask]
            values_filtered = band1[mask]

            # 1. Get native coordinates (Mollweide for GHS, Lat/Lon for others)
            native_xs, native_ys = rasterio.transform.xy(src.transform, rows_filtered, cols_filtered)

            # 2. ALIGNMENT STEP: Convert native coordinates to EPSG:4326 (Lat/Lon)
            # This handles the Mollweide (meters) to Lat/Lon conversion automatically
            lons, lats = transform(src.crs, dst_crs, native_xs, native_ys)

            df = pd.DataFrame({
                'longitude': lons,
                'latitude': lats,
                base_filename: values_filtered
            })

            df.to_csv(output_path, index=False)
            print(f"Successfully aligned and created: {os.path.basename(output_path)}")

    print("\n--- Coordinate Alignment & CSV Conversion Complete ---")

if __name__ == "__main__":
    run_raster_to_csv_conversion()