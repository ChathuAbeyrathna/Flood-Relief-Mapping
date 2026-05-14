import os
import glob # For file searching and path handling
import rasterio # For reading raster data and handling geospatial transformations
import pandas as pd # For DataFrame creation and CSV output
import numpy as np # For efficient array handling and masking
from rasterio.warp import transform  # Essential for coordinate alignment

def run_raster_to_csv_conversion():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    processed_dir = os.path.join(base_dir, "data", "processed", "population")
    tif_files = [f for f in glob.glob(os.path.join(processed_dir, "*")) if f.lower().endswith('.tif')] # Search for .tif files in the processed directory

    if not tif_files:
        print("No clipped .tif files found.")
        return

    # Target CRS is standard Latitude/Longitude (WGS84)
    dst_crs = 'EPSG:4326'

    for tif_path in tif_files: # Process each .tif file found
        base_filename = os.path.splitext(os.path.basename(tif_path))[0] # Extract the base filename without extension for use in the CSV output
        output_path = os.path.join(processed_dir, f"{base_filename}.csv") # Define the output path for the CSV file, using the same base filename as the .tif but with a .csv extension

        print(f"Aligning and Converting: {base_filename}.tif")

        with rasterio.open(tif_path) as src: # Open the .tif file using rasterio to read the raster data and access geospatial metadata
            band1 = src.read(1) # Read the first band of the raster, which contains the population data
            mask = (band1 != src.nodata) # Create a mask to filter out NoData values, ensuring we only process valid population data points

            # Get pixel indices
            height, width = band1.shape # Get the dimensions of the raster to create a grid of pixel coordinates
            cols, rows = np.meshgrid(np.arange(width), np.arange(height)) # Create a grid of column and row indices corresponding to each pixel in the raster

            rows_filtered = rows[mask] # Filter the row and column indices using the mask to keep only valid data points
            cols_filtered = cols[mask]
            values_filtered = band1[mask] # Filter the population values using the same mask to keep only valid data points

            # Get native coordinates (Mollweide for GHS, Lat/Lon for others)
            native_xs, native_ys = rasterio.transform.xy(src.transform, rows_filtered, cols_filtered)

            # ALIGNMENT STEP: Convert native coordinates to EPSG:4326 (Lat/Lon)
            lons, lats = transform(src.crs, dst_crs, native_xs, native_ys) # Use rasterio's transform function to convert the native coordinates to the target CRS (EPSG:4326)

            df = pd.DataFrame({ # Create a DataFrame with the aligned coordinates and population values
                'longitude': lons,
                'latitude': lats,
                base_filename: values_filtered # Use the base filename as the column name for the population values, ensuring that each CSV file has a unique column name corresponding to its source .tif file
            })

            df.to_csv(output_path, index=False) # Save the DataFrame to a CSV file at the defined output path, without including the index in the CSV
            print(f"Successfully aligned and created: {os.path.basename(output_path)}")

    print("\n--- Coordinate Alignment & CSV Conversion Complete ---")

if __name__ == "__main__":
    run_raster_to_csv_conversion()