# Automation geospatial pipeline - clip & save
import os # file paths
import glob # find files recursively
import geopandas as gpd # handle shapefiles
import pandas as pd # csv handling
import rasterio # handle raster data (.tif)
from rasterio.mask import mask # clip rasters by mask

def run_preprocessing():
    # This finds the absolute path to the 'code' folder (finds project root directory)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))

    # Update paths to use base_dir (path to shapefile - Gampaha DS boundaries, raw data folder (input), and output folder)
    boundary_path = os.path.join(base_dir, "data", "processed", "population", "Gampaha_DS_Boundaries", "Gampaha_DS_Boundaries.shp") # Path to Gampaha DS boundary shapefile
    raw_root = os.path.join(base_dir, "data", "raw", "population") # data/raw/population/ is the root folder where all raw files are stored
    output_dir = os.path.join(base_dir, "data", "processed", "population") # data/processed/population/ is the output folder where clipped files will be saved

    # Check if the boundary file actually exists before trying to read it
    if not os.path.exists(boundary_path):
        print(f"ERROR: Shapefile not found at: {boundary_path}")
        return

    # Load the Gampaha Boundary (Loads shapefile as GeoDataFrame)
    gdf = gpd.read_file(boundary_path)
    print(f"Loaded boundary with {len(gdf)} DS divisions.") #number of DS divisions

    # Process Raster Files (.tif)
    # This finds all .tif files in any subfolder of raw/population/
    raster_files = glob.glob(os.path.join(raw_root, "**/*.tif"), recursive=True)

    # Loop through each raster file
    for r_file in raster_files:
        filename = os.path.basename(r_file)
        output_name = filename.replace(".tif", "_Gampaha.tif") # Rename output
        output_path = os.path.join(output_dir, output_name)

        print(f"Clipping Raster: {filename}")
        with rasterio.open(r_file) as src: # Open the raster file
            # Ensure CRS matches (GHS is usually EPSG:54009)
            gdf_proj = gdf.to_crs(src.crs) # Ensure both raster + shapefile use same coordiante system (CRS matching)
            shapes = [feature["geometry"] for feature in gdf_proj.__geo_interface__["features"]] # Extract geometries from the GeoDataFrame to use as masks for clipping

            out_image, out_transform = mask(src, shapes, crop=True) # Clip raster to Gampaha area using the mask function from rasterio
            out_meta = src.meta.copy() # Update metadata
            out_meta.update({  # Update metadata - adjust size + location info
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save clipped raster - writes processed file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

    # Process CSV Files (.csv) for spatial clipping (if they contain lat/lon columns)
    csv_files = glob.glob(os.path.join(raw_root, "**/*.csv"), recursive=True)

    for c_file in csv_files:
        filename = os.path.basename(c_file)
        output_name = filename.replace(".csv", "_Gampaha.csv")
        output_path = os.path.join(output_dir, output_name)

        print(f"Clipping CSV: {filename}")
        df = pd.read_csv(c_file)

        # NOTE: This assumes your CSV has 'lat' and 'lon' columns. (If spatial data exists, it will be clipped to Gampaha. If not, it will just be copied over.)
        if 'lat' in df.columns and 'lon' in df.columns:
            # convert CSV --> Spatial points
            gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
            # Spatial join to keep only points within Gampaha
            clipped_df = gpd.sjoin(gdf_pts, gdf.to_crs("EPSG:4326"), how="inner", predicate="within")
            # Save clipped CSV (drop geometry column since it's not needed in the final CSV)
            clipped_df.drop(columns='geometry').to_csv(output_path, index=False)
        else:
            # If no coordinates, just copy the cleaned file over
            df.to_csv(output_path, index=False)

    print("\n--- Pipeline Complete: All years clipped and saved to data/processed/population/ ---")

if __name__ == "__main__":
    run_preprocessing()


# ETL pipeline for geospatial data: Extract raw data (rasters + csvs) --> Transform (clip to Gampaha boundary) --> Load (save processed files to data/processed/population/)