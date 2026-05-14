import pandas as pd
import geopandas as gpd
from shapely.geometry import Point # For creating geometry points from lat/lon in the CSV
import os

def generate_coordinate_lookup():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))

    # 1. Load your Gampaha DS Boundary Shapefile
    # Ensure this shapefile has a column like 'DIVISION_NAME' or 'DS_NAME'
    shapefile_path = os.path.join(base_dir, "data", "processed", "population", "Gampaha_DS_Boundaries", "Gampaha_DS_Boundaries.shp")
    gdf_divisions = gpd.read_file(shapefile_path)

    # Ensure the shapefile is in the same coordinate system as your CSVs (WGS84)
    if gdf_divisions.crs != "EPSG:4326":
        gdf_divisions = gdf_divisions.to_crs("EPSG:4326")

    # 2. Load one of your 100m CSV files to get the "Master Grid"
    # We use GHS_POP because it contains all 13,000+ valid coordinates
    sample_csv = os.path.join(base_dir, "data", "processed", "population", "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_Gampaha.csv")
    df_grid = pd.read_csv(sample_csv)

    # 3. Convert CSV coordinates into Geometry Points
    geometry = [Point(xy) for xy in zip(df_grid['longitude'], df_grid['latitude'])]
    gdf_points = gpd.GeoDataFrame(df_grid, geometry=geometry, crs="EPSG:4326") #

    # 4. Point-in-Polygon Join
    # This "tags" every point with the attributes of the division it sits inside
    print("Performing Spatial Join (Tagging pixels with DS Division names)...")
    joined = gpd.sjoin(gdf_points, gdf_divisions, how="left", predicate="within")

    # 5. Clean and Save the Lookup Table
    # Keep only the essential columns: Lon, Lat, and the Division Name
    # (Replace 'DS_Name' with the actual column name in your shapefile)
    lookup_table = joined[['longitude', 'latitude', 'adm3_name']].rename(columns={'adm3_name': 'DS_Division_Name'})

    # --- DUPLICATE CHECK START ---
    duplicate_count = lookup_table.duplicated(subset=['longitude', 'latitude']).sum()

    if duplicate_count > 0:
        print(f"WARNING: Found {duplicate_count} duplicate coordinates.")
        # Optional: Remove duplicates and keep the first one found
        lookup_table = lookup_table.drop_duplicates(subset=['longitude', 'latitude'], keep='first')
        print("Duplicates have been removed (kept the first occurrence).")
    else:
        print("Validation Passed: No duplicate coordinates found.")
    # --- DUPLICATE CHECK END ---

    output_path = os.path.join(base_dir, "data", "processed", "master", "Gampaha_Coordinate_Lookup.csv")
    lookup_table.to_csv(output_path, index=False)

    print(f"Success! Lookup table created with {len(lookup_table)} points.")

if __name__ == "__main__":
    generate_coordinate_lookup()