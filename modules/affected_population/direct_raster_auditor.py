# direct_raster_auditor.py  (CORRECTED VERSION)
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform as warp_transform
from rasterio.enums import Resampling
from scipy.spatial import cKDTree

current_module_dir = os.path.dirname(os.path.abspath(__file__))
if current_module_dir not in sys.path:
    sys.path.insert(0, current_module_dir)


def run_direct_raster_audit():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    flood_tiff_path = os.path.join(base_dir, "interface", "backend", "outputs", "flood_extent_4326.tif")
    layer5_file = os.path.join(base_dir, "data", "processed", "master", "layer5N_2025_spatial_predictions.csv")

    print("=" * 85)
    print("🎯 RAW MODULE 1 RASTER VS LAYER 5 SPATIAL INTERSECT AUDITOR (CORRECTED)")
    print("=" * 85)

    if not os.path.exists(flood_tiff_path):
        print(f"❌ Error: Module 1 output raster missing at: {flood_tiff_path}")
        return
    if not os.path.exists(layer5_file):
        print(f"❌ Error: Layer 5 spatial prediction log missing at: {layer5_file}")
        return

    # -------------------------------------------------------------------------
    # STEP 1: READ AND RESAMPLE THE RAW RASTER EXACTLY AS THE LIVE ENDPOINT DOES
    # -------------------------------------------------------------------------
    print("📖 Step 1: Processing raw Module 1 TIFF footprint...")
    TARGET_RESOLUTION_M = 90
    PROJECTED_CRS = "EPSG:32644"  # UTM Zone 44N for Sri Lanka

    with rasterio.open(flood_tiff_path) as src:
        raw_data = src.read(1)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, PROJECTED_CRS, src.width, src.height, *src.bounds,
            resolution=(TARGET_RESOLUTION_M, TARGET_RESOLUTION_M)
        )

        flood_band = np.zeros((dst_height, dst_width), dtype=raw_data.dtype)

        reproject(
            source=raw_data, destination=flood_band,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=PROJECTED_CRS,
            resampling=Resampling.max
        )

        rows, cols = np.where(flood_band == 255)
        utm_x, utm_y = rasterio.transform.xy(dst_transform, rows, cols)

        if len(utm_x) > 0:
            raster_lons, raster_lats = warp_transform(PROJECTED_CRS, "EPSG:4326", utm_x, utm_y)
        else:
            raster_lons, raster_lats = [], []

    raster_lons = np.array(raster_lons)
    raster_lats = np.array(raster_lats)
    total_raster_flooded_cells = len(raster_lons)
    print(f"   -> Found {total_raster_flooded_cells:,} flooded grid cells at 90m resolution inside the TIFF.")

    if total_raster_flooded_cells == 0:
        print("⚠️ No flooded cells found in raster. Stopping audit.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: LOAD LAYER 5 OUTPUT (the reference population grid)
    # -------------------------------------------------------------------------
    print("\n📖 Step 2: Loading Layer 5 spatial prediction matrix...")
    df_l5 = pd.read_csv(layer5_file)
    print(f"   -> Layer 5 file contains {len(df_l5):,} total grid rows (district-wide, NOT flood-filtered yet).")

    required_cols = {'Longitude', 'Latitude', 'Ghs_Pop_Baseline'}
    missing = required_cols - set(df_l5.columns)
    if missing:
        print(f"❌ Error: Layer 5 file is missing required columns: {missing}")
        return

    # -------------------------------------------------------------------------
    # STEP 3: REAL SPATIAL JOIN — snap each flooded raster cell to its nearest
    # Layer 5 grid point, with a distance cutoff so we don't silently match
    # cells that are actually far away (same method used in the live endpoint).
    # -------------------------------------------------------------------------
    print("\n🔗 Step 3: Performing nearest-neighbor spatial join (raster -> Layer 5 grid)...")

    # Tolerance based on the master grid's real ~90m spacing confirmed earlier
    # (~0.000811 deg lat). Half-cell-diagonal tolerance ~0.00058 deg (~65m).
    MAX_SNAP_DEGREES = 0.00058

    l5_tree = cKDTree(df_l5[['Longitude', 'Latitude']].values)
    flood_coords = np.column_stack([raster_lons, raster_lats])

    distances, nearest_idx = l5_tree.query(flood_coords, k=1)
    within_range = distances <= MAX_SNAP_DEGREES

    matched_rows = df_l5.iloc[nearest_idx[within_range]].copy()
    matched_rows['_snap_distance_deg'] = distances[within_range]

    # De-duplicate: if multiple raster cells snapped to the same Layer 5 row,
    # keep only the closest match so population isn't double-counted.
    matched_rows = (
        matched_rows
        .sort_values('_snap_distance_deg')
        .drop_duplicates(subset=['Longitude', 'Latitude'], keep='first')
        .drop(columns=['_snap_distance_deg'])
    )

    num_flood_cells_matched = within_range.sum()
    num_flood_cells_unmatched = total_raster_flooded_cells - num_flood_cells_matched

    print(f"   -> {num_flood_cells_matched:,} / {total_raster_flooded_cells:,} flooded raster cells matched "
          f"to a Layer 5 grid point within {MAX_SNAP_DEGREES} deg (~65m).")
    if num_flood_cells_unmatched > 0:
        print(f"   -> {num_flood_cells_unmatched:,} flooded cells had NO nearby Layer 5 point "
              f"(outside district grid coverage or grid misalignment).")
    print(f"   -> After de-duplication: {len(matched_rows):,} unique Layer 5 rows represent the flood footprint.")

    # -------------------------------------------------------------------------
    # STEP 4: REPORT — stats computed ONLY on the actually-matched flood footprint
    # -------------------------------------------------------------------------
    zero_pop_cells = matched_rows[matched_rows['Ghs_Pop_Baseline'] == 0]
    populated_cells = matched_rows[matched_rows['Ghs_Pop_Baseline'] > 0]

    num_zero = len(zero_pop_cells)
    num_populated = len(populated_cells)
    zero_percentage = (num_zero / len(matched_rows)) * 100 if len(matched_rows) > 0 else 0

    baseline_pop_sum = matched_rows['Ghs_Pop_Baseline'].sum()
    predicted_sum = (
        matched_rows['Predicted_Mean_Affected'].sum()
        if 'Predicted_Mean_Affected' in matched_rows.columns else None
    )

    print("\n" + "=" * 85)
    print("📊 FINAL SPATIAL VERIFICATION REPORT (based on ACTUAL flood-footprint intersection)")
    print("=" * 85)
    print(f"🔹 Total 90m Flooded Footprint Cells (From TIFF)         : {total_raster_flooded_cells:,} cells")
    print(f"🔹 Flooded Cells Successfully Matched to Layer 5 Grid    : {len(matched_rows):,} cells")
    print(f"🔹 Empty Flooded Footprint Cells (Zero Population)       : {num_zero:,} cells ({zero_percentage:.2f}%)")
    print(f"🔹 Populated Flooded Footprint Cells                     : {num_populated:,} cells ({(100 - zero_percentage):.2f}%)")
    print(f"🔹 Total Baseline Population INSIDE Flood Footprint Only : {baseline_pop_sum:,.0f} people")
    if predicted_sum is not None:
        print(f"🔹 Total Model Predicted Affected Population              : {predicted_sum:,.0f} people")
        if baseline_pop_sum > 0:
            print(f"🔹 Predicted / Baseline Ratio                             : {100*predicted_sum/baseline_pop_sum:.2f}%")

    print("-" * 85)
    print("💡 HOW TO READ THIS FOR VIVA:")
    print(f"   Of the {total_raster_flooded_cells:,} cells Module 1 flagged as flooded, {num_zero:,} "
          f"({zero_percentage:.1f}%) contain")
    print("   zero recorded baseline population in the GHS-POP layer — i.e. water bodies, paddy fields,")
    print("   roads, or other unpopulated land. This is a factual count from the intersection above,")
    print("   not an assumption. Report the two percentages and the baseline/predicted totals directly;")
    print("   avoid asserting the model's output is 'correct' — state what the data shows and let the")
    print("   panel judge whether the ratio is plausible for Gampaha's known land use.")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # OPTIONAL: sample of matched populated cells, for spot-checking in viva
    # -------------------------------------------------------------------------
    if num_populated > 0:
        print("\n📍 Sample of matched FLOODED + POPULATED cells (first 10):")
        cols_to_show = [c for c in ['Ds_Division_Name', 'Longitude', 'Latitude',
                                    'Ghs_Pop_Baseline', 'Predicted_Mean_Affected']
                        if c in matched_rows.columns]
        print(populated_cells[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    run_direct_raster_audit()