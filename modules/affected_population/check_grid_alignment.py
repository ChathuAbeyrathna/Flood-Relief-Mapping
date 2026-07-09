# =============================================================================
# DIAGNOSTIC + FIX for Step 2 (grid alignment between resampled flood cells
# and the master reference CSV). Two parts:
#
#   PART A - Diagnostic: run this once, standalone, to confirm the grid
#            spacing/origin mismatch theory before changing production logic.
#
#   PART B - Drop-in replacement for the "Step 2/3" coordinate-matching block
#            (df_flooded_pixels -> chunked merge -> df_work) using a KD-tree
#            nearest-neighbor snap instead of exact rounded-coordinate merge.
#            Everything before it (Step 1 resampling) and after it
#            (Step 4 ML prediction onward) is untouched.
# =============================================================================


# -----------------------------------------------------------------------
# PART A - DIAGNOSTIC (run separately, e.g. in a notebook or a quick script)
# -----------------------------------------------------------------------

import pandas as pd
import numpy as np

master_file = "G:\A_FYP_PROJECT\code\data\processed\master\Final_Training_Dataset_Gampaha.csv"

# Only pull the two coordinate columns, in chunks, to avoid loading 11M rows
# with every feature column.
lon_vals = set()
lat_vals = set()
zero_pop_count = 0
total_rows = 0

for chunk in pd.read_csv(master_file, usecols=["Longitude", "Latitude", "Ghs_Pop_Baseline"],
                          chunksize=1_000_000):
    lon_vals.update(np.round(chunk["Longitude"].unique(), 6))
    lat_vals.update(np.round(chunk["Latitude"].unique(), 6))
    zero_pop_count += (chunk["Ghs_Pop_Baseline"] == 0).sum()
    total_rows += len(chunk)

lon_sorted = np.sort(list(lon_vals))
lat_sorted = np.sort(list(lat_vals))

lon_diffs = np.diff(lon_sorted)
lat_diffs = np.diff(lat_sorted)

print(f"Total rows scanned: {total_rows:,}")
print(f"Rows with Ghs_Pop_Baseline == 0: {zero_pop_count:,} "
      f"({100*zero_pop_count/total_rows:.1f}%)")
print()
print(f"Master grid longitude spacing (should be ~constant if it's a regular grid):")
print(f"  min={lon_diffs.min():.6f}  median={np.median(lon_diffs):.6f}  max={lon_diffs.max():.6f}")
print(f"Master grid latitude spacing:")
print(f"  min={lat_diffs.min():.6f}  median={np.median(lat_diffs):.6f}  max={lat_diffs.max():.6f}")
print()
print(f"Master grid origin (min lon, min lat): ({lon_sorted[0]:.6f}, {lat_sorted[0]:.6f})")

# For reference: at ~7 deg N latitude, 100m corresponds to roughly:
#   longitude: 100 / (111320 * cos(radians(7))) ~= 0.000905 deg
#   latitude:  100 / 110574                      ~= 0.000905 deg
# Compare the printed median spacings above against ~0.0009 to confirm
# the master CSV really is on a 100m grid, and compare the origin against
# your resampled raster's transform origin to check for an offset.