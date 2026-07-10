import os
import pandas as pd
import numpy as np
import gc
import duckdb

def generate_deduplicated_disaggregated_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_dir = os.path.join(base_dir, "data", "processed", "master")

    input_path = os.path.join(master_dir, "Final_Training_Dataset_Gampaha.csv")
    dedup_path = os.path.join(master_dir, "_dedup_temp_Gampaha.csv")  # intermediate
    output_path = os.path.join(master_dir, "FinalN_Training_Dataset_Gampaha.csv")

    if not os.path.exists(input_path):
        print(f"❌ Error: Source dataset not found at {input_path}")
        return

    group_cols = ['Longitude', 'Latitude', 'Event_Date', 'Ds_Division_Name']

    print("🦆 Running out-of-core deduplication with DuckDB (handles data larger than RAM)...")

    con = duckdb.connect()
    # limit DuckDB's own memory use; it spills the rest to disk automatically
    con.execute("PRAGMA memory_limit='2GB'")

    con.execute(f"""
        COPY (
            SELECT
                Longitude, Latitude, Ds_Division_Name,
                MAX(Ghs_Pop_Baseline)      AS Ghs_Pop_Baseline,
                MAX(Ghs_Built_S_Total)     AS Ghs_Built_S_Total,
                MAX(Ghs_Built_V_Total)     AS Ghs_Built_V_Total,
                MAX(Ghs_Built_S_NonRes)    AS Ghs_Built_S_NonRes,
                MAX(Ghs_Built_V_NonRes)    AS Ghs_Built_V_NonRes,
                MAX(Ghs_Settlement_Type)   AS Ghs_Settlement_Type,
                MAX(Nightlight_Intensity)  AS Nightlight_Intensity,
                MAX(Affected_People)       AS Affected_People,
                MAX(Affected_Families)     AS Affected_Families,
                Event_Date,
                MAX(Precip_Mm)             AS Precip_Mm,
                MAX(Is_Holiday)            AS Is_Holiday,
                MAX(Is_Weekend)            AS Is_Weekend,
                MAX(Severity_Weight)       AS Severity_Weight,
                MAX(Occupancy_Adj)         AS Occupancy_Adj,
                MAX(Built_Up_Ratio)        AS Built_Up_Ratio,
                MAX(Weighted_Pop_Engineered) AS Weighted_Pop_Engineered,
                MAX(Data_Year)             AS Data_Year,
                MAX(Ambient_Pop_Landscan)  AS Ambient_Pop_Landscan
            FROM read_csv_auto('{input_path}')
            GROUP BY {", ".join(group_cols)}
        ) TO '{dedup_path}' (HEADER, DELIMITER ',')
    """)
    con.close()

    print(f"   -> Deduplication complete. Loading result for downstream steps...")

    reload_dtype_map = {
        'Ghs_Pop_Baseline': 'float32', 'Ghs_Built_S_Total': 'float32',
        'Ghs_Built_V_Total': 'float32', 'Ghs_Built_S_NonRes': 'float32',
        'Ghs_Built_V_NonRes': 'float32', 'Ghs_Settlement_Type': 'float32',
        'Nightlight_Intensity': 'float32',
        'Affected_People': 'float32', 'Affected_Families': 'float32',
        'Precip_Mm': 'float32', 'Is_Holiday': 'int8', 'Is_Weekend': 'int8',
        'Severity_Weight': 'float32', 'Occupancy_Adj': 'float32',
        'Built_Up_Ratio': 'float32', 'Weighted_Pop_Engineered': 'float32',
        'Data_Year': 'int16', 'Ambient_Pop_Landscan': 'float32',
        'Ds_Division_Name': 'category',
        # Longitude/Latitude omitted -> stays float64, full precision
    }
    df_dedup = pd.read_csv(dedup_path, dtype=reload_dtype_map)
    print(f"   -> Final matrix shape: {df_dedup.shape}")

    # --- STEP 2: CALL POPULATION DISAGGREGATION UTILITY ---
    print("\n🌊 Applying population disaggregation rules...")
    try:
        from disaggregation_utility import apply_bounded_disaggregation
        df_disagg = apply_bounded_disaggregation(df_dedup)
        print("   -> Success: Bounded population disaggregation applied.")
    except ImportError:
        print("⚠️ Warning: disaggregation_utility.py could not be imported. Using inline fallback...")
        df_disagg = df_dedup.copy()
        is_impacted = (df_disagg['Affected_People'] > 0)
        df_disagg['_flooded_pop_subtotal'] = df_disagg['Ghs_Pop_Baseline'] * is_impacted

        div_pop_totals = df_disagg.groupby(['Ds_Division_Name', 'Data_Year'])['_flooded_pop_subtotal'].transform('sum')
        div_pixel_counts = df_disagg.groupby(['Ds_Division_Name', 'Data_Year'])['_flooded_pop_subtotal'].transform('count')

        df_disagg['_pixel_share'] = np.where(
            is_impacted & (div_pop_totals > 0),
            df_disagg['Ghs_Pop_Baseline'] / div_pop_totals,
            0.0
        )
        df_disagg['_pixel_share'] = np.where(
            is_impacted & (div_pop_totals == 0),
            1.0 / (div_pixel_counts + 1e-6),
            df_disagg['_pixel_share']
        )

        if 'Affected_People' in df_disagg.columns:
            df_disagg['Affected_People'] = df_disagg['Affected_People'] * df_disagg['_pixel_share']
        if 'Affected_Families' in df_disagg.columns:
            df_disagg['Affected_Families'] = df_disagg['Affected_Families'] * df_disagg['_pixel_share']

        df_disagg.drop(columns=['_flooded_pop_subtotal', '_pixel_share'], inplace=True, errors='ignore')

    # --- STEP 3: ENFORCE ZERO POPULATION CAP ---
    print("\n🛑 Enforcing absolute population baseline boundaries...")
    zero_pop_mask = (df_disagg['Ghs_Pop_Baseline'] == 0)
    df_disagg.loc[zero_pop_mask, 'Affected_People'] = 0.0
    if 'Affected_Families' in df_disagg.columns:
        df_disagg.loc[zero_pop_mask, 'Affected_Families'] = 0.0

    print(f"   -> Reset affected values to 0 where Ghs_Pop_Baseline == 0.")

    # --- STEP 4: EXPORT RETAINING COLUMN ORDER ---
    expected_order = [
        'Longitude', 'Latitude', 'Ds_Division_Name', 'Ghs_Pop_Baseline',
        'Ghs_Built_S_Total', 'Ghs_Built_V_Total', 'Ghs_Built_S_NonRes',
        'Ghs_Built_V_NonRes', 'Ghs_Settlement_Type', 'Nightlight_Intensity',
        'Affected_People', 'Affected_Families', 'Event_Date', 'Precip_Mm',
        'Is_Holiday', 'Is_Weekend', 'Severity_Weight', 'Occupancy_Adj',
        'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Data_Year', 'Ambient_Pop_Landscan'
    ]

    final_cols = [col for col in expected_order if col in df_disagg.columns]
    df_final = df_disagg[final_cols]

    print(f"\n💾 Saving cleaned dataset to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print("=" * 60 + "\n🎉 SUCCESS: FinalN_Training_Dataset_Gampaha.csv Generated!\n" + "=" * 60)

if __name__ == "__main__":
    generate_deduplicated_disaggregated_dataset()