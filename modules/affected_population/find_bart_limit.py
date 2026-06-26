import os
import pandas as pd
import numpy as np
from bartpy.sklearnmodel import SklearnModel  # imports the BART model

def find_hardware_limit():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    master_file = os.path.join(base_dir, "data", "processed", "master", "Final_Training_Dataset_Gampaha.csv")

    df = pd.read_csv(master_file)

    features = [
        'Ghs_Pop_Baseline', 'Ghs_Built_S_Total', 'Ghs_Built_V_Total',
        'Ghs_Built_S_NonRes', 'Ghs_Settlement_Type', 'Nightlight_Intensity',
        'Precip_Mm', 'Is_Holiday', 'Is_Weekend', 'Severity_Weight',
        'Occupancy_Adj', 'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Ambient_Pop_Landscan'
    ]

    # Isolate training years pool (Excluding 2025 Hold-out)
    train_pool = df[(df['Data_Year'] != 2025)]
    unique_years = [2000, 2005, 2010, 2015, 2020]

    # Step-by-step test increments (Per-year allocations)
    test_scales = [3000, 6000, 10000, 15000, 20000, 30000]

    # Loop through sizes
    for rows_per_year in test_scales:
        total_rows = rows_per_year * len(unique_years)
        print(f"\n[TESTING SCALE] Pulling {rows_per_year} rows per year. Total Matrix Size = {total_rows} rows...")

        try:
            # Execute stratified sample collection
            sampled_dfs = []
            for year in unique_years: # year loop
                year_subset = train_pool[train_pool['Data_Year'] == year]  # extract one year
                sampled_dfs.append(year_subset.sample(n=rows_per_year, random_state=42)) # random sampling following rows_per_year

            test_df = pd.concat(sampled_dfs, ignore_index=True) # combines all data rows into one dataset.
            X_test = test_df[features] # input matrix (X matrix or predictor variables)
            y_test = test_df['Affected_People']  # y Vector or target variable

            # Test allocation boundaries on the Mean Ensemble
            print(f" -> Attempting MCMC root node initialization for {total_rows} rows...")
            model = SklearnModel(n_trees=50, n_samples=10, n_burn=5, n_jobs=1)
            model.fit(X_test, y_test) # use to test hardware limits

            print(f" SUCCESS: Your hardware safely supports a total of {total_rows} training rows!")

        except Exception as e:
            print("\n" + "!"*60)
            print(f"HARDWARE LIMIT REACHED AT {total_rows} ROWS ({rows_per_year} per year)!")
            print(f"Error Details: {type(e).__name__} - {e}")
            print("!"*60)
            break  # If the current size fails, there is no point trying even larger sizes, so the loop ends.

if __name__ == "__main__":   # This ensures the function runs only when this file is executed directly.
    find_hardware_limit()