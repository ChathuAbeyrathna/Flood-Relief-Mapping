import os
import pandas as pd
import glob

def generate_final_training_set():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    master_dir = os.path.join(base_dir, "data", "processed", "master")
    output_path = os.path.join(master_dir, "Final_Training_Dataset_Gampaha.csv")

    study_years = ["2000", "2005", "2010", "2015", "2020", "2025"]
    all_year_dfs = []

    print("Starting Global Dataset Consolidation...")

    for year in study_years:
        file_path = os.path.join(master_dir, f"Master_Feature_Matrix_{year}.csv")

        if not os.path.exists(file_path):
            print(f"Warning: Missing file for {year}. Skipping.")
            continue

        df = pd.read_csv(file_path)
        print(f"Processing {year}: {len(df)} rows found.")

        # --- STANDARDIZATION LOGIC ---
        # Rename year-specific columns to generic feature names
        rename_map = {}
        for col in df.columns:
            if 'Ghs_Pop' in col:
                rename_map[col] = 'Ghs_Pop_Baseline'
            elif 'Ghs_Built_S_Nres' in col:
                rename_map[col] = 'Ghs_Built_S_NonRes'
            elif 'Ghs_Built_S' in col:
                rename_map[col] = 'Ghs_Built_S_Total'
            elif 'Ghs_Built_V_Nres' in col:
                rename_map[col] = 'Ghs_Built_V_NonRes'
            elif 'Ghs_Built_V' in col:
                rename_map[col] = 'Ghs_Built_V_Total'
            elif 'Ghs_Smod' in col:
                rename_map[col] = 'Ghs_Settlement_Type'
            elif 'Vnl' in col:
                rename_map[col] = 'Nightlight_Intensity'
            elif 'Landscan' in col:
                rename_map[col] = 'Ambient_Pop_Landscan'

        # Apply the renaming operation
        df = df.rename(columns=rename_map)

        # Add a "Data_Year" column
        df['Data_Year'] = int(year)

        # Store processed DataFrame
        all_year_dfs.append(df)

    # --- FINAL CONCATENATION ---
    if all_year_dfs:
        final_master = pd.concat(all_year_dfs, ignore_index=True)

        # Final cleanup
        # and remove the redundant 'Ds_Division' (Ground Truth is usually Affected_People)
        if 'Ds_Division' in final_master.columns:
            final_master = final_master.drop(columns=['Ds_Division'])

        # Save the unified FINAL dataset
        final_master.to_csv(output_path, index=False)

        print("\n" + "="*30)
        print(f"SUCCESS: Final Training Dataset Generated!")
        print(f"Path: {output_path}")
        print(f"Total Observations: {len(final_master)}")
        print(f"Columns: {list(final_master.columns)}")
        print("="*30)

if __name__ == "__main__":
    generate_final_training_set()