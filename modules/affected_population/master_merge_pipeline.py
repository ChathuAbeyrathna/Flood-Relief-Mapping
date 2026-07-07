import os
import glob
import pandas as pd
import numpy as np

def standardize_names(df, column_name):
    """Standardizes DS Division names based on the user's mapping."""
    mapping = {
        'Meerigame': 'Mirigama',
        'Ja-Ela': 'Ja Ela',
        'Attanagalle': 'Attanagalla',
        'Biyagame': 'Biyagama'
    }
    # Clean column headers (strip spaces, Title Case)
    df.columns = [c.strip().title() for c in df.columns]

    # Standardize the actual division names
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.strip().str.title()
        df[column_name] = df[column_name].replace(mapping)
    return df

def run_dynamic_master_merge():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    processed_dir = os.path.join(base_dir, "data", "processed", "population")
    lookup_path = os.path.join(base_dir, "data", "processed", "master", "Gampaha_Coordinate_Lookup.csv")
    output_dir = os.path.join(base_dir, "data", "processed", "master")
    os.makedirs(output_dir, exist_ok=True)

    # Load lookup once as the spatial backbone
    lookup_df = standardize_names(pd.read_csv(lookup_path), 'DS_Division_Name')
    study_years = ["2000", "2005", "2010", "2015", "2020", "2025"]

    for year in study_years:
        print(f"\n--- DYNAMIC PROCESSING YEAR: {year} ---")

        # 1. LOAD YEAR-SPECIFIC DATA
        try:
            rain_df = standardize_names(pd.read_csv(os.path.join(processed_dir, f"{year}_rainfall_Gampaha.csv")), 'None')
            aff_df_raw = standardize_names(pd.read_csv(os.path.join(processed_dir, f"affected_population_{year}.csv")), 'Ds_Division')
            cal_df = standardize_names(pd.read_csv(os.path.join(processed_dir, f"sl_calendar_{year}_Gampaha.csv")), 'None')
        except FileNotFoundError as e:
            print(f"Skipping {year} - File not found: {e}")
            continue

        # 2. IDENTIFY ALL UNIQUE FLOOD DATES
        # We only consider dates where the Disaster is "Flood"
        flood_dates = aff_df_raw[aff_df_raw['Disaster'].str.contains('Flood', case=False, na=False)]['Date'].unique()
        if len(flood_dates) == 0:
            print(f"No flood events found in the dataset for {year}.")
            continue
        print(f"Found {len(flood_dates)} flood event dates: {flood_dates}")

        # 3. SPATIAL BACKBONE LOAD (Yearly static features)
        spatial_patterns = ["GHS_POP", "GHS_BUILT_S", "GHS_BUILT_V", "GHS_BUILT_S_NRES", "GHS_BUILT_V_NRES", "GHS_SMOD", "landscan"]
        spatial_base = lookup_df.copy() # Start with the coordinate lookup as the base, which has Lon, Lat, and DS Division Name

        for pattern in spatial_patterns:
            files = glob.glob(os.path.join(processed_dir, f"*{pattern}*{year}*.csv"))
            if files:
                temp_df = standardize_names(pd.read_csv(files[0]), 'None') # Load the spatial feature CSV and standardize names
                spatial_base = pd.merge(spatial_base, temp_df, on=['Longitude', 'Latitude'], how='left') # Merge spatial features into the base using Lon/Lat as the key, ensuring we keep all coordinates from the lookup and add spatial attributes where available

        # 4. NIGHTTIME LIGHTS PROXIES
        vnl_year = "2020" if int(year) < 2024 else "2024"
        vnl_files = glob.glob(os.path.join(processed_dir, f"*VNL*{vnl_year}*.csv"))
        if vnl_files:
            vnl_df = standardize_names(pd.read_csv(vnl_files[0]), 'None')
            spatial_base = pd.merge(spatial_base, vnl_df, on=['Longitude', 'Latitude'], how='left')

        # 5. ITERATE THROUGH EVERY FLOOD DATE FOUND
        yearly_event_collection = [] # We will collect the event-specific DataFrames in this list and concatenate at the end of the year.

        for f_date in flood_dates:
            print(f"Processing Event: {f_date}") # For each flood date, we will create a DataFrame that combines the spatial features with the affected population data, rainfall, and calendar information specific to that date. This allows us to capture the dynamic nature of each flood event while maintaining the spatial context provided by the coordinate lookup and spatial features.

            # Filter tabular data for THIS specific date
            day_aff = aff_df_raw[aff_df_raw['Date'] == f_date].groupby('Ds_Division')[['Affected_People', 'Affected_Families']].max().reset_index() # We take the max to ensure we capture the highest reported impact for that division on that date, in case there are multiple entries.
            day_rain = rain_df[rain_df['Datetime'] == f_date] # Standardized to Datetime
            day_cal = cal_df[cal_df['Date'] == f_date]

            # Merge spatial with division-level affected data
            event_df = pd.merge(spatial_base, day_aff, left_on='Ds_Division_Name', right_on='Ds_Division', how='left') # This merge will add the 'Affected_People' and 'Affected_Families' columns to our spatial base, aligned by the DS Division Name. We use a left merge to keep all spatial points.

            # Broadcast Date, Rain, and Calendar
            event_df['Event_Date'] = f_date
            event_df['Precip_Mm'] = day_rain['Precip'].values[0] if not day_rain.empty else 0
            event_df['Is_Holiday'] = day_cal['Is_Public_Holiday'].values[0] if not day_cal.empty else 0
            event_df['Is_Weekend'] = day_cal['Is_Weekend'].values[0] if not day_cal.empty else 0

            yearly_event_collection.append(event_df) # We add this event-specific DataFrame to our yearly collection. Each event_df contains the spatial features, the affected population data for that specific flood date, and the relevant rainfall and calendar information.

        # 6. VALIDATION & DUPLICATE CHECK
        if yearly_event_collection:
            final_year_master = pd.concat(yearly_event_collection, ignore_index=True)

            print("Before duplicate removal:", final_year_master.shape)

            dupes = final_year_master.duplicated(subset=['Longitude', 'Latitude', 'Ds_Division_Name', 'Event_Date']).sum()
            if dupes > 0:
                print(f"Removing {dupes} duplicates.")
                final_year_master = final_year_master.drop_duplicates(subset=['Longitude', 'Latitude', 'Ds_Division_Name', 'Event_Date'])

            # Check for unmerged divisions (no victims reported)
            unmerged = final_year_master[final_year_master['Affected_People'].isna()]['Ds_Division_Name'].unique()
            if len(unmerged) > 0:
                missing_dates = final_year_master[
                    final_year_master['Affected_People'].isna()
                ]['Event_Date'].unique()

                print(
                    f"Info: {len(unmerged)} divisions had no reported victims on dates: {missing_dates}"
                )

            print("After duplicate removal:", final_year_master.shape)

            print(f"Applying DMC-aligned Heuristic for {year}...")

            # Identify columns dynamically by year
            try:
                pop_col = [c for c in final_year_master.columns if 'Ghs_Pop' in c][0]
                built_s_col = [c for c in final_year_master.columns if 'Ghs_Built_S' in c and 'Nres' not in c][0]

                print(f"Using columns: {pop_col} and {built_s_col}")
            except IndexError:
                print(f"ERROR: Could not find Ghs_Pop or Ghs_Built_S columns for {year}")
                continue

            # Heuristic Part A: Severity Weight (DMC Refined)
            def get_sev_weight(mm):
                if mm <= 50: return 0.3      # Moderate and below
                elif mm <= 100: return 0.6   # Fairly heavy
                else: return 1.0             # Heavy / Very Heavy

            # Heuristic Part B: Occupancy (LandScan Diurnal Logic)
            def get_occ_adj(row):
                return 1.2 if (row['Is_Holiday'] == 1 or row['Is_Weekend'] == 1) else 0.85

            final_year_master['Severity_Weight'] = final_year_master['Precip_Mm'].apply(get_sev_weight)
            final_year_master['Occupancy_Adj'] = final_year_master.apply(get_occ_adj, axis=1)
            final_year_master['Built_Up_Ratio'] = final_year_master[built_s_col] / 10000

            # The Main Heuristic Equation
            # Weighted Pop = min(max(Population * Ratio * Sev * Occ, 0), Population)
            weighted_calc = (
                    final_year_master[pop_col] * final_year_master['Built_Up_Ratio'] * final_year_master['Severity_Weight'] * final_year_master['Occupancy_Adj']
            )

            # Constraint: min(max(A, 0), GHS_POP)
            final_year_master['Weighted_Pop_Engineered'] = np.minimum(np.maximum(weighted_calc, 0), final_year_master[pop_col])

            # 7. FILLING NaNs (Important for BART)
            # CRITICAL: Fill NaNs with 0 for BART compatibility
            numeric_cols = final_year_master.select_dtypes(include=[np.number]).columns # Identify numeric columns dynamically
            final_year_master[numeric_cols] = final_year_master[numeric_cols].fillna(0)

            output_file = os.path.join(output_dir, f"Master_Feature_Matrix_{year}.csv")
            final_year_master.to_csv(output_file, index=False)
            print(f"SUCCESS: {year} Master saved with {len(final_year_master)} total observations.")

if __name__ == "__main__":
    run_dynamic_master_merge()