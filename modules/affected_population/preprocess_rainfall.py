import pandas as pd
import os

def categorize_rain(mm):
    #Applies DMC standard rainfall categories based on mm values
    if mm == 0:
        return "No Rain"
    elif 0 < mm <= 12.5:
        return "Light"
    elif 12.5 < mm <= 25:
        return "Light to moderate"
    elif 25 < mm <= 50:
        return "Moderate"
    elif 50 < mm <= 100:
        return "Fairly heavy rain"
    elif 100 < mm <= 150:
        return "Heavy rain"
    else:
        return "Very heavy rain"

def process_rainfall():
    # Define Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, "data", "raw", "population")
    output_dir = os.path.join(base_dir, "data", "processed", "population")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all CSV files related to rainfall
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory {raw_dir} not found.")
        return

    for file_name in os.listdir(raw_dir):
        # Load the data
        file_path = os.path.join(raw_dir, file_name)

        if os.path.isfile(file_path) and file_name.endswith('_rainfall.csv'):
            print(f"Processing rainfall file: {file_name}...")

            df = pd.read_csv(file_path)

            # Data Cleaning
            # Remove unwanted columns if they exist
            cols_to_remove = ['name', 'severerisk']
            df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

            # Add 'rain_type' column & apply the DMC categorization
            # We use 'precip' for the calculation
            df['rain_type'] = df['precip'].apply(categorize_rain)

            # Save to processed folder
            # Appending '_Gampaha' to distinguish from raw files
            output_name = file_name.replace("_rainfall.csv", "_rainfall_Gampaha.csv")
            output_path = os.path.join(output_dir, output_name)

            df.to_csv(output_path, index=False)
            print(f"Successfully saved to: {output_path}")

    print("\n--- Rainfall Processing Complete! ---")

if __name__ == "__main__":
    process_rainfall()