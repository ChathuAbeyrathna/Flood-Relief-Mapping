import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

class ReliefDataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

        # Scalers
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()

        # Encoding
        self.severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

        # Features (final model inputs)
        self.feature_columns = [
            'Affected_Population',
            'Children_%',
            'Elderly_%',
            'Severity_Code'
        ]

        # Targets (relief items)
        self.target_columns = [
            'Cooked Food Packs', 'Water Bottles', 'Milk Powder Packs',
            'Infant Milk Powder Packs', 'Biscuits Packs', 'Noodles Packs',
            'Tea Powder Packets', 'Sanitary', 'Soap', 'Toothpaste', 'Toothbrushes'
        ]

    # ---------------- LOAD DATA ----------------
    def load_data(self):
        raw = pd.read_excel(self.file_path, sheet_name='Sheet1', header=None)

        rows = []
        headers_found = False
        headers = []
        current_district = None

        for _, row in raw.iterrows():

            # Detect district
            if pd.notna(row.iloc[0]):
                val = str(row.iloc[0]).strip()
                if val in ['Gampaha', 'Colombo']:
                    current_district = val
                    continue

            # Detect header row
            if not headers_found and pd.notna(row.iloc[0]):
                if 'Year' in str(row.iloc[0]):
                    headers = [str(c).strip() if pd.notna(c) else '' for c in row]
                    headers_found = True
                    continue

            # Extract data
            if headers_found and pd.notna(row.iloc[0]):
                if str(row.iloc[0]).isdigit():
                    record = {'District': current_district}

                    for i, col in enumerate(headers):
                        if col:
                            record[col] = row.iloc[i]

                    rows.append(record)

        self.data = pd.DataFrame(rows)

        # Clean column names
        self.data.columns = [
            c.replace(' ', '_').replace('%', 'Pct') for c in self.data.columns
        ]

        # Rename columns
        rename_map = {
            'Children_Pct': 'Children_%',
            'Elderly_Pct': 'Elderly_%',
            'Cooked_Food_Packs': 'Cooked Food Packs',
            'Water_Bottles': 'Water Bottles',
            'Milk_Powder_Packs': 'Milk Powder Packs',
            'Infant_Milk_Powder_Packs': 'Infant Milk Powder Packs',
            'Biscuits_Packs': 'Biscuits Packs',
            'Noodles_Packs': 'Noodles Packs',
            'Tea_Powder_Packets': 'Tea Powder Packets',
        }

        self.data.rename(columns=rename_map, inplace=True)

        # Convert numeric
        numeric_cols = ['Year', 'Affected_Population', 'Children_%', 'Elderly_%'] + self.target_columns

        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Drop missing essentials
        self.data.dropna(subset=['Year', 'DS_Division', 'Affected_Population'], inplace=True)

        print(f"\nLoaded dataset shape: {self.data.shape}")
        return self.data

    # ---------------- CLEAN ----------------
    def clean_data(self):
        targets = [c for c in self.target_columns if c in self.data.columns]
        self.data.dropna(subset=targets, inplace=True)

        # Convert percentages
        for col in ['Children_%', 'Elderly_%']:
            if col in self.data.columns:
                self.data[col] = self.data[col] / 100

        return self.data

    # ---------------- ENCODE ----------------
    def encode_severity(self):
        self.data['Severity_Code'] = self.data['Severity'].map(self.severity_map)
        self.data.dropna(subset=['Severity_Code'], inplace=True)
        return self.data

    # ---------------- TIME-BASED SPLIT (RETURNS RAW DATA) ----------------
    def time_split(self, test_year=2025):
        """Returns RAW (unscaled) data"""
        train_data = self.data[self.data['Year'] < test_year]
        test_data = self.data[self.data['Year'] == test_year]

        # Clean year display
        train_years = sorted(train_data['Year'].astype(int).unique())
        test_years = sorted(test_data['Year'].astype(int).unique())

        print("\nTime-Based Split:")
        print("Training years:", train_years)
        print("Testing years :", test_years)

        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_columns]

        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_columns]

        return X_train, X_test, y_train, y_test

    # ---------------- SCALING (OPTIONAL - FOR DIRECT ML USE) ----------------
    def scale_data(self, X_train, X_test, y_train, y_test):
        """Scale data for direct ML use (without internal scaling)"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        y_train_scaled = self.normalizer.fit_transform(y_train)
        y_test_scaled = self.normalizer.transform(y_test)

        y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
        y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    # ---------------- GET DIVISION LIST ----------------
    def get_division_list(self):
        """Return list of all DS Divisions with their districts"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        divisions = []
        for _, row in self.data[['District', 'DS_Division']].drop_duplicates().iterrows():
            divisions.append({
                'name': row['DS_Division'],
                'district': row['District']
            })
        return divisions

    # ---------------- GET DIVISIONS BY DISTRICT ----------------
    def get_divisions_by_district(self, district):
        """Get divisions for a specific district"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        divisions = self.data[self.data['District'] == district]['DS_Division'].unique().tolist()
        return divisions

    # ---------------- GET STATISTICS BY DIVISION ----------------
    def get_statistics_by_division(self, division_name):
        """Get comprehensive statistics for a DS Division"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        division_data = self.data[self.data['DS_Division'] == division_name]
        if len(division_data) == 0:
            return None
        
        stats = {
            'district': division_data['District'].iloc[0] if 'District' in division_data.columns else 'Unknown',
            'years_of_data': division_data['Year'].tolist(),
            'sample_count': len(division_data),
            'avg_affected_population': float(division_data['Affected_Population'].mean()),
            'avg_children_pct': float(division_data['Children_%'].mean() * 100),
            'avg_elderly_pct': float(division_data['Elderly_%'].mean() * 100)
        }
        
        # Add average relief items
        for col in self.target_columns:
            if col in division_data.columns:
                stats[f'avg_{col.replace(" ", "_")}'] = float(round(division_data[col].mean(), 2))
        
        return stats

    # ---------------- FULL PIPELINE ----------------
    def run_pipeline(self, test_year=2025, scale=False):
        """Run preprocessing pipeline"""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)

        self.load_data()
        self.clean_data()
        self.encode_severity()

        X_train, X_test, y_train, y_test = self.time_split(test_year)

        if scale:
            X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test)
            print("\nData scaled for direct ML use")
        else:
            print("\nReturning RAW data (predictor will handle scaling internally)")
        
        print(f"\nDataset Split Summary:")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples : {len(X_test)}")
        print(f"Total samples: {len(self.data)}")

        print("\n" + "=" * 60 + "\n")

        return X_train, X_test, y_train, y_test, self.data


# ---------------- RUN ----------------
if __name__ == "__main__":

    file_path = "Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx"

    processor = ReliefDataPreprocessor(file_path)

    # Get RAW data (scale=False) for ReliefPredictor
    X_train, X_test, y_train, y_test, data = processor.run_pipeline(test_year=2025, scale=False)

    print("Final Shapes (RAW Data):")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)
    
    # Show sample raw values
    print("\nSample RAW values:")
    print("Affected_Population sample:", X_train['Affected_Population'].head(3).tolist())
    print("Cooked Food Packs sample:", y_train['Cooked Food Packs'].head(3).tolist())