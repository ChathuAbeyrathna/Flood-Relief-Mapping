import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class ReliefDataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        self.severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

        self.feature_columns = [
            'Affected_Population',
            'Children_%',
            'Elderly_%',
            'Female %',
            'Severity_Code'
        ]

        self.target_columns = [
            'Cooked Food Packs', 'Water Bottles', 'Milk Powder Packs',
            'Infant Milk Powder Packs', 'Biscuits Packs', 'Noodles Packs',
            'Tea Powder Packets', 'Sanitary', 'Soap', 'Toothpaste', 'Toothbrushes'
        ]

    def load_data(self):
        raw = pd.read_excel(self.file_path, sheet_name='Sheet1', header=None)

        rows = []
        headers_found = False
        headers = []
        current_district = None

        for _, row in raw.iterrows():
            if pd.notna(row.iloc[0]):
                val = str(row.iloc[0]).strip()
                if val in ['Gampaha', 'Colombo']:
                    current_district = val
                    continue

            if not headers_found and pd.notna(row.iloc[0]):
                if 'Year' in str(row.iloc[0]):
                    headers = [str(c).strip() if pd.notna(c) else '' for c in row]
                    headers_found = True
                    continue

            if headers_found and pd.notna(row.iloc[0]):
                if str(row.iloc[0]).isdigit():
                    record = {'District': current_district}
                    for i, col in enumerate(headers):
                        if col:
                            record[col] = row.iloc[i]
                    rows.append(record)

        self.data = pd.DataFrame(rows)
        self.data.columns = [c.replace(' ', '_').replace('%', 'Pct') for c in self.data.columns]

        rename_map = {
            'Children_Pct': 'Children_%', 
            'Elderly_Pct': 'Elderly_%',
            'Female_Pct': 'Female %',
            'Cooked_Food_Packs': 'Cooked Food Packs', 
            'Water_Bottles': 'Water Bottles',
            'Milk_Powder_Packs': 'Milk Powder Packs', 
            'Infant_Milk_Powder_Packs': 'Infant Milk Powder Packs',
            'Biscuits_Packs': 'Biscuits Packs', 
            'Noodles_Packs': 'Noodles Packs',
            'Tea_Powder_Packets': 'Tea Powder Packets',
        }
        self.data.rename(columns=rename_map, inplace=True)

        numeric_cols = ['Year', 'Affected_Population', 'Children_%', 'Elderly_%', 'Female %'] + self.target_columns
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.data.dropna(subset=['Year', 'DS_Division', 'Affected_Population'], inplace=True)
        print(f"Loaded: {self.data.shape}")
        return self.data

    def clean_data(self):
        targets = [c for c in self.target_columns if c in self.data.columns]
        self.data.dropna(subset=targets, inplace=True)
        
        for col in ['Children_%', 'Elderly_%', 'Female %']:
            if col in self.data.columns:
                self.data[col] = self.data[col] / 100
        
        if 'Female %' in self.data.columns:
            missing_count = self.data['Female %'].isna().sum()
            if missing_count > 0:
                avg_female = self.data['Female %'].mean()
                self.data['Female %'] = self.data['Female %'].fillna(avg_female)
                print(f"✅ Filled {missing_count} missing Female % values")
        
        return self.data

    def encode_severity(self):
        self.data['Severity_Code'] = self.data['Severity'].map(self.severity_map)
        self.data.dropna(subset=['Severity_Code'], inplace=True)
        return self.data

    def time_split(self, test_year=2025):
        train_data = self.data[self.data['Year'] < test_year]
        test_data = self.data[self.data['Year'] == test_year]

        print(f"Training years: {train_data['Year'].unique().tolist()}")
        print(f"Testing years: {test_data['Year'].unique().tolist()}")

        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_columns]
        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_columns]

        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test, y_train, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        y_train_scaled = self.normalizer.fit_transform(y_train)
        y_test_scaled = self.normalizer.transform(y_test)
        y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
        y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def run_pipeline(self, test_year=2025, scale=True):
        print("\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)

        self.load_data()
        self.clean_data()
        self.encode_severity()

        X_train, X_test, y_train, y_test = self.time_split(test_year)

        if scale:
            X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test)
            print("Returning SCALED data")
        else:
            print("Returning RAW data")

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test, self.data

    def get_division_list(self):
        divisions = []
        for _, row in self.data[['District', 'DS_Division']].drop_duplicates().iterrows():
            divisions.append({'name': row['DS_Division'], 'district': row['District']})
        return divisions