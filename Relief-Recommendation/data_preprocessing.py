import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ReliefDataPreprocessor:
    """
    Preprocesses historical flood relief data for ML or Causal Bayesian Network.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.feature_columns = ['Affected_Population', 'Children_%', 'Elderly_%']
        self.target_columns = [
            'Cooked Food Packs',
            'Water Bottles',
            'Milk Powder Packs',
            'Infant Milk Powder Packs',
            'Biscuits Packs',
            'Noodles Packs',
            'Tea Powder Packets',
            'Sanitary',
            'Soap',
            'Toothpaste',
            'Toothbrushes'
        ]
    
    def load_data(self):
        """Load Excel data."""
        try:
            self.data = pd.read_excel(self.file_path, sheet_name='Sheet1')
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """Clean dataset: remove missing rows and convert percentages."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_shape = self.data.shape
        self.data = self.data.dropna()
        print(f"Removed {initial_shape[0] - self.data.shape[0]} rows with missing values")
        
        # Convert percentages to decimals (0-1) for ML
        for col in ['Children_%', 'Elderly_%']:
            if col in self.data.columns:
                self.data[col] = self.data[col] / 100
        
        print(f"Data cleaned. New shape: {self.data.shape}")
        return self.data
    
    def prepare_features_for_modeling(self):
        """Separate features (X) and targets (y)."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_targets = [col for col in self.target_columns if col not in self.data.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        
        X = self.data[self.feature_columns].copy()
        y = self.data[self.target_columns].copy()
        
        print(f"\nData prepared for modeling:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        print(f"   Features: {self.feature_columns}")
        print(f"   Targets: {self.target_columns}")
        
        return X, y
    
    def assign_train_test_sets(self):
        """
        Assign training/testing sets exactly based on NDRSC years:
        Training: 2019-13, 2023-13, 2024-8, 2025-7
        Testing: 2024-5, 2025-6
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Default all to Training
        self.data['Set'] = 'Training'
        
        # 2024 testing: first 5 rows of 2024
        test_2024_idx = self.data[(self.data['Year'] == 2024)].index[:5]
        # 2025 testing: first 6 rows of 2025
        test_2025_idx = self.data[(self.data['Year'] == 2025)].index[:6]
        
        self.data.loc[test_2024_idx, 'Set'] = 'Testing'
        self.data.loc[test_2025_idx, 'Set'] = 'Testing'
        
        print(f"Training samples: {len(self.data[self.data['Set']=='Training'])}")
        print(f"Testing samples: {len(self.data[self.data['Set']=='Testing'])}")
        
        return self.data
    
    def get_division_list(self):
        """Return list of DS Divisions."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data['DS_Division'].unique().tolist()
    
    def get_statistics_by_division(self, division_name):
        """Get averages for a DS Division."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        division_data = self.data[self.data['DS_Division'] == division_name]
        if len(division_data) == 0:
            return None
        
        stats = {
            'years_of_data': division_data['Year'].tolist(),
            'sample_count': len(division_data),
            'avg_affected_population': division_data['Affected_Population'].mean(),
            'avg_children_pct': division_data['Children_%'].mean() * 100,
            'avg_elderly_pct': division_data['Elderly_%'].mean() * 100
        }
        for col in self.target_columns:
            if col in division_data.columns:
                stats[f'avg_{col}'] = round(division_data[col].mean(), 2)
        
        return stats
    
    def run_full_preprocessing(self):
        """Run full pipeline: load, clean, features, assign train/test sets."""
        print("="*60)
        print("STARTING DATA PREPROCESSING")
        print("="*60)
        
        self.load_data()
        self.clean_data()
        X, y = self.prepare_features_for_modeling()
        self.assign_train_test_sets()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X, y, self.data


# -------------------------------
# Run the preprocessor
# -------------------------------
if __name__ == "__main__":
    preprocessor = ReliefDataPreprocessor('Gampaha_DS_Flood_Relief_2019_2025.xlsx')
    
    # Run full preprocessing
    X, y, full_data = preprocessor.run_full_preprocessing()
    
    # Display sample features and targets
    print("\nSample Features (First 3 rows):")
    print(X.head(3))
    
    print("\nSample Targets (First 3 rows):")
    print(y.head(3))
    
    print("\nSample Train/Test sets (First 10 rows):")
    print(full_data[['Year', 'DS_Division', 'Set']].head(10))