import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ReliefDataPreprocessor:
    """
    Preprocesses historical flood relief data for ML and Causal Bayesian Network
    """
    
    def __init__(self, file_path):
        """
        Initialize the preprocessor with data file path
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing historical relief data
        """
        self.file_path = file_path
        self.data = None
        self.feature_columns = None
        self.target_columns = None
        
    def load_data(self):
        """
        Load data from Excel file
        """
        try:
            self.data = pd.read_excel(self.file_path, sheet_name='Sheet1')
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """
        Clean the dataset: handle missing values and convert percentages
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove rows with missing values
        initial_shape = self.data.shape
        self.data = self.data.dropna()
        print(f"Removed {initial_shape[0] - self.data.shape[0]} rows with missing values")
        
        # Convert percentage columns to decimal (0-1 range)
        percentage_columns = ['Children_%', 'Elderly_%']
        for col in percentage_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col] / 100
        
        print(f"Data cleaned. New shape: {self.data.shape}")
        return self.data
    
    def prepare_features_for_modeling(self):
        """
        Separate features (X) and targets (y) for machine learning
        
        Features (inputs): What we know before a flood
        - Affected_Population: Number of people affected
        - Children_%: Percentage of children in affected population
        - Elderly_%: Percentage of elderly in affected population
        
        Targets (outputs): Relief items we need to predict
        - All relief items from the dataset including sanitation
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Define features (inputs)
        self.feature_columns = [
            'Affected_Population',
            'Children_%',
            'Elderly_%'
        ]
        
        # Define targets (outputs) - all relief items
        self.target_columns = [
            'Cooked Food Packs',
            'Water Bottles (litres)',
            'Tea Powder Packets',
            'Milk Flour Packs',
            'Baby Formula Packs',
            'Biscuits Packs',
            'Noodles Packs',
            'Soap',
            'Toothpaste',
            'Toothbrushes'
        ]
        
        # Check if all target columns exist in data
        missing_targets = [col for col in self.target_columns if col not in self.data.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            # Keep only existing targets
            self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        
        # Extract features and targets
        X = self.data[self.feature_columns].copy()
        y = self.data[self.target_columns].copy()
        
        print(f"\nData prepared for modeling:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        print(f"   Features: {self.feature_columns}")
        print(f"   Targets: {self.target_columns}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : DataFrame
            Target data
        test_size : float
            Proportion of data to use for testing (default: 0.2 = 20%)
        random_state : int
            Random seed for reproducible split
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : DataFrames
            Split datasets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"\nData split:")
        print(f"   Training set: {X_train.shape[0]} samples ({int((1-test_size)*100)}%)")
        print(f"   Testing set: {X_test.shape[0]} samples ({int(test_size*100)}%)")
        
        return X_train, X_test, y_train, y_test
    
    def get_division_list(self):
        """
        Get list of all DS Divisions in the dataset
        
        Returns:
        --------
        list : Unique division names
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        divisions = self.data['DS_Division'].unique().tolist()
        print(f"DS Divisions ({len(divisions)}): {divisions}")
        return divisions
    
    def get_statistics_by_division(self, division_name):
        """
        Get historical statistics for a specific division
        
        Parameters:
        -----------
        division_name : str
            Name of the DS Division
        
        Returns:
        --------
        dict : Statistics for that division
        """
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
        
        # Add average relief items for this division
        for col in self.target_columns:
            if col in division_data.columns:
                stats[f'avg_{col}'] = round(division_data[col].mean(), 2)
        
        return stats
    
    def run_full_preprocessing(self):
        """
        Run the complete preprocessing pipeline
        
        Returns:
        --------
        X, y : DataFrames
            Features and targets ready for modeling
        """
        print("=" * 60)
        print("STARTING DATA PREPROCESSING")
        print("=" * 60)
        
        self.load_data()
        self.clean_data()
        X, y = self.prepare_features_for_modeling()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return X, y


# Test the preprocessor
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ReliefDataPreprocessor('Gampaha_DS_Flood_Relief_2019_2022.xlsx')
    
    # Run full preprocessing
    X, y = preprocessor.run_full_preprocessing()
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Display sample datas
    print("\nSample Features (First 3 rows):")
    print(X.head(3))
    
    print("\nSample Targets (First 3 rows):")
    print(y.head(3))