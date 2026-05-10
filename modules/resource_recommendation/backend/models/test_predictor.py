from data_preprocessing import ReliefDataPreprocessor
from relief_predictor import ReliefPredictor
import pandas as pd
import numpy as np
import os
import shutil

# Delete old models to ensure fresh training
if os.path.exists('models_saved'):
    shutil.rmtree('models_saved')

# Load RAW data 
print("=" * 60)
print("STEP 1: Loading RAW Data")
print("=" * 60)

preprocessor = ReliefDataPreprocessor("../../data/Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx")
X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)

print(f"RAW data loaded!")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing:  {len(X_test)} samples")
print(f"   X_train sample: {X_train['Affected_Population'].iloc[0]:,} people")
print(f"   y_train sample: {y_train['Cooked Food Packs'].iloc[0]:,} food packs")

# Train the predictor 
print("\n" + "=" * 60)
print("STEP 2: Training Models (Predictor handles scaling)")
print("=" * 60)

predictor = ReliefPredictor()
predictor.train_models(X_train, y_train, X_test, y_test)

# Test prediction on Gampaha 2025
print("\n" + "=" * 60)
print("STEP 3: Making a Prediction")
print("=" * 60)

# Get REAL 2025 data for Gampaha
real_2025_data = full_data[full_data['Year'] == 2025]
gampaha_2025 = real_2025_data[real_2025_data['DS_Division'] == 'Gampaha']

if len(gampaha_2025) > 0:
    real_pop = gampaha_2025['Affected_Population'].iloc[0]
    real_children = gampaha_2025['Children_%'].iloc[0]
    real_elderly = gampaha_2025['Elderly_%'].iloc[0]
    real_severity = gampaha_2025['Severity'].iloc[0]
    
    print(f"\n REAL 2025 Data for Gampaha:")
    print(f"   Population: {real_pop:,}")
    print(f"   Children %: {real_children*100:.1f}%")
    print(f"   Elderly %: {real_elderly*100:.1f}%")
    print(f"   Severity: {real_severity}")
    
    # Get prediction
    result = predictor.predict_with_analysis(
        affected_population=real_pop,
        children_pct=real_children,
        elderly_pct=real_elderly,
        flood_severity=real_severity
    )
    
    print("\n PREDICTION RESULTS:")
    print("-" * 60)
    
    print("\nItem                          Predicted     Actual      Error")
    print("-" * 60)
    
    for item in ['Cooked Food Packs', 'Water Bottles', 'Sanitary', 'Soap']:
        if item in result['predictions']:
            pred_qty = result['predictions'][item]['quantity']
            actual_qty = gampaha_2025[item].iloc[0]
            error_pct = abs(pred_qty - actual_qty) / actual_qty * 100
            status = "OK" if error_pct < 20 else "HIGH ERROR"
            
            print(f"{item:28s} {pred_qty:10,}  {actual_qty:10,}  {error_pct:5.1f}%  {status}")
    
    print(f"\n Overall Priority: {result['overall_priority']}")

else:
    print(" Gampaha 2025 data not found")

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)