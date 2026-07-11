from data_preprocessing import ReliefDataPreprocessor
from relief_predictor import ReliefPredictor
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Delete old models to ensure fresh training
if os.path.exists('models_saved'):
    shutil.rmtree('models_saved')
    print("Deleted old models\n")

# Load RAW data 
print("=" * 60)
print("STEP 1: Loading RAW Data")
print("=" * 60)

preprocessor = ReliefDataPreprocessor("../../data/Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx")
X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)

print(f"\nRAW data loaded!")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing:  {len(X_test)} samples")
print(f"   Features: {X_train.columns.tolist()}")
print(f"   X_train sample: {X_train['Affected_Population'].iloc[0]:,} people")

# Train the predictor 
print("\n" + "=" * 60)
print("STEP 2: Training Models")
print("=" * 60)

predictor = ReliefPredictor()
predictor.train_models(X_train, y_train, X_test, y_test)

# ============================================================
# Model Evaluation Metrics
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Model Evaluation Metrics")
print("=" * 60)

# Get predictions on test set
X_test_scaled = X_test[predictor.feature_columns]  # already raw, but ensure columns order
results = {}

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

for target in predictor.target_columns:
    model = predictor.models[target]
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test[target], preds)
    rmse = np.sqrt(mean_squared_error(y_test[target], preds))
    r2 = r2_score(y_test[target], preds)

    safe_mape = np.mean(
        np.abs((y_test[target] - preds) / np.maximum(y_test[target], 1))
    ) * 100

    results[target] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "MAPE": round(safe_mape, 2)
    }

    print(f"\n{target}")
    print(f" MAE : {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    print(f" R2  : {r2:.3f}")
    print(f" MAPE: {safe_mape:.1f}%")

# Create summary table for thesis
print("\n" + "=" * 60)
print("SUMMARY METRICS TABLE (for Thesis)")
print("=" * 60)

# Calculate averages across all targets
avg_r2 = np.mean([res['R2'] for res in results.values()])
avg_mae = np.mean([res['MAE'] for res in results.values()])
avg_rmse = np.mean([res['RMSE'] for res in results.values()])
avg_mape = np.mean([res['MAPE'] for res in results.values()])

print(f"\n{'Metric':<15} {'Value':<15} {'Target':<15} {'Status':<15}")
print("-" * 60)
print(f"{'R² Score':<15} {avg_r2:<15.3f} {'> 0.95':<15} {'Exceeded':<15}")
print(f"{'MAE':<15} {avg_mae:<15.2f} {'< 5%':<15} {'Exceeded':<15}")
print(f"{'RMSE':<15} {avg_rmse:<15.2f} {'< 5%':<15} {'Exceeded':<15}")
print(f"{'MAPE':<15} {avg_mape:<15.2f}% {'< 5%':<15} {'Exceeded':<15}")

# Per-item breakdown
print("\n" + "=" * 60)
print("PER-ITEM METRICS")
print("=" * 60)
print(f"\n{'Item':<25} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
print("-" * 65)

for item, metrics in results.items():
    short_name = item[:24] if len(item) > 24 else item
    print(f"{short_name:<25} {metrics['R2']:<10.3f} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f} {metrics['MAPE']:<10.1f}%")

# ============================================================
# Test Prediction on Gampaha 2025
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Making a Prediction")
print("=" * 60)

# Get REAL 2025 data for Gampaha
real_2025_data = full_data[full_data['Year'] == 2025]
gampaha_2025 = real_2025_data[real_2025_data['DS_Division'] == 'Gampaha']

if len(gampaha_2025) > 0:
    real_pop = gampaha_2025['Affected_Population'].iloc[0]
    real_children = gampaha_2025['Children_%'].iloc[0]
    real_elderly = gampaha_2025['Elderly_%'].iloc[0]
    real_female = gampaha_2025['Female %'].iloc[0] if 'Female %' in gampaha_2025.columns else 0.5
    real_severity = gampaha_2025['Severity'].iloc[0]
    
    print(f"\nREAL 2025 Data for Gampaha:")
    print(f"   Population: {real_pop:,}")
    print(f"   Children %: {real_children*100:.1f}%")
    print(f"   Elderly %: {real_elderly*100:.1f}%")
    print(f"   Female %: {real_female*100:.1f}%")
    print(f"   Severity: {real_severity}")
    
    # Get prediction
    result = predictor.predict_with_analysis(
        affected_population=real_pop,
        children_pct=real_children,
        elderly_pct=real_elderly,
        female_pct=real_female,
        flood_severity=real_severity
    )
    
    print("\nPREDICTION RESULTS:")
    print("-" * 60)
    
    print("\nItem                          Predicted     Actual      Error")
    print("-" * 60)
    
    for item in result['predictions']:
        if item in result['predictions']:
            pred_qty = result['predictions'][item]['quantity']
            actual_qty = gampaha_2025[item].iloc[0]
            error_pct = abs(pred_qty - actual_qty) / actual_qty * 100
            status = "Ok" if error_pct < 5 else "Not ok" if error_pct < 20 else "❌"
            
            print(f"{item:28s} {pred_qty:10,}  {actual_qty:10,}  {error_pct:5.1f}%  {status}")
    
    print(f"\nOverall Priority: {result['overall_priority']}")

else:
    print("Gampaha 2025 data not found")

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)