import os
import joblib
import pandas as pd
import numpy as np
import gc
from sklearn.impute import SimpleImputer
# Import your model definition components directly from your Layer 5 file
from layer5_inference_engine import HeteroscedasticBARTInference, stratified_sample

def execute_and_serialize_production_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_file = os.path.join(base_dir, "data", "processed", "master", "Final_Training_Dataset_Gampaha.csv")

    # Target Production Export Location
    model_dir = os.path.join(base_dir, "models", "production")

    # CRITICAL FIX: Automatically force the OS to generate the directory path structural boundaries
    print(f"Verifying system model directories at: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    print("Reading Consolidated Master Matrix for production lock...")
    df_raw = pd.read_csv(master_file)

    # Apply Option A Bounded Disaggregation to production pipeline target pools
    print("Structuring production matrices using Option A metrics...")
    from disaggregation_utility import apply_bounded_disaggregation
    df_raw = apply_bounded_disaggregation(df_raw)

    features = [
        'Ghs_Pop_Baseline', 'Ghs_Built_S_Total', 'Ghs_Built_S_NonRes', 'Ghs_Built_V_Total',
        'Ghs_Settlement_Type', 'Nightlight_Intensity', 'Precip_Mm', 'Is_Holiday', 'Is_Weekend',
        'Severity_Weight', 'Occupancy_Adj', 'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Ambient_Pop_Landscan'
    ]

    imputer = SimpleImputer(strategy='median')
    df_raw[features] = imputer.fit_transform(df_raw[features])

    historical_years = [2000, 2005, 2010, 2015, 2020]
    ROWS_PER_YEAR = 12000

    historical_samples = []
    for y in historical_years:
        year_data = df_raw[(df_raw['Data_Year'] == y) & (df_raw['Affected_People'] > 0)].copy()
        sampled_year_data = stratified_sample(year_data, n=min(ROWS_PER_YEAR, len(year_data)), random_state=y)
        historical_samples.append(sampled_year_data)

    df_train_pool = pd.concat(historical_samples, ignore_index=True)
    X_train = df_train_pool[features].values
    y_train = np.log1p(df_train_pool['Affected_People'].values)

    del df_raw, df_train_pool, historical_samples
    gc.collect()

    print("Fitting production ensemble layers...")
    model = HeteroscedasticBARTInference(n_trees_mean=50, n_samples=200, n_burn=200, n_chains=4, n_trees_var=200, max_depth_var=4)
    model.fit_production_model(X_train, y_train)

    # ------------------------------------------------------------------
    # JOBLIB SERIALIZATION EXPORT CORE
    # ------------------------------------------------------------------
    print("\nSerializing engine architectures to portables...")
    joblib.dump(model.mean_ensemble, os.path.join(model_dir, "g_mu_bart_model.pkl"))
    joblib.dump(model.var_ensemble, os.path.join(model_dir, "g_sigma_gbt_model.pkl"))
    joblib.dump(model.calibrated_multiplier, os.path.join(model_dir, "calibrated_multiplier.pkl"))

    print("=" * 85)
    print(f"[Success] All 3 blueprints locked inside directory link: {model_dir}")
    print("=" * 85)

# This block triggers the processing routine automatically on runtime execution
if __name__ == "__main__":
    execute_and_serialize_production_models()