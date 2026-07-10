import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    #visualization
from bartpy.sklearnmodel import SklearnModel    # BART model implementation
from bartpy.diagnostics.features import feature_split_proportions    # Diagnostic utility for extracting feature split frequencies

def layer3_stratified_diagnostics():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".." ))
    master_file = os.path.join(base_dir, "data", "processed", "master", "FinalN_Training_Dataset_Gampaha.csv")
    output_dir = os.path.join(base_dir, "data", "processed", "master")

    print("Loading Final Unified Dataset for Layer 3...")
    df = pd.read_csv(master_file)   #Load data

    # DEFINE INPUT FEATURES
    features = [
        'Ghs_Pop_Baseline', 'Ghs_Built_S_Total', 'Ghs_Built_V_Total',
        'Ghs_Built_S_NonRes', 'Ghs_Settlement_Type', 'Nightlight_Intensity',
        'Precip_Mm', 'Is_Holiday', 'Is_Weekend', 'Severity_Weight',
        'Occupancy_Adj', 'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Ambient_Pop_Landscan'
    ]

    # --- UPDATED STRATIFIED SAMPLING LOGIC (Aligned to 100k Hardware Limits) ---
    # We pool data only from the historical training horizon (Excluding 2025 Hold-out context)
    unique_train_years = [2000, 2005, 2010, 2015, 2020]

    # Scale allocation to exactly 20,000 rows per year based on hardware test limits
    ROWS_PER_YEAR = 10000

    print(f"Executing Scaled Stratified Sampling across training epochs ({ROWS_PER_YEAR} rows/year)...")
    sampled_dfs = []

    for year in unique_train_years:
        year_subset = df[df['Data_Year'] == year]

        # Pull 20,000 rows safely, dropping down to full availability if a year contains fewer rows
        sample_n = min(ROWS_PER_YEAR, len(year_subset))
        sampled_dfs.append(year_subset.sample(n=sample_n, random_state=42))

    # Combine all yearly samples
    df_stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Stratified sample completed. Total diagnostic rows: {len(df_stratified_sample)}")

    # PREPARE TRAINING MATRICES
    X = df_stratified_sample[features]
    y = df_stratified_sample['Affected_People']

    # --- FIT EXPLORATORY BART MODEL ---
    print("Fitting Baseline BART for Feature Importance extraction...")
    # n_trees: Number of trees in the ensemble.
    # n_samples:Number of posterior MCMC samples.
    # n_burn: Number of burn-in samples to discard. (Burn-in iterations discarded before inference.)
    # n_jobs: Number of parallel jobs to run for both `fit` and `predict`. -1 means using all processors.
    model = SklearnModel(n_trees=50, n_samples=100, n_burn=50, n_jobs=1)
    model.fit(X, y) # Fit the BART model to the stratified sample. Fit means the model learns the relationship between features and target variable.

    # --- EXTRACT FIS (feature importance score) LOGIC---
    # feature_split_proportions()
    # Returns:
    # How often each feature was used in tree splits
    # across all trees and posterior samples.
    proportions = feature_split_proportions(model)
    # 1. Map columns using X.columns natively to prevent index shifting bugs (Convert feature index → feature name)
    raw_scores = {X.columns[col]: prop for col, prop in proportions.items() if col is not None}
    # 2. Add features that got 0 splits back into the dictionary explicitly
    for col in X.columns:
        if col not in raw_scores:
            raw_scores[col] = 0.0
    # 3. Normalize scores so total = 100%
    total_raw_sum = sum(raw_scores.values())
    fis_scores = {feat: (val / total_raw_sum) * 100 if total_raw_sum > 0 else 0
                  for feat, val in raw_scores.items()}
    # 4. Create sorted feature importance dataframe
    fis_df = pd.DataFrame(list(fis_scores.items()), columns=['Feature', 'Importance_Score']).sort_values(by='Importance_Score', ascending=False)

    print("\n" + "="*40)
    print(X.columns.tolist()) # Print the list of feature names to verify correct mapping
    print(sorted(proportions.keys()))
    print("*********")
    print(proportions)
    print(len(proportions), len(features))
    print("STRATIFIED LAYER 3 FEATURE IMPORTANCE")
    print("="*40)
    print(fis_df.to_string(index=False))
    print("="*40)
    # Save feature importance table
    fis_df.to_csv(os.path.join(output_dir, "layer3N_stratified_feature_importance.csv"), index=False)

    # --- 5. GENERATE PARTIAL DEPENDENCE PLOT ---
    print("\nGenerating Partial Dependence Plot for Weighted_Pop_Engineered...")
    target_feature = 'Weighted_Pop_Engineered'

    # Generate 10 uniform testing points across your engineered feature spectrum
    grid = np.linspace(X[target_feature].min(), X[target_feature].max(), 10)
    pdp_values = []
    X_temp = X.copy()
    # Evaluate model at each grid point
    for point in grid:
        X_temp[target_feature] = point  #Force all rows to same feature value
        predictions = model.predict(X_temp)  # Predict outcomes
        pdp_values.append(np.mean(predictions)) # Average prediction

    # Construct the plot
    plt.figure(figsize=(8, 5))
    plt.plot(grid, pdp_values, marker='o', linewidth=2, color='darkblue')
    plt.title('Layer 3 Partial Dependence Plot', fontsize=12, fontweight='bold')
    plt.xlabel('Weighted Population Engineered Value (Layer 2 Heuristic)', fontsize=10)
    plt.ylabel('Marginal Value on Predicted Affected Counts', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    # Save plot
    plot_path = os.path.join(output_dir, "layer3n_pdp_engineered_pop.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SUCCESS: Partial dependence plot saved to {plot_path}")

    print("Layer 3 diagnostics safely concluded.")

if __name__ == "__main__":
    layer3_stratified_diagnostics()