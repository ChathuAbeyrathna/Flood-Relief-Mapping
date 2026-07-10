import os
import pandas as pd
import numpy as np
import gc
from bartpy.sklearnmodel import SklearnModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

# ==============================================================================
# TECHNICAL NOVELTY ENGINE: HeteroscedasticBART (Layer 5 Self-Correcting Production Engine)
# ==============================================================================
class HeteroscedasticBARTInference:
    def __init__(self, n_trees_mean=50, n_samples=200, n_burn=200, n_chains=4, n_trees_var=200, max_depth_var=4, calib_frac=0.10, calib_target_coverage=0.95):
        self.n_trees_mean = n_trees_mean
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.n_chains = n_chains
        self.n_trees_var = n_trees_var
        self.max_depth_var = max_depth_var
        self.calib_frac = calib_frac
        self.calib_target_coverage = calib_target_coverage

        self.mean_ensemble = None
        self.var_ensemble = None
        self.calibrated_multiplier = 1.96

    def fit_production_model(self, X, y):
        rng = np.random.default_rng(seed=42)
        n = len(X)
        idx = rng.permutation(n)
        n_calib = max(1, int(n * self.calib_frac))
        calib_idx = idx[:n_calib]
        fit_idx = idx[n_calib:]

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_calib, y_calib = X[calib_idx], y[calib_idx]

        print(f"   -> Production Training Split: {len(X_fit)} fit rows | {len(X_calib)} calibration rows")

        print("   -> Stage 1: Training Production g_mu (BART) on full historical horizon...")
        self.mean_ensemble = SklearnModel(
            n_trees=self.n_trees_mean, n_samples=self.n_samples, n_burn=self.n_burn, n_chains=self.n_chains,
            sigma_a=2.0, sigma_b=0.75, store_in_sample_predictions=True, n_jobs=1
        )
        self.mean_ensemble.fit(X_fit, y_fit)

        mu_fit = self.mean_ensemble.predict()
        residuals_fit = y_fit - mu_fit
        residuals_fit = np.where(np.isfinite(residuals_fit), residuals_fit, 0.0)
        log_variance_target = np.log(residuals_fit ** 2 + 1e-6)
        log_variance_target = np.clip(log_variance_target, -20, 20)
        del mu_fit
        gc.collect()

        print("   -> Stage 2: Training Production g_sigma (GBT) on log(residual²)...")
        self.var_ensemble = GradientBoostingRegressor(
            n_estimators=self.n_trees_var, max_depth=self.max_depth_var, learning_rate=0.05, subsample=0.8, random_state=42
        )
        self.var_ensemble.fit(X_fit, log_variance_target)
        del residuals_fit, log_variance_target
        gc.collect()

        print("   -> Step 4: Calibrating Production scale parameters...")
        mu_calib = self.mean_ensemble.predict(X_calib)
        log_var_calib = self.var_ensemble.predict(X_calib)
        sigma_calib = np.sqrt(np.exp(np.clip(log_var_calib, -10, 10)))
        residuals_calib = np.where(np.isfinite(y_calib - mu_calib), y_calib - mu_calib, 0.0)

        z_scores = np.abs(residuals_calib) / (sigma_calib + 1e-6)
        self.calibrated_multiplier = float(np.percentile(z_scores, self.calib_target_coverage * 100))
        print(f"   -> Finalized Production Multiplier Locked: {self.calibrated_multiplier:.4f}")
        del mu_calib, log_var_calib, sigma_calib, residuals_calib, z_scores
        gc.collect()

    def generate_predictions(self, X):
        mu_log = self.mean_ensemble.predict(X)
        log_var_pred = self.var_ensemble.predict(X)
        sigma_log = np.sqrt(np.exp(np.clip(log_var_pred, -10, 10)))

        lower_log = mu_log - (self.calibrated_multiplier * sigma_log)
        upper_log = mu_log + (self.calibrated_multiplier * sigma_log)
        return mu_log, sigma_log, lower_log, upper_log


# ==============================================================================
# STRATIFIED SAMPLING HELPER
# ==============================================================================
def stratified_sample(df, n, random_state):
    try:
        df = df.copy()
        df['_sev_bin'] = pd.qcut(df['Severity_Weight'], q=4, labels=False, duplicates='drop')
        df['_stratum'] = df['Ghs_Settlement_Type'].astype(str) + '_' + df['_sev_bin'].astype(str)

        counts = df['_stratum'].value_counts(normalize=True)
        per_stratum = (counts * n).round().astype(int)
        diff = n - per_stratum.sum()
        if diff != 0:
            per_stratum[per_stratum.idxmax()] += diff

        parts = []
        for stratum, count in per_stratum.items():
            pool = df[df['_stratum'] == stratum]
            take = min(count, len(pool))
            if take > 0:
                parts.append(pool.sample(n=take, random_state=random_state))

        result = pd.concat(parts).drop(columns=['_sev_bin', '_stratum'])
        return result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    except Exception:
        return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)


# ==============================================================================
# MAIN PRODUCTION INFERENCE PIPELINE (WITH CHUNK APPENDING & INLINE CORRECTION)
# ==============================================================================
def run_layer5_inference_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_file = os.path.join(base_dir, "data", "processed", "master", "Final_Training_Dataset_Gampaha.csv")
    output_dir = os.path.join(base_dir, "data", "processed", "master")
    out_path = os.path.join(output_dir, "layer5_2025_spatial_predictions.csv")

    if os.path.exists(out_path):
        os.remove(out_path)

    print("=" * 85)
    print("LAYER 5 — INFERENCE ENGINE: SELF-CORRECTING DATA STREAM ENGINE")
    print("=" * 85)

    print("Reading Consolidated Master Matrix...")
    df_raw = pd.read_csv(master_file)

    # ------------------------------------------------------------------
    # INLINE CORRECTOR: UPGRADED BOUNDED DISAGGREGATION (OPTION A)
    # ------------------------------------------------------------------
    print("Executing Option A Bounded Footprint Disaggregation...")
    from disaggregation_utility import apply_bounded_disaggregation
    df_raw = apply_bounded_disaggregation(df_raw)
    print(" -> Success: Macro targets locked strictly to historical wet footprints.")
    # ------------------------------------------------------------------

    features = [
        'Ghs_Pop_Baseline', 'Ghs_Built_S_Total', 'Ghs_Built_S_NonRes', 'Ghs_Built_V_Total',
        'Ghs_Settlement_Type', 'Nightlight_Intensity', 'Precip_Mm', 'Is_Holiday', 'Is_Weekend',
        'Severity_Weight', 'Occupancy_Adj', 'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Ambient_Pop_Landscan'
    ]

    print("Executing Global Anchor Imputation over consolidated data bounds...")
    imputer = SimpleImputer(strategy='median')
    df_raw[features] = imputer.fit_transform(df_raw[features])

    historical_years = [2000, 2005, 2010, 2015, 2020]
    ROWS_PER_YEAR = 12000

    print("Pre-extracting training vectors with strict footprint boundaries...")
    historical_samples = []
    for y in historical_years:
        # Enforce a strict human baseline capacity threshold.
        # This isolates pixels that actually had historical macro-displaced targets assigned
        # BEFORE disaggregation diluted them into decimal fractions.
        year_data = df_raw[(df_raw['Data_Year'] == y) & (df_raw['Affected_People'] >= 1.0)].copy()

        # Fallback safeguard: if a historical year has highly sparse cell distributions,
        # catch the top percentile of active targets instead of dropping the epoch.
        if len(year_data) < 500:
            year_data = df_raw[(df_raw['Data_Year'] == y) & (df_raw['Affected_People'] > 0.05)].copy()

        sampled_year_data = stratified_sample(year_data, n=min(ROWS_PER_YEAR, len(year_data)), random_state=y)
        historical_samples.append(sampled_year_data)

    df_train_pool = pd.concat(historical_samples, ignore_index=True)

    df_2025_target = df_raw[df_raw['Data_Year'] == 2025].copy()
    if len(df_2025_target) == 0:
        print("\n[Simulation Mode] Pulling 2020 matrix as 2025 proxy...")
        df_2025_target = df_raw[df_raw['Data_Year'] == 2020].copy()
        df_2025_target['Data_Year'] = 2025
        df_2025_target['Precip_Mm'] *= 1.10

    print(f"\n -> Unified Training Matrix Size  : {len(df_train_pool)} rows")
    print(f" -> Target 2025 Inference Footprint: {len(df_2025_target)} pixels")

    X_train = df_train_pool[features].values
    y_train = np.log1p(df_train_pool['Affected_People'].values)

    del df_raw, df_train_pool, historical_samples
    gc.collect()

    # Train Model
    model = HeteroscedasticBARTInference(n_trees_mean=50, n_samples=200, n_burn=200, n_chains=4, n_trees_var=200, max_depth_var=4)
    model.fit_production_model(X_train, y_train)

    # ------------------------------------------------------------------
    # CHUNKED STREAMING EXECUTION ENGINE
    # ------------------------------------------------------------------
    CHUNK_SIZE = 100000
    total_rows = len(df_2025_target)
    num_chunks = int(np.ceil(total_rows / CHUNK_SIZE))

    print(f"\nStreaming forward-looking inference in {num_chunks} individual streaming chunks...")

    export_columns = [
        'Longitude', 'Latitude', 'Ds_Division_Name', 'Data_Year', 'Precip_Mm',
        'Ghs_Pop_Baseline', 'Predicted_Mean_Affected', 'Predicted_Sigma_Log',
        'Predicted_Lower_Bound', 'Predicted_Upper_Bound', 'Uncertainty_Range_Width'
    ]

    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_rows)

        print(f"   -> Processing Streaming Chunk {i+1}/{num_chunks}...")

        df_chunk = df_2025_target.iloc[start_idx:end_idx].copy()
        X_chunk = df_chunk[features].values

        mu_log, sigma_log, lower_log, upper_log = model.generate_predictions(X_chunk)

        # Spatial inversion adjustments
        df_chunk['Predicted_Mean_Affected'] = np.maximum(0, np.expm1(mu_log))
        df_chunk['Predicted_Sigma_Log']      = sigma_log
        df_chunk['Predicted_Lower_Bound']    = np.maximum(0, np.expm1(lower_log))
        df_chunk['Predicted_Upper_Bound']    = np.maximum(0, np.expm1(upper_log))
        df_chunk['Uncertainty_Range_Width']  = df_chunk['Predicted_Upper_Bound'] - df_chunk['Predicted_Lower_Bound']

        df_chunk_out = df_chunk[export_columns]

        if not os.path.exists(out_path):
            df_chunk_out.to_csv(out_path, index=False)
        else:
            df_chunk_out.to_csv(out_path, mode='a', header=False, index=False)

        del df_chunk, X_chunk, df_chunk_out, mu_log, sigma_log, lower_log, upper_log
        gc.collect()

    print("\n" + "=" * 85)
    print(f"[Success] Balanced Layer 5 spatial predictions saved → {out_path}")
    print("=" * 85)

    # ==============================================================================
    # INTEGRATED VALIDATION AUDITOR BLOCK (Layer 5 Sanity Check)
    # ==============================================================================
    print("\nExecuting Macro-Level Validation Audit on Generated 2025 Deliverables...")
    df_audit = pd.read_csv(out_path)

    total_baseline_pop = df_audit['Ghs_Pop_Baseline'].sum()
    total_predicted_mean = df_audit['Predicted_Mean_Affected'].sum()
    total_upper_bound = df_audit['Predicted_Upper_Bound'].sum()

    print("-" * 85)
    print(f" -> Total Regional Population Baseline : {total_baseline_pop:,.2f} citizens")
    print(f" -> Projected Total Mean Affected      : {total_predicted_mean:,.2f} citizens")
    print(f" -> Max Conservative Risk Exposure     : {total_upper_bound:,.2f} citizens")
    print(f" -> Regional Macro Affected Ratio      : {(total_predicted_mean / (total_baseline_pop + 1e-6)) * 100:.2f}%")
    print("-" * 85)

    # DS-Division Level Aggregate Audit
    print("\nSummary Check by DS Division Name:")
    ds_summary = df_audit.groupby('Ds_Division_Name').agg(
        Total_Baseline=('Ghs_Pop_Baseline', 'sum'),
        Mean_Projected_Affected=('Predicted_Mean_Affected', 'sum'),
        Upper_Risk_Limit=('Predicted_Upper_Bound', 'sum')
    ).round(2)
    print(ds_summary.to_string())
    # ==============================================================================

if __name__ == "__main__":
    run_layer5_inference_pipeline()