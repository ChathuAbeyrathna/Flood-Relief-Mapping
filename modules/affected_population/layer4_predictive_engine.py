import os
import pandas as pd
import numpy as np
import gc
from bartpy.sklearnmodel import SklearnModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from scipy.stats import spearmanr

class HeteroscedasticBART:
    """
    TECHNICAL NOVELTY IMPLEMENTATION: Dual-Ensemble Heteroscedastic BART.
    g_mu (BART) + g_sigma (GBT) -> Spatially-adaptive localized uncertainty quantification.
    """
    def __init__(self, n_trees_mean=50, n_samples=100, n_burn=50, n_chains=4, n_trees_var=200, max_depth_var=4, calib_frac=0.10, calib_target_coverage=0.95):
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

    def fit(self, X, y):
        rng = np.random.default_rng(seed=42)
        n = len(X)
        idx = rng.permutation(n)
        n_calib = max(1, int(n * self.calib_frac))
        calib_idx = idx[:n_calib]
        fit_idx = idx[n_calib:]

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_calib, y_calib = X[calib_idx], y[calib_idx]

        print(f"   -> Calibration split: {len(X_fit)} fit rows | {len(X_calib)} calib rows")

        print("   -> Stage 1: Training g_mu (BART) on log1p(Affected_People)...")
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

        print("   -> Stage 2: Training g_sigma (GBT) on log(residual²)...")
        self.var_ensemble = GradientBoostingRegressor(
            n_estimators=self.n_trees_var, max_depth=self.max_depth_var, learning_rate=0.05, subsample=0.8, random_state=42
        )
        self.var_ensemble.fit(X_fit, log_variance_target)
        del residuals_fit, log_variance_target
        gc.collect()

        print("   -> Step 4: Calibrating CI multiplier on held-out set...")
        mu_calib = self.mean_ensemble.predict(X_calib)
        log_var_calib = self.var_ensemble.predict(X_calib)
        sigma_calib = np.sqrt(np.exp(np.clip(log_var_calib, -10, 10)))
        residuals_calib = np.where(np.isfinite(y_calib - mu_calib), y_calib - mu_calib, 0.0)

        z_scores = np.abs(residuals_calib) / (sigma_calib + 1e-6)
        self.calibrated_multiplier = float(np.percentile(z_scores, self.calib_target_coverage * 100))
        print(f"   -> CI Multiplier calibrated on held-out data: {self.calibrated_multiplier:.4f}")
        del mu_calib, log_var_calib, sigma_calib, residuals_calib, z_scores
        gc.collect()

    def predict(self, X):
        mu_log = self.mean_ensemble.predict(X)
        log_var_pred = self.var_ensemble.predict(X)
        sigma_log = np.sqrt(np.exp(np.clip(log_var_pred, -10, 10)))
        lower_log = mu_log - (self.calibrated_multiplier * sigma_log)
        upper_log = mu_log + (self.calibrated_multiplier * sigma_log)
        return mu_log, sigma_log, lower_log, upper_log


def run_layer4_validation_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_file = os.path.join(base_dir, "data", "processed", "master", "FinalN_Training_Dataset_Gampaha.csv")
    output_dir = os.path.join(base_dir, "data", "processed", "master")

    print("=" * 70)
    print("LAYER 4 — COMPREHENSIVE BENCHMARK: Novel Architecture vs Standard BART")
    print("=" * 70)
    df_raw = pd.read_csv(master_file)

    features = [
        'Ghs_Pop_Baseline', 'Ghs_Built_S_Total', 'Ghs_Built_S_NonRes', 'Ghs_Built_V_Total',
        'Ghs_Settlement_Type', 'Nightlight_Intensity', 'Precip_Mm', 'Is_Holiday', 'Is_Weekend',
        'Severity_Weight', 'Occupancy_Adj', 'Built_Up_Ratio', 'Weighted_Pop_Engineered', 'Ambient_Pop_Landscan'
    ]

    print("Executing Global Anchor Imputation across master scope...")
    imputer = SimpleImputer(strategy='median')
    df_raw[features] = imputer.fit_transform(df_raw[features])

    validation_years = [2000, 2005, 2010, 2015, 2020]
    year_seeds = {2000: 0, 2005: 5, 2010: 10, 2015: 15, 2020: 20}

    ROWS_PER_YEAR = 10000
    TRAIN_SPLIT   = 8000

    year_samples = {}
    for y in validation_years:
        year_data = df_raw[(df_raw['Data_Year'] == y) & (df_raw['Affected_People'] > 0)].copy()
        take = min(ROWS_PER_YEAR, len(year_data))
        year_samples[y] = year_data.sample(n=take, random_state=year_seeds[y])

    del df_raw
    gc.collect()

    cv_metrics = []

    for test_year in validation_years:
        print(f"\n{'='*70}\nFOLD: Leaving out year {test_year}\n{'='*70}")
        train_df = pd.concat([year_samples[y].iloc[:TRAIN_SPLIT] for y in validation_years if y != test_year], ignore_index=True)
        test_df  = year_samples[test_year].iloc[TRAIN_SPLIT:].copy()

        X_train, y_train = train_df[features].values, np.log1p(train_df['Affected_People'].values)
        X_test, y_test_raw = test_df[features].values, test_df['Affected_People'].values

        # ------------------------------------------------------------------
        # BASELINE MODEL: Standard Homoscedastic BART Instance
        # ------------------------------------------------------------------
        print(" -> Fitting Baseline Standard Homoscedastic BART...")
        std_bart = SklearnModel(n_trees=50, n_samples=100, n_burn=50, n_chains=4, sigma_a=2.0, sigma_b=0.75, n_jobs=1)
        std_bart.fit(X_train, y_train)

        # Standard BART uses global residual variance (homoscedastic sigma)
        std_mu_log = std_bart.predict(X_test)
        std_sigma_global = std_bart.model.sigma.current_value() # Fixed global noise variance parameter

        std_lower_log = std_mu_log - (1.96 * std_sigma_global)
        std_upper_log = std_mu_log + (1.96 * std_sigma_global)

        std_mu_pred = np.maximum(0, np.expm1(std_mu_log))
        std_lower = np.maximum(0, np.expm1(std_lower_log))
        std_upper = np.expm1(std_upper_log)

        std_coverage = np.mean((y_test_raw >= std_lower) & (y_test_raw <= std_upper)) * 100
        std_ci_width = np.median(std_upper - std_lower)
        del std_bart
        gc.collect()

        # ------------------------------------------------------------------
        # NOVEL ARCHITECTURE: Dual-Ensemble Heteroscedastic BART
        # ------------------------------------------------------------------
        model = HeteroscedasticBART(n_trees_mean=50, n_samples=100, n_burn=50, n_chains=4, n_trees_var=200, max_depth_var=4)
        model.fit(X_train, y_train)

        mu_log, sigma_log, lower_log, upper_log = model.predict(X_test)
        mu_pred = np.maximum(0, np.expm1(mu_log))
        lower, upper = np.maximum(0, np.expm1(lower_log)), np.expm1(upper_log)

        mae  = mean_absolute_error(y_test_raw, mu_pred)
        rmse = root_mean_squared_error(y_test_raw, mu_pred)
        r2   = r2_score(y_test_raw, mu_pred)
        rho, _ = spearmanr(y_test_raw, mu_pred) if np.var(mu_pred) >= 1e-10 else (0.0, 0)
        novel_coverage = np.mean((y_test_raw >= lower) & (y_test_raw <= upper)) * 100
        novel_ci_width = np.median(upper - lower)

        print(f"\n     [Standard BART] Coverage: {std_coverage:.2f}% | Median Width: {std_ci_width:.2f}")
        print(f"     [Novel Model  ] Coverage: {novel_coverage:.2f}% | Median Width: {novel_ci_width:.2f}")

        cv_metrics.append({
            'Year': test_year, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Spearman_Rho': rho,
            'Std_BART_Coverage': std_coverage, 'Novel_BART_Coverage': novel_coverage,
            'Std_BART_Width': std_ci_width, 'Novel_BART_Width': novel_ci_width
        })
        del train_df, test_df, X_train, y_train, X_test, model
        gc.collect()

    cv_df = pd.DataFrame(cv_metrics)
    print("\n" + "=" * 85 + "\nFINAL LAYER 4 ARCHITECTURAL NOVELTY COMPARISON REPORT\n" + "=" * 85)
    print(cv_df.to_string(index=False))
    print("-" * 85)
    print(f"Mean Standard BART Coverage : {cv_df['Std_BART_Coverage'].mean():.2f}% | Mean Width: {cv_df['Std_BART_Width'].mean():.2f}")
    print(f"Mean Novel Model Coverage   : {cv_df['Novel_BART_Coverage'].mean():.2f}% | Mean Width: {cv_df['Novel_BART_Width'].mean():.2f}")
    print("=" * 85)
    cv_df.to_csv(os.path.join(output_dir, "layer4N_novelty_comparison_report.csv"), index=False)

if __name__ == "__main__":
    run_layer4_validation_pipeline()