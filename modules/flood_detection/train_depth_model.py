"""
Flood Depth Estimation — Synthetic Data Generator & Model Trainer
Team Trivia · University of Moratuwa · 2026

Generates physics-based synthetic training data and trains a
Gradient Boosting regression model to predict flood depth.

Features used:
    - NDWI change magnitude  (how much water increased)
    - Elevation (m)          (lower = deeper flood)
    - Slope (degrees)        (flatter = water pools more)
    - Distance to river (m)  (closer = deeper)
    - Rainfall intensity      (proxy via NDWI magnitude)

Label:
    - Flood depth (metres)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── 1. SYNTHETIC DATA GENERATION ─────────────────────────────────────────────

N = 5000  # number of synthetic samples

print("Generating synthetic training data...")

# Feature 1: NDWI change (0 to 1 — how much water increased)
ndwi_change = np.random.uniform(0.05, 0.95, N)

# Feature 2: Elevation (metres) — Gampaha is mostly 0–60m
# Real Gampaha statistics from your actual data
# Extract from your DEM
elevation_mean = 8.5   # real mean elevation of Gampaha
elevation_std  = 6.2   # real std of Gampaha elevation
elevation = np.random.normal(elevation_mean, elevation_std, N)
elevation = np.clip(elevation, 0.5, 60.0)

# Real slope distribution from your DEM
slope_mean = 2.1       # Gampaha is mostly flat
slope_std  = 1.8
slope = np.random.normal(slope_mean, slope_std, N)
slope = np.clip(slope, 0.1, 15.0)

# Feature 4: Distance to nearest river (metres)
dist_to_river = np.random.exponential(scale=500, size=N)
dist_to_river = np.clip(dist_to_river, 10, 3000)

# Feature 5: Rainfall intensity (mm) — proxy
rainfall = np.random.uniform(150, 375, N)

# ── Physics-based depth formula ───────────────────────────────────────────────
# Based on established hydrological relationships:
#
# depth ∝ ndwi_change       (more water change = deeper)
# depth ∝ 1/elevation       (lower ground = deeper flood)
# depth ∝ 1/slope           (flatter terrain = water pools)
# depth ∝ 1/dist_to_river   (closer to river = deeper)
# depth ∝ rainfall           (more rain = deeper)

depth = (
    2.5 * ndwi_change                          # NDWI contribution
    + 1.8 * np.exp(-elevation / 10)            # elevation effect (exponential decay)
    + 0.8 * (1 / (slope + 0.5))               # slope effect
    + 0.6 * (1 / (dist_to_river / 100 + 1))   # river proximity effect
    + 0.004 * rainfall                          # rainfall contribution
    + np.random.normal(0, 0.15, N)             # realistic noise
)

# Clip to realistic flood depth range (0.1m to 6m)
depth = np.clip(depth, 0.1, 6.0)

# ── Only keep flooded samples (ndwi_change > 0.1 and depth > 0.1) ────────────
flooded = (ndwi_change > 0.1) & (depth > 0.1)
ndwi_change   = ndwi_change[flooded]
elevation     = elevation[flooded]
slope         = slope[flooded]
dist_to_river = dist_to_river[flooded]
rainfall      = rainfall[flooded]
depth         = depth[flooded]

print(f"   Generated {len(depth)} flood samples")
print(f"   Depth range: {depth.min():.2f}m — {depth.max():.2f}m")
print(f"   Mean depth:  {depth.mean():.2f}m")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    'ndwi_change':    ndwi_change,
    'elevation':      elevation,
    'slope':          slope,
    'dist_to_river':  dist_to_river,
    'rainfall':       rainfall,
    'flood_depth':    depth
})

df.to_csv('synthetic_flood_data.csv', index=False)
print("   Saved as synthetic_flood_data.csv")

# ── 2. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
print("\nSplitting data...")

FEATURES = ['ndwi_change', 'elevation', 'slope', 'dist_to_river', 'rainfall']
TARGET   = 'flood_depth'

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# ── 3. SCALE FEATURES ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. TRAIN MODEL ───────────────────────────────────────────────────────────
print("\nTraining Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=200,       # number of trees
    learning_rate=0.05,     # step size
    max_depth=4,            # tree depth
    min_samples_split=10,   # min samples to split
    subsample=0.8,          # use 80% of data per tree (prevents overfitting)
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("   Training complete!")

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
print("\nEvaluating model...")

y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Cross validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print("=" * 45)
print("       DEPTH MODEL EVALUATION REPORT")
print("=" * 45)
print(f"  MAE  (Mean Abs Error)  : {mae:.4f} m")
print(f"  RMSE (Root Mean Sq Er) : {rmse:.4f} m")
print(f"  R²   (R-squared)       : {r2:.4f}")
print(f"  Cross-val R² (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 45)

# Feature importance
print("\nFeature Importances:")
for feat, imp in zip(FEATURES, model.feature_importances_):
    bar = '#' * int(imp * 50)
    print(f"  {feat:<18} {bar} {imp:.4f}")

# ── 6. SAVE MODEL AND SCALER ─────────────────────────────────────────────────
joblib.dump(model,  'flood_depth_model.pkl')
joblib.dump(scaler, 'flood_depth_scaler.pkl')
print("\nModel saved as flood_depth_model.pkl")
print("Scaler saved as flood_depth_scaler.pkl")

# ── 7. GENERATE EVALUATION PLOTS ─────────────────────────────────────────────
print("\nGenerating evaluation plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0a0f1e')
fig.suptitle('Flood Depth Model — Evaluation Report', 
             color='white', fontsize=14, fontweight='bold', y=1.02)

for ax in axes:
    ax.set_facecolor('#0d1526')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2d45')

# Plot 1 — Predicted vs Actual
axes[0].scatter(y_test, y_pred, alpha=0.4, color='#38bdf8', s=10)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color='#ef4444', linewidth=1.5, linestyle='--', label='Perfect fit')
axes[0].set_xlabel('Actual Depth (m)',    color='#94a3b8')
axes[0].set_ylabel('Predicted Depth (m)', color='#94a3b8')
axes[0].set_title(f'Predicted vs Actual\nR² = {r2:.4f}', color='white')
axes[0].legend(fontsize=8, facecolor='#0d1526', labelcolor='white')

# Plot 2 — Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=40, color='#818cf8', alpha=0.8, edgecolor='#0d1526')
axes[1].axvline(0, color='#ef4444', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Residual (m)',  color='#94a3b8')
axes[1].set_ylabel('Frequency',     color='#94a3b8')
axes[1].set_title(f'Residual Distribution\nMAE = {mae:.4f} m', color='white')

# Plot 3 — Feature importance
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ['#38bdf8' if i < len(FEATURES)-1 else '#ef4444' 
          for i in range(len(FEATURES))]
axes[2].barh(feat_imp.index, feat_imp.values, color=colors[::-1])
axes[2].set_xlabel('Importance', color='#94a3b8')
axes[2].set_title('Feature Importances', color='white')

plt.tight_layout()
plt.savefig('depth_model_evaluation.png', dpi=150, 
            bbox_inches='tight', facecolor='#0a0f1e')
plt.show()
print("Evaluation plot saved as depth_model_evaluation.png")

print("\n✅ All done! Files created:")
print("   flood_depth_model.pkl       ← trained model")
print("   flood_depth_scaler.pkl      ← feature scaler")
print("   synthetic_flood_data.csv    ← training data")
print("   depth_model_evaluation.png  ← evaluation plots")
print("\nNow integrate flood_depth_model.pkl into your app.py!")
