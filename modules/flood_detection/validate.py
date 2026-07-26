# modules/flood_detection/validate.py
"""
Division-Level Flood Validation using RAPIDA dataset

Validates:
1. Flooded DS divisions detected
2. Flood area ranking similarity
3. Correlation with RAPIDA

Run:
python modules/flood_detection/validate.py
"""

import os 
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr, pearsonr

# -------------------------------------------------------------------
# Project path setup
# -------------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# -------------------------------------------------------------------
# RAPIDA reference data
# -------------------------------------------------------------------

# RAPIDA flood area values
# (Use your official values here)

RAPIDA_DATA = {
    "Dompe": 6997.8,
    "Katana": 4736.5,
    "Attanagalla": 4593.4,
    "Wattala": 4311.3,
    "Minuwangoda": 4262.0,
    "Gampaha": 4130.3,
    "Ja Ela": 3617.0,
    "Mirigama": 3469.5,
    "Divulapitiya": 3457.7,
    "Mahara": 2711.0,
    "Biyagama": 2660.5,
    "Kelaniya": 1553.8,
    "Negombo": 509.4,
}

# -------------------------------------------------------------------
# Load YOUR results from GeoJSON
# -------------------------------------------------------------------

print("\nLoading your flood results...")

geojson_path = project_root / "interface" / "backend" / "outputs" / "flood_results.geojson"

if not geojson_path.exists():
    print(f"❌ GeoJSON not found: {geojson_path}")
    exit(1)

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson["features"]

your_results = {}

for feat in features:

    props = feat["properties"]

    division = props.get("adm3_name")

    if division is None:
        continue

    flood_area = props.get("flood_area_ha", 0)

    your_results[division] = flood_area

print(f"✓ Loaded {len(your_results)} divisions")

# -------------------------------------------------------------------
# Prepare comparison dataframe
# -------------------------------------------------------------------

all_divisions = sorted(
    set(RAPIDA_DATA.keys()) |
    set(your_results.keys())
)

rows = []

for div in all_divisions:

    rapida_area = RAPIDA_DATA.get(div, 0)
    your_area = your_results.get(div, 0)

    rows.append({
        "Division": div,
        "RAPIDA": rapida_area,
        "Yours": your_area,
        "RAPIDA_Flooded": 1 if rapida_area > 0 else 0,
        "Your_Flooded": 1 if your_area > 0 else 0,
    })

df = pd.DataFrame(rows)

# -------------------------------------------------------------------
# Binary classification metrics
# -------------------------------------------------------------------

print("\nCalculating division-level metrics...")

y_true = df["RAPIDA_Flooded"]
y_pred = df["Your_Flooded"]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

# -------------------------------------------------------------------
# Correlation metrics
# -------------------------------------------------------------------

# Only divisions flooded in either dataset
valid = df[
    (df["RAPIDA"] > 0) |
    (df["Yours"] > 0)
]

spearman_corr, spearman_p = spearmanr(
    valid["RAPIDA"],
    valid["Yours"]
)

pearson_corr, pearson_p = pearsonr(
    valid["RAPIDA"],
    valid["Yours"]
)

# -------------------------------------------------------------------
# Ranking comparison
# -------------------------------------------------------------------

df["RAPIDA_Rank"] = df["RAPIDA"].rank(ascending=False)
df["Your_Rank"] = df["Yours"].rank(ascending=False)

# -------------------------------------------------------------------
# Print report
# -------------------------------------------------------------------

print("\n" + "=" * 60)
print("🌊 DIVISION-LEVEL FLOOD VALIDATION REPORT")
print("Reference Dataset: RAPIDA")
print("=" * 60)

print(f"\nClassification Metrics")
print("-" * 40)

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"IoU           : {iou:.4f}")

print(f"\nCorrelation Metrics")
print("-" * 40)

print(f"Spearman Corr : {spearman_corr:.4f}")
print(f"Pearson Corr  : {pearson_corr:.4f}")

print(f"\nConfusion Matrix")
print("-" * 40)

print(f"TP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

print("\nTop Division Comparison")
print("-" * 40)

print(
    df[
        [
            "Division",
            "RAPIDA",
            "Yours",
            "RAPIDA_Rank",
            "Your_Rank",
        ]
    ]
    .sort_values("RAPIDA", ascending=False)
    .to_string(index=False)
)

# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------

print("\nGenerating plots...")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 12),
    facecolor="#0a0f1e"
)

axes = axes.flatten()

for ax in axes:
    ax.set_facecolor("#0d1526")

# ----------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Flood", "Flood"],
    yticklabels=["No Flood", "Flood"],
    ax=axes[0]
)

axes[0].set_title(
    "Division Classification",
    color="white"
)

# ----------------------------------------------------------
# Flood area comparison
# ----------------------------------------------------------

axes[1].scatter(
    df["RAPIDA"],
    df["Yours"]
)

for _, row in df.iterrows():
    axes[1].text(
        row["RAPIDA"],
        row["Yours"],
        row["Division"],
        fontsize=8
    )

axes[1].set_title(
    "Flood Area Correlation",
    color="white"
)

axes[1].set_xlabel("RAPIDA")
axes[1].set_ylabel("Your Detection")

# ----------------------------------------------------------
# Ranking comparison
# ----------------------------------------------------------

x = np.arange(len(df))

axes[2].bar(
    x - 0.2,
    df["RAPIDA_Rank"],
    width=0.4,
    label="RAPIDA"
)

axes[2].bar(
    x + 0.2,
    df["Your_Rank"],
    width=0.4,
    label="Yours"
)

axes[2].set_xticks(x)
axes[2].set_xticklabels(
    df["Division"],
    rotation=90
)

axes[2].legend()

axes[2].set_title(
    "Division Ranking Comparison",
    color="white"
)

# ----------------------------------------------------------
# Area comparison
# ----------------------------------------------------------

axes[3].barh(
    df["Division"],
    df["RAPIDA"],
    alpha=0.7,
    label="RAPIDA"
)

axes[3].barh(
    df["Division"],
    df["Yours"],
    alpha=0.7,
    label="Yours"
)

axes[3].legend()

axes[3].set_title(
    "Flood Area by Division",
    color="white"
)

plt.tight_layout()

# -------------------------------------------------------------------
# Save outputs
# -------------------------------------------------------------------

output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

report_path = output_dir / "rapida_validation_report.png"

plt.savefig(
    report_path,
    dpi=150,
    bbox_inches="tight"
)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "iou": round(float(iou), 4),
    "spearman_corr": round(float(spearman_corr), 4),
    "pearson_corr": round(float(pearson_corr), 4),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
}

metrics_path = output_dir / "rapida_validation_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Validation complete!")

print(f"\nSaved:")
print(f"   Report  : {report_path}")
print(f"   Metrics : {metrics_path}")