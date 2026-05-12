"""
validate_adam.py
Pixel-level flood map validation against WFP ADAM ground truth.
Run from project root: python validate_adam.py
Team Trivia · University of Moratuwa · 2026
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Setup paths ───────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask as rasterio_mask
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import mapping, shape
from sklearn.metrics import confusion_matrix

# ── Config ────────────────────────────────────────────────────
ADAM_KEY        = 'FL-20251201-LKA-00.tiff'   # in Supabase flood-data bucket
FLOOD_RASTER    = project_root / 'interface' / 'backend' / 'outputs' / 'flood_extent_4326.tif'
OUTPUT_DIR      = project_root / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Import Supabase helpers ───────────────────────────────────
sys.path.insert(0, str(project_root / 'modules' / 'flood_detection'))
from config import FloodDetectionConfig, get_file_from_supabase

config = FloodDetectionConfig()

# ── Step 1: Load ADAM from Supabase ──────────────────────────
print("Step 1: Loading WFP ADAM from Supabase...")

temp_dir = tempfile.mkdtemp()

try:
    adam_bytes = get_file_from_supabase(config.SUPABASE_BUCKET, ADAM_KEY)
    adam_temp  = os.path.join(temp_dir, ADAM_KEY)
    with open(adam_temp, 'wb') as f:
        f.write(adam_bytes)
    print(f"   ✓ Downloaded {ADAM_KEY}")
except Exception as e:
    print(f"   ❌ Cannot download ADAM: {e}")
    print(f"   Upload {ADAM_KEY} to Supabase bucket '{config.SUPABASE_BUCKET}' first.")
    shutil.rmtree(temp_dir)
    sys.exit(1)

# ── Step 2: Load and binarise ADAM ───────────────────────────
print("\nStep 2: Loading and binarising ADAM...")

with rasterio.open(adam_temp) as src:
    print(f"   CRS:   {src.crs}")
    print(f"   Size:  {src.width} x {src.height}")
    adam_data     = src.read(1)
    adam_crs      = src.crs
    adam_transform= src.transform
    adam_profile  = src.profile.copy()
    nodata_val    = src.nodata

print(f"   Unique values: {np.unique(adam_data)[:10]}")

adam_valid  = (adam_data != nodata_val) if nodata_val is not None else np.ones_like(adam_data, dtype=bool)
# CORRECT — only values 1, 2, 3 are flood in WFP ADAM products
# 1 = low flood severity
# 2 = medium flood severity  
# 3 = high flood severity
# 9, 10, 11 = population/infrastructure categories, NOT flood extent
adam_binary = np.where(adam_valid & (adam_data >= 1) & (adam_data <= 3), 1, 0).astype(np.uint8)
print(f"   ADAM flooded pixels: {np.sum(adam_binary == 1):,} ({np.mean(adam_binary)*100:.2f}%)")

# ── Step 3: Reproject ADAM to EPSG:4326 ──────────────────────
print("\nStep 3: Reprojecting ADAM to EPSG:4326...")

transform_4326, w4326, h4326 = calculate_default_transform(
    adam_crs, 'EPSG:4326',
    adam_data.shape[1], adam_data.shape[0],
    *rasterio.transform.array_bounds(adam_data.shape[0], adam_data.shape[1], adam_transform)
)

adam_4326 = np.zeros((h4326, w4326), dtype=np.uint8)
reproject(
    source      = adam_binary.astype(np.float32),
    destination = adam_4326,
    src_transform = adam_transform,
    src_crs       = adam_crs,
    dst_transform = transform_4326,
    dst_crs       = 'EPSG:4326',
    resampling    = Resampling.nearest
)
print(f"   ADAM 4326 shape: {adam_4326.shape}")
print(f"   ADAM 4326 flooded: {np.sum(adam_4326 == 1):,}")

# ── Step 4: Load your flood extent raster ────────────────────
print(f"\nStep 4: Loading your flood extent raster...")
print(f"   Path: {FLOOD_RASTER}")

if not FLOOD_RASTER.exists():
    print(f"   ❌ File not found: {FLOOD_RASTER}")
    print("   Run the flood detection pipeline first.")
    shutil.rmtree(temp_dir)
    sys.exit(1)

with rasterio.open(FLOOD_RASTER) as src:
    your_flood     = src.read(1)
    your_transform = src.transform
    your_crs       = src.crs
    print(f"   Shape: {your_flood.shape}")
    print(f"   Unique values: {np.unique(your_flood)}")

your_binary = (your_flood > 0).astype(np.uint8)
print(f"   Your flooded pixels: {np.sum(your_binary == 1):,} ({np.mean(your_binary)*100:.2f}%)")

# ── Step 5: Align grids ───────────────────────────────────────
print("\nStep 5: Aligning ADAM to your raster grid...")

adam_resampled = np.zeros_like(your_binary, dtype=np.uint8)
reproject(
    source        = adam_4326.astype(np.float32),
    destination   = adam_resampled,
    src_transform = transform_4326,
    src_crs       = 'EPSG:4326',
    dst_transform = your_transform,
    dst_crs       = your_crs,
    resampling    = Resampling.nearest
)

print(f"   Aligned shape:     {adam_resampled.shape}")
print(f"   Your flooded:      {np.sum(your_binary == 1):,}")
print(f"   ADAM flooded:      {np.sum(adam_resampled == 1):,}")
print(f"   Overlap (TP):      {np.sum((your_binary == 1) & (adam_resampled == 1)):,}")

# ── Step 6: Compute metrics ───────────────────────────────────
print("\nStep 6: Computing accuracy metrics...")

y_true = adam_resampled.flatten()
y_pred = your_binary.flatten()

cm          = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy  = (tp + tn) / (tp + tn + fp + fn) * 100
precision = tp / (tp + fp)  if (tp + fp) > 0 else 0
recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

print("\n" + "=" * 55)
print("     🌊 PIXEL-LEVEL FLOOD VALIDATION REPORT")
print("     Reference: WFP ADAM FL-20251201-LKA-00")
print("=" * 55)
print(f"  Overall Accuracy  : {accuracy:6.2f}%")
print(f"  Precision         : {precision:6.4f}")
print(f"  Recall            : {recall:6.4f}")
print(f"  F1 Score          : {f1:6.4f}")
print(f"  IoU (Jaccard)     : {iou:6.4f}")
print("-" * 55)
print(f"  True Positives    : {tp:10,}  (correctly detected flood)")
print(f"  True Negatives    : {tn:10,}  (correctly detected non-flood)")
print(f"  False Positives   : {fp:10,}  (false alarm)")
print(f"  False Negatives   : {fn:10,}  (missed flood)")
print("=" * 55)

# ── Step 7: Generate plots ────────────────────────────────────
print("\nStep 7: Generating validation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0a0f1e')
axes = axes.flatten()

for ax in axes:
    ax.set_facecolor('#0d1526')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2d45')

# Confusion matrix
sns.heatmap(
    cm, annot=True, fmt=',d', cmap='Blues',
    xticklabels=['No Flood', 'Flood'],
    yticklabels=['No Flood', 'Flood'],
    ax=axes[0], linewidths=0.5,
    annot_kws={'size': 13, 'color': 'white', 'fontweight': 'bold'}
)
axes[0].set_title('Confusion Matrix\nvs WFP ADAM', color='white', fontsize=11)
axes[0].set_ylabel('WFP ADAM (Ground Truth)', color='#94a3b8')
axes[0].set_xlabel('Your Detection (SW-GAT)', color='#94a3b8')
axes[0].tick_params(colors='white')

# Metrics bar chart
metrics_vals = {
    'Accuracy\n(%)': accuracy,
    'Precision\n(×100)': precision * 100,
    'Recall\n(×100)': recall * 100,
    'F1 Score\n(×100)': f1 * 100,
    'IoU\n(×100)': iou * 100
}
colors = ['#38bdf8', '#818cf8', '#22c55e', '#f59e0b', '#2dd4bf']
bars = axes[1].bar(metrics_vals.keys(), metrics_vals.values(), color=colors)
axes[1].set_ylim(0, 115)
axes[1].set_title('Performance Metrics', color='white', fontsize=11)
axes[1].set_ylabel('Score', color='#94a3b8')
for bar, val in zip(bars, metrics_vals.values()):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f'{val:.1f}', ha='center', va='bottom',
        color='white', fontsize=9, fontweight='bold'
    )

# ADAM map
axes[2].imshow(adam_4326, cmap='Blues', interpolation='nearest')
axes[2].set_title('WFP ADAM Ground Truth\n(Blue = Flood)', color='white', fontsize=11)
axes[2].axis('off')

# Your map
axes[3].imshow(your_binary, cmap='Reds', interpolation='nearest')
axes[3].set_title('Your Detection (SW-GAT)\n(Red = Flood)', color='white', fontsize=11)
axes[3].axis('off')

plt.suptitle(
    f'Pixel-Level Flood Validation — Gampaha District · Dec 2025\n'
    f'Accuracy: {accuracy:.1f}%  |  F1: {f1:.3f}  |  IoU: {iou:.3f}',
    color='white', fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()

report_path = OUTPUT_DIR / 'validation_report_adam.png'
plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='#0a0f1e')
plt.show()
print(f"   ✓ Saved: {report_path}")

# ── Step 8: Save metrics JSON ─────────────────────────────────
metrics_out = {
    'reference':        'WFP ADAM FL-20251201-LKA-00',
    'accuracy':         round(accuracy, 2),
    'precision':        round(precision, 4),
    'recall':           round(recall, 4),
    'f1_score':         round(f1, 4),
    'iou':              round(iou, 4),
    'true_positives':   int(tp),
    'true_negatives':   int(tn),
    'false_positives':  int(fp),
    'false_negatives':  int(fn),
}

metrics_path = OUTPUT_DIR / 'validation_metrics_adam.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"   ✓ Metrics saved: {metrics_path}")

# Cleanup
shutil.rmtree(temp_dir, ignore_errors=True)

print("\n✅ ADAM validation complete!")
print(f"   Plot:    {report_path}")
print(f"   Metrics: {metrics_path}")