"""
=============================================================
Disaster Prediction - Random Forest Model Training
Predicts: Flood, Landslide, Fog, Warning, No Risk
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH    = "train_split.csv"
VAL_PATH      = "val_split.csv"
MODEL_OUTPUT  = "disaster_rf_model.pkl"
REPORT_OUTPUT = "model_report.png"

LABEL_MAP = {
    0: 'No Risk',
    1: 'Warning',
    2: 'Flood',
    3: 'Landslide',
    4: 'Fog'
}

FEATURE_COLS = [
    'Rainfall', 'Rain_3d', 'Rain_7d', 'Temperature', 'Humidity',
    'Latitude', 'Longitude', 'Elevation', 'Month', 'DayOfYear',
    'DayOfWeek', 'Is_Monsoon', 'Rain_Accel', 'Humid_Rain_Index',
    'Heat_Index', 'Elev_Risk'
]

# ─────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading train/val data")
print("=" * 60)

df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)

feat_train = [c for c in FEATURE_COLS if c in df_train.columns]
feat_val   = [c for c in FEATURE_COLS if c in df_val.columns]

X_train = df_train[feat_train]
y_train = df_train['Event_Label']

X_val   = df_val[feat_val]
y_val   = df_val['Event_Label']

print(f"  Training samples   : {len(X_train)}")
print(f"  Validation samples : {len(X_val)}")
print(f"  Features used      : {len(feat_train)}")

# ─────────────────────────────────────────────
# STEP 2: Train Random Forest
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training Random Forest model")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=10,            # prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced', # handles class imbalance (No Risk >> Flood)
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

print("  Training... (this may take a few seconds)")
model.fit(X_train, y_train)
print("  Training complete!")

# ─────────────────────────────────────────────
# STEP 3: Evaluate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_val)
acc    = accuracy_score(y_val, y_pred)

print(f"\n  Overall Accuracy: {acc * 100:.2f}%\n")

present_labels = sorted(y_val.unique())
target_names   = [LABEL_MAP[l] for l in present_labels]

print("  Per-class Report:")
print(classification_report(y_val, y_pred, labels=present_labels, target_names=target_names))

# ─────────────────────────────────────────────
# STEP 4: Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Importance (Top 10)")
print("=" * 60)

importances = pd.Series(model.feature_importances_, index=feat_train)
top10 = importances.sort_values(ascending=False).head(10)
print()
for feat, score in top10.items():
    bar = "█" * int(score * 100)
    print(f"  {feat:<22}: {score:.4f}  {bar}")

# ─────────────────────────────────────────────
# STEP 5: Save visuals
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving report chart")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Disaster Prediction Model - Random Forest Report', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_val, y_pred, labels=present_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n(Accuracy: {acc*100:.2f}%)', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

top_feats = importances.sort_values(ascending=True).tail(10)
colors = ['#2196F3' if v < 0.1 else '#FF5722' if v > 0.2 else '#4CAF50' for v in top_feats.values]
axes[1].barh(top_feats.index, top_feats.values, color=colors)
axes[1].set_title('Top 10 Feature Importances', fontsize=12)
axes[1].set_xlabel('Importance Score')
axes[1].axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='0.1 threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig(REPORT_OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {REPORT_OUTPUT}")

# ─────────────────────────────────────────────
# STEP 6: Save model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving trained model")
print("=" * 60)

with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump({
        'model'        : model,
        'feature_cols' : feat_train,
        'label_map'    : LABEL_MAP,
        'accuracy'     : acc
    }, f)

print(f"  Model saved : {MODEL_OUTPUT}")
print(f"  Accuracy    : {acc * 100:.2f}%")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"""
  Output files:
    disaster_rf_model.pkl  <- load this in your app/component
    model_report.png       <- open this to see charts

  Next step:
    Run predict.py to test predictions on new input data.
""")
