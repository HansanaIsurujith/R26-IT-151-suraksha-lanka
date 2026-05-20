import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_CLEAN_PATH = "dataset_clean.csv"
GAMPAHA_GRID_PATH  = "gampaha_grid_dataset.csv"
OUTPUT_TRAIN_PATH  = "preprocessed_train.csv"
OUTPUT_GRID_PATH   = "preprocessed_grid.csv"
RANDOM_STATE         = 42

# ─────────────────────────────────────────────
# STEP 1: Load datasets
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading datasets")
print("=" * 60)

df_clean = pd.read_csv(DATASET_CLEAN_PATH)
df_grid  = pd.read_csv(GAMPAHA_GRID_PATH)

print(f"[dataset_clean]  Shape: {df_clean.shape}")
print(f"[gampaha_grid]   Shape: {df_grid.shape}")

# ─────────────────────────────────────────────
# STEP 2: Parse Date column (YYYYMMDD → features)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Parsing Date column")
print("=" * 60)

def parse_date_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
    df['Month']      = df['Date'].dt.month
    df['DayOfYear']  = df['Date'].dt.dayofyear
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    # Monsoon flag — SW monsoon: May-Sep (5-9), NE monsoon: Dec-Feb (12,1,2)
    df['Is_Monsoon'] = df['Month'].apply(
        lambda m: 1 if m in [5, 6, 7, 8, 9, 12, 1, 2] else 0
    )
    df = df.drop(columns=['Date'])
    return df

df_clean = parse_date_features(df_clean)
df_grid  = parse_date_features(df_grid)
print("✓ Extracted: Month, DayOfYear, DayOfWeek, Is_Monsoon")

# ─────────────────────────────────────────────
# STEP 3: Drop/fix low-quality columns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Dropping low-quality columns")
print("=" * 60)

# gampaha_grid: Slope is all zeros — useless
if df_grid['Slope'].nunique() == 1:
    df_grid = df_grid.drop(columns=['Slope'])
    print("✓ Dropped 'Slope' from gampaha_grid (all zeros — no variance)")

# dataset_clean: Drop Location (encoded as lat/lon already)
if 'Location' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Location'])
    print("✓ Dropped 'Location' from dataset_clean (redundant with Lat/Lon)")

# ─────────────────────────────────────────────
# STEP 4: Synthetic Landslide & Fog events
# (dataset_clean has only Flood/Warning/No Risk)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Generating synthetic Landslide & Fog events")
print("=" * 60)

def add_synthetic_events(df):
    """
    Rules derived from domain knowledge for Sri Lanka:

    LANDSLIDE:
      - Rain_7d > 120mm  (sustained heavy rainfall)
      - Elevation > 15m  (hilly terrain)
      - Humidity > 80%
      → Overrides existing label if conditions met

    FOG:
      - Temperature < 26°C
      - Humidity > 82%
      - Rainfall < 2mm  (calm, moist conditions)
      - Is_Monsoon == 0  (inter-monsoon / dry season mornings)
      → Applied only to No Risk rows
    """
    df = df.copy()
    original_dist = df['Event'].value_counts().to_dict()

    # --- Landslide rule ---
    landslide_mask = (
        (df['Rain_7d'] > 120) &
        (df['Elevation'] > 15) &
        (df['Humidity'] > 80)
    )
    df.loc[landslide_mask, 'Event'] = 'Landslide'

    # --- Fog rule (only on No Risk rows) ---
    fog_mask = (
        (df['Event'] == 'No Risk') &
        (df['Temperature'] < 26.0) &
        (df['Humidity'] > 82) &
        (df['Rainfall'] < 2.0) &
        (df['Is_Monsoon'] == 0)
    )
    df.loc[fog_mask, 'Event'] = 'Fog'

    new_dist = df['Event'].value_counts().to_dict()
    print(f"\n  Before → After synthetic injection:")
    all_events = set(list(original_dist.keys()) + list(new_dist.keys()))
    for ev in sorted(all_events):
        before = original_dist.get(ev, 0)
        after  = new_dist.get(ev, 0)
        diff   = after - before
        sign   = f"+{diff}" if diff >= 0 else str(diff)
        print(f"    {ev:<12}: {before:>4} → {after:>4}  ({sign})")
    return df

df_clean = add_synthetic_events(df_clean)

# ─────────────────────────────────────────────
# STEP 5: Engineer additional features
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature engineering")
print("=" * 60)

def engineer_features(df):
    df = df.copy()

    # Rainfall acceleration (how fast rain is building up)
    df['Rain_Accel']      = df['Rain_3d'] - (df['Rain_7d'] / 7) * 3
    df['Rain_Accel']      = df['Rain_Accel'].round(4)

    # Humidity-Rainfall interaction (saturation index)
    df['Humid_Rain_Index'] = (df['Humidity'] * df['Rain_3d']) / 100
    df['Humid_Rain_Index'] = df['Humid_Rain_Index'].round(4)

    # Heat index proxy (Temperature × Humidity)
    df['Heat_Index']      = (df['Temperature'] * df['Humidity']) / 100
    df['Heat_Index']      = df['Heat_Index'].round(4)

    # Elevation risk band (lower = higher flood risk)
    df['Elev_Risk']       = np.where(df['Elevation'] <= 9, 2,
                            np.where(df['Elevation'] <= 14, 1, 0))

    return df

df_clean = engineer_features(df_clean)

# Apply same features to grid (no River_Distance col in clean, add dummy if needed)
df_grid  = engineer_features(df_grid)

print("✓ Added features: Rain_Accel, Humid_Rain_Index, Heat_Index, Elev_Risk")

# ─────────────────────────────────────────────
# STEP 6: Label Encoding for Event
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Label encoding")
print("=" * 60)

# Fixed mapping so it's reproducible
EVENT_LABEL_MAP = {
    'No Risk'   : 0,
    'Warning'   : 1,
    'Flood'     : 2,
    'Landslide' : 3,
    'Fog'       : 4,
}

df_clean['Event_Label'] = df_clean['Event'].map(EVENT_LABEL_MAP)
df_grid['Event_Label']  = df_grid['Event'].map(EVENT_LABEL_MAP).fillna(0).astype(int)

print("  Label mapping:")
for k, v in EVENT_LABEL_MAP.items():
    print(f"    {v} → {k}")

# ─────────────────────────────────────────────
# STEP 7: Normalize numerical features
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: MinMax scaling numerical features")
print("=" * 60)

SCALE_COLS = [
    'Rainfall', 'Rain_3d', 'Rain_7d', 'Temperature', 'Humidity',
    'Elevation', 'Latitude', 'Longitude',
    'Rain_Accel', 'Humid_Rain_Index', 'Heat_Index'
]

# Only scale columns that exist in each df
scale_cols_clean = [c for c in SCALE_COLS if c in df_clean.columns]
scale_cols_grid  = [c for c in SCALE_COLS + ['River_Distance'] if c in df_grid.columns]

scaler_clean = MinMaxScaler()
scaler_grid  = MinMaxScaler()

df_clean[scale_cols_clean] = scaler_clean.fit_transform(df_clean[scale_cols_clean])
df_grid[scale_cols_grid]   = scaler_grid.fit_transform(df_grid[scale_cols_grid])

print(f"✓ Scaled {len(scale_cols_clean)} columns in dataset_clean")
print(f"✓ Scaled {len(scale_cols_grid)} columns in gampaha_grid")

# ─────────────────────────────────────────────
# STEP 8: Train / Validation split (dataset_clean only)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Train/Val split (80/20, stratified)")
print("=" * 60)

feature_cols = [c for c in df_clean.columns if c not in ['Event', 'Event_Label']]
X = df_clean[feature_cols]
y = df_clean['Event_Label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Training samples   : {len(X_train)}")
print(f"  Validation samples : {len(X_val)}")
print(f"  Feature count      : {len(feature_cols)}")
print(f"  Features           : {feature_cols}")

# ─────────────────────────────────────────────
# STEP 9: Save outputs
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving preprocessed files")
print("=" * 60)

# Full preprocessed training data
df_clean.to_csv(OUTPUT_TRAIN_PATH, index=False)
print(f"✓ Saved: {OUTPUT_TRAIN_PATH}  ({df_clean.shape})")

# Save train/val splits separately
X_train_out = X_train.copy(); X_train_out['Event_Label'] = y_train
X_val_out   = X_val.copy();   X_val_out['Event_Label']   = y_val
X_train_out.to_csv("train_split.csv", index=False)
X_val_out.to_csv("val_split.csv",   index=False)
print(f"✓ Saved: train_split.csv  ({X_train_out.shape})")
print(f"✓ Saved: val_split.csv    ({X_val_out.shape})")

# Grid dataset (for spatial inference / map prediction)
df_grid.to_csv(OUTPUT_GRID_PATH, index=False)
print(f"✓ Saved: {OUTPUT_GRID_PATH}  ({df_grid.shape})")

# ─────────────────────────────────────────────
# STEP 10: Final summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("\n[Training Dataset - preprocessed_train.csv]")
print(f"  Total rows     : {len(df_clean)}")
print(f"  Total features : {len(feature_cols)}")
print("\n  Event distribution:")
event_counts = df_clean['Event'].value_counts()
for evt, cnt in event_counts.items():
    pct = cnt / len(df_clean) * 100
    bar = "█" * int(pct / 2)
    print(f"    {evt:<12}: {cnt:>4}  ({pct:5.1f}%)  {bar}")

print("\n[Gampaha Grid - preprocessed_grid.csv]")
print(f"  Total rows     : {len(df_grid)}")
print(f"  Purpose        : Spatial inference / map-based prediction")

print("\n[RECOMMENDATION]")
print("  • Use preprocessed_train.csv  → Train your model (RF/XGBoost/LSTM)")
print("  • Use train_split.csv         → Fit the model")
print("  • Use val_split.csv           → Evaluate performance")
print("  • Use preprocessed_grid.csv   → Run spatial predictions for map")
print("\n  ⚠ Note: Landslide & Fog are SYNTHETIC rules.")
print("    Validate with domain experts before production use.")
print("=" * 60)
print("Preprocessing complete ✓")
