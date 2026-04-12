"""
FILE 1 (IMPROVED): Data Preprocessing — Home Credit Dataset
============================================================
IMPROVEMENTS:
  ✅ Median imputation instead of dropping NaN rows
  ✅ IQR-based outlier capping (Winsorization)
  ✅ Feature engineering: debt-to-income, annuity ratio, employment ratio
  ✅ Logs shape/class balance at every step

Run FIRST in the pipeline.
Saves: processed_data.pkl
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/application_train.csv"   # adjust if your path differs
TARGET_COL   = "TARGET"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

NUM_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
]
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Load Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Raw shape            : {df.shape}")
print(f"  Target distribution  :\n{df[TARGET_COL].value_counts()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# Debt-to-income ratio
df["DEBT_TO_INCOME"]    = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

# Annuity-to-income ratio (monthly payment burden)
df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

# Credit-to-annuity ratio (loan term proxy)
df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

# Employment ratio — fraction of life spent employed (DAYS are negative)
df["EMPLOY_TO_AGE"]     = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] - 1)

# Age in years (DAYS_BIRTH is negative)
df["AGE_YEARS"]         = (-df["DAYS_BIRTH"]) / 365

ENGINEERED = [
    "DEBT_TO_INCOME", "ANNUITY_TO_INCOME",
    "CREDIT_TO_ANNUITY", "EMPLOY_TO_AGE", "AGE_YEARS"
]

ALL_FEATURES = NUM_FEATURES + ENGINEERED
print(f"  Engineered features  : {ENGINEERED}")
print(f"  Total features used  : {len(ALL_FEATURES)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: MISSING VALUE IMPUTATION (Median)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Missing Value Imputation")
print("=" * 60)

missing_before = df[ALL_FEATURES].isnull().sum()
print("  Missing values before imputation:")
print(missing_before[missing_before > 0].to_string() if missing_before.any() else "  None")

medians = df[ALL_FEATURES].median()
df[ALL_FEATURES] = df[ALL_FEATURES].fillna(medians)

print(f"\n  Missing values after imputation: {df[ALL_FEATURES].isnull().sum().sum()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: OUTLIER CAPPING (IQR Winsorization)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: Outlier Capping (IQR Winsorization at 1.5×IQR)")
print("=" * 60)

for col in ALL_FEATURES:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    if clipped > 0:
        print(f"  {col:<25}: {clipped:>6} values capped  [{lo:.1f}, {hi:.1f}]")

print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: Train / Test Split")
print("=" * 60)

X = df[ALL_FEATURES]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,
)

print(f"  Train shape : {X_train.shape}   Default rate: {y_train.mean():.4f}")
print(f"  Test shape  : {X_test.shape}    Default rate: {y_test.mean():.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SCALING (StandardScaler)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6: Feature Scaling (StandardScaler)")
print("=" * 60)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train mean (post-scale, col 0): {X_train_scaled[:, 0].mean():.6f}")
print(f"  Train std  (post-scale, col 0): {X_train_scaled[:, 0].std():.6f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open("processed_data.pkl", "wb") as f:
    pickle.dump({
        "df":             df,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "scaler":         scaler,
        "NUM_FEATURES":   NUM_FEATURES,
        "ALL_FEATURES":   ALL_FEATURES,
        "ENGINEERED":     ENGINEERED,
        "medians":        medians,
    }, f)

print("✅ Saved processed_data.pkl  — run 2_train_models.py next")
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)