"""
PSI Drift Detection — simulates an economic shock and measures
feature distribution shift using Population Stability Index.
Computes PSI on both numeric and one-hot encoded features.
"""

import numpy as np
import pandas as pd
import warnings
import mlflow
warnings.filterwarnings("ignore")

from data_preprocess import get_home_credit_data, NUM_FEATURES, CAT_FEATURES, TARGET

def compute_psi(expected, actual, bins=10):
    """PSI between two distributions. Handles both continuous and binary features."""
    n_unique = len(np.unique(expected))

    if n_unique <= 2:
        # Binary feature: compare proportions directly
        p_exp = np.clip(expected.mean(), 1e-4, 1 - 1e-4)
        p_act = np.clip(actual.mean(),   1e-4, 1 - 1e-4)
        expected_perc = np.array([1 - p_exp, p_exp])
        actual_perc   = np.array([1 - p_act, p_act])
    else:
        breakpoints = np.linspace(0, 100, bins + 1)
        bin_edges = np.unique(np.percentile(expected, breakpoints))
        if len(bin_edges) < 2:
            return 0.0
        expected_perc = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        actual_perc   = np.histogram(actual,   bins=bin_edges)[0] / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-4, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-4, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)

def psi_status(psi):
    if psi > 0.2:  return "DRIFT"
    if psi > 0.1:  return "WARN"
    return "STABLE"

# Prepare original data (Baseline from Home Credit)
X_train, _, _, _ = get_home_credit_data()
original = X_train.copy()

# Load real incoming data (German Credit conceptual batch)
GERMAN_PATH = "data/german_data.csv"
try:
    df_german = pd.read_csv(GERMAN_PATH)
    # Fix col names if they don't exist
    if "checking_status" not in df_german.columns:
        df_german.columns = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class"
        ]
except Exception as e:
    print(f"Error loading {GERMAN_PATH}: {e}")
    df_german = pd.DataFrame()

# Create conceptual aligned DataFrames for PSI using real shared features
df_old = pd.DataFrame()
df_old["credit_amount"] = original["AMT_CREDIT"]
df_old["age"] = original["DAYS_BIRTH"].abs() / 365
df_old["installment_amount"] = original["AMT_ANNUITY"]

df_new = pd.DataFrame()
if not df_german.empty:
    df_new["credit_amount"] = df_german["credit_amount"]
    df_new["age"] = df_german["age"]
    df_new["installment_amount"] = df_german["credit_amount"] / df_german["duration"].clip(lower=1)
else:
    df_new["credit_amount"] = df_old["credit_amount"]
    df_new["age"] = df_old["age"]
    df_new["installment_amount"] = df_old["installment_amount"]


df_old.fillna(df_old.median(), inplace=True)
df_new.fillna(df_new.median(), inplace=True)

selected_features = [
    "credit_amount", 
    "age", 
    "installment_amount"
]


for col in selected_features:
    old_mean = df_old[col].mean()
    old_std = df_old[col].std() if df_old[col].std() != 0 else 1.0
    
    df_old[col] = (df_old[col] - old_mean) / old_std
    df_new[col] = (df_new[col] - old_mean) / old_std

original_selected = df_old
shocked_selected  = df_new

psi_results = []
any_drift   = False

with mlflow.start_run(run_name="PSI_Drift_Detection") as run:
    for col in selected_features:
        psi    = compute_psi(original_selected[col].values, shocked_selected[col].values)
        status = psi_status(psi)
        psi_results.append({"feature": col, "psi": round(psi, 4), "status": status, "drift": psi > 0.2})
        if psi > 0.2:
            any_drift = True
            
        mlflow.log_metric(f"psi_{col}", psi)
        
    drifted_features = [r["feature"] for r in psi_results if r["drift"]]
    mlflow.log_param("any_drift", any_drift)
    mlflow.log_param("drifted_features", ",".join(drifted_features))

print(f"{'Feature':<40} {'PSI':>8}  Status")
print("-" * 58)
for r in psi_results:
    if r["psi"] >= 0:
        print(f"{r['feature']:<40} {r['psi']:>8.4f}  {r['status']}")
print(f"Checked features: {selected_features}")
print(f"Drifted features: {drifted_features}")
print(f"Overall drift: {'YES' if any_drift else 'NO'}")