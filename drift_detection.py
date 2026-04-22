"""
PSI Drift Detection — Home Credit (baseline) vs German Credit (incoming batch).

Computes Population Stability Index (PSI) on conceptually aligned features
shared between the two datasets.  Results are logged to MLflow so that
german_credit.py can read them and decide whether to retrain.

PSI thresholds (standard industry rule):
  PSI < 0.10  → STABLE   (no action)
  PSI < 0.20  → WARN     (monitor closely)
  PSI >= 0.20 → DRIFT    (trigger retraining)

Run:
    python drift_detection.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow

from data_preprocess import get_home_credit_data


# ── PSI helpers ──────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between two 1-D distributions.

    For binary features (≤2 unique values) the proportion of the positive
    class is compared directly; for continuous features percentile-based
    binning is used so the bins are always equi-populated on the reference.
    """
    n_unique = len(np.unique(expected))

    if n_unique <= 2:
        p_exp = np.clip(expected.mean(), 1e-4, 1 - 1e-4)
        p_act = np.clip(actual.mean(),   1e-4, 1 - 1e-4)
        exp_pct = np.array([1 - p_exp, p_exp])
        act_pct = np.array([1 - p_act, p_act])
    else:
        breakpoints = np.linspace(0, 100, bins + 1)
        bin_edges   = np.unique(np.percentile(expected, breakpoints))
        if len(bin_edges) < 2:
            return 0.0
        exp_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        act_pct = np.histogram(actual,   bins=bin_edges)[0] / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 6)


def psi_status(psi: float) -> str:
    if psi >= 0.20:
        return "DRIFT"
    if psi >= 0.10:
        return "WARN"
    return "STABLE"


# ── load datasets ────────────────────────────────────────────────────────────

# Baseline: Home Credit training split
X_train_hc, _, _, _ = get_home_credit_data()

# Incoming batch: German Credit
GERMAN_PATH = "data/german_data.csv"
try:
    df_german = pd.read_csv(GERMAN_PATH)
    if "checking_status" not in df_german.columns:
        df_german.columns = [
            "checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings_status", "employment",
            "installment_commitment", "personal_status", "other_parties",
            "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class",
        ]
    print(f"German Credit loaded: {df_german.shape}")
except Exception as exc:
    print(f"ERROR loading {GERMAN_PATH}: {exc}")
    df_german = pd.DataFrame()


# ── build aligned feature frames ─────────────────────────────────────────────
# We align on 5 conceptually equivalent features:
#   credit_amount      ↔  AMT_CREDIT
#   age                ↔  |DAYS_BIRTH| / 365
#   installment_amount ↔  AMT_ANNUITY
#   income_proxy       ↔  AMT_INCOME_TOTAL
#   employment_years   ↔  |DAYS_EMPLOYED| / 365  (NaN → median)

df_baseline = pd.DataFrame({
    "credit_amount":      X_train_hc["AMT_CREDIT"],
    "age":                X_train_hc["DAYS_BIRTH"].abs() / 365,
    "installment_amount": X_train_hc["AMT_ANNUITY"],
    "income_proxy":       X_train_hc["AMT_INCOME_TOTAL"],
    "employment_years":   X_train_hc["DAYS_EMPLOYED"].abs() / 365,
})

if not df_german.empty:
    df_incoming = pd.DataFrame({
        "credit_amount":      df_german["credit_amount"],
        "age":                df_german["age"],
        "installment_amount": df_german["credit_amount"] / df_german["duration"].clip(lower=1),
        "income_proxy":       df_german["credit_amount"] * 3,          # rough proxy
        "employment_years":   df_german["employment"].map(            # ordinal decode
            lambda x: {"A11": 1, "A12": 3, "A13": 5, "A14": 8}.get(str(x), 3)
        ),
    })
else:
    # Fallback: use baseline itself (PSI will be ~0)
    df_incoming = df_baseline.copy()

df_baseline.fillna(df_baseline.median(), inplace=True)
df_incoming.fillna(df_incoming.median(), inplace=True)

FEATURES = ["credit_amount", "age", "installment_amount", "income_proxy", "employment_years"]

# ── z-score normalise using baseline statistics ───────────────────────────────
for col in FEATURES:
    mu  = df_baseline[col].mean()
    std = df_baseline[col].std() or 1.0
    df_baseline[col] = (df_baseline[col] - mu) / std
    df_incoming[col] = (df_incoming[col] - mu) / std


# ── compute PSI and log to MLflow ─────────────────────────────────────────────

psi_results   = []
drifted_feats = []
any_drift     = False

with mlflow.start_run(run_name="PSI_Drift_Detection") as run:
    for col in FEATURES:
        psi    = compute_psi(df_baseline[col].values, df_incoming[col].values)
        status = psi_status(psi)
        drifted = psi >= 0.20

        psi_results.append({
            "feature": col,
            "psi":     psi,
            "status":  status,
            "drift":   drifted,
        })

        if drifted:
            any_drift = True
            drifted_feats.append(col)

        mlflow.log_metric(f"psi_{col}", psi)

    mlflow.log_param("any_drift",        str(any_drift))
    mlflow.log_param("drifted_features", ",".join(drifted_feats) if drifted_feats else "none")
    mlflow.log_param("psi_threshold",    0.20)
    mlflow.log_param("n_features_checked", len(FEATURES))

    print(f"\n{'Feature':<25} {'PSI':>8}   Status")
    print("-" * 45)
    for r in psi_results:
        marker = " ◄ RETRAIN" if r["drift"] else ""
        print(f"{r['feature']:<25} {r['psi']:>8.4f}   {r['status']}{marker}")

    print(f"\nFeatures checked  : {FEATURES}")
    print(f"Drifted features  : {drifted_feats if drifted_feats else 'None'}")
    print(f"Overall drift     : {'YES — retraining will be triggered' if any_drift else 'NO — champion stays'}")
    print(f"\nMLflow run ID     : {run.info.run_id}")
