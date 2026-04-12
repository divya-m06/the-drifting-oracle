"""
FILE 3 (IMPROVED): PSI Drift Detection
=======================================
IMPROVEMENTS:
  ✅ Also checks engineered features (DEBT_TO_INCOME, ANNUITY_TO_INCOME, etc.)
  ✅ KS-test added alongside PSI for richer drift signal
  ✅ More granular economic shock scenarios
  ✅ Saves psi_metrics.csv with both PSI and KS statistics

PSI Guide:
  PSI < 0.1   → 🟢 STABLE  (no significant shift)
  PSI 0.1-0.2 → 🟡 WARNING (moderate shift)
  PSI > 0.2   → 🔴 DRIFT   (significant — triggers SVM retrain in File 4)

Run AFTER: 1_data_preprocess.py
Saves: drift_data.pkl, psi_metrics.csv
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Load processed data from Step 1 ──────────────────────────────────────────
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

df           = data["df"]
NUM_FEATURES = data["NUM_FEATURES"]
ALL_FEATURES = data.get("ALL_FEATURES", NUM_FEATURES)   # falls back if old pkl
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index between two distributions.
    Uses percentile-based binning on the expected (training) distribution.

    PSI < 0.1   → Stable
    PSI 0.1-0.2 → Warning
    PSI > 0.2   → Drift
    """
    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges   = np.unique(np.percentile(expected, breakpoints))

    if len(bin_edges) < 2:
        return 0.0   # degenerate distribution — treat as stable

    exp_counts = np.histogram(expected, bins=bin_edges)[0]
    act_counts = np.histogram(actual,   bins=bin_edges)[0]

    exp_perc = exp_counts / len(expected)
    act_perc = act_counts / len(actual)

    # Replace zeros to avoid log(0)
    exp_perc = np.where(exp_perc == 0, 1e-4, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-4, act_perc)

    psi = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
    return float(psi)


def compute_ks(expected, actual):
    """
    Kolmogorov-Smirnov two-sample test.
    Returns (ks_statistic, p_value).
    A small p-value (< 0.05) means the distributions differ significantly.
    """
    ks_stat, p_val = stats.ks_2samp(expected, actual)
    return round(float(ks_stat), 4), round(float(p_val), 6)


def psi_status(psi):
    if psi > 0.2:  return "🔴 DRIFT"
    if psi > 0.1:  return "🟡 WARN"
    return "🟢 STABLE"


def ks_status(p_val):
    if p_val < 0.01:  return "🔴 DRIFT"
    if p_val < 0.05:  return "🟡 WARN"
    return "🟢 STABLE"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: SIMULATE ECONOMIC SHOCK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 7: PSI + KS Drift Detection (Simulated Economic Shock)")
print("=" * 65)

drift_df = df[ALL_FEATURES].copy()

# Core financial shock (same as original)
if "AMT_INCOME_TOTAL" in drift_df.columns:
    drift_df["AMT_INCOME_TOTAL"] *= 0.7    # income drops 30%
if "AMT_CREDIT" in drift_df.columns:
    drift_df["AMT_CREDIT"]       *= 1.3    # loan amounts increase 30%
if "AMT_ANNUITY" in drift_df.columns:
    drift_df["AMT_ANNUITY"]      *= 1.2    # EMI increases 20%
# DAYS_EMPLOYED, DAYS_BIRTH unchanged → should show STABLE

# Engineered features will shift automatically because they depend on above
if "DEBT_TO_INCOME" in drift_df.columns:
    drift_df["DEBT_TO_INCOME"]    = (
        drift_df.get("AMT_CREDIT",  df["AMT_CREDIT"]  * 1.3) /
        (drift_df.get("AMT_INCOME_TOTAL", df["AMT_INCOME_TOTAL"] * 0.7) + 1)
    )
if "ANNUITY_TO_INCOME" in drift_df.columns:
    drift_df["ANNUITY_TO_INCOME"] = (
        drift_df.get("AMT_ANNUITY", df["AMT_ANNUITY"] * 1.2) /
        (drift_df.get("AMT_INCOME_TOTAL", df["AMT_INCOME_TOTAL"] * 0.7) + 1)
    )

print("\nSimulated economic shock applied:")
print("  AMT_INCOME_TOTAL × 0.7  (income drops 30%)")
print("  AMT_CREDIT       × 1.3  (loans increase 30%)")
print("  AMT_ANNUITY      × 1.2  (EMI increases 20%)")
print("  DAYS_EMPLOYED    —      (unchanged → should be STABLE)")
print("  DAYS_BIRTH       —      (unchanged → should be STABLE)")
print("  Engineered ratios recalculated from shifted raw values\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7b: COMPUTE PSI + KS PER FEATURE
# ══════════════════════════════════════════════════════════════════════════════
print(f"  {'Feature':<25} {'PSI':>8}  PSI Status    {'KS':>8}  {'p-value':>10}  KS Status")
print("  " + "─" * 80)

psi_results = []
any_drift   = False

for col in ALL_FEATURES:
    if col not in drift_df.columns or col not in df.columns:
        continue

    expected = df[col].dropna().values
    actual   = drift_df[col].dropna().values

    psi    = compute_psi(expected, actual)
    ks, pv = compute_ks(expected, actual)
    p_stat = psi_status(psi)
    k_stat = ks_status(pv)
    is_drift = psi > 0.2

    print(f"  {col:<25} {psi:>8.4f}  {p_stat:<12}  {ks:>8.4f}  {pv:>10.6f}  {k_stat}")

    psi_results.append({
        "feature":    col,
        "psi":        round(psi, 4),
        "ks_stat":    ks,
        "ks_pvalue":  pv,
        "psi_status": p_stat,
        "ks_status":  k_stat,
        "status":     p_stat,   # kept for backward compat with File 4
        "drift":      is_drift,
    })
    if is_drift:
        any_drift = True

print("\n  PSI Guide : < 0.1 = Stable | 0.1–0.2 = Warning | > 0.2 = Drift")
print("  KS Guide  : p > 0.05 = Stable | 0.01–0.05 = Warning | < 0.01 = Drift")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
drifted_features = [r["feature"] for r in psi_results if r["drift"]]
warned_features  = [r["feature"] for r in psi_results if 0.1 < r["psi"] <= 0.2]

print(f"\n{'─' * 65}")
print(f"  Total features checked  : {len(psi_results)}")
print(f"  🟡 Warning (0.1–0.2 PSI): {len(warned_features)}  {warned_features}")
print(f"  🔴 Drifted  (> 0.2 PSI) : {len(drifted_features)}  {drifted_features}")
print(f"  Overall drift flag      : "
      f"{'⚠️  YES — SVM retrain will trigger in File 4' if any_drift else '✅  NO — models remain stable'}")
print(f"{'─' * 65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
psi_df = pd.DataFrame(psi_results)
psi_df.to_csv("psi_metrics.csv", index=False)

with open("drift_data.pkl", "wb") as f:
    pickle.dump({
        "psi_results":      psi_results,
        "psi_df":           psi_df,
        "drift_df":         drift_df,
        "original_df":      df[ALL_FEATURES],
        "any_drift":        any_drift,
        "drifted_features": drifted_features,
        "warned_features":  warned_features,
    }, f)

print("✅ Saved psi_metrics.csv")
print("✅ Saved drift_data.pkl — run 4_german_credit.py next")
print("\n" + "=" * 65)
print("DRIFT DETECTION COMPLETE")
print("=" * 65)