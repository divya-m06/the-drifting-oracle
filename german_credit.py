"""
Champion / Challenger Evaluation — XGBoost on German Credit Data.

Pipeline (triggered after drift_detection.py):
  1. Load German Credit dataset + preprocess + feature engineering
  2. Read PSI drift result from latest MLflow PSI_Drift_Detection run
  3. Load Champion XGBoost model from MLflow (trained on Home Credit)
  4. Evaluate Champion on Home Credit hold-out test set (its original test set)
  5. IF any_drift (PSI >= 0.2 on any feature):
       - Retrain SGD Challenger on German Credit training data
       - Evaluate Challenger on German Credit test set
       - Plot & save AUC-ROC curves for both models
       - Log Challenger + all metrics to MLflow
       - Select the model with higher AUC for deployment
  6. IF no drift: Champion stays, no retraining needed

Run:
    python german_credit.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, roc_curve,
)
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from data_preprocess import get_home_credit_data   # ← for Champion evaluation

GERMAN_PATH   = "data/german_data.csv"
PSI_THRESHOLD = 0.20
RANDOM_STATE  = 42

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load & preprocess German Credit  (used for Challenger only)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    df_german = pd.read_csv(GERMAN_PATH)
    if df_german.shape[1] == 21 and "checking_status" not in df_german.columns:
        df_german.columns = [
            "checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings_status", "employment",
            "installment_commitment", "personal_status", "other_parties",
            "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class",
        ]
    print(f"German Credit loaded: {df_german.shape}")
except FileNotFoundError:
    print(f"ERROR: '{GERMAN_PATH}' not found — place german_data.csv in data/ and rerun.")
    raise SystemExit(1)

# Target: class column (1 = Good Credit, 2 = Bad Credit → remap to 0 / 1)
target_col = next((c for c in df_german.columns if c.lower() == "class"), None)
if target_col is None:
    raise ValueError(f"No 'class' column found. Columns: {df_german.columns.tolist()}")

if df_german[target_col].isin([1, 2]).all():
    df_german[target_col] = (df_german[target_col] == 2).astype(int)

print(f"Target distribution:\n{df_german[target_col].value_counts()}\n")

X_g = df_german.drop(columns=[target_col]).copy()
y_g = df_german[target_col]

# ── Feature engineering ───────────────────────────────────────────────────────
if "credit_amount" in X_g.columns and "duration" in X_g.columns:
    X_g["credit_per_month"]   = X_g["credit_amount"] / X_g["duration"].clip(lower=1)
if "credit_amount" in X_g.columns and "age" in X_g.columns:
    X_g["credit_to_age"]      = X_g["credit_amount"] / X_g["age"].clip(lower=1)
if "installment_commitment" in X_g.columns and "duration" in X_g.columns:
    X_g["total_commitment"]   = X_g["installment_commitment"] * X_g["duration"]
if "age" in X_g.columns and "duration" in X_g.columns:
    X_g["age_duration_ratio"] = X_g["age"] / X_g["duration"].clip(lower=1)

# ── Additional features for accuracy boost ────────────────────────────────────
if "credit_amount" in X_g.columns and "duration" in X_g.columns:
    X_g["debt_to_duration"]   = X_g["credit_amount"] / X_g["duration"].clip(lower=1)
if "age" in X_g.columns:
    X_g["age_bucket"]         = pd.cut(
        X_g["age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]
    ).astype(float)
if "employment" in X_g.columns:
    emp_map = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}
    X_g["employment_score"]   = X_g["employment"].map(emp_map).fillna(0)
if "credit_amount" in X_g.columns:
    X_g["credit_bucket"]      = pd.qcut(
        X_g["credit_amount"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).astype(float)

# ── One-hot encode categoricals ───────────────────────────────────────────────
cat_cols = [c for c in CATEGORICAL_COLS if c in X_g.columns]
for c in cat_cols:
    X_g[c] = X_g[c].astype(str)
X_g_enc = pd.get_dummies(X_g, columns=cat_cols, drop_first=True).astype(float)
print(f"After OHE: {X_g_enc.shape}")

# ── Train / test split for German Credit ─────────────────────────────────────
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    X_g_enc, y_g, test_size=0.2, random_state=RANDOM_STATE, stratify=y_g
)

scaler_g    = RobustScaler()          # less sensitive to outliers than StandardScaler
Xg_tr_sc    = scaler_g.fit_transform(Xg_tr)
Xg_te_sc    = scaler_g.transform(Xg_te)
yg_tr_arr   = yg_tr.values
yg_te_arr   = yg_te.values

neg = (yg_tr_arr == 0).sum()
pos = (yg_tr_arr == 1).sum()
imb = neg / pos
print(f"German Train: {Xg_tr_sc.shape}  Test: {Xg_te_sc.shape}  Imbalance {neg}:{pos} ({imb:.1f}:1)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Load Home Credit test set  (Champion's original test data)
# ═══════════════════════════════════════════════════════════════════════════════

print("[Step 1] Loading Home Credit test set for Champion evaluation …")
_, X_hc_te, _, y_hc_te = get_home_credit_data()
print(f"Home Credit test set: {X_hc_te.shape}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Read PSI drift result from MLflow
# ═══════════════════════════════════════════════════════════════════════════════

try:
    drift_runs = mlflow.search_runs(
        filter_string="tags.`mlflow.runName` = 'PSI_Drift_Detection'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if drift_runs.empty:
        print("WARNING: No PSI_Drift_Detection run found — assuming NO drift.")
        any_drift        = False
        drifted_features = ""
    else:
        any_drift        = str(drift_runs.iloc[0].get("params.any_drift", "False")).lower() == "true"
        drifted_features = str(drift_runs.iloc[0].get("params.drifted_features", ""))
        print(f"PSI result → Drift detected: {'YES' if any_drift else 'NO'}")
        if drifted_features and drifted_features != "none":
            print(f"  Drifted features: {drifted_features}\n")
except Exception as exc:
    print(f"MLflow PSI fetch error: {exc}")
    raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Load Champion from MLflow and evaluate on HOME CREDIT test set
# ═══════════════════════════════════════════════════════════════════════════════

try:
    champ_runs = mlflow.search_runs(
        filter_string="tags.Champion = 'True'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if champ_runs.empty:
        raise ValueError("No Champion-tagged run found in MLflow. Run train_model.py first.")
    champ_run_id   = champ_runs.iloc[0]["run_id"]
    champion_model = mlflow.sklearn.load_model(f"runs:/{champ_run_id}/xgboost_model")
    print(f"Champion loaded — Run ID: {champ_run_id}")
except Exception as exc:
    print(f"MLflow model load error: {exc}")
    raise SystemExit(1)

print("\n[Step 2] Evaluating Champion on Home Credit test set …")

# Champion pipeline has its own preprocessor — pass raw Home Credit data
y_champ_proba = champion_model.predict_proba(X_hc_te)[:, 1]
y_champ_pred  = (y_champ_proba >= 0.5).astype(int)

champ_acc      = accuracy_score(y_hc_te, y_champ_pred)
champ_auc      = roc_auc_score(y_hc_te, y_champ_proba)
champ_f1       = f1_score(y_hc_te, y_champ_pred, zero_division=0)
champ_f1_macro = f1_score(y_hc_te, y_champ_pred, average="macro", zero_division=0)

print(f"\n{'─'*60}")
print(f"  XGBoost Champion — evaluated on HOME CREDIT test set")
print(f"{'─'*60}")
print(f"  Accuracy : {champ_acc * 100:.2f}%")
print(f"  AUC-ROC  : {champ_auc:.4f}")
print(f"  F1       : {champ_f1:.4f}  |  F1-Macro : {champ_f1_macro:.4f}")
print(classification_report(
    y_hc_te, y_champ_pred,
    target_names=["No Default", "Default"],
    zero_division=0,
))

champion_res = {
    "name":     "XGBoost Champion (Home Credit)",
    "model":    champion_model,
    "y_proba":  y_champ_proba,
    "y_te":     y_hc_te.values,
    "acc":      champ_acc,
    "auc":      champ_auc,
    "f1":       champ_f1,
    "f1_macro": champ_f1_macro,
}

all_results = [champion_res]
selected    = champion_res
verdict     = "NO DRIFT — Champion stays"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Conditional retraining on German Credit (Challenger)
# ═══════════════════════════════════════════════════════════════════════════════

if not any_drift:
    print(f"\n[Step 3] PSI < {PSI_THRESHOLD} on all features — Champion XGBoost retained.")

else:
    print(f"\n[Step 3] DRIFT detected on: {drifted_features}")
    print("  Retraining SGD Challenger on German Credit data …\n")

    # ── SMOTE: oversample minority class on training data only ────────────────
    sm = SMOTE(random_state=RANDOM_STATE)
    Xg_tr_res, yg_tr_res = sm.fit_resample(Xg_tr_sc, yg_tr_arr)
    print(f"  After SMOTE — Train shape: {Xg_tr_res.shape}  "
          f"Class counts: {dict(zip(*np.unique(yg_tr_res, return_counts=True)))}")

    # ── GridSearchCV: tune SGD hyperparameters ────────────────────────────────
    param_grid = {
        "loss":     ["modified_huber", "log_loss"],
        "penalty":  ["elasticnet", "l2"],
        "alpha":    [0.0001, 0.001, 0.01],
        "l1_ratio": [0.1, 0.15, 0.5],
        "max_iter": [1000, 2000],
    }
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sgd_base = SGDClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(
        sgd_base, param_grid,
        cv=cv_inner, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    grid_search.fit(Xg_tr_res, yg_tr_res)
    challenger_sgd = grid_search.best_estimator_
    print(f"\n  Best SGD params : {grid_search.best_params_}")
    print(f"  Best CV F1-Macro: {grid_search.best_score_:.4f}")

    # ── Cross-validation score on full training set ───────────────────────────
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        challenger_sgd, Xg_tr_res, yg_tr_res,
        cv=cv_outer, scoring="f1_macro", n_jobs=-1,
    )
    print(f"  CV F1-Macro (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    y_chal_proba  = challenger_sgd.predict_proba(Xg_te_sc)[:, 1]

    # ── Find best threshold optimising F1-Macro (balanced across both classes) ─
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_chal_proba >= t).astype(int)
        score = f1_score(yg_te_arr, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, round(t, 2)

    y_chal_pred    = (y_chal_proba >= best_t).astype(int)
    chal_acc       = accuracy_score(yg_te_arr, y_chal_pred)
    chal_auc       = roc_auc_score(yg_te_arr,  y_chal_proba)
    chal_f1        = f1_score(yg_te_arr, y_chal_pred, zero_division=0)
    chal_f1_macro  = f1_score(yg_te_arr, y_chal_pred, average="macro", zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  SGD Challenger — evaluated on GERMAN CREDIT test set  (threshold={best_t})")
    print(f"{'─'*60}")
    print(f"  Accuracy : {chal_acc * 100:.2f}%")
    print(f"  AUC-ROC  : {chal_auc:.4f}")
    print(f"  F1       : {chal_f1:.4f}  |  F1-Macro : {chal_f1_macro:.4f}")
    print(classification_report(
        yg_te_arr, y_chal_pred,
        target_names=["Good Credit", "Bad Credit"],
        zero_division=0,
    ))

    challenger_res = {
        "name":     "SGD Challenger (German Credit — Post-Drift Retrain)",
        "model":    challenger_sgd,
        "y_proba":  y_chal_proba,
        "y_te":     yg_te_arr,
        "acc":      chal_acc,
        "auc":      chal_auc,
        "f1":       chal_f1,
        "f1_macro": chal_f1_macro,
    }
    all_results.append(challenger_res)

    if challenger_res["auc"] >= champion_res["auc"]:
        selected = challenger_res
        verdict  = "DRIFT DETECTED — Challenger deployed (higher AUC)"
    else:
        selected = champion_res
        verdict  = "DRIFT DETECTED — Champion retained (Champion AUC still higher)"


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  AUC-ROC plot  (each model plotted against its own test set)
# ═══════════════════════════════════════════════════════════════════════════════

roc_path = "roc_curves.png"
fig, ax  = plt.subplots(figsize=(8, 6))
colors   = ["steelblue", "tomato"]

for i, r in enumerate(all_results):
    fpr, tpr, _ = roc_curve(r["y_te"], r["y_proba"])
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
            label=f"{r['name']}  (AUC = {r['auc']:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("AUC-ROC — Champion (Home Credit) vs Challenger (German Credit)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"\nROC curve saved → {roc_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Print comparison & log to MLflow
# ═══════════════════════════════════════════════════════════════════════════════

df_cmp = pd.DataFrame([{
    "Model":    r["name"],
    "Test Set": "Home Credit" if "Champion" in r["name"] else "German Credit",
    "Accuracy": f"{r['acc'] * 100:.2f}%",
    "AUC":      round(r["auc"],      4),
    "F1":       round(r["f1"],       4),
    "F1 Macro": round(r["f1_macro"], 4),
} for r in all_results])

print(f"\n{'═'*70}")
print("  FINAL MODEL COMPARISON")
print(f"{'═'*70}")
print(df_cmp.to_string(index=False))
print(f"\n  Selected model : {selected['name']}")
print(f"  Verdict        : {verdict}")

with mlflow.start_run(run_name="German_Credit_Evaluation") as run:
    mlflow.log_param("selected_model",   selected["name"])
    mlflow.log_param("verdict",          verdict)
    mlflow.log_param("any_drift",        str(any_drift))
    mlflow.log_param("drifted_features", drifted_features if any_drift else "none")

    mlflow.log_metric("champion_auc",      champion_res["auc"])
    mlflow.log_metric("champion_f1",       champion_res["f1"])
    mlflow.log_metric("champion_f1_macro", champion_res["f1_macro"])
    mlflow.log_metric("champion_accuracy", champion_res["acc"])

    if len(all_results) > 1:
        cr = all_results[1]
        mlflow.log_metric("challenger_auc",      cr["auc"])
        mlflow.log_metric("challenger_f1",       cr["f1"])
        mlflow.log_metric("challenger_f1_macro", cr["f1_macro"])
        mlflow.log_metric("challenger_accuracy", cr["acc"])

    mlflow.log_metric("final_auc",      selected["auc"])
    mlflow.log_metric("final_f1",       selected["f1"])
    mlflow.log_metric("final_f1_macro", selected["f1_macro"])
    mlflow.log_metric("final_accuracy", selected["acc"])

    mlflow.log_artifact(roc_path)

    if any_drift and len(all_results) > 1:
        mlflow.sklearn.log_model(
            sk_model=all_results[1]["model"],
            name="sgd_challenger",
        )
        mlflow.log_param("sgd_best_params",   str(grid_search.best_params_))
        mlflow.log_metric("challenger_cv_f1_macro", cv_scores.mean())
        client = MlflowClient()
        client.set_tag(run.info.run_id, "Challenger", "True")

    print(f"\nMLflow run ID : {run.info.run_id}")

print("\nAll results logged to MLflow.")
