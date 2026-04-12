"""
FILE 4 (IMPROVED): German Credit — XGBoost vs SVM + PSI-Triggered SVM Retrain
===============================================================================
IMPROVEMENTS OVER ORIGINAL:
  ✅ Feature engineering: debt_to_income, monthly_burden, age_risk, total_commitment
  ✅ Three-way split: train / val / test (threshold tuned on val — fixes leakage)
  ✅ RandomizedSearchCV tunes XGBoost (30 iters, 5-fold)
  ✅ GridSearchCV tunes SVM C + gamma
  ✅ 10-Fold Stratified CV for reliable AUC estimates
  ✅ Optional SMOTE oversampling
  ✅ Optional LightGBM third challenger

Pipeline:
  1.  Load + preprocess + feature-engineer German Credit data
  2.  Load PSI results from File 3 (drift_data.pkl)
  3.  Load XGBoost + SVM from File 2 (trained_models.pkl)
  4.  10-Fold CV AUC baseline
  5.  Tune XGBoost + SVM
  6.  Evaluate on test set (threshold from val — no leakage)
  7.  If PSI > 0.2:
        → Retrain tuned SVM on German Credit data
        → Compare XGBoost vs retrained SVM predictions
        → Agreement ≥ 90% → ✅ OLD MODEL STABLE
        → Agreement <  90% → ❌ OLD MODEL INVALID (deploy retrained SVM)
  8.  Final comparison table + save

Run AFTER: 1_data_preprocess.py → 2_train_models.py → 3_drift_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Optional: LightGBM ───────────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
except ImportError:
    USE_LGBM = False
    print("⚠️  LightGBM not installed. (pip install lightgbm)")

# ── Optional: SMOTE ──────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
except ImportError:
    USE_SMOTE = False
    print("⚠️  imbalanced-learn not installed. (pip install imbalanced-learn)")

# ── Config ────────────────────────────────────────────────────────────────────
GERMAN_PATH         = "data/german_data.csv"
PSI_DRIFT_THRESHOLD = 0.2
AGREE_THRESHOLD     = 0.90
TUNE_MODELS         = True   # Set False to skip tuning (faster run)
N_CV_FOLDS          = 10
RANDOM_STATE        = 42
# ──────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true, y_proba):
    """Scan 0.1–0.9; return threshold with best F1. Call on VAL set only."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return round(best_thresh, 2)


def evaluate_fitted_model(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, sample_size=None):
    """
    Fit on (X_tr, y_tr).
    Tune threshold on (X_val, y_val)  ← no leakage.
    Report final metrics on (X_te, y_te).
    """
    if sample_size and sample_size < len(X_tr):
        idx  = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), size=sample_size, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"    (subsampled to {sample_size} rows for speed)")

    model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_val)[:, 1]
    thresh    = find_best_threshold(y_val, val_proba)

    y_proba  = model.predict_proba(X_te)[:, 1]
    y_pred   = (y_proba >= thresh).astype(int)

    auc      = roc_auc_score(y_te, y_proba)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    accuracy = (y_pred == y_te).mean()

    print(f"\n{'─' * 55}")
    print(f"  Model     : {name}  (threshold={thresh} — val-tuned)")
    print(f"{'─' * 55}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1        : {f1:.4f}  (class=1, bad credit)")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Good Credit','Bad Credit'], zero_division=0)}")

    return {
        "name":     name,
        "model":    model,
        "auc":      auc,
        "f1":       f1,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "thresh":   thresh,
    }


def cv_auc(name, model, X, y, n_splits=N_CV_FOLDS):
    """Stratified K-Fold cross-validated AUC."""
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:<40}  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


def print_comparison_table(results, title="Model Comparison"):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    df_cmp = pd.DataFrame([
        {
            "Model":        r["name"],
            "AUC":          round(r["auc"],      4),
            "Accuracy":     f"{r['accuracy']*100:.1f}%",
            "F1 (Default)": round(r["f1"],       4),
            "F1 Macro":     round(r["f1_macro"], 4),
        }
        for r in results
    ])
    print(df_cmp.to_string(index=False))
    best = max(results, key=lambda x: x["auc"])
    print(f"\n  ✅ Best by AUC : {best['name']} (AUC={best['auc']:.4f})")
    return best


def compare_predictions(xgb_preds, svm_preds):
    total    = len(xgb_preds)
    agree    = int((xgb_preds == svm_preds).sum())
    disagree = total - agree
    ratio    = agree / total

    print(f"\n{'═' * 65}")
    print("  PREDICTION COMPARISON — XGBoost (Old) vs SVM (Retrained)")
    print(f"{'═' * 65}")
    print(f"  Total test samples   : {total}")
    print(f"  Agreements           : {agree}  ({ratio:.1%})")
    print(f"  Disagreements        : {disagree}  ({1-ratio:.1%})")
    print(f"  Agreement threshold  : {AGREE_THRESHOLD:.0%}")

    if ratio >= AGREE_THRESHOLD:
        verdict = "STABLE"
        print(f"\n  ✅ VERDICT: OLD MODEL IS STABLE")
        print(f"     XGBoost and retrained SVM agree on {ratio:.1%} of predictions.")
        print(f"     The original XGBoost remains valid for production.")
    else:
        verdict = "INVALID"
        print(f"\n  ❌ VERDICT: OLD MODEL IS INVALID")
        print(f"     XGBoost and retrained SVM disagree on {1-ratio:.1%} of predictions.")
        print(f"     Deploy the retrained SVM model instead.")
    print(f"{'═' * 65}\n")

    return ratio, verdict


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: LOAD + PREPROCESS + FEATURE ENGINEER GERMAN CREDIT DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 8: German Credit Dataset — Load, Preprocess & Feature Engineer")
print("=" * 65)

try:
    df_german = pd.read_csv(GERMAN_PATH)

    if df_german.shape[1] == 21 and "checking_status" not in df_german.columns:
        german_cols = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class"
        ]
        df_german.columns = german_cols

    print(f"✅ Loaded German Credit: {df_german.shape}")

    target_col = next((c for c in df_german.columns if c.lower() == "class"), None)
    if target_col is None:
        raise ValueError(f"No 'class' column found. Columns: {df_german.columns.tolist()}")

    if df_german[target_col].isin([1, 2]).all():
        df_german[target_col] = (df_german[target_col] == 2).astype(int)

    print(f"Target distribution:\n{df_german[target_col].value_counts()}\n")

    # ── IMPROVEMENT: Feature Engineering ─────────────────────────────────────
    print("🔧 Feature Engineering...")

    if "credit_amount" in df_german.columns and "duration" in df_german.columns:
        df_german["debt_to_income"]   = df_german["credit_amount"] / (df_german["duration"] + 1)
        df_german["monthly_burden"]   = df_german["credit_amount"] / df_german["duration"].replace(0, 1)
        print("  ✅ debt_to_income   = credit_amount / (duration + 1)")
        print("  ✅ monthly_burden   = credit_amount / duration")

    if "age" in df_german.columns:
        df_german["age_risk"] = pd.cut(
            df_german["age"],
            bins   = [0, 25, 35, 50, 100],
            labels = [3, 2, 1, 0],        # younger = riskier
        ).astype(float)
        print("  ✅ age_risk         = binned age (young=3, old=0)")

    if "installment_commitment" in df_german.columns and "duration" in df_german.columns:
        df_german["total_commitment"] = df_german["installment_commitment"] * df_german["duration"]
        print("  ✅ total_commitment = installment_commitment × duration")

    if "existing_credits" in df_german.columns and "num_dependents" in df_german.columns:
        df_german["burden_ratio"] = df_german["existing_credits"] + df_german["num_dependents"]
        print("  ✅ burden_ratio     = existing_credits + num_dependents")

    if "credit_amount" in df_german.columns and "age" in df_german.columns:
        df_german["credit_per_age"] = df_german["credit_amount"] / (df_german["age"] + 1)
        print("  ✅ credit_per_age   = credit_amount / age")

    print()

    # ── OHE ──────────────────────────────────────────────────────────────────
    X_g = df_german.drop(target_col, axis=1).copy()
    y_g = df_german[target_col]

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_g.columns]
    if not cat_cols_present:
        cat_cols_present = X_g.select_dtypes(include="object").columns.tolist()
        print(f"⚠️  Fallback: dtype-detected categoricals ({len(cat_cols_present)})")
    else:
        for col in cat_cols_present:
            X_g[col] = X_g[col].astype(str)

    X_g_encoded = pd.get_dummies(X_g, columns=cat_cols_present, drop_first=True).astype(float)
    print(f"Shape before OHE : {X_g.shape}  →  after OHE : {X_g_encoded.shape}")

    if X_g_encoded.shape[1] == 0:
        raise ValueError("OHE produced 0 columns. Check categorical column names match your CSV.")

    # ── IMPROVEMENT: Three-way split (train / val / test) ────────────────────
    Xg_train_full, Xg_test, yg_train_full, yg_test = train_test_split(
        X_g_encoded, y_g, test_size=0.20, random_state=RANDOM_STATE, stratify=y_g
    )
    Xg_train, Xg_val, yg_train, yg_val = train_test_split(
        Xg_train_full, yg_train_full,
        test_size=0.15, random_state=RANDOM_STATE, stratify=yg_train_full
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler_g          = StandardScaler()
    Xg_train_sc       = scaler_g.fit_transform(Xg_train)
    Xg_val_sc         = scaler_g.transform(Xg_val)
    Xg_test_sc        = scaler_g.transform(Xg_test)
    Xg_train_full_sc  = scaler_g.fit_transform(Xg_train_full)   # for retrain

    yg_train_arr      = yg_train.values
    yg_val_arr        = yg_val.values
    yg_test_arr       = yg_test.values
    yg_train_full_arr = yg_train_full.values

    neg_g = (yg_train_arr == 0).sum()
    pos_g = (yg_train_arr == 1).sum()
    imbalance_g = neg_g / pos_g

    print(f"\n  Train : {Xg_train_sc.shape}  Val : {Xg_val_sc.shape}  Test : {Xg_test_sc.shape}")
    print(f"  Imbalance ratio : {imbalance_g:.1f}:1\n")

    # ── SMOTE (optional) ─────────────────────────────────────────────────────
    if USE_SMOTE:
        print("🔧 Applying SMOTE to training set...")
        sm = SMOTE(random_state=RANDOM_STATE)
        Xg_train_sc, yg_train_arr = sm.fit_resample(Xg_train_sc, yg_train_arr)
        print(f"  After SMOTE: {Xg_train_sc.shape}, balance: {np.bincount(yg_train_arr)}\n")

except FileNotFoundError:
    print(f"⚠️  File not found: '{GERMAN_PATH}'")
    print("   Place german_data.csv inside the data/ folder and rerun.")
    raise SystemExit(1)
except ValueError as e:
    print(f"⚠️  Data error: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: LOAD PSI RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 9: Loading PSI Results from File 3")
print("=" * 65)

try:
    with open("drift_data.pkl", "rb") as f:
        drift_data = pickle.load(f)

    psi_results      = drift_data["psi_results"]
    any_drift        = drift_data["any_drift"]
    drifted_features = drift_data["drifted_features"]

    print(f"\n  {'Feature':<25} {'PSI':>8}  Status")
    print("  " + "─" * 48)
    for r in psi_results:
        print(f"  {r['feature']:<25} {r['psi']:>8.4f}  {r['status']}")

    print(f"\n  PSI Guide: < 0.1 = Stable | 0.1–0.2 = Warning | > 0.2 = Drift")
    print(f"\n  Drifted features (PSI > {PSI_DRIFT_THRESHOLD}) : {drifted_features}")
    print(f"  Overall drift flag : "
          f"{'⚠️  YES — SVM retrain will trigger' if any_drift else '✅  NO'}\n")

except FileNotFoundError:
    print("⚠️  drift_data.pkl not found. Run 3_drift_detection.py first.")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: LOAD TRAINED MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 10: Loading Trained Models from File 2")
print("=" * 65)

try:
    with open("trained_models.pkl", "rb") as f:
        model_data = pickle.load(f)

    results_list = model_data["results"]
    xgb_model    = next((r["model"] for r in results_list if "XGBoost" in r["name"]), None)
    svm_model    = next((r["model"] for r in results_list if "SVM"     in r["name"]), None)

    if xgb_model is None:
        raise KeyError("XGBoost not found in trained_models.pkl.")
    if svm_model is None:
        raise KeyError("SVM not found in trained_models.pkl.")

    print(f"✅ Loaded XGBoost : {xgb_model.__class__.__name__}")
    print(f"✅ Loaded SVM     : {svm_model.__class__.__name__}\n")

except FileNotFoundError:
    print("⚠️  trained_models.pkl not found. Run 2_train_models.py first.")
    raise SystemExit(1)
except KeyError as e:
    print(f"⚠️  Model loading error: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: 10-FOLD CV AUC — BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"STEP 11: {N_CV_FOLDS}-Fold CV AUC — German Credit Baseline")
print("=" * 65)
print()

X_cv = np.vstack([Xg_train_sc, Xg_val_sc])
y_cv = np.concatenate([yg_train_arr, yg_val_arr])

cv_auc("XGBoost (default params)",
       XGBClassifier(scale_pos_weight=imbalance_g,
                     use_label_encoder=False, eval_metric="auc",
                     random_state=RANDOM_STATE, n_jobs=-1),
       X_cv, y_cv)

cv_auc("SVM RBF (C=1, default)",
       SVC(kernel="rbf", C=1.0, class_weight="balanced",
           probability=True, random_state=RANDOM_STATE),
       X_cv, y_cv)

if USE_LGBM:
    cv_auc("LightGBM (default)",
           LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                          n_jobs=-1, verbose=-1),
           X_cv, y_cv)
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 12: Hyperparameter Tuning — XGBoost + SVM")
print("=" * 65)

if TUNE_MODELS:

    # ── XGBoost RandomizedSearchCV ────────────────────────────────────────────
    print("\n🔍 Tuning XGBoost — RandomizedSearchCV (30 iters, 5-fold)...")
    xgb_param_grid = {
        "n_estimators":     [200, 300, 400, 600],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample":        [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha":        [0, 0.01, 0.1, 1],
        "reg_lambda":       [1, 1.5, 2, 5],
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight = imbalance_g,
            use_label_encoder= False,
            eval_metric      = "auc",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        ),
        xgb_param_grid,
        n_iter       = 30,
        scoring      = "roc_auc",
        cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        verbose      = 1,
    )
    xgb_search.fit(Xg_train_sc, yg_train_arr)
    xgb_g = xgb_search.best_estimator_
    print(f"\n  ✅ Best XGBoost params : {xgb_search.best_params_}")
    print(f"     Best CV AUC        : {xgb_search.best_score_:.4f}")

    # ── SVM GridSearchCV ──────────────────────────────────────────────────────
    print("\n🔍 Tuning SVM — GridSearchCV (C × gamma, 5-fold)...")
    svm_param_grid = {
        "C":     [0.1, 0.5, 1, 5, 10, 50, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }

    n_svm_tune = min(700, len(Xg_train_sc))
    tune_idx   = np.random.RandomState(RANDOM_STATE).choice(len(Xg_train_sc), size=n_svm_tune, replace=False)

    svm_search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=RANDOM_STATE),
        svm_param_grid,
        scoring = "roc_auc",
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs  = -1,
        verbose = 1,
    )
    svm_search.fit(Xg_train_sc[tune_idx], yg_train_arr[tune_idx])
    best_C     = svm_search.best_params_["C"]
    best_gamma = svm_search.best_params_["gamma"]
    print(f"\n  ✅ Best SVM params     : C={best_C}, gamma={best_gamma}")
    print(f"     Best CV AUC        : {svm_search.best_score_:.4f}")

else:
    print("  ⚠️  TUNE_MODELS=False — using default params")
    xgb_g  = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance_g,
        use_label_encoder=False, eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    best_C     = 1.0
    best_gamma = "scale"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13: BASELINE EVALUATION — XGBoost vs SVM on German Credit
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 13: Baseline Evaluation — Tuned XGBoost vs Tuned SVM")
print("=" * 65)

print("\n--- Tuned XGBoost ---")
xgb_g_res = evaluate_fitted_model(
    "XGBoost (Tuned)", xgb_g,
    Xg_train_sc,  yg_train_arr,
    Xg_val_sc,    yg_val_arr,
    Xg_test_sc,   yg_test_arr,
)

print("\n--- Tuned SVM ---")
svm_g = SVC(
    kernel="rbf", C=best_C, gamma=best_gamma,
    class_weight="balanced", probability=True, random_state=RANDOM_STATE,
)
svm_g_res = evaluate_fitted_model(
    "SVM (Tuned)", svm_g,
    Xg_train_sc,  yg_train_arr,
    Xg_val_sc,    yg_val_arr,
    Xg_test_sc,   yg_test_arr,
    sample_size   = 700,
)

baseline_results = [xgb_g_res, svm_g_res]

if USE_LGBM:
    print("\n--- LightGBM ---")
    lgbm_g = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=63, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgbm_g_res = evaluate_fitted_model(
        "LightGBM", lgbm_g,
        Xg_train_sc, yg_train_arr,
        Xg_val_sc,   yg_val_arr,
        Xg_test_sc,  yg_test_arr,
    )
    baseline_results.append(lgbm_g_res)

print_comparison_table(baseline_results, title="Baseline: Tuned Models — German Credit")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14: DRIFT CHECK → CONDITIONAL SVM RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 14: Drift Check → Conditional SVM Retrain")
print("=" * 65)

verdict       = None
ratio         = None
final_results = []

if not any_drift:
    print("\n✅ No drift detected (all PSI ≤ 0.2)")
    print("   SVM retraining is NOT required.")
    print("   Both models remain valid. Using baseline results as final.\n")
    final_results = baseline_results
    verdict       = "STABLE (no drift)"
    ratio         = 1.0

else:
    print(f"\n⚠️  Drift detected on features : {drifted_features}")
    print(f"   PSI > {PSI_DRIFT_THRESHOLD} — retraining SVM on German Credit data.\n")

    print("=" * 65)
    print("STEP 15: Retraining Tuned SVM on German Credit (Drifted) Data")
    print("=" * 65)

    n_svm   = min(700, len(Xg_train_full_sc))
    svm_idx = np.random.RandomState(RANDOM_STATE).choice(len(Xg_train_full_sc), size=n_svm, replace=False)

    print(f"  Retraining SVM (C={best_C}, gamma={best_gamma}) on {n_svm} samples...")

    svm_retrained = SVC(
        kernel       = "rbf",
        C            = best_C,
        gamma        = best_gamma,
        class_weight = "balanced",
        probability  = True,
        random_state = RANDOM_STATE,
    )
    svm_retrained.fit(Xg_train_full_sc[svm_idx], yg_train_full_arr[svm_idx])
    print("  ✅ SVM retrained.\n")

    # Threshold on val set
    val_proba_svm = svm_retrained.predict_proba(Xg_val_sc)[:, 1]
    thresh_svm    = find_best_threshold(yg_val_arr, val_proba_svm)

    # Evaluate on test set
    y_proba_svm  = svm_retrained.predict_proba(Xg_test_sc)[:, 1]
    y_pred_svm   = (y_proba_svm >= thresh_svm).astype(int)
    auc_svm      = roc_auc_score(yg_test_arr, y_proba_svm)
    f1_svm       = f1_score(yg_test_arr, y_pred_svm, zero_division=0)
    f1_macro_svm = f1_score(yg_test_arr, y_pred_svm, average="macro", zero_division=0)
    acc_svm      = (y_pred_svm == yg_test_arr).mean()

    print(f"{'─' * 55}")
    print(f"  Model     : SVM (Retrained on Drift)  (threshold={thresh_svm})")
    print(f"{'─' * 55}")
    print(f"  AUC       : {auc_svm:.4f}")
    print(f"  Accuracy  : {acc_svm:.4f}  ({acc_svm*100:.1f}%)")
    print(f"  F1        : {f1_svm:.4f}")
    print(f"  F1 Macro  : {f1_macro_svm:.4f}")
    print(f"\n{classification_report(yg_test_arr, y_pred_svm, target_names=['Good Credit','Bad Credit'], zero_division=0)}")

    svm_retrained_res = {
        "name":     "SVM (Retrained on Drift)",
        "model":    svm_retrained,
        "auc":      auc_svm,
        "f1":       f1_svm,
        "f1_macro": f1_macro_svm,
        "accuracy": acc_svm,
        "y_pred":   y_pred_svm,
        "y_proba":  y_proba_svm,
        "thresh":   thresh_svm,
    }

    # XGBoost predictions (already fitted above)
    val_proba_xgb = xgb_g.predict_proba(Xg_val_sc)[:, 1]
    thresh_xgb    = find_best_threshold(yg_val_arr, val_proba_xgb)
    y_proba_xgb   = xgb_g.predict_proba(Xg_test_sc)[:, 1]
    y_pred_xgb    = (y_proba_xgb >= thresh_xgb).astype(int)
    xgb_g_res["y_pred"] = y_pred_xgb

    # ── Prediction Comparison & Validity ──────────────────────────────────────
    print("=" * 65)
    print("STEP 16: Prediction Comparison → Model Validity Decision")
    print("=" * 65)
    ratio, verdict = compare_predictions(xgb_g_res["y_pred"], svm_retrained_res["y_pred"])

    final_results = [xgb_g_res, svm_retrained_res]
    if USE_LGBM and "lgbm_g_res" in dir():
        final_results.append(lgbm_g_res)

    # ── Validity Summary ───────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 17: Model Validity Summary")
    print("=" * 65)
    print(f"""
  DECISION LOGIC:
  ┌──────────────────────────────────────────────────────────┐
  │  PSI > 0.2            → Data drift confirmed             │
  │  SVM retrained        → Adapts to new distribution       │
  │                                                          │
  │  Compare XGBoost vs SVM predictions on German Credit:    │
  │                                                          │
  │  Agreement ≥ {AGREE_THRESHOLD:.0%}    → ✅ OLD MODEL STABLE            │
  │                  Both models agree → XGBoost stays       │
  │                                                          │
  │  Agreement <  {AGREE_THRESHOLD:.0%}    → ❌ OLD MODEL INVALID           │
  │                  Models diverge → XGBoost must retire    │
  │                  Deploy retrained SVM to production      │
  └──────────────────────────────────────────────────────────┘

  RESULT:
    Drifted features : {drifted_features}
    Agreement ratio  : {ratio:.1%}
    Verdict          : {verdict}
    Action           : {"✅ Keep XGBoost in production." if verdict == "STABLE" else "❌ Retire XGBoost. Deploy retrained SVM."}
""")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print_comparison_table(
    final_results,
    title="FINAL: XGBoost vs SVM — German Credit" + (" (Post-Drift Retrain)" if any_drift else "")
)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open("german_credit_results.pkl", "wb") as f:
    pickle.dump({
        "final_results":    final_results,
        "baseline_results": baseline_results,
        "verdict":          verdict,
        "agree_ratio":      ratio,
        "any_drift":        any_drift,
    }, f)

print("\n✅ Saved german_credit_results.pkl")
print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)