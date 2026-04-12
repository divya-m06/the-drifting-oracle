"""
FILE 2 (IMPROVED): Model Training — XGBoost + SVM
==================================================
IMPROVEMENTS:
  ✅ RandomizedSearchCV tunes XGBoost (30 iterations, 5-fold CV)
  ✅ GridSearchCV tunes SVM C + gamma
  ✅ Threshold tuned on validation set (not test set — fixes data leakage)
  ✅ 10-Fold Stratified CV for reliable AUC estimates
  ✅ Optional LightGBM third challenger
  ✅ Optional SMOTE oversampling

Run AFTER: 1_data_preprocess.py
Saves: trained_models.pkl
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    GridSearchCV, StratifiedKFold, cross_val_score
)
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
    print("⚠️  LightGBM not installed. Skipping. (pip install lightgbm)")

# ── Optional: SMOTE ──────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
except ImportError:
    USE_SMOTE = False
    print("⚠️  imbalanced-learn not installed. Skipping SMOTE. (pip install imbalanced-learn)")

# ── Config ────────────────────────────────────────────────────────────────────
TUNE_MODELS  = True   # Set False to skip hyperparameter search (much faster)
N_CV_FOLDS   = 10     # Stratified K-Fold folds for CV AUC estimate
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────

# ── Load preprocessed data ───────────────────────────────────────────────────
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train_full = data["X_train_scaled"]
X_test       = data["X_test_scaled"]
y_train_full = data["y_train"].values
y_test       = data["y_test"].values
# ──────────────────────────────────────────────────────────────────────────────

# ── Split out a validation set from training (for threshold tuning) ───────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size    = 0.15,
    random_state = RANDOM_STATE,
    stratify     = y_train_full,
)

neg             = (y_train == 0).sum()
pos             = (y_train == 1).sum()
imbalance_ratio = neg / pos

print("=" * 60)
print("STEP 5: Model Training — XGBoost + SVM (Improved)")
print("=" * 60)
print(f"  Train : {X_train.shape}   Val : {X_val.shape}   Test : {X_test.shape}")
print(f"  Class counts → No Default: {neg}, Default: {pos}")
print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1\n")


# ── SMOTE (optional) ─────────────────────────────────────────────────────────
if USE_SMOTE:
    print("🔧 Applying SMOTE to training set...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    imbalance_ratio = neg / pos
    print(f"  After SMOTE: {X_train.shape}, class balance: {np.bincount(y_train)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true, y_proba):
    """Scan thresholds 0.1–0.9; return the one with best F1 on y_true."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return round(best_thresh, 2)


def evaluate_model(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, sample_size=None):
    """
    Fit on X_tr/y_tr.
    Tune threshold on X_val/y_val  ← no leakage.
    Evaluate on X_te/y_te.
    """
    if sample_size and sample_size < len(X_tr):
        idx  = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), size=sample_size, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"    (subsampled to {sample_size} rows for speed)")

    model.fit(X_tr, y_tr)

    # Threshold on VALIDATION set
    val_proba = model.predict_proba(X_val)[:, 1]
    thresh    = find_best_threshold(y_val, val_proba)

    # Final metrics on TEST set
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
    print(f"  F1        : {f1:.4f}  (class=1, defaulters)")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['No Default','Default'], zero_division=0)}")

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
    """Stratified K-Fold CV AUC — more reliable than a single split."""
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:<38}  CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5a: CROSS-VALIDATION AUC (before tuning — baseline reference)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"STEP 5a: {N_CV_FOLDS}-Fold CV AUC — Baseline (default params)")
print("=" * 60)

X_cv = np.vstack([X_train, X_val])
y_cv = np.concatenate([y_train, y_val])

cv_auc("XGBoost (default)",
       XGBClassifier(scale_pos_weight=imbalance_ratio,
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
# STEP 5b: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5b: Hyperparameter Tuning")
print("=" * 60)

if TUNE_MODELS:

    # ── XGBoost: RandomizedSearchCV ──────────────────────────────────────────
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
            scale_pos_weight = imbalance_ratio,
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
    xgb_search.fit(X_train, y_train)
    xgb = xgb_search.best_estimator_
    print(f"\n  ✅ Best XGBoost params : {xgb_search.best_params_}")
    print(f"     Best CV AUC        : {xgb_search.best_score_:.4f}")

    # ── SVM: GridSearchCV ────────────────────────────────────────────────────
    print("\n🔍 Tuning SVM — GridSearchCV (C × gamma grid, 5-fold)...")
    svm_param_grid = {
        "C":     [0.1, 0.5, 1, 5, 10, 50, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }

    n_svm = min(20000, len(X_train))
    svm_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=n_svm, replace=False)

    svm_search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=RANDOM_STATE),
        svm_param_grid,
        scoring = "roc_auc",
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs  = -1,
        verbose = 1,
    )
    svm_search.fit(X_train[svm_idx], y_train[svm_idx])
    svm = SVC(
        kernel       = "rbf",
        C            = svm_search.best_params_["C"],
        gamma        = svm_search.best_params_["gamma"],
        class_weight = "balanced",
        probability  = True,
        random_state = RANDOM_STATE,
    )
    print(f"\n  ✅ Best SVM params     : {svm_search.best_params_}")
    print(f"     Best CV AUC        : {svm_search.best_score_:.4f}")

else:
    print("  ⚠️  TUNE_MODELS=False — using default params")
    xgb = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = imbalance_ratio,
        use_label_encoder= False,
        eval_metric      = "auc",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced",
        probability=True, random_state=RANDOM_STATE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: TRAIN + EVALUATE — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Train + Evaluate — XGBoost (Tuned)")
print("=" * 60)

xgb_result = evaluate_model(
    "XGBoost (Tuned)", xgb,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6b: TRAIN + EVALUATE — SVM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6b: Train + Evaluate — SVM (Tuned RBF)")
print("=" * 60)

svm_result = evaluate_model(
    "SVM (Tuned RBF)", svm,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    sample_size=20000,
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6c: TRAIN + EVALUATE — LightGBM (optional)
# ══════════════════════════════════════════════════════════════════════════════
results = [xgb_result, svm_result]

if USE_LGBM:
    print("\n" + "=" * 60)
    print("STEP 6c: Train + Evaluate — LightGBM")
    print("=" * 60)

    lgbm = LGBMClassifier(
        n_estimators  = 500,
        learning_rate = 0.05,
        max_depth     = 6,
        num_leaves    = 63,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
        verbose       = -1,
    )
    lgbm_result = evaluate_model(
        "LightGBM", lgbm,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )
    results.append(lgbm_result)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Model Comparison")
print("=" * 60)

comparison_df = pd.DataFrame([
    {
        "Model":        r["name"],
        "AUC":          round(r["auc"],      4),
        "Accuracy":     f"{r['accuracy']*100:.1f}%",
        "F1 (Default)": round(r["f1"],       4),
        "F1 Macro":     round(r["f1_macro"], 4),
        "Threshold":    r["thresh"],
    }
    for r in results
])
print(comparison_df.to_string(index=False))

best_auc = max(results, key=lambda x: x["auc"])
best_f1  = max(results, key=lambda x: x["f1"])
print(f"\n  Champion (best AUC) : {best_auc['name']}   AUC={best_auc['auc']:.4f}")
print(f"  Best F1 on Defaulters: {best_f1['name']}   F1={best_f1['f1']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
svm_idx_save = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=min(20000, len(X_train)), replace=False)

with open("trained_models.pkl", "wb") as f:
    pickle.dump({
        "results":         results,
        "xgb_result":      xgb_result,
        "svm_result":      svm_result,
        "best_model_name": best_auc["name"],
        "champion":        best_auc,
        "X_train_svm":     X_train[svm_idx_save],
        "y_train_svm":     y_train[svm_idx_save],
    }, f)

print("\n✅ Saved trained_models.pkl — run 3_drift_detection.py next")
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)