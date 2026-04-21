"""
German Credit — XGBoost (Champion) vs SDG (Challenger) + PSI-Triggered Model Switching

Pipeline:
  1. Load + preprocess German Credit data
  2. Load PSI drift results from drift_detection.py
  3. Train XGBoost (champion) on original German Credit data
  4. If PSI >= 0.2 (drift detected):
     - Apply same economic shock to German Credit training data
     - Retrain SDG (challenger) on drifted training data
     - Select challenger for deployment
  5. If PSI < 0.2: champion stays, no retraining needed
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

GERMAN_PATH = "data/german_data.csv"
PSI_DRIFT_THRESHOLD = 0.2

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]


def find_best_threshold(y_true, y_proba):
    """Scan thresholds 0.1-0.9, return the one with best F1."""
    best_thresh, best_f1 = 0.5, 0.0
    from sklearn.metrics import f1_score
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return round(best_thresh, 2)


def evaluate_model(name, model, X_te, y_te, threshold=None):
    """Evaluate a fitted model on test set, return metrics dict."""
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_te)
    thresh = threshold if threshold is not None else find_best_threshold(y_te, y_proba)
    y_pred = (y_proba >= thresh).astype(int)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"\n{name} (threshold={thresh})")
    print(f"  Unique Predictions: {np.unique(y_pred)}")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  AUC: {auc:.4f}  F1: {f1:.4f}  F1 Macro: {f1_macro:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Good Credit', 'Bad Credit'], zero_division=0))

    return {
        "name": name, "model": model,
        "auc": auc, "f1": f1, "f1_macro": f1_macro,
        "accuracy": acc, "y_pred": y_pred, "y_proba": y_proba,
    }


def print_comparison(results, title="Model Comparison"):
    """Print a side-by-side comparison table."""
    print(f"\n{title}")
    df = pd.DataFrame([{
        "Model": r["name"],
        "Accuracy": f"{r['accuracy'] * 100:.2f}%",
        "AUC": round(r["auc"], 4),
        "F1": round(r["f1"], 4),
        "F1 Macro": round(r["f1_macro"], 4),
    } for r in results])
    print(df.to_string(index=False))



try:
    df_german = pd.read_csv(GERMAN_PATH)

    if df_german.shape[1] == 21 and "checking_status" not in df_german.columns:
        df_german.columns = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class"
        ]

    print(f"Loaded German Credit: {df_german.shape}")

    target_col = next((c for c in df_german.columns if c.lower() == "class"), None)
    if target_col is None:
        raise ValueError(f"No 'class' column found. Columns: {df_german.columns.tolist()}")

    if df_german[target_col].isin([1, 2]).all():
        df_german[target_col] = (df_german[target_col] == 2).astype(int)

    print(f"Target distribution:\n{df_german[target_col].value_counts()}\n")

    X_g = df_german.drop(target_col, axis=1).copy()
    y_g = df_german[target_col]

    if 'credit_amount' in X_g.columns and 'duration' in X_g.columns:
        X_g['credit_per_month'] = X_g['credit_amount'] / X_g['duration'].clip(lower=1)
    if 'credit_amount' in X_g.columns and 'age' in X_g.columns:
        X_g['credit_to_age'] = X_g['credit_amount'] / X_g['age'].clip(lower=1)
    if 'installment_commitment' in X_g.columns and 'duration' in X_g.columns:
        X_g['total_commitment'] = X_g['installment_commitment'] * X_g['duration']
    if 'age' in X_g.columns and 'duration' in X_g.columns:
        X_g['age_duration_ratio'] = X_g['age'] / X_g['duration'].clip(lower=1)

    print(f"After feature engineering: {X_g.shape}")

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_g.columns]
    if len(cat_cols_present) == 0:
        cat_cols_present = X_g.select_dtypes(include="object").columns.tolist()
    else:
        for col in cat_cols_present:
            X_g[col] = X_g[col].astype(str)

    X_g_encoded = pd.get_dummies(X_g, columns=cat_cols_present, drop_first=True).astype(float)
    print(f"After OHE: {X_g_encoded.shape}")

    if X_g_encoded.shape[1] == 0:
        raise ValueError("OHE produced 0 columns — check categorical column names.")

    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_g_encoded, y_g, test_size=0.2, random_state=42, stratify=y_g
    )

    scaler_g = StandardScaler()
    Xg_train_sc = scaler_g.fit_transform(Xg_train)
    Xg_test_sc = scaler_g.transform(Xg_test)
    yg_train_arr = yg_train.values
    yg_test_arr = yg_test.values

    neg_g = (yg_train_arr == 0).sum()
    pos_g = (yg_train_arr == 1).sum()
    print(f"Train: {Xg_train_sc.shape}, Test: {Xg_test_sc.shape}, Imbalance: {neg_g/pos_g:.1f}:1\n")

except FileNotFoundError:
    print(f"File not found: '{GERMAN_PATH}' — place german_data.csv in data/ and rerun.")
    raise SystemExit(1)
except ValueError as e:
    print(f"Data error: {e}")
    raise SystemExit(1)


import mlflow
from mlflow.tracking import MlflowClient

try:
    drift_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'PSI_Drift_Detection'", order_by=["start_time DESC"], max_results=1)
    if not drift_runs.empty:
        any_drift = str(drift_runs.iloc[0].get("params.any_drift", "False")).lower() == "true"
        drifted_features = str(drift_runs.iloc[0].get("params.drifted_features", ""))
        print(f"Drift detected according to MLflow: {'YES' if any_drift else 'NO'}\n")
    else:
        print("No PSI_Drift_Detection run found in MLflow!")
        any_drift = False
        drifted_features = ""
except Exception as e:
    print(f"MLflow fetch error: {e}")
    raise SystemExit(1)

try:
    champ_runs = mlflow.search_runs(filter_string="tags.Champion = 'True'", order_by=["start_time DESC"], max_results=1)
    if champ_runs.empty:
        print("No Champion model found in MLflow!")
        raise SystemExit(1)
        
    champ_run_id = champ_runs.iloc[0].run_id
    champion_name = str(champ_runs.iloc[0].get("params.model_name", "Champion Model"))
    
    artifact_path = "xgboost_model" if "xgb" in champion_name.lower() else "sgd_model"
    champion_model = mlflow.sklearn.load_model(f"runs:/{champ_run_id}/{artifact_path}")
except Exception as e:
    print(f"MLflow model load error: {e}")
    raise SystemExit(1)

from data_preprocess import get_home_credit_data
_, X_hc_test, _, y_hc_test = get_home_credit_data()

X_hc_eval = X_hc_test.copy()
if any_drift:
    X_hc_eval['AMT_INCOME_TOTAL'] *= 0.7
    X_hc_eval['AMT_CREDIT'] *= 1.3
    X_hc_eval['AMT_ANNUITY'] *= 1.2

champion_res = evaluate_model(f"{champion_name} (Loaded Baseline)", champion_model, X_hc_eval, y_hc_test.values, threshold=0.5)


selected = champion_res
verdict = "NO DRIFT — champion stays"
all_results = [champion_res]

if not any_drift:
    print("\nNo drift detected (PSI < 0.2) — champion XGBoost stays in production.")

else:
    print(f"\nDrift detected on: {drifted_features}")
    print("Retraining SGD challenger on drifted data...\n")

    Xg_train_drifted = Xg_train.copy()
    Xg_test_drifted = Xg_test.copy()

    for df_ in [Xg_train_drifted, Xg_test_drifted]:
        if 'credit_amount' in df_.columns:
            df_['credit_amount'] *= 1.3
        if 'installment_commitment' in df_.columns:
            df_['installment_commitment'] *= 1.2

       
        if 'credit_per_month' in df_.columns:
            df_['credit_per_month'] = df_['credit_amount'] / df_['duration'].clip(lower=1)
        if 'credit_to_age' in df_.columns:
            df_['credit_to_age'] = df_['credit_amount'] / df_['age'].clip(lower=1)
        if 'total_commitment' in df_.columns:
            df_['total_commitment'] = df_['installment_commitment'] * df_['duration']

    scaler_new = StandardScaler()
    Xg_train_drifted_sc = scaler_new.fit_transform(Xg_train_drifted)
    Xg_test_drifted_sc = scaler_new.transform(Xg_test_drifted)
    
    print(f"Drifted training data: {Xg_train_drifted_sc.shape}")

    from sklearn.linear_model import SGDClassifier
    
    
    sgd_search = RandomizedSearchCV(
        SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=3000, tol=1e-3),
        param_distributions={
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'class_weight': ['balanced', None],
        },
        n_iter=20, cv=5, scoring='accuracy',
        random_state=42, n_jobs=-1,
    )
    sgd_search.fit(Xg_train_drifted_sc, yg_train_arr)
    print(f"Best params: {sgd_search.best_params_}")
    print(f"Best CV accuracy: {sgd_search.best_score_:.4f}")

    challenger_res = evaluate_model("SGD (Challenger)", sgd_search.best_estimator_, Xg_test_drifted_sc, yg_test_arr)

    all_results = [champion_res, challenger_res]

    selected = challenger_res
    verdict = "DRIFT DETECTED — switched to challenger"



print_comparison(all_results, title="Final Comparison" + (" (Post-Drift)" if any_drift else ""))
print(f"\nSelected model: {selected['name']} | Verdict: {verdict}")

with mlflow.start_run(run_name="German_Credit_Evaluation") as run:
    mlflow.log_param("selected_model", selected["name"])
    mlflow.log_param("verdict", verdict)
    mlflow.log_param("any_drift", any_drift)
    mlflow.log_metric("final_auc", selected["auc"])
    mlflow.log_metric("final_f1", selected["f1"])
    mlflow.log_metric("final_accuracy", selected["accuracy"])
    
print("Logged final evaluation to MLflow.")
