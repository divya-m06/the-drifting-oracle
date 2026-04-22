"""
Model Training — XGBoost Champion on Home Credit data.

Pipeline:
  1. Load Home Credit train/test splits from data_preprocess.py
  2. Build sklearn Pipeline: impute → scale / OHE → XGBoost
  3. Evaluate on hold-out test set
  4. Log params, metrics, and model artifact to MLflow
  5. Tag the run as Champion in the MLflow tracking server

Run:
    python train_model.py
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier

from data_preprocess import get_home_credit_data, NUM_FEATURES, CAT_FEATURES


# ── helpers ──────────────────────────────────────────────────────────────────

def build_xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    """Returns a fully-defined sklearn Pipeline for XGBoost."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,    NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            use_label_encoder=False,
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate_pipeline(name: str, pipeline: Pipeline,
                      X_tr, y_tr, X_te, y_te) -> dict:
    """Fit pipeline, evaluate, print report, log to MLflow. Returns metrics."""
    with mlflow.start_run(run_name=name) as run:
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")
        pipeline.fit(X_tr, y_tr)

        y_pred  = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1]

        acc      = accuracy_score(y_te,  y_pred)
        prec     = precision_score(y_te, y_pred,  zero_division=0)
        rec      = recall_score(y_te,    y_pred,  zero_division=0)
        f1       = f1_score(y_te,        y_pred,  zero_division=0)
        f1_macro = f1_score(y_te,        y_pred,  average="macro", zero_division=0)
        auc      = roc_auc_score(y_te,   y_proba)

        print(f"  Accuracy  : {acc  * 100:.2f}%")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}  |  F1-Macro : {f1_macro:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(classification_report(
            y_te, y_pred,
            target_names=["No Default", "Default"],
            zero_division=0,
        ))

        # ── MLflow logging ────────────────────────────────────────────────
        mlflow.log_params({
            "model_name":     name,
            "dataset":        "HomeCredit",
            "n_estimators":   300,
            "max_depth":      6,
            "learning_rate":  0.05,
        })
        mlflow.log_metrics({
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "f1_macro":  f1_macro,
            "auc":       auc,
        })
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="xgboost_model",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        run_id = run.info.run_id

    return {
        "name":   name,
        "auc":    auc,
        "f1":     f1,
        "acc":    acc,
        "run_id": run_id,
        "model":  pipeline,
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_home_credit_data()

    # Class imbalance ratio  (neg / pos)
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio (neg/pos): {imbalance_ratio:.1f}")

    pipeline = build_xgb_pipeline(scale_pos_weight=imbalance_ratio)
    result   = evaluate_pipeline(
        "XGBoost_Champion",
        pipeline,
        X_train, y_train,
        X_test,  y_test,
    )

    # ── Tag Champion ─────────────────────────────────────────────────────
    client = MlflowClient()
    client.set_tag(result["run_id"], "Champion", "True")
    print(f"\nChampion model tagged — Run ID: {result['run_id']}")
    print(f"AUC={result['auc']:.4f}  |  F1={result['f1']:.4f}  |  Accuracy={result['acc']*100:.2f}%")
