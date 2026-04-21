"""
Model Training — Baseline Models on Home Credit data.
Uses scikit-learn Pipeline and MLflow tracking.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, accuracy_score
import mlflow

import mlflow.sklearn
from mlflow.tracking import MlflowClient

from data_preprocess import get_home_credit_data, NUM_FEATURES, CAT_FEATURES

def evaluate_and_log_pipeline(name, pipeline, X_tr, y_tr, X_te, y_te):
    """Fit pipeline, evaluate on test set, print report, and log strictly to MLflow."""
    with mlflow.start_run(run_name=name) as run:
        print(f"\nTraining {name}...")
        pipeline.fit(X_tr, y_tr)
        
        y_pred = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_proba)
        
        print(f"{name} Results:")
        print(f"  Unique Predictions: {np.unique(y_pred)}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(classification_report(y_te, y_pred, target_names=["No Default", "Default"], zero_division=0))
        
        mlflow.log_params({
            "model_name": name,
            "pipeline_steps": str([step[0] for step in pipeline.steps])
        })
        
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        })
        
        model_name = f"{name.lower()}_model"
        if "sgd" in model_name or "svm" in model_name: 
            model_name = "sgd_model"  
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            name=model_name, 
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        )
        
        run_id = run.info.run_id
    
    return {
        "name": name,
        "auc": auc,
        "run_id": run_id
    }

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_home_credit_data()
    
    
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES)
    ])
    
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            use_label_encoder=False, eval_metric="auc", 
            scale_pos_weight=imbalance_ratio, random_state=42, n_jobs=-1
        ))
    ])
    
    sgd_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=3000,
            tol=1e-3,
            random_state=42
        ))
    ])
    
    results = []
    results.append(evaluate_and_log_pipeline("XGBoost", xgb_pipeline, X_train, y_train, X_test, y_test))
    results.append(evaluate_and_log_pipeline("SGD", sgd_pipeline, X_train, y_train, X_test, y_test))
    
    best_model = max(results, key=lambda x: x["auc"])
    print(f"\nChampion Model: {best_model['name']} with AUC={best_model['auc']:.4f}")
    
    client = MlflowClient()
    client.set_tag(best_model["run_id"], "Champion", "True")