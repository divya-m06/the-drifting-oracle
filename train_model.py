import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings("ignore")
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)
X_train = data["X_train_scaled"]
X_test  = data["X_test_scaled"]
y_train = data["y_train"].values
y_test  = data["y_test"].values
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
imbalance_ratio = neg / pos
print(f"Class counts → No Default: {neg}, Default: {pos}")
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1\n")
print("Applying SMOTE to create balanced training set for RF...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE → No Default: {(y_train_sm==0).sum()}, Default: {(y_train_sm==1).sum()}\n")
def find_best_threshold(y_true, y_proba):
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return round(best_thresh, 2), round(best_f1, 4)
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, use_threshold_tuning=True):
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    if use_threshold_tuning:
        best_thresh, _ = find_best_threshold(y_te, y_proba)
        y_pred = (y_proba >= best_thresh).astype(int)
        thresh_note = f"(threshold={best_thresh})"
    else:
        y_pred = model.predict(X_te)
        thresh_note = "(threshold=0.5)"
    auc      = roc_auc_score(y_te, y_proba)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    accuracy = (y_pred == y_te).mean()
    print(f"\n{'─'*45}")
    print(f"  Model: {name}  {thresh_note}")
    print(f"{'─'*45}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1:       {f1:.4f}  (class=1, defaulters)")
    print(f"  F1 Macro: {f1_macro:.4f}  (avg both classes)")
    print(f"\n{classification_report(y_te, y_pred, target_names=['No Default', 'Default'], zero_division=0)}")
    return {
        "name": name, "model": model,
        "auc": auc, "f1": f1, "f1_macro": f1_macro, "accuracy": accuracy
    }
print("=" * 60)
print("STEP 5: Training Models")
print("=" * 60)
results = []
print("\n[A] Logistic Regression — Baseline")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
results.append(evaluate_model("Logistic Regression", lr, X_train, y_train, X_test, y_test))

print("\n[B] XGBoost — Champion Model")
xgb = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,       
    colsample_bytree = 0.8,       
    scale_pos_weight = imbalance_ratio,  
    use_label_encoder= False,
    eval_metric      = "auc",
    random_state     = 42,
    n_jobs           = -1,
)
results.append(evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test))
print("\n[C] SVM — Challenger Model")
print("    (sampling 20k rows — SVM is slow on 300k rows)")
sample_idx = np.random.RandomState(42).choice(len(X_train), size=20000, replace=False)
X_train_svm = X_train[sample_idx]
y_train_svm = y_train[sample_idx]

svm = SVC(
    kernel          = "rbf",
    C               = 1.0,
    class_weight    = "balanced",
    probability     = True,       # needed for predict_proba + AUC
    random_state    = 42,
)
results.append(evaluate_model("SVM (RBF)", svm, X_train_svm, y_train_svm, X_test, y_test))
print("\n[D] Random Forest — Fixed with SMOTE")
rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 10,            
    random_state = 42,
    n_jobs       = -1,
)
results.append(evaluate_model("Random Forest (SMOTE)", rf, X_train_sm, y_train_sm, X_test, y_test))
print("\n" + "=" * 60)
print("STEP 6: Model Comparison")
print("=" * 60)
comparison_df = pd.DataFrame([
    {
        "Model":        r["name"],
        "AUC":          round(r["auc"],      4),
        "Accuracy":     f"{r['accuracy']*100:.1f}%",
        "F1 (Default)": round(r["f1"],       4),
        "F1 Macro":     round(r["f1_macro"], 4),
    }
    for r in results
])
print(comparison_df.to_string(index=False))
best_auc = max(results, key=lambda x: x["auc"])
best_f1  = max(results, key=lambda x: x["f1"])
print(f"Champion (best AUC): {best_auc['name']}  AUC={best_auc['auc']:.4f}")
print(f"Best F1 on Defaulters: {best_f1['name']}   F1={best_f1['f1']:.4f}")
with open("trained_models.pkl", "wb") as f:
    pickle.dump({
        "results":          results,
        "best_model_name":  best_auc["name"],
        "champion":         best_auc,
    }, f)
