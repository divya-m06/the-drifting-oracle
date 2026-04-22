# Drift Oracle — XGBoost Champion / Challenger Pipeline

## Problem Statement
A credit scoring model trained on **Home Credit** historical data is scoring
applicants in a post-inflation economy.  Feature distributions may have shifted
and nobody noticed.  This pipeline:

1. Trains an **XGBoost Champion** on Home Credit data and registers it in MLflow.
2. Detects **distribution drift** (PSI) between Home Credit and a German Credit
   incoming batch on 5 aligned features.
3. **Triggers retraining** of an XGBoost Challenger (on German Credit) only when
   PSI ≥ 0.20 on any feature.
4. Compares Champion vs Challenger and logs **AUC-ROC curves**, accuracy, and F1
   for both to MLflow.

---

## Directory layout

```
drift-oracle/
├── data/
│   ├── application_train.csv   ← Home Credit (Kaggle)
│   └── german_data.csv         ← German Credit (UCI)
├── data_preprocess.py
├── train_model.py
├── drift_detection.py
├── german_credit.py
└── README.md
```

## Dataset download

| Dataset | Link |
|---------|------|
| Home Credit Default Risk | https://www.kaggle.com/competitions/home-credit-default-risk/data |
| German Credit | https://www.kaggle.com/datasets/uciml/german-credit |

Place both CSV files in `data/`.

---

## Execution order

```bash
# 1. Sanity-check preprocessing
python data_preprocess.py

# 2. Train XGBoost Champion on Home Credit → tags run as Champion in MLflow
python train_model.py

# 3. Compute PSI (Home Credit vs German Credit) → logs drift results to MLflow
python drift_detection.py

# 4. Evaluate + conditionally retrain → logs AUC curves + metrics to MLflow
python german_credit.py
```

---

## PSI thresholds

| PSI | Status | Action |
|-----|--------|--------|
| < 0.10 | STABLE | No action |
| 0.10 – 0.20 | WARN | Monitor |
| ≥ 0.20 | **DRIFT** | **Retrain Challenger** |

---

## MLflow runs created

| Run name | Contents |
|----------|----------|
| `XGBoost_Champion` | Champion model artifact + Home Credit metrics |
| `PSI_Drift_Detection` | PSI per feature, `any_drift` flag, drifted feature list |
| `German_Credit_Evaluation` | Champion & (optional) Challenger metrics, ROC PNG, selected model |

---

## Key design decisions

* **XGBoost only** — SGD / SVM removed; a single model family makes the
  Champion/Challenger comparison meaningful.
* **German Credit as drift batch only** — Home Credit is never modified;
  the German Credit dataset is the incoming distribution.
* **Challenger trained only on drift** — if PSI < 0.20 for all features the
  pipeline exits after Champion evaluation with no unnecessary retraining.
* **AUC-ROC plot** — saved as `roc_curves.png` and logged as an MLflow artifact;
  shows both curves when a Challenger is trained, single curve otherwise.
