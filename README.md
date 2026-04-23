# Drift Oracle — XGBoost Champion / Challenger Pipeline

## Problem Statement

A credit scoring model trained on Home Credit historical data is scoring applicants in a post-inflation economy. Feature distributions may have shifted and nobody noticed. This pipeline:

1. Trains an XGBoost Champion on Home Credit data and registers it in MLflow.
2. Detects distribution drift (PSI) between Home Credit and a German Credit incoming batch on 5 aligned features.
3. Triggers retraining of an XGBoost Challenger (on German Credit) only when PSI ≥ 0.20 on any feature.
4. Compares Champion vs Challenger and logs AUC-ROC curves, accuracy, and F1 for both to MLflow.

---

## Directory Layout

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

---

## Dataset Download

| Dataset | Link |
|---|---|
| Home Credit Default Risk | https://www.kaggle.com/competitions/home-credit-default-risk/data |
| German Credit | https://www.kaggle.com/datasets/uciml/german-credit |

Place both CSV files in `data/`.

---

## Execution Order

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

## PSI Thresholds

| PSI | Status | Action |
|---|---|---|
| < 0.10 | STABLE | No action |
| 0.10 – 0.20 | WARN | Monitor |
| ≥ 0.20 | DRIFT | Retrain Challenger |

---

## MLflow Runs Created

| Run name | Contents |
|---|---|
| `XGBoost_Champion` | Champion model artifact + Home Credit metrics |
| `PSI_Drift_Detection` | PSI per feature, `any_drift` flag, drifted feature list |
| `German_Credit_Evaluation` | Champion & (optional) Challenger metrics, ROC PNG, selected model |

---

## Dependencies

```bash
pip install xgboost scikit-learn mlflow pandas numpy matplotlib
```

---

## Key Design Decisions

- **XGBoost only** — a single model family makes the Champion/Challenger comparison meaningful.
- **German Credit as drift batch only** — Home Credit is never modified; German Credit is the incoming production distribution.
- **Challenger trained only on drift** — if PSI < 0.20 for all features, the pipeline exits after Champion evaluation with no unnecessary retraining.
- **AUC-ROC plot** — saved as `roc_curves.png` and logged as an MLflow artifact. Shows both Champion and Challenger curves when drift is detected and a Challenger is trained; shows the Champion curve only otherwise.

## 📌 Use Cases

- Credit scoring systems in changing economic conditions  
- Fraud detection models affected by shifting user behavior  
- Production ML systems requiring continuous monitoring and retraining  
- Financial risk modeling under dynamic market environments  

## 🚀 Future Improvements

- Integrate real-time streaming data for continuous drift monitoring  
- Support additional drift detection techniques (KS test, KL divergence)  
- Extend pipeline to handle multi-model ensembles beyond XGBoost  
- Automate deployment using CI/CD pipelines with MLflow Model Registry  
- Add dashboard visualization for drift metrics and model performance  
