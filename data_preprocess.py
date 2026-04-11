import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")
HOME_CREDIT_PATH = "C:/Users/drait/Desktop/drift oracle/drift-oracle/data/application_train.csv"
NUM_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
]
CAT_FEATURES = [
    "NAME_CONTRACT_TYPE",    
    "CODE_GENDER",           
    "FLAG_OWN_CAR",          
    "FLAG_OWN_REALTY",       
    "NAME_INCOME_TYPE",      
    "NAME_EDUCATION_TYPE",   
    "NAME_FAMILY_STATUS",    
    "NAME_HOUSING_TYPE",     
]
TARGET = "TARGET"
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)
df_hc = pd.read_csv(HOME_CREDIT_PATH)
print(f"Home Credit shape: {df_hc.shape}")
print(f"Target distribution:\n{df_hc[TARGET].value_counts()}\n")
print("=" * 60)
print("STEP 2: Feature Selection & Imputation")
print("=" * 60)
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
df = df_hc[ALL_FEATURES + [TARGET]].copy()
print(f"Before imputation: {df.shape}")
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
num_imputer = SimpleImputer(strategy="median")
df[NUM_FEATURES] = num_imputer.fit_transform(df[NUM_FEATURES])
cat_imputer = SimpleImputer(strategy="most_frequent")
df[CAT_FEATURES] = cat_imputer.fit_transform(df[CAT_FEATURES])
print(f"After imputation (no rows dropped): {df.shape}")
print(f"Missing values remaining: {df.isnull().sum().sum()}\n")
print("=" * 60)
print("STEP 3: One-Hot Encoding")
print("=" * 60)
df_encoded = pd.get_dummies(df, columns=CAT_FEATURES, drop_first=True)
print(f"Shape before OHE: {df.shape}")
print(f"Shape after OHE:  {df_encoded.shape}")
print(f"Columns added:    {df_encoded.shape[1] - df.shape[1]}\n")
print("=" * 60)
print("STEP 4: Train-Test Split & Scaling")
print("=" * 60)
feature_cols = [c for c in df_encoded.columns if c != TARGET]
X = df_encoded[feature_cols].astype(float)
y = df_encoded[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled  = scaler.transform(X_test)         
print(f"Train size: {X_train_scaled.shape}")
print(f"Test size:  {X_test_scaled.shape}")
print(f"\nClass distribution in train:\n{pd.Series(y_train).value_counts()}")
print(f"\nImbalance ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.1f}:1")
with open("processed_data.pkl", "wb") as f:
    pickle.dump({
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "y_train":        y_train,
        "y_test":         y_test,
        "df":             df,            
        "NUM_FEATURES":   NUM_FEATURES,
        "feature_cols":   feature_cols,
    }, f)
print("\nSaved processed_data.pkl — run 2_train_models.py next")
