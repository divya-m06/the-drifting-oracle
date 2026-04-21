import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

HOME_CREDIT_PATH = "data/application_train.csv"

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

def get_home_credit_data(path=HOME_CREDIT_PATH):
    """Loads Home Credit dataset and returns raw train/test splits for the pipeline."""
    df_hc = pd.read_csv(path)
    
    ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
    df = df_hc[ALL_FEATURES + [TARGET]].copy()
    
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    
    X = df[ALL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_home_credit_data()
    print(f"Train split shape: {X_train.shape}")
    print(f"Test split shape: {X_test.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}\n")
