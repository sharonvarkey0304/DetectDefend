import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#  PATH SETUP 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH  = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")


# LOAD DATA 
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


# MAIN PREPROCESSING 
def load_and_preprocess_data():
    train_df = load_dataset(TRAIN_PATH)
    test_df  = load_dataset(TEST_PATH)

    # Drop ID column if present
    for df in [train_df, test_df]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)
        df.fillna(0, inplace=True)

    # Validate label column
    if "label" not in train_df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    if "label" not in test_df.columns:
        raise ValueError("Test dataset must contain a 'label' column.")

    # Separate features and labels
    y_train = train_df["label"]
    y_test  = test_df["label"]

    X_train = train_df.drop(columns=["label", "attack_cat"], errors="ignore")
    X_test  = test_df.drop(columns=["label", "attack_cat"], errors="ignore")

    # Align columns — test must match train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    #  Encode categorical columns 
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols]  = encoder.transform(X_test[categorical_cols])

        # Save encoder for use in predict.py
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

    #  Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler and feature names for predict.py
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


# STANDALONE TEST 
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)
    print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Preprocessing completed successfully.")
