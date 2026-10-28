import os
import pickle
import pandas as pd
import numpy as np

# CONFIG 
THRESHOLD    = 0.5
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
NEW_DATA_PATH   = os.path.join(DATA_DIR, "new_traffic.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")
MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH   = os.path.join(MODELS_DIR, "feature_names.pkl") 


# LOAD ARTIFACTS 
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}\nRun train_model.py first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}\nRun train_model.py first.")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature names not found at: {FEATURES_PATH}\nRun train_model.py first.")

    with open(MODEL_PATH,    "rb") as f: model         = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler        = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feature_names = pickle.load(f)

    return model, scaler, feature_names


#PREDICT FROM DATAFRAME
def predict_from_df(df: pd.DataFrame, model, scaler, feature_names, threshold=THRESHOLD):
    """
    Core prediction function.
    Accepts a raw DataFrame, returns DataFrame with predictions appended.
    """
    # Drop label columns if present (in case test CSV included them)
    df = df.drop(columns=["label", "attack_cat"], errors="ignore")

    # Align to training features
    df = df.reindex(columns=feature_names, fill_value=0)

    # Fill any NaNs
    df.fillna(0, inplace=True)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    result = df.copy()
    result["attack_probability"] = np.round(y_prob, 4)
    result["predicted_label"]    = y_pred
    result["prediction_text"]    = result["predicted_label"].map({0: "NORMAL", 1: "ATTACK"})

    return result


# PREDICT FROM CSV FILE 
def predict_from_csv(csv_path=NEW_DATA_PATH, threshold=THRESHOLD):
    model, scaler, feature_names = load_artifacts()

    if not os.path.exists(csv_path):
        print(f" {csv_path} not found — generating dummy data for testing...")
        os.makedirs(DATA_DIR, exist_ok=True)
        dummy = pd.DataFrame(
            np.random.rand(10, len(feature_names)),
            columns=feature_names
        )
        dummy.to_csv(csv_path, index=False)
        print(f"   Dummy file created: {csv_path}")

    df = pd.read_csv(csv_path)
    result = predict_from_df(df, model, scaler, feature_names, threshold)

    # Save predictions
    result.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved → {PREDICTIONS_CSV}")
    print(result[["attack_probability", "predicted_label", "prediction_text"]].head(10))

    return result

if __name__ == "__main__":
    predict_from_csv()
