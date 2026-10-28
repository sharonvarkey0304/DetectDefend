import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_and_preprocess_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train_and_save_model():
    print("=" * 55)
    print(" Loading and preprocessing data")
    print("=" * 55)
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Ensure integer labels
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)

    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test samples  : {X_test.shape[0]}")
    print(f"  Features      : {X_train.shape[1]}")

    # Handle class imbalance
    neg = len(y_train[y_train == 0])
    pos = len(y_train[y_train == 1])
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"  Class balance — Normal: {neg}, Attack: {pos}, Weight: {scale_pos_weight:.2f}")

    print("\n Training XGBoost model")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=4,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(" Model training complete")

    # Evaluate
    print("\n  Evaluating on test set")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n Model saved → {model_path}")

    return model, accuracy


if __name__ == "__main__":
    train_and_save_model()
