import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from preprocessing import load_and_preprocess_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")


def generate_adversarial_data(X, noise_level=0.1):
    """Add Gaussian noise to simulate adversarial evasion."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def run_adversarial_attack(noise_level=0.1):
    print("=" * 55)
    print("  ADVERSARIAL ATTACK SIMULATION")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    _, X_test, _, y_test, _ = load_and_preprocess_data()
    y_test = y_test.astype(int)

    # Baseline accuracy
    y_pred_base = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, y_pred_base)
    print(f"  Baseline Accuracy      : {baseline_acc * 100:.2f}%")

    # Adversarial accuracy
    X_test_adv = generate_adversarial_data(X_test, noise_level)
    y_pred_adv = (model.predict_proba(X_test_adv)[:, 1] > 0.5).astype(int)
    adv_acc = accuracy_score(y_test, y_pred_adv)
    print(f"  Adversarial Accuracy   : {adv_acc * 100:.2f}%")
    print(f"  Accuracy Drop          : {(baseline_acc - adv_acc) * 100:.2f}%")
    print(f"  Attack demonstrates model vulnerability.")

    # Save adversarial test set for defense step
    adv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "X_test_adv.npy")
    np.save(adv_path, X_test_adv)
    print(f"  Adversarial data saved → {adv_path}")

    return {
        "baseline_accuracy": round(baseline_acc * 100, 2),
        "adversarial_accuracy": round(adv_acc * 100, 2),
        "accuracy_drop": round((baseline_acc - adv_acc) * 100, 2)
    }


if __name__ == "__main__":
    run_adversarial_attack()
