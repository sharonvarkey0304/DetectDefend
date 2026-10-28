import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_and_preprocess_data
from adversary import generate_adversarial_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
DEFENDED_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model_defended.pkl")


def run_defensive_retraining(noise_level=0.1):
    print("=" * 55)
    print("  DEFENSIVE RETRAINING")
    print("=" * 55)

    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)

    # Load original model for comparison
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Original model not found. Run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        original_model = pickle.load(f)

    # Generate adversarial training data
    X_train_adv = generate_adversarial_data(X_train, noise_level)

    # Combine original + adversarial training data
    X_combined = np.vstack([X_train, X_train_adv])
    y_combined = np.concatenate([y_train, y_train])

    print(f"  Combined training set size: {X_combined.shape[0]} samples")

    # Retrain
    neg = len(y_combined[y_combined == 0])
    pos = len(y_combined[y_combined == 1])
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    defended_model = XGBClassifier(
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
    defended_model.fit(X_combined, y_combined)
    print(" Defensive retraining complete.")

    # Evaluate on adversarial test set
    X_test_adv = generate_adversarial_data(X_test, noise_level)

    # Before defense (original model on adversarial data)
    y_pred_before = (original_model.predict_proba(X_test_adv)[:, 1] > 0.5).astype(int)
    acc_before = accuracy_score(y_test, y_pred_before)

    # After defense (new model on adversarial data)
    y_pred_after = (defended_model.predict_proba(X_test_adv)[:, 1] > 0.5).astype(int)
    acc_after = accuracy_score(y_test, y_pred_after)

    # On clean data
    y_pred_clean = (defended_model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    acc_clean = accuracy_score(y_test, y_pred_clean)

    print(f"\n  Accuracy before defense (on adv data) : {acc_before * 100:.2f}%")
    print(f"  Accuracy after defense  (on adv data) : {acc_after  * 100:.2f}%")
    print(f"  Accuracy after defense  (clean data)  : {acc_clean  * 100:.2f}%")
    print(f"  Improvement: +{(acc_after - acc_before) * 100:.2f}%")
    print("\n  Classification Report (defended model, adversarial test):")
    print(classification_report(y_test, y_pred_after, target_names=["Normal", "Attack"]))

    # Save defended model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(DEFENDED_MODEL_PATH, "wb") as f:
        pickle.dump(defended_model, f)
    print(f" Defended model saved → {DEFENDED_MODEL_PATH}")

    return {
        "accuracy_before_defense": round(acc_before * 100, 2),
        "accuracy_after_defense":  round(acc_after  * 100, 2),
        "accuracy_clean":          round(acc_clean   * 100, 2),
        "improvement":             round((acc_after - acc_before) * 100, 2)
    }


if __name__ == "__main__":
    run_defensive_retraining()
