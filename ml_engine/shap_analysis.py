import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import shap
from preprocessing import load_and_preprocess_data

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
SHAP_PLOT_PATH = os.path.join(STATIC_DIR, "shap_plot.png")


def run_shap_analysis(max_samples=200):
    print("=" * 55)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load feature names
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "rb") as f:
            feature_names = pickle.load(f)
    else:
        feature_names = None

    _, X_test, _, _, _ = load_and_preprocess_data()

    # Use a sample for speed
    sample = X_test[:max_samples]

    print(f"  Computing SHAP values on {max_samples} samples...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Handle binary classification output shape
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Save SHAP summary plot
    os.makedirs(STATIC_DIR, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        sample,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance — IDS Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f" SHAP plot saved → {SHAP_PLOT_PATH}")

    # Top features summary
    mean_shap = np.abs(shap_vals).mean(axis=0)
    if feature_names:
        top_indices = np.argsort(mean_shap)[::-1][:10]
        print("\n  Top 10 Most Important Features:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"    {rank:2d}. {feature_names[idx]:<30s} SHAP: {mean_shap[idx]:.4f}")

    return SHAP_PLOT_PATH


if __name__ == "__main__":
    run_shap_analysis()
