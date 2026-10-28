from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__, static_folder="static")
CORS(app)

model = None
scaler = None
feature_columns = None
X_test_global = None
y_test_global = None


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn)
        }
    }


def apply_attack(X, epsilon=0.4):
    noise = np.random.normal(0, epsilon, X.shape)
    return X + noise


def load_or_train_model():
    global model, scaler, feature_columns, X_test_global, y_test_global

    print("Loading lightweight demo model...")

    np.random.seed(42)

    # SMALLER DATA → faster
    normal = np.random.normal(0, 1.5, (1000, 20))
    attack = np.random.normal(1.5, 1.8, (800, 20))

    X = np.vstack([normal, attack])
    y = np.array([0]*1000 + [1]*800)

    feature_columns = [f"feature_{i}" for i in range(20)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMALLER MODEL → faster
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    model.fit(X_train, y_train)

    X_test_global = X_test
    y_test_global = y_test

    print(" Model ready (fast mode).")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/baseline")
def baseline():
    y_pred = model.predict(X_test_global)
    return jsonify(compute_metrics(y_test_global, y_pred))


@app.route("/attack")
def attack():
    X_adv  = apply_attack(X_test_global)
    y_pred = model.predict(X_adv)
    return jsonify(compute_metrics(y_test_global, y_pred))


@app.route("/defense")
def defense():
    X_adv      = apply_attack(X_test_global)
    X_combined = np.vstack([X_test_global, X_adv])
    y_combined = np.concatenate([y_test_global, y_test_global])
    defended_model = RandomForestClassifier(n_estimators=120)
    defended_model.fit(X_combined, y_combined)
    y_pred = defended_model.predict(apply_attack(X_test_global))
    return jsonify(compute_metrics(y_test_global, y_pred))


@app.route("/upload_pipeline", methods=["POST"])
def upload_pipeline():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 400

    # Find label column automatically
    label_col = None
    for col in df.columns:
        if col.strip().lower() in ("label", "class", "target", "attack", "attack_type"):
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]

    # Convert label to binary
    y_raw = df[label_col].astype(str).str.strip().str.lower()
    y = y_raw.apply(lambda x: 0 if x in ("normal", "0", "benign", "none") else 1).values

    # Prepare features
    X_df = df.drop(columns=[label_col])
    le = LabelEncoder()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    X_df = X_df.fillna(0).replace([np.inf, -np.inf], 0)

    if len(X_df) < 10:
        return jsonify({"error": "Dataset too small (need at least 10 rows)"}), 400

    upload_scaler = StandardScaler()
    X_scaled = upload_scaler.fit_transform(X_df.values)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train on uploaded data
    upload_model = RandomForestClassifier(n_estimators=100, random_state=42)
    upload_model.fit(X_train, y_train)

    # Baseline
    base = compute_metrics(y_test, upload_model.predict(X_test))

    # Attack
    X_adv = apply_attack(X_test)
    atk   = compute_metrics(y_test, upload_model.predict(X_adv))

    # Defense
    X_comb = np.vstack([X_test, X_adv])
    y_comb = np.concatenate([y_test, y_test])
    def_model = RandomForestClassifier(n_estimators=100)
    def_model.fit(X_comb, y_comb)
    dfn = compute_metrics(y_test, def_model.predict(apply_attack(X_test)))

    return jsonify({
        "baseline": base,
        "attack":   atk,
        "defense":  dfn,
        "rows_evaluated": len(y_test),
        "total_rows": len(y)
    })


if __name__ == "__main__":
    load_or_train_model()
    app.run(debug=True, port=5000)
