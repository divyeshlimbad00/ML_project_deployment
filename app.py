from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import urllib.request
import urllib.parse
import time

app = Flask(__name__)

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cardio_model.pkl")

MODEL_URL = os.environ.get("MODEL_URL")  # optional (for large models)

# ------------------------
# Helpers
# ------------------------
def download_model(url, dest, attempts=3, backoff=1.0):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)
            return True, None
        except Exception as e:
            err = e
            time.sleep(backoff * (2 ** i))
    return False, err


# ------------------------
# Load model
# ------------------------
model = None
model_error = None

if not os.path.exists(MODEL_PATH) and MODEL_URL:
    parsed = urllib.parse.urlparse(MODEL_URL)
    filename = os.path.basename(parsed.path) or "cardio_model.pkl"
    MODEL_PATH = os.path.join(MODEL_DIR, filename)

    ok, err = download_model(MODEL_URL, MODEL_PATH)
    if not ok:
        model_error = f"Failed to download model: {err}"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        model_error = f"Failed to load model: {e}"
else:
    model_error = f"Model not found at {MODEL_PATH}"

if model is None:
    print(f"❌ Model not loaded: {model_error}")

# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    risk = None
    error = None

    if request.method == "POST":
        if model is None:
            return render_template("predict.html", error=model_error)

        try:
            # ⚠️ MUST MATCH training features
            age_years = float(request.form["age"])
            gender = int(request.form["gender"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            ap_hi = float(request.form["ap_hi"])
            ap_lo = float(request.form["ap_lo"])
            cholesterol = int(request.form["cholesterol"])
            smoke = int(request.form["smoke"])
            alco = int(request.form["alco"])
            active = int(request.form["active"])

            X = np.array([[
                age_years,
                gender,
                height,
                weight,
                ap_hi,
                ap_lo,
                cholesterol,
                smoke,
                alco,
                active
            ]])

            probability = model.predict_proba(X)[0][1]
            risk = round(probability * 100, 2)

            return render_template("results.html", risk=risk)

        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template("predict.html", risk=risk, error=error)


# ------------------------
# API Endpoint
# ------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": model_error}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    required = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol",
        "smoke", "alco", "active"
    ]

    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    try:
        X = np.array([[data[k] for k in required]])
        probability = model.predict_proba(X)[0][1]
        return jsonify({
            "cardio_risk": int(probability >= 0.5),
            "risk_probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
