from openpyxl.worksheet import datavalidation
from openpyxl.worksheet import datavalidation
from openpyxl.worksheet import datavalidation
from openpyxl.worksheet import datavalidation
from openpyxl.worksheet import datavalidation
from flask import Flask, request, jsonify
import joblib
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Create flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")


# ------ Prometheus metrics ------
REQUEST_COUNT = Counter(
    "prediction_request_total",
    "Total number of prediction requests"
)
REQUEST_LATENCY = Histogram(
    "prediction_request_duration_seconds",
    "prediction request latency"
)

# ---- Home Route ----- 
@app.route("/")
def home():
    return "Flask API is  running.."

# ---- Prediction route -----

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' key"
        }), 400
        
        features = data['features']

        prediction = model.predict([features])[0]

        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# --- Metrcs Route -----

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

# ----- Run app -----

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)