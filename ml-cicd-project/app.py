from flask import Flask, request, jsonify
import joblib
from prometheus_client import Counter, Histogram, generate_latest

app = Flask(__name__)

model = joblib.load("model.pkl")

REQUESTS = Counter("prediction_requests_total", "Total_prediction Requests") 

LATENCY = Histogram(
    "prediction_latency_seconds",
    "prediction Latency"
)

@app.route("/predict", methods=["POST"])
@LATENCY.time()
def predict():
    REQUESTS.inc()

    data = request.json["features"]
    prediction = model.predict([data])

    return jsonify(
        {
            "prediction": int(prediction[0])
        }
    )
@app.route("/metrics")
def metrics():
    return generate_latest()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)