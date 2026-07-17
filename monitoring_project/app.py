from flask import Flask, request, jsonify
import time
import logging
import random

from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import Gauge
from prometheus_client import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

app = Flask(__name__)

##################################################
# Logging Configuration
##################################################

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

##################################################
# Prometheus Metrics
##################################################

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total Prediction Requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Latency"
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Prediction Errors"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current Model Accuracy"
)

MODEL_ACCURACY.set(0.94)

##################################################
# Prediction API
##################################################

@app.route("/predict", methods=["POST"])
def predict():

    REQUEST_COUNT.inc()

    start = time.time()

    try:

        data = request.get_json()

        x = data["input"]

        time.sleep(random.uniform(0.1,0.8))

        prediction = sum(x)

        latency = time.time() - start

        REQUEST_LATENCY.observe(latency)

        logging.info(f"Input={x} Prediction={prediction}")

        return jsonify({
            "prediction": prediction,
            "latency": latency
        })

    except Exception as e:

        ERROR_COUNT.inc()

        logging.error(str(e))

        return jsonify({
            "error": str(e)
        }),500

##################################################
# Metrics Endpoint
##################################################

@app.route("/metrics")
def metrics():

    return generate_latest(),200,{"Content-Type":CONTENT_TYPE_LATEST}

##################################################
# Home
##################################################

@app.route("/")
def home():

    return "Monitoring Example Running"

##################################################
# Run
##################################################

if __name__=="__main__":

    app.run(host="0.0.0.0",port=5000)