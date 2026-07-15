from flask import Flask, request
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "Flask API is running"

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method =="GET":
        return "Use a POST request with JSON data"
        
    data = request.json["features"]
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}

app.run(host="0.0.0.0", port=5000)