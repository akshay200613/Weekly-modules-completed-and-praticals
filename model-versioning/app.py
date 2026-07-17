from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

model = mlflow.pyfunc.load_model(
    "models:/IrisClassifier/Production"
)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["features"]

    prediction = model.predict([data])

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)