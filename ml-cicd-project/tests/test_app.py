import joblib

def test_model():
    model = joblib.load("model.pkl")

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert prediction[0] == 0