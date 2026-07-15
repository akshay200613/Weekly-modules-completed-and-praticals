import joblib

def test_model():
    model = joblib.load("model/model.pkl")
    pred = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert pred[0] == 0