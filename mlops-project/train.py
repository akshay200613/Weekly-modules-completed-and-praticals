from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X,y)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Training completed")