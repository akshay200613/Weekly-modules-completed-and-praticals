import os
import shutil
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============================================
# Configuration
# ============================================

TRAIN_DATA = "data/train.csv"
NEW_DATA = "data/new_train.csv"
MODEL_PATH = "models/model.pkl"
ARCHIVE_FOLDER = "data/archive"
TARGET_COLUMN = "Purchased"


# ============================================
# Function to Train Model
# ============================================

def train_model(dataset_path):
    """Train a Random Forest model and return the model and accuracy."""

    print(f"\nLoading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


# ============================================
# STEP 1 : Train Current Production Model
# ============================================

print("=" * 60)
print("TRAINING CURRENT PRODUCTION MODEL")
print("=" * 60)

old_model, old_accuracy = train_model(TRAIN_DATA)

print(f"\nProduction Model Accuracy : {old_accuracy:.4f}")

# ============================================
# STEP 2 : Check for New Dataset
# ============================================

print("\nChecking for new dataset...")

if not os.path.exists(NEW_DATA):
    print("No new dataset found.")
    exit()

print("New dataset detected.")

# ============================================
# STEP 3 : Create Timestamp
# ============================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

timestamped_dataset = f"data/train_{timestamp}.csv"

# Rename uploaded dataset
os.rename(NEW_DATA, timestamped_dataset)

print(f"Dataset version created : {timestamped_dataset}")

# ============================================
# STEP 4 : Retrain Model
# ============================================

print("\n" + "=" * 60)
print("AUTOMATED RETRAINING STARTED")
print("=" * 60)

new_model, new_accuracy = train_model(timestamped_dataset)

print(f"\nNew Model Accuracy : {new_accuracy:.4f}")

# ============================================
# STEP 5 : Compare Models
# ============================================

print("\nComparing Models...")

if new_accuracy > old_accuracy:

    print("\nNew model performs better.")
    print("Updating production model...")

    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(new_model, MODEL_PATH)

    # Replace production dataset
    shutil.copy(timestamped_dataset, TRAIN_DATA)

    # Create archive folder
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

    # Move processed dataset
    shutil.move(
        timestamped_dataset,
        os.path.join(
            ARCHIVE_FOLDER,
            os.path.basename(timestamped_dataset)
        )
    )

    # MLflow
    mlflow.set_experiment("Automated Retraining")

    with mlflow.start_run():

        mlflow.log_param(
            "production_dataset",
            TRAIN_DATA
        )

        mlflow.log_param(
            "candidate_dataset",
            os.path.basename(timestamped_dataset)
        )

        mlflow.log_metric(
            "production_accuracy",
            old_accuracy
        )

        mlflow.log_metric(
            "candidate_accuracy",
            new_accuracy
        )

        mlflow.sklearn.log_model(
            new_model,
            artifact_path="model"
        )

    print("\nProduction model updated successfully.")
    print("Model saved to:", MODEL_PATH)
    print("Production dataset updated.")
    print("Dataset archived.")
    print("Model logged to MLflow.")

else:

    print("\nNew model is NOT better.")
    print("Keeping existing production model.")

    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

    shutil.move(
        timestamped_dataset,
        os.path.join(
            ARCHIVE_FOLDER,
            os.path.basename(timestamped_dataset)
        )
    )

    print("Rejected dataset archived.")

print("\nPipeline completed successfully.")