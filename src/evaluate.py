import os
import json
import pandas as pd
import joblib
import mlflow
import dagshub

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("CVD_Heart_Disease_Evaluation")

DATA_PATH = "data/processed/preprocessed_dataset.csv"
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "metrics.json"

def evaluate():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Heart_Disease"])
    y = df["Heart_Disease"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": joblib.load(os.path.join(MODEL_DIR, "LogisticRegression.pkl")),
        "RandomForest": joblib.load(os.path.join(MODEL_DIR, "RandomForest.pkl")),
        "GradientBoosting": joblib.load(os.path.join(MODEL_DIR, "GradientBoosting.pkl")),
    }

    all_metrics = {}
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=f"eval_{name}"):
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
            }

            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)

            all_metrics[name] = metrics

            print(f"{name} evaluated | F1={metrics['f1_score']:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model_name = name
                best_model = model

    joblib.dump(best_model, BEST_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_f1_score": best_f1,
                "all_models": all_metrics,
            },
            f,
            indent=4
        )

    print(f"\n Best model: {best_model_name}")
    print(f"Saved to: {BEST_MODEL_PATH}")
    print(f" Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    evaluate()
