import pandas as pd
import joblib
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = "data/processed/preprocessed_dataset.csv"
MODEL_DIR = "models/"

mlflow.set_experiment("CVD_Heart_Disease")

def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Heart_Disease"])
    y = df["Heart_Disease"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=10000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"train_{name}"):
            model.fit(X_train, y_train)
            mlflow.log_param("model_type", name)
            mlflow.sklearn.log_model(model, "model")

            joblib.dump(model, f"{MODEL_DIR}{name}.pkl")
            print(f"{name} trained and saved")

if __name__ == "__main__":
    train()
