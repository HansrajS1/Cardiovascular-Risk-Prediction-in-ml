import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/processed/preprocessed_dataset.csv"
MODEL_PATH = "models/best_model.pkl"

mlflow.set_experiment("CVD_Heart_Disease")

def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Heart_Disease"])
    y = df["Heart_Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = LogisticRegression(
            max_iter=10000,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, MODEL_PATH)

        print("Model training completed")
        print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}")

if __name__ == "__main__":
    train()
