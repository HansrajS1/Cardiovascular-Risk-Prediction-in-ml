import os
import pandas as pd
import mlflow

SOURCE_PATH = "data/source/CVD_cleaned.csv"
RAW_PATH = "data/raw/CVD_cleaned.csv"

def ingest_data():
    os.makedirs("data/raw", exist_ok=True)

    df = pd.read_csv(SOURCE_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    if "Heart_Disease" not in df.columns:
        raise ValueError("Target column missing")

    df.to_csv(RAW_PATH, index=False)

    with mlflow.start_run(run_name="Data_Ingestion"):
        mlflow.log_param("source", SOURCE_PATH)
        mlflow.log_metric("rows", df.shape[0])
        mlflow.log_metric("columns", df.shape[1])

    print(" Data ingestion completed")

if __name__ == "__main__":
    ingest_data()
