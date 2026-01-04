import pandas as pd
import os
import mlflow

RAW_DATA_PATH = "data/raw/CVD_cleaned.csv"
VALIDATED_DATA_PATH = "data/validated/data.csv"

def validate_data():
    df = pd.read_csv(RAW_DATA_PATH)

    duplicate_count = df.duplicated().sum()

    mlflow.log_metric("duplicate_rows", duplicate_count)

    if duplicate_count > 0:
        print(f" Duplicate rows found: {duplicate_count}. Removing duplicates...")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found")

    os.makedirs("data/validated", exist_ok=True)
    df.to_csv(VALIDATED_DATA_PATH, index=False)

    print(" Data validation & cleaning completed")

if __name__ == "__main__":
    validate_data()
