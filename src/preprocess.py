import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib

INPUT_PATH = "data/validated/data.csv"
OUTPUT_PATH = "data/processed/preprocessed_dataset.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def preprocess():
    df = pd.read_csv(INPUT_PATH)

    binary_cols = [
        "Exercise","Heart_Disease","Skin_Cancer","Other_Cancer",
        "Depression","Arthritis","Sex","Smoking_History"
    ]
    label_encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    categorical_cols = ["General_Health","Checkup","Diabetes","Age_Category"]
    df = pd.get_dummies(df, columns=categorical_cols)

    scaler_columns = ["Height_(cm)","Weight_(kg)","BMI"]
    scaler = StandardScaler()
    df[scaler_columns] = scaler.fit_transform(df[scaler_columns])

    consumption_cols = [
        "Alcohol_Consumption","Fruit_Consumption",
        "Green_Vegetables_Consumption","FriedPotato_Consumption"
    ]
    mm = MinMaxScaler()
    df[consumption_cols] = mm.fit_transform(df[consumption_cols])

    df.to_csv(OUTPUT_PATH, index=False)
    print("Data preprocessing completed")

    preprocessor = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "scaler_columns": scaler_columns,
        "minmax": mm,
        "minmax_columns": consumption_cols,
        "dummy_columns": list(df.columns.drop(binary_cols + scaler_columns + consumption_cols + ["Heart_Disease"]))
    }
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    preprocess()
