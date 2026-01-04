import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

INPUT_PATH = "data/validated/data.csv"
OUTPUT_PATH = "data/processed/preprocessed_dataset.csv"

def preprocess():
    df = pd.read_csv(INPUT_PATH)

    binary_cols = [
        "Exercise","Heart_Disease","Skin_Cancer","Other_Cancer",
        "Depression","Arthritis","Sex","Smoking_History"
    ]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(
        df,
        columns=["General_Health","Checkup","Diabetes","Age_Category"]
    )

    scaler = StandardScaler()
    df[["Height_(cm)","Weight_(kg)","BMI"]] = scaler.fit_transform(
        df[["Height_(cm)","Weight_(kg)","BMI"]]
    )

    mm = MinMaxScaler()
    consumption_cols = [
        "Alcohol_Consumption","Fruit_Consumption",
        "Green_Vegetables_Consumption","FriedPotato_Consumption"
    ]
    df[consumption_cols] = mm.fit_transform(df[consumption_cols])

    df.to_csv(OUTPUT_PATH, index=False)
    print(" Data preprocessing completed")

if __name__ == "__main__":
    preprocess()
