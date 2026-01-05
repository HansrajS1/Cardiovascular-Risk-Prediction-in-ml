from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import gc

MODEL_PATH = "models/RandomForest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
model = None
preprocessor = None


def load_artifacts():
    global model, preprocessor
    if model is None or preprocessor is None:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        gc.collect()


app = FastAPI(title="Cardiovascular Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    Exercise: str
    Heart_Disease: str
    Skin_Cancer: str
    Other_Cancer: str
    Depression: str
    Arthritis: str
    Sex: str

    Height_cm: float = Field(..., alias="Height_(cm)")
    Weight_kg: float = Field(..., alias="Weight_(kg)")
    BMI: float

    Smoking_History: str
    Alcohol_Consumption: float
    Fruit_Consumption: float
    Green_Vegetables_Consumption: float
    FriedPotato_Consumption: float

    General_Health_Excellent: bool = False
    General_Health_Fair: bool = False
    General_Health_Good: bool = False
    General_Health_Poor: bool = False
    General_Health_Very_Good: bool = Field(False, alias="General_Health_Very Good")

    Checkup_5_or_more_years_ago: bool = Field(False, alias="Checkup_5 or more years ago")
    Checkup_Never: bool = False
    Checkup_Within_the_past_2_years: bool = Field(False, alias="Checkup_Within the past 2 years")
    Checkup_Within_the_past_5_years: bool = Field(False, alias="Checkup_Within the past 5 years")
    Checkup_Within_the_past_year: bool = Field(False, alias="Checkup_Within the past year")

    Diabetes_No: bool = False
    Diabetes_Pre: bool = Field(False, alias="Diabetes_No, pre-diabetes or borderline diabetes")
    Diabetes_Yes: bool = False
    Diabetes_Pregnancy: bool = Field(False, alias="Diabetes_Yes, but female told only during pregnancy")

    Age_Category_18_24: bool = Field(False, alias="Age_Category_18-24")
    Age_Category_25_29: bool = Field(False, alias="Age_Category_25-29")
    Age_Category_30_34: bool = Field(False, alias="Age_Category_30-34")
    Age_Category_35_39: bool = Field(False, alias="Age_Category_35-39")
    Age_Category_40_44: bool = Field(False, alias="Age_Category_40-44")
    Age_Category_45_49: bool = Field(False, alias="Age_Category_45-49")
    Age_Category_50_54: bool = Field(False, alias="Age_Category_50-54")
    Age_Category_55_59: bool = Field(False, alias="Age_Category_55-59")
    Age_Category_60_64: bool = Field(False, alias="Age_Category_60-64")
    Age_Category_65_69: bool = Field(False, alias="Age_Category_65-69")
    Age_Category_70_74: bool = Field(False, alias="Age_Category_70-74")
    Age_Category_75_79: bool = Field(False, alias="Age_Category_75-79")
    Age_Category_80_plus: bool = Field(False, alias="Age_Category_80+")

@app.get("/")
def home():
    return {"message": "CVD Risk Prediction API is running"}


@app.post("/predict")
def predict(data: PatientInput):
    load_artifacts()

    df = pd.DataFrame([data.dict(by_alias=True)])

    for col, le in preprocessor["label_encoders"].items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    scaler_cols = list(preprocessor["scaler"].feature_names_in_)
    df[scaler_cols] = preprocessor["scaler"].transform(df[scaler_cols])

    if "minmax" in preprocessor and "minmax_columns" in preprocessor:
        df[preprocessor["minmax_columns"]] = preprocessor["minmax"].transform(
            df[preprocessor["minmax_columns"]]
        )

    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = float(model.predict_proba(df)[0][1])

    if prob >= 0.20:
        risk = "High Risk"
        label = 1
    elif prob >= 0.10:
        risk = "Moderate Risk"
        label = 1
    else:
        risk = "Low Risk"
        label = 0

    return {
        "heart_disease_prediction": label,
        "risk_probability": round(prob, 3),
        "result": risk
    }
