# Cardiovascular Risk Prediction using Machine Learning

This project predicts the **10-year risk of coronary heart disease (CHD)** using machine learning algorithms based on clinical and demographic data. It aims to assist healthcare professionals in early risk assessment and preventive care decisions.

The project is **fully MLOps-enabled** with **DVC pipelines** and **MLflow experiment tracking via DAGsHub**.

live link : https://cardiovascular-risk-prediction.hansrajvvs.me/
---

## Key Features

- **Data ingestion & validation** using Python scripts  
- **Preprocessing**: handling missing values, scaling, and encoding  
- **Exploratory Data Analysis (EDA)** with visual insights  
- **Feature selection** using correlation, ANOVA F-test, RFE, and Random Forest importance  
- **Model training** with:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Experiment tracking** with **MLflow** on **DAGsHub**  
- **Evaluation metrics**: Accuracy, ROC-AUC, Precision, Recall, F1 Score  
- **Risk probability prediction** for individual cases  
- **DVC pipelines** for reproducible workflows  

---

## Dataset

- **Source**: [CVD_cleaned.zip] (include a link if possible)  
- **Attributes**: Age, Sex, Blood Pressure, Cholesterol, Smoking, Diabetes, and more  
- **Size**: ~308,854 samples × 19 features  

---

## Tech Stack

| Category           | Tools / Libraries                 |
|-------------------|----------------------------------|
| Language           | Python 3.10                     |
| Data Handling      | Pandas, NumPy                   |
| Visualization      | Matplotlib, Seaborn             |
| ML Algorithms      | scikit-learn                    |
| Experiment Tracking| MLflow + DAGsHub                 |
| Pipelines          | DVC                             |
| Environment        | Jupyter Notebook / VS Code       |

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|-------------------|---------|----------|--------|----------|
| Logistic Regression| 0.9188  | 0.5169   | 0.0648 | 0.1152   |
| Random Forest      | 0.9177  | 0.4528   | 0.0445 | 0.0810   |
| Gradient Boosting  | 0.9187  | 0.5125   | 0.0515 | 0.0936   |

> *Random Forest performed best in terms of balanced accuracy and ROC-AUC.*

---

## DAGsHub Integration

All experiments and metrics are tracked on **DAGsHub MLflow**:

[https://dagshub.com/HansrajS1/Cardiovascular-Risk-Prediction-in-ml.mlflow](https://dagshub.com/HansrajS1/Cardiovascular-Risk-Prediction-in-ml.mlflow)

- Logs hyperparameters, metrics, and models automatically  
- Compare experiment runs visually  
- Download trained models from the Artifacts tab  

---

## How to Run

1. **Clone the repo**

```bash
git clone https://github.com/HansrajS1/Cardiovascular-Risk-Prediction-in-ml.git
cd Cardiovascular-Risk-Prediction-in-ml
```

2. Create a virtual environment

```bash
python -m venv venv

venv\Scripts\activate

source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```
4. Setup DVC remote (optional)

```bash
dvc remote add -d storage <remote-url>
dvc pull
```
5. Setup DVC remote (optional)

```bash
dvc repro
```

6. Check MLflow experiments on DAGsHub

https://dagshub.com/HansrajS1/Cardiovascular-Risk-Prediction-in-ml.mlflow

