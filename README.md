# ❤️ Cardiovascular Risk Prediction using Machine Learning

This project predicts the **10-year risk of coronary heart disease (CHD)** using machine learning algorithms based on clinical and demographic data from the **Framingham Heart Study** dataset. It aims to assist healthcare professionals in early risk assessment and preventive care decisions.

---

## 🧠 Key Features

- ✅ Preprocessing and handling of missing medical data
- ✅ Exploratory Data Analysis (EDA) with visual insights
- ✅ Feature selection based on correlation and domain knowledge
- ✅ Model training using:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- ✅ Evaluation using Accuracy, ROC-AUC, Precision, Recall, and F1 Score
- ✅ Risk probability prediction for individual cases

---

## 📊 Dataset

- **Source**: [Framingham Heart Study](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)
- **Attributes**: Age, Sex, Blood Pressure, Cholesterol, Smoking, Diabetes, and more
- **Size**: ~4,000 samples × 15 features

---

## 🛠️ Tech Stack

| Category           | Tools / Libraries                 |
|--------------------|-----------------------------------|
| Language           | Python 3.10                       |
| Data Handling      | Pandas, NumPy                     |
| Visualization      | Matplotlib, Seaborn               |
| ML Algorithms      | scikit-learn                      |
| Evaluation Metrics | ROC-AUC, Confusion Matrix, F1     |
| Environment        | Jupyter Notebook / VS Code        |

---

## 📈 Results

| Model               | Accuracy | Precision | F1 Score |
|--------------------|----------|---------|----------|
| Logistic Regression| 0.918797    | 0.516878    |  0.115240    |
| Random Forest      | 0.917696    | 0.452830    | 0.080983     |
| SVM                | 0.918657    | 0.512516    | 0.093566     |

> 📌 *Logistic Regression performed best in terms of balanced accuracy and AUC.*

---

## ▶️ How to Run

```bash
git clone https://github.com/HansrajS1/Cardiovascular-Risk-Prediction-in-ml.git
cd Cardiovascular-Risk-Prediction-in-ml
pip install -r requirements.txt
