# ❤️ Heart Disease Prediction App

A Streamlit web app that predicts the likelihood of heart disease using a
Logistic Regression model trained on the UCI Heart Disease dataset.

## 🔍 Overview

- **Model:** Logistic Regression (Pipeline with OneHotEncoder + StandardScaler)
- **Accuracy:** 88% on held-out test set
- **Dataset:** UCI Heart Disease dataset (1,025 rows → cleaned to ~300 unique patients)

## 📊 Project Workflow

1. Data cleaning — removed duplicates and invalid category codes (`ca=4`, `thal=0`)
1. Exploratory Data Analysis — outlier detection, correlation analysis
1. Preprocessing — OneHotEncoding for categorical features, StandardScaler for numeric
1. Model comparison — Logistic Regression vs Random Forest (cross-validated)
1. Deployment — Streamlit app with interactive input form

## 🚀 Run Locally

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Files

|File                        |Purpose                               |
|----------------------------|--------------------------------------|
|`app.py`                    |Streamlit application                 |
|`heart_disease_pipeline.pkl`|Trained model + preprocessing pipeline|
|`requirements.txt`          |Python dependencies                   |

## ⚠️ Disclaimer

This app is for educational/demo purposes only and is not a substitute for
professional medical advice, diagnosis, or treatment.