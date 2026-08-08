# 🏥 Synthea Healthcare ETL & Readmission Prediction

An end-to-end healthcare analytics project using **synthetic Synthea data** for ETL, 30-day readmission prediction, explainable AI, fairness analysis, and interactive visualization.

## 🔄 Pipeline

```text
Synthea Data
     ↓
ETL & Feature Engineering
     ↓
Readmission Prediction
     ↓
ML Models
     ↓
SHAP Explainability
     ↓
Fairness Audit
     ↓
Streamlit Dashboard
```

## 📊 Results

| Model               |   ROC-AUC |    PR-AUC |
| ------------------- | --------: | --------: |
| Logistic Regression |     0.783 |     0.238 |
| XGBoost             | **0.826** | **0.333** |

**Dataset:** 1,476 encounters
**30-day readmissions:** 122
**Readmission rate:** 8.3%

## 🔍 Features

* Healthcare data ETL
* Feature engineering
* Logistic Regression & XGBoost
* SHAP-based explainability
* Fairness analysis by gender, race, and age
* Interactive data explorer
* Streamlit dashboard

## 🛠️ Technology

**Python · Pandas · DuckDB · Scikit-learn · XGBoost · SHAP · Plotly · Streamlit · GitHub**

## 🚀 Run Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## ☁️ Deployment

The dashboard is deployed using **Streamlit Community Cloud**.

> **Note:** This project uses synthetic Synthea data for research and educational purposes only and is not intended for clinical decision-making.
