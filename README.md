# 🧠 Customer Churn Prediction & Retention System

A machine learning project to predict telecom customer churn and explain the key factors influencing that risk — presented via an interactive, Streamlit-powered dashboard.

---

## 📌 Objective

- Predict customer churn probability with high recall
- Understand what drives churn at an individual level
- Build an explainable, lightweight web app using Streamlit
- Package it as a professional, portfolio-grade ML project

---

## 📊 Dataset

- Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Records: 7,043 customers
- Target: `Churn` (Yes/No)
- Features: Demographics, subscription plans, billing & support info

---

## 🚀 How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/kuldeepnethues/customerchurn.git
cd customerchurn
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit App

```bash
streamlit run app/app.py
```

---

## 🧪 Model Pipeline Overview

* **EDA**: Exploratory visualizations and pattern mining
* **Preprocessing**: OneHotEncoding + StandardScaling using ColumnTransformer
* **Modeling**: Logistic Regression, Decision Tree, Random Forest
* **Tuning**: GridSearchCV + class balancing (SMOTE, weights)
* **Explainability**: SHAP for local + global feature impact

---

## 📈 Sample Output

Here’s a sample prediction with SHAP-based explanation:

![Churn Prediction Output](https://github.com/kuldeepnethues/customerchurn/blob/main/sample_churn_prediction.png)

> In this case, the customer is predicted to churn with 72.92% probability — largely influenced by short tenure, monthly contract, and lack of online security.

---

## ✅ Future Enhancement

* Integrate GenAI to suggest personalized retention actions based on prediction and SHAP drivers

---

## 📄 License

MIT License © kuldeepnethues
