# app/app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt

# Local modules
from src.input_schema import create_df
from src.preprocessing import preprocess_data
from src.predict import make_predictions
from src.explain import explain

# Constants
MODEL_PATH = "models/randomforest.pkl"
TRANSFORMER_PATH = "models/transformer.pkl"
FEATURE_NAMES_PATH = "data/processed/feature_columns.json"

# App layout
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")

with st.form("user_input_form"):
    st.subheader("📥 Enter Customer Info")

    # Example inputs (customize these!)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (in months)", 0, 150, 1)
    phone = st.selectbox("Phone Service", ['Yes', 'No'])
    multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No Internet Service'])
    internet = st.selectbox("Internet Service", ['DSL', 'Fiber Optic', 'No'])
    onlinesecurity = st.selectbox("Online Security", ['Yes', 'No', 'No Internet Service'])
    onlinebackup = st.selectbox("Online Backup", ['Yes', 'No', 'No Internet Service'])
    deviceprotection = st.selectbox("Device Protection", ['Yes', 'No', 'No Internet Service'])
    techsupport = st.selectbox("Tech Support", ['Yes', 'No', 'No Internet Service'])
    tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No Internet Service'])
    movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No Internet Service'])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
    paymentmethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    
    

    submitted = st.form_submit_button("Predict")

if submitted:
    # 1. Format user input
    user_input = {
        "GENDER": gender,
        "SENIORCITIZEN": senior,
        "PARTNER": partner,  # Add based on your form
        "DEPENDENTS": dependents,
        "TENURE": tenure,
        "PHONESERVICE": phone,
        "MULTIPLELINES": multiplelines,
        "INTERNETSERVICE": internet,
        "ONLINESECURITY": onlinesecurity,
        "ONLINEBACKUP": onlinebackup,
        "DEVICEPROTECTION": deviceprotection,
        "TECHSUPPORT": techsupport,
        "STREAMINGTV": tv,
        "STREAMINGMOVIES": movies,
        "CONTRACT": contract,
        "PAPERLESSBILLING": paperless,
        "PAYMENTMETHOD": paymentmethod,
        "MONTHLYCHARGES": monthly_charges,
        "TOTALCHARGES": total_charges,
    }

    try:
        input_df = create_df(user_input)
        X = preprocess_data(input_df, TRANSFORMER_PATH)
        st.write(make_predictions(X, MODEL_PATH))
        
        # SHAP
        st.subheader("🔍 Feature Impact (SHAP)")
        shap_values, X_df = explain(MODEL_PATH, X, FEATURE_NAMES_PATH)
        
        fig, ax = plt.subplots()  # create a thread-safe figure object
        shap.summary_plot(shap_values[1], X_df, plot_type='bar', show=False)  # modify global figure
        st.pyplot(fig)  # safely render the figure
        plt.clf()

    except Exception as e:
        st.error(f"Something went wrong: {e}")
