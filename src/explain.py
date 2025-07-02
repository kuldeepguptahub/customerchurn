import shap
import joblib
import numpy as np
import pandas as pd
import json
import streamlit as st
import matplotlib.pyplot as plt

def explain(model_path: str, input_array: np.ndarray, feature_path: str):
    '''
    Uses SHAP to explain the prediction from a trained model.

    Displays the SHAP summary plot directly in Streamlit.
    '''

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Load feature names
    try:
        with open(feature_path, 'r') as f:
            features = json.load(f)
        feature_names = [name.replace("cat__", "").replace("num__", "").replace("_", " ") for name in features]
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return

    # Create DataFrame
    try:
        input_df = pd.DataFrame(input_array, columns=feature_names)
    except Exception as e:
        st.error("Mismatch between input shape and feature names.")
        return

    # Compute SHAP values and plot
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = np.transpose(shap_values, (2, 0, 1))  # shape: (2, n_samples, n_features)

        # Display SHAP summary plot in Streamlit
        st.subheader("SHAP Summary Plot (Prediction Explanation)")
        plt.figure()
        shap.summary_plot(shap_values[1], input_df, show=False)
        st.pyplot(plt)
        plt.clf()

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
