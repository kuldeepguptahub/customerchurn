import shap
import joblib
import numpy as np
import pandas as pd
import json

def explain(model_path: str, input_array: np.ndarray, feature_path: str) -> np.ndarray:
    '''
    Uses SHAP to explain the prediction from a trained model.

    Return shap values for summary plots.
    '''

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise FileNotFoundError(f'Model not found at {model_path}') from e

    # Load feature names
    try:
        with open(feature_path, 'r') as f:
            features = json.load(f)
        feature_names = [name.replace("cat__", "").replace("num__", "").replace("_", " ") for name in features]
    except Exception as e:
        raise FileNotFoundError(f'Features not found at {feature_path}') from e

    # Create DataFrame
    try:
        input_df = pd.DataFrame(input_array, columns=feature_names)
    except Exception as e:
        raise ValueError("Mismatch between input shape and feature names.") from e
        

    # Compute SHAP values and plot
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        shap_values = np.transpose(shap_values, (2, 0, 1))  # shape: (2, n_samples, n_features)

    except Exception as e:
        raise ValueError('Error during SHAP calculations') from e
    
    try:
        data_df = pd.DataFrame(input_array, columns=feature_names)
    except Exception as e:
        raise ValueError from e

    return shap_values, data_df
