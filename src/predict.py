import joblib
import numpy as np


def make_predictions(input_array: np.ndarray, model_path: str) -> str:
    '''
    This function makes prediction for churn probability based on user input based. 
    Uses the pre-trained classification model.

    Parameters:
    input_array (np.ndarray): numpy array of preprocessed data.
    model_path (str): Path where pre-trained model is saved

    Returns:
    String predicting class and probablity
    '''
    assert isinstance(input_array, np.ndarray)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise FileNotFoundError(f'Model file not found at {model_path}') from e
    
    try:
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)
    except Exception as e:
        raise ValueError('Error during prediction.') from e
    
    return f'Prediction: {"Churn" if prediction[0] == 1 else "Not Churn"}, Churn Probability: {probability[0][1]:.2%}'