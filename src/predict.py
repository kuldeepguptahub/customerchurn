from math import e
import joblib
import numpy as np


def make_predictions(input_array: np.ndarray, model_path: str) -> float:
    '''
    This function makes prediction for churn probability based on user input based. 
    Uses the pre-trained classification model.

    Parameters:
    input_array (np.ndarray): numpy array of preprocessed data.
    model_path (str): Path where pre-trained model is saved

    Returns:
    Churn Probability (float) 
    '''
    assert isinstance(input_array, np.ndarray)

    try:
        model = joblib.load(model_path)
    except:
        raise FileNotFoundError(f'Model file not found at {model_path}') from e
    
    try:
        prediction = model.predict(input_array)
    except:
        raise ValueError('Error during prediction.') from e
    
    return prediction