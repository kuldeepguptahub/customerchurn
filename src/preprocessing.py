import pandas as pd
import numpy as np
import joblib


def preprocess_data(input_df: pd.DataFrame, transformer_path: str) -> np.ndarray:
    '''
    This function transforms the input data into model compliant numpy array using
    the pre-trained transformer.

    Parameters:
    input_df (pandas dataframe): Single-rowed dataframe prepared from user inputs

    transformer_path (str): Path to saved transformer 
    
    Returns:
    Numpy Array ready for predictions
    '''

    assert isinstance(input_df, pd.DataFrame)
    assert input_df.shape[0] == 1

    try:
        transformer = joblib.load(transformer_path)
    except Exception as e:
        raise FileNotFoundError("Transformer file not found or corrupted.") from e

    try:
        input_array = transformer.transform(input_df)
    except Exception as e:
        raise ValueError("Error during transformation.") from e

    return input_array