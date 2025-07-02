import pandas as pd
import numpy as np

def create_df(user_input: dict) -> pd.DataFrame:
    '''
    Accepts dict of user input from Streamlit form and returns a one-row DataFrame
    with correct structure, column names, and default values if needed.

    Parameters:
    user-input (dict): Key value pairs for customer features

    Returns:
    pd.DataFrame: Single row input in correct format 
    '''

    expected_columns = [
        'GENDER', 'SENIORCITIZEN', 'PARTNER', 'DEPENDENTS', 'TENURE',
        'PHONESERVICE', 'MULTIPLELINES', 'INTERNETSERVICE', 'ONLINESECURITY',
        'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT', 'STREAMINGTV',
        'STREAMINGMOVIES', 'CONTRACT', 'PAPERLESSBILLING', 'PAYMENTMETHOD',
        'MONTHLYCHARGES', 'TOTALCHARGES'
    ]
    
    # validate all columns are present

    missing = [col for col in expected_columns if col not in user_input]

    if missing:
        raise ValueError(f'Missing Data in Input: {[missing]}')
    
    input_df =  pd.DataFrame(user_input, columns=expected_columns)

    return input_df


