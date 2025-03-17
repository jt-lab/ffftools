import pandas as pd
from os.path import join, dirname

def drop_unavailable(df, column):
    df = df[df[column] != -1]
    return df

def nicify(df):
    def rename_columns(df, columns_data):
        for _, row in columns_data.iterrows():
            if row['Column name'] in df.columns:
                df.rename(columns={row['Column name']: row['Label name']}, inplace=True)

    path = join(dirname(__file__), 'columns')
    mdf = pd.read_csv(join(path, 'mandatory.csv')).reset_index(drop=True)
    cdf = pd.read_csv(join(path, 'computable.csv')).reset_index(drop=True)

    rename_columns(df, mdf)
    rename_columns(df, cdf)
    
    return df

def make_label(column_name, keep_level=False):
    cn = column_name[2:]                    # Cut off the status flags ('M_', etc)
    if not keep_level:
        cn = cn.replace('Trial_', '')       # Remove the level information
        cn = cn.replace('Selection_', '')   # Remove the level information
    cn = cn.replace('_', ' ')               # Replace all _ with spaces
    return cn

def extract_condition(df, name):
    df = df.query('M_Condition_Name == @name')
    return df

def extract_trial(df, index):
    df = df.query('M_Trial_Index == @index')
    return df

def has_valid_name(column_name):
    return column_name.startswith(('M_', 'C_', '!C_', 'S_'))

def is_mandatory(column_name):
    return column_name.startswith('M_')


import numpy as np

def arcsine_sqrt_transform(x):
    """
    Performs the arcsine square root transformation on a given value or array.

    Args:
        x (float or np.ndarray): Input value or array of values to be transformed. Values should be in the range [0, 1].
        
    Returns:
        float or np.ndarray: Transformed value or array.
    
    Raises:
        ValueError: If any value in x is not in the range [0, 1].
    """
    
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Input values must be in the range [0, 1].")
    
    return np.arcsin(np.sqrt(x))
