import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def delete_max_rows(df, max_rows):
    """
    Use to delete the rows we found to be maximum and reset the index

    Args:
        df (DataFrame): combined data
        max_rows (Dataframe): of rows to be deleted.
    
    Returns: 
        df: processed without the rows
    """
    
    unique_max_rows = max_rows.unique()
    df = df.drop(unique_max_rows).reset_index(drop=True)
    return df

def log_scale_encode(df, numerical_cols, categorical_cols):
    """
    Use to scale, transform numeric values and one hot encode the categorical ones

    Args:
        df (DataFrame): to encode and transfrom
        numerical_cols (list): of the numerical columns to be transformed
        categorical_cols (list): of the categorical columns to be encoded

    Returns:
        df: encoded and scaled
    """

    scalar = StandardScaler()
    X = pd.get_dummies(df, columns=categorical_cols)
    X[numerical_cols] = np.log1p(X[numerical_cols])   
    X_numer_scaled = scalar.fit_transform(X[numerical_cols])
    X[numerical_cols] = pd.DataFrame(X_numer_scaled, index=X[numerical_cols].index, columns=X[numerical_cols].columns)

    return X