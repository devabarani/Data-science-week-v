"""
Data Transformation Module
Handles normalization and standardization of numeric features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to normalize
    columns : list
        Columns to normalize. If None, normalizes all numeric columns
    method : str
        Normalization method: 'minmax', 'zscore', 'robust'
    
    Returns:
    --------
    pd.DataFrame
        Normalized dataset
    """
    print(f"\n{'='*60}")
    print(f"Normalizing Data using {method}")
    print(f"{'='*60}")
    
    df_transformed = df.copy()
    
    if columns is None:
        columns = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
        print(f"Applied Min-Max normalization to: {columns}")
    elif method == 'zscore':
        scaler = StandardScaler()
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
        print(f"Applied Z-score standardization to: {columns}")
    elif method == 'robust':
        scaler = RobustScaler()
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
        print(f"Applied Robust scaling to: {columns}")
    
    return df_transformed, scaler


def standardize_data(df, columns=None):
    """
    Standardize numeric features (Z-score normalization)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to standardize
    columns : list
        Columns to standardize. If None, standardizes all numeric columns
    
    Returns:
    --------
    pd.DataFrame
        Standardized dataset
    """
    return normalize_data(df, columns, method='zscore')[0]


def log_transform(df, columns=None):
    """
    Apply log transformation to numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to transform
    columns : list
        Columns to transform. If None, transforms all numeric columns
    
    Returns:
    --------
    pd.DataFrame
        Transformed dataset
    """
    print(f"\n{'='*60}")
    print("Applying Log Transformation")
    print(f"{'='*60}")
    
    df_transformed = df.copy()
    
    if columns is None:
        columns = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if df_transformed[col].min() > 0:
            df_transformed[col] = np.log1p(df_transformed[col])
            print(f"Applied log transformation to {col}")
        else:
            print(f"Skipped {col} (contains non-positive values)")
    
    return df_transformed


def encode_categorical(df, columns=None, method='onehot'):
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to encode
    columns : list
        Categorical columns to encode. If None, encodes all object columns
    method : str
        Encoding method: 'onehot', 'label'
    
    Returns:
    --------
    pd.DataFrame
        Encoded dataset
    """
    print(f"\n{'='*60}")
    print(f"Encoding Categorical Variables using {method}")
    print(f"{'='*60}")
    
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns)
        print(f"Applied one-hot encoding to: {columns}")
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"Applied label encoding to {col}")
    
    return df_encoded

