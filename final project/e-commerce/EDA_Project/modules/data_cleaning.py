"""
Data Cleaning Module
Handles missing values, duplicates, and outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    strategy : str
        Strategy to handle missing values: 'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
    columns : list
        Specific columns to handle. If None, handles all columns
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.columns
    
    print(f"\n{'='*60}")
    print("Handling Missing Values")
    print(f"{'='*60}")
    print(f"\nMissing values before cleaning:")
    missing_before = df_cleaned[columns].isnull().sum()
    print(missing_before[missing_before > 0])
    
    for col in columns:
        if df_cleaned[col].isnull().sum() > 0:
            if strategy == 'mean' and df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                print(f"Filled {col} with mean: {df_cleaned[col].mean():.2f}")
            elif strategy == 'median' and df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                print(f"Filled {col} with median: {df_cleaned[col].median():.2f}")
            elif strategy == 'mode':
                mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None
                if mode_value is not None:
                    df_cleaned[col].fillna(mode_value, inplace=True)
                    print(f"Filled {col} with mode: {mode_value}")
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)
                print(f"Dropped rows with missing values in {col}")
            elif strategy == 'forward_fill':
                df_cleaned[col].ffill(inplace=True)
                print(f"Forward filled {col}")
            elif strategy == 'backward_fill':
                df_cleaned[col].bfill(inplace=True)
                print(f"Backward filled {col}")
    
    print(f"\nMissing values after cleaning:")
    missing_after = df_cleaned[columns].isnull().sum()
    print(missing_after[missing_after > 0])
    
    return df_cleaned


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    
    Returns:
    --------
    pd.DataFrame
        Dataset without duplicates
    """
    print(f"\n{'='*60}")
    print("Removing Duplicates")
    print(f"{'='*60}")
    
    duplicates_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    
    if duplicates_count > 0:
        df_cleaned = df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate rows")
        print(f"Shape before: {df.shape}, Shape after: {df_cleaned.shape}")
        return df_cleaned
    else:
        print("No duplicates found")
        return df.copy()


def detect_outliers_iqr(df, columns=None):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to check for outliers. If None, checks all numeric columns
    
    Returns:
    --------
    dict
        Dictionary with column names as keys and outlier indices as values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outliers[col] = outlier_indices
        
        if len(outlier_indices) > 0:
            print(f"\n{col}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
            print(f"  Number of outliers: {len(outlier_indices)}")
    
    return outliers


def handle_outliers(df, columns=None, method='cap', visualize=True):
    """
    Handle outliers in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    columns : list
        Columns to handle outliers. If None, handles all numeric columns
    method : str
        Method to handle outliers: 'cap', 'remove', 'log_transform'
    visualize : bool
        Whether to create visualization of outliers
    
    Returns:
    --------
    pd.DataFrame
        Dataset with handled outliers
    """
    print(f"\n{'='*60}")
    print("Handling Outliers")
    print(f"{'='*60}")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_cleaned = df.copy()
    outliers = detect_outliers_iqr(df_cleaned, columns)
    
    if visualize:
        for col in columns:
            if len(outliers.get(col, [])) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Box plot
                axes[0].boxplot(df_cleaned[col].dropna())
                axes[0].set_title(f'Box Plot - {col}')
                axes[0].set_ylabel('Value')
                
                # Histogram
                axes[1].hist(df_cleaned[col].dropna(), bins=50, edgecolor='black')
                axes[1].set_title(f'Histogram - {col}')
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(f'outputs/outliers_{col}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    for col in columns:
        if col in outliers and len(outliers[col]) > 0:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in {col}")
            elif method == 'remove':
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                print(f"Removed outliers in {col}")
            elif method == 'log_transform':
                if df_cleaned[col].min() > 0:
                    df_cleaned[col] = np.log1p(df_cleaned[col])
                    print(f"Applied log transformation to {col}")
    
    return df_cleaned


def clean_dataset(df, handle_missing=True, remove_dup=True, handle_out=True):
    """
    Complete data cleaning pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    handle_missing : bool
        Whether to handle missing values
    remove_dup : bool
        Whether to remove duplicates
    handle_out : bool
        Whether to handle outliers
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    df_cleaned = df.copy()
    
    if handle_missing:
        df_cleaned = handle_missing_values(df_cleaned, strategy='mean')
    
    if remove_dup:
        df_cleaned = remove_duplicates(df_cleaned)
    
    if handle_out:
        df_cleaned = handle_outliers(df_cleaned, method='cap', visualize=True)
    
    return df_cleaned

