"""
Descriptive Statistics Module
Calculates and interprets summary statistics
"""

import pandas as pd
import numpy as np


def calculate_descriptive_stats(df, columns=None):
    """
    Calculate descriptive statistics for numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to analyze. If None, analyzes all numeric columns
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    print(f"\n{'='*60}")
    print("Descriptive Statistics")
    print(f"{'='*60}")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = pd.DataFrame({
        'Mean': df[columns].mean(),
        'Median': df[columns].median(),
        'Mode': [df[col].mode()[0] if not df[col].mode().empty else np.nan for col in columns],
        'Std Dev': df[columns].std(),
        'Variance': df[columns].var(),
        'Min': df[columns].min(),
        'Max': df[columns].max(),
        'Range': df[columns].max() - df[columns].min(),
        'Q1': df[columns].quantile(0.25),
        'Q3': df[columns].quantile(0.75),
        'IQR': df[columns].quantile(0.75) - df[columns].quantile(0.25),
        'Skewness': df[columns].skew(),
        'Kurtosis': df[columns].kurtosis()
    })
    
    print(stats)
    return stats


def interpret_statistics(stats_df):
    """
    Provide brief interpretation of key statistics
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        Statistics dataframe from calculate_descriptive_stats
    """
    print(f"\n{'='*60}")
    print("Statistical Interpretation")
    print(f"{'='*60}")
    
    for col in stats_df.index:
        mean = stats_df.loc[col, 'Mean']
        median = stats_df.loc[col, 'Median']
        std = stats_df.loc[col, 'Std Dev']
        skew = stats_df.loc[col, 'Skewness']
        
        print(f"\n{col}:")
        print(f"  Mean: {mean:.2f}, Median: {median:.2f}")
        print(f"  Standard Deviation: {std:.2f}")
        
        # Interpret skewness
        if abs(skew) < 0.5:
            print(f"  Distribution: Approximately symmetric (skewness: {skew:.2f})")
        elif abs(skew) < 1:
            print(f"  Distribution: Moderately skewed (skewness: {skew:.2f})")
        else:
            print(f"  Distribution: Highly skewed (skewness: {skew:.2f})")
        
        # Compare mean and median
        if abs(mean - median) < 0.1 * std:
            print(f"  Mean and median are close, suggesting symmetric distribution")
        else:
            print(f"  Mean and median differ, suggesting skewed distribution")


def calculate_correlation(df, columns=None):
    """
    Calculate correlation matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to include. If None, includes all numeric columns
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    return corr_matrix


def calculate_covariance(df, columns=None):
    """
    Calculate covariance matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to include. If None, includes all numeric columns
    
    Returns:
    --------
    pd.DataFrame
        Covariance matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cov_matrix = df[columns].cov()
    return cov_matrix

