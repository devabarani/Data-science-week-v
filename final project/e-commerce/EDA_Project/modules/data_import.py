"""
Data Import Module
Handles loading datasets from CSV, Excel, or SQL database
"""

import pandas as pd
import os


def load_data_from_csv(file_path):
    """
    Load dataset from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def load_data_from_excel(file_path, sheet_name=0):
    """
    Load dataset from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str or int
        Sheet name or index to load
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def display_data_info(df, name="Dataset"):
    """
    Display basic information about the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to display
    name : str
        Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:")
    print(df.columns.tolist())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())


def export_data_to_csv(df, file_path):
    """
    Export dataset to CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to export
    file_path : str
        Path where to save the CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Successfully exported data to {file_path}")
    except Exception as e:
        print(f"Error exporting data: {str(e)}")


def export_data_to_excel(df, file_path, sheet_name='Sheet1'):
    """
    Export dataset to Excel file
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to export
    file_path : str
        Path where to save the Excel file
    sheet_name : str
        Sheet name
    """
    try:
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        print(f"Successfully exported data to {file_path}")
    except Exception as e:
        print(f"Error exporting data: {str(e)}")


def load_all_datasets(data_dir):
    """
    Load all CSV files from a directory
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
    
    Returns:
    --------
    dict
        Dictionary with dataset names as keys and DataFrames as values
    """
    datasets = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        dataset_name = file.replace('.csv', '')
        df = load_data_from_csv(file_path)
        if df is not None:
            datasets[dataset_name] = df
            display_data_info(df, dataset_name)
    
    return datasets

