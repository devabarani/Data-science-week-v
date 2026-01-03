# E-commerce Transaction Data Analysis and Insights

## Project Overview

This project performs a comprehensive Exploratory Data Analysis (EDA) on e-commerce transaction data, covering data import, cleaning, transformation, statistical analysis, visualization, and machine learning modeling.

## Project Structure

```
EDA_Project/
├── data/                          # CSV data files
│   ├── Customers.csv
│   ├── Products.csv
│   └── Transactions.csv
├── modules/                       # Python modules
│   ├── data_import.py            # Data loading and export
│   ├── data_cleaning.py          # Missing values, duplicates, outliers
│   ├── transformation.py          # Normalization and standardization
│   ├── stats_analysis.py         # Descriptive statistics
│   ├── visualization.py          # Basic, advanced, and interactive plots
│   └── modeling.py               # k-NN and k-Means models
├── main.py                        # Main script to run the complete workflow
├── outputs/                       # Generated outputs
│   ├── *.png                      # Visualization plots
│   ├── *.html                    # Interactive dashboards
│   ├── *.csv                     # Exported datasets
│   └── summary_report.txt        # Summary insights
└── README.md                      # This file
```

## Features

### 1. Data Import
- Load datasets from CSV files
- Display dataset structure and basic information
- Export cleaned/preprocessed data

### 2. Data Cleaning
- Handle missing values (mean, median, mode, forward/backward fill)
- Remove duplicate records
- Detect and handle outliers using IQR method
- Visualize outliers

### 3. Data Transformation
- Normalization (Min-Max, Z-score, Robust scaling)
- Standardization
- Log transformation
- Categorical encoding

### 4. Descriptive Statistics
- Mean, median, mode, standard deviation
- Variance, range, quartiles, IQR
- Skewness and kurtosis
- Correlation and covariance matrices
- Statistical interpretation

### 5. Basic Visualization
- Line plots
- Bar charts
- Histograms
- Customized titles, labels, and legends

### 6. Advanced Visualization
- Pair plots
- Correlation heatmaps
- Violin plots
- Probability distributions

### 7. Interactive Visualization
- Interactive dashboards using Plotly
- Interactive scatter plots
- HTML-based visualizations

### 8. Probability Analysis
- Probability distribution visualizations
- Q-Q plots for normality testing

### 9. Modeling - Classification (k-NN)
- k-Nearest Neighbors classification
- Train-test split
- Accuracy evaluation
- Confusion matrix visualization

### 10. Modeling - Clustering (k-Means)
- k-Means clustering
- Elbow method for optimal k
- Cluster visualization (2D scatter, pair plots)
- Cluster interpretation

## Requirements

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

## Usage

Run the main script to execute the complete EDA workflow:

```bash
cd EDA_Project
python main.py
```

The script will:
1. Load all datasets from the `data/` directory
2. Merge datasets for comprehensive analysis
3. Clean the data (handle missing values, duplicates, outliers)
4. Transform and normalize features
5. Calculate descriptive statistics
6. Create various visualizations
7. Build and evaluate k-NN classification model
8. Apply k-Means clustering
9. Generate summary report with insights

## Outputs

All outputs are saved in the `outputs/` directory:

- **Visualizations**: PNG files for all plots
- **Interactive Dashboards**: HTML files for interactive visualizations
- **Exported Data**: CSV files with cleaned/processed data
- **Summary Report**: Text file with key insights

## Key Insights

The analysis provides insights on:
- Total revenue and average transaction values
- Top-selling products and categories
- Regional sales performance
- Customer behavior patterns
- Product price distributions
- Transaction trends over time

## Modules Documentation

### data_import.py
- `load_data_from_csv()`: Load CSV files
- `load_data_from_excel()`: Load Excel files
- `display_data_info()`: Show dataset information
- `export_data_to_csv()`: Export to CSV
- `load_all_datasets()`: Load all datasets from directory

### data_cleaning.py
- `handle_missing_values()`: Handle missing data
- `remove_duplicates()`: Remove duplicate rows
- `detect_outliers_iqr()`: Detect outliers using IQR
- `handle_outliers()`: Handle outliers (cap, remove, transform)
- `clean_dataset()`: Complete cleaning pipeline

### transformation.py
- `normalize_data()`: Normalize numeric features
- `standardize_data()`: Standardize features
- `log_transform()`: Apply log transformation
- `encode_categorical()`: Encode categorical variables

### stats_analysis.py
- `calculate_descriptive_stats()`: Calculate summary statistics
- `interpret_statistics()`: Interpret statistical results
- `calculate_correlation()`: Calculate correlation matrix
- `calculate_covariance()`: Calculate covariance matrix

### visualization.py
- `create_line_plot()`: Create line plots
- `create_bar_chart()`: Create bar charts
- `create_histogram()`: Create histograms
- `create_pair_plot()`: Create pair plots
- `create_heatmap()`: Create correlation heatmaps
- `create_violin_plot()`: Create violin plots
- `create_probability_distribution()`: Visualize probability distributions
- `create_interactive_dashboard()`: Create interactive dashboards
- `create_interactive_scatter()`: Create interactive scatter plots

### modeling.py
- `knn_classification()`: k-NN classification
- `kmeans_clustering()`: k-Means clustering
- `find_optimal_k()`: Find optimal number of clusters

## Author

EDA Project - E-commerce Transaction Data Analysis

## Date

January 2026

