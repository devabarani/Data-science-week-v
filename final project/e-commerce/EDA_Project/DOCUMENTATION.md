# E-commerce Transaction Data Analysis - Code Documentation

## Overview

This documentation explains how each step from the project requirements (project_details.pdf) is implemented in the codebase. The project follows a modular structure where each step is implemented in separate Python modules.

---

## Step 1: Import Data

### Requirement
- Load the dataset from CSV, Excel, or a SQL database into Python
- Display the first few rows and check the structure of the dataset

### Implementation Location
**File:** `modules/data_import.py`

### Key Functions

#### `load_data_from_csv(file_path)`
- **Purpose:** Loads a CSV file into a pandas DataFrame
- **Parameters:** 
  - `file_path`: Path to the CSV file
- **Returns:** DataFrame or None if error occurs
- **Features:**
  - Error handling with try-except blocks
  - Prints success message with file path
  - Displays dataset shape

#### `load_data_from_excel(file_path, sheet_name=0)`
- **Purpose:** Loads an Excel file into a pandas DataFrame
- **Parameters:**
  - `file_path`: Path to the Excel file
  - `sheet_name`: Sheet name or index (default: 0)
- **Returns:** DataFrame or None if error occurs

#### `display_data_info(df, name="Dataset")`
- **Purpose:** Displays comprehensive information about the dataset
- **Shows:**
  - First 5 rows using `df.head()`
  - Dataset shape (rows, columns)
  - Column names
  - Data types
  - Basic statistics using `df.describe()`
  - Missing values count

#### `load_all_datasets(data_dir)`
- **Purpose:** Automatically loads all CSV files from a directory
- **Returns:** Dictionary with dataset names as keys and DataFrames as values
- **Usage in main.py:**
  ```python
  datasets = load_all_datasets('data')
  customers_df = datasets.get('Customers', None)
  products_df = datasets.get('Products', None)
  transactions_df = datasets.get('Transactions', None)
  ```

### Code Flow
1. `main.py` calls `load_all_datasets('data')`
2. Function scans the data directory for CSV files
3. Each CSV is loaded using `load_data_from_csv()`
4. `display_data_info()` is called for each dataset
5. Datasets are merged for comprehensive analysis

---

## Step 2: Export Data

### Requirement
- Save a copy of the dataset (after any preprocessing) to CSV or Excel

### Implementation Location
**File:** `modules/data_import.py`

### Key Functions

#### `export_data_to_csv(df, file_path)`
- **Purpose:** Exports DataFrame to CSV format
- **Parameters:**
  - `df`: DataFrame to export
  - `file_path`: Destination path
- **Features:**
  - Saves without index (`index=False`)
  - Error handling with informative messages

#### `export_data_to_excel(df, file_path, sheet_name='Sheet1')`
- **Purpose:** Exports DataFrame to Excel format
- **Parameters:**
  - `df`: DataFrame to export
  - `file_path`: Destination path
  - `sheet_name`: Name of the Excel sheet

### Usage in main.py
```python
# After merging and preprocessing
export_data_to_csv(merged_df, 'outputs/merged_dataset.csv')
```

---

## Step 3: Data Cleaning

### Requirement
- Handle missing values (impute or remove)
- Remove duplicates if present
- Identify and handle outliers using visualization or statistical methods

### Implementation Location
**File:** `modules/data_cleaning.py`

### Key Functions

#### `handle_missing_values(df, strategy='mean', columns=None)`
- **Purpose:** Handles missing values using various strategies
- **Parameters:**
  - `df`: DataFrame to clean
  - `strategy`: One of 'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
  - `columns`: Specific columns to handle (None = all columns)
- **Strategies:**
  - **mean**: Fills numeric columns with mean value
  - **median**: Fills numeric columns with median value
  - **mode**: Fills with most frequent value
  - **drop**: Removes rows with missing values
  - **forward_fill**: Uses previous value (ffill)
  - **backward_fill**: Uses next value (bfill)
- **Features:**
  - Shows missing values before and after cleaning
  - Handles numeric and categorical columns appropriately

#### `remove_duplicates(df)`
- **Purpose:** Removes duplicate rows from the dataset
- **Returns:** DataFrame without duplicates
- **Features:**
  - Reports number of duplicates found
  - Shows shape before and after removal
  - Uses pandas `drop_duplicates()` method

#### `detect_outliers_iqr(df, columns=None)`
- **Purpose:** Detects outliers using Interquartile Range (IQR) method
- **Method:**
  - Calculates Q1 (25th percentile) and Q3 (75th percentile)
  - IQR = Q3 - Q1
  - Lower bound = Q1 - 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR
  - Identifies values outside these bounds
- **Returns:** Dictionary with column names and outlier indices

#### `handle_outliers(df, columns=None, method='cap', visualize=True)`
- **Purpose:** Handles detected outliers
- **Parameters:**
  - `method`: 'cap' (clip to bounds), 'remove' (delete rows), 'log_transform'
- **Features:**
  - Creates box plots and histograms for visualization
  - Saves outlier visualizations to outputs directory
  - Provides detailed statistics for each column

#### `clean_dataset(df, handle_missing=True, remove_dup=True, handle_out=True)`
- **Purpose:** Complete data cleaning pipeline
- **Returns:** Fully cleaned DataFrame
- **Pipeline:**
  1. Handle missing values
  2. Remove duplicates
  3. Handle outliers

### Usage in main.py
```python
# Handle missing values
cleaned_df = handle_missing_values(merged_df, strategy='mean')

# Remove duplicates
cleaned_df = remove_duplicates(cleaned_df)

# Handle outliers
cleaned_df = handle_outliers(cleaned_df, columns=numeric_cols, method='cap')
```

---

## Step 4: Data Transformation

### Requirement
- Apply normalization or standardization to numeric features if required

### Implementation Location
**File:** `modules/transformation.py`

### Key Functions

#### `normalize_data(df, columns=None, method='minmax')`
- **Purpose:** Normalizes numeric features
- **Parameters:**
  - `method`: 'minmax', 'zscore', or 'robust'
- **Methods:**
  - **minmax**: Scales to [0, 1] range using MinMaxScaler
    - Formula: (x - min) / (max - min)
  - **zscore**: Standardizes using mean and std (StandardScaler)
    - Formula: (x - mean) / std
  - **robust**: Uses median and IQR (RobustScaler)
    - Resistant to outliers
- **Returns:** Tuple of (transformed DataFrame, scaler object)

#### `standardize_data(df, columns=None)`
- **Purpose:** Wrapper for Z-score standardization
- **Returns:** Standardized DataFrame

#### `log_transform(df, columns=None)`
- **Purpose:** Applies log transformation (log1p = log(1+x))
- **Use case:** Reduces skewness in highly skewed distributions
- **Note:** Only applies to positive values

#### `encode_categorical(df, columns=None, method='onehot')`
- **Purpose:** Encodes categorical variables
- **Methods:**
  - **onehot**: Creates binary columns for each category
  - **label**: Assigns numeric labels to categories

### Usage in main.py
```python
# Normalize numeric features
normalized_df, scaler = normalize_data(cleaned_df, 
                                       columns=transform_cols, 
                                       method='minmax')
```

---

## Step 5: Descriptive Statistics

### Requirement
- Calculate mean, median, mode, standard deviation, and other summary statistics
- Provide a brief interpretation of key statistics

### Implementation Location
**File:** `modules/stats_analysis.py`

### Key Functions

#### `calculate_descriptive_stats(df, columns=None)`
- **Purpose:** Calculates comprehensive descriptive statistics
- **Statistics Calculated:**
  - **Central Tendency:** Mean, Median, Mode
  - **Dispersion:** Standard Deviation, Variance, Range, IQR
  - **Quartiles:** Q1 (25th percentile), Q3 (75th percentile)
  - **Shape:** Skewness, Kurtosis
  - **Extremes:** Min, Max
- **Returns:** DataFrame with statistics for each column

#### `interpret_statistics(stats_df)`
- **Purpose:** Provides human-readable interpretation of statistics
- **Interpretations:**
  - Compares mean and median to detect skewness
  - Interprets skewness values:
    - |skew| < 0.5: Approximately symmetric
    - |skew| < 1: Moderately skewed
    - |skew| ≥ 1: Highly skewed
  - Explains distribution characteristics

#### `calculate_correlation(df, columns=None)`
- **Purpose:** Calculates Pearson correlation matrix
- **Returns:** Correlation matrix DataFrame
- **Use case:** Identifies relationships between numeric variables

#### `calculate_covariance(df, columns=None)`
- **Purpose:** Calculates covariance matrix
- **Returns:** Covariance matrix DataFrame
- **Use case:** Measures joint variability between variables

### Usage in main.py
```python
# Calculate statistics
stats_df = calculate_descriptive_stats(cleaned_df, columns=stats_cols)
interpret_statistics(stats_df)

# Correlation and covariance
corr_matrix = calculate_correlation(cleaned_df, columns=stats_cols)
cov_matrix = calculate_covariance(cleaned_df, columns=stats_cols)
```

---

## Step 6: Basic Visualization

### Requirement
- Create line plots, bar charts, and histograms
- Customize titles, labels, and legends

### Implementation Location
**File:** `modules/visualization.py`

### Key Functions

#### `create_line_plot(df, x_col, y_col, title, save_path)`
- **Purpose:** Creates a line plot with markers
- **Customization:**
  - Custom title, x-label, y-label
  - Grid lines with transparency
  - Rotated x-axis labels
  - High-resolution output (300 DPI)
- **Use case:** Time series data, trends over time

#### `create_bar_chart(df, x_col, y_col, title, save_path)`
- **Purpose:** Creates a bar chart
- **Customization:**
  - Colored bars with black edges
  - Grid on y-axis
  - Rotated category labels
- **Use case:** Categorical comparisons

#### `create_histogram(df, column, bins, title, save_path)`
- **Purpose:** Creates a histogram
- **Parameters:**
  - `bins`: Number of bins (default: 30)
- **Customization:**
  - Sky blue color with black edges
  - Frequency on y-axis
- **Use case:** Distribution visualization

### Usage in main.py
```python
# Line plot - Sales over time
create_line_plot(daily_sales, 'Date', 'TotalSales', 
                "Daily Sales Over Time", 'outputs/line_plot_sales.png')

# Bar chart - Sales by category
create_bar_chart(category_sales, 'Category', 'TotalValue',
                "Total Sales by Category", 'outputs/bar_chart_category.png')

# Histogram - Price distribution
create_histogram(cleaned_df, 'Price_x', bins=50,
                "Price Distribution", 'outputs/histogram_price.png')
```

---

## Step 7: Advanced Visualization

### Requirement
- Create pair plots, heatmaps, and violin plots
- Analyze correlations and covariance between numeric features

### Implementation Location
**File:** `modules/visualization.py`

### Key Functions

#### `create_pair_plot(df, columns, save_path)`
- **Purpose:** Creates a pair plot (scatter matrix)
- **Features:**
  - Shows relationships between all pairs of variables
  - Diagonal shows kernel density estimation (KDE)
  - Uses seaborn for styling
- **Limitation:** Limited to 5 columns for readability
- **Use case:** Exploratory analysis of multiple variables

#### `create_heatmap(df, columns, title, save_path)`
- **Purpose:** Creates a correlation heatmap
- **Features:**
  - Color-coded correlation values (-1 to 1)
  - Annotated with correlation coefficients
  - Coolwarm colormap (blue = negative, red = positive)
  - Centered at 0
- **Use case:** Visualizing relationships between numeric variables

#### `create_violin_plot(df, x_col, y_col, title, save_path)`
- **Purpose:** Creates a violin plot
- **Features:**
  - Shows distribution shape and density
  - Combines box plot and KDE
  - Useful for comparing distributions across categories
- **Use case:** Comparing distributions by category

### Usage in main.py
```python
# Pair plot
create_pair_plot(cleaned_df[pair_cols], columns=pair_cols, 
                'outputs/pair_plot.png')

# Heatmap
create_heatmap(cleaned_df, columns=stats_cols,
              "Correlation Heatmap", 'outputs/heatmap.png')

# Violin plot
create_violin_plot(df_top_cat, 'Category', 'TotalValue',
                  "Total Value Distribution by Category", 
                  'outputs/violin_plot_category.png')
```

---

## Step 8: Interactive Visualization

### Requirement
- Use Plotly or Plotly Dash to create at least one interactive visualization or simple dashboard

### Implementation Location
**File:** `modules/visualization.py`

### Key Functions

#### `create_interactive_dashboard(df, save_path)`
- **Purpose:** Creates a multi-panel interactive dashboard
- **Features:**
  - 2×2 subplot layout
  - **Panel 1:** Line plot (sales over time)
  - **Panel 2:** Bar chart (category distribution)
  - **Panel 3:** Histogram (price distribution)
  - **Panel 4:** Top values bar chart
- **Technology:** Plotly's `make_subplots`
- **Output:** HTML file with interactive features
- **Interactivity:**
  - Zoom, pan, hover tooltips
  - Click to isolate data
  - Download as PNG

#### `create_interactive_scatter(df, x_col, y_col, color_col, title, save_path)`
- **Purpose:** Creates an interactive scatter plot
- **Features:**
  - Color coding by category (optional)
  - Hover tooltips showing all data
  - Zoom and pan capabilities
  - Legend for color categories
- **Technology:** Plotly Express
- **Output:** HTML file

### Usage in main.py
```python
# Interactive dashboard
create_interactive_dashboard(cleaned_df.head(1000), 
                            'outputs/interactive_dashboard.html')

# Interactive scatter plot
create_interactive_scatter(scatter_df, 'Quantity', 'TotalValue', 
                          color_col='Category',
                          "Quantity vs Total Value", 
                          'outputs/interactive_scatter.html')
```

---

## Step 9: Probability Analysis

### Requirement
- Visualize probability distributions for numeric features

### Implementation Location
**File:** `modules/visualization.py`

### Key Function

#### `create_probability_distribution(df, column, title, save_path)`
- **Purpose:** Visualizes probability distributions
- **Features:**
  - **Left panel:** Histogram with density curve
    - Shows actual distribution shape
    - Density on y-axis (not frequency)
  - **Right panel:** Q-Q (Quantile-Quantile) plot
    - Compares data distribution to normal distribution
    - Points on diagonal line = normal distribution
    - Deviations indicate non-normality
- **Technology:** Matplotlib and SciPy stats
- **Use case:** 
  - Check if data follows normal distribution
  - Identify distribution type
  - Detect skewness visually

### Usage in main.py
```python
# Probability distributions for key numeric features
for col in ['TotalValue', 'Price_x', 'Quantity']:
    create_probability_distribution(cleaned_df, col,
                                  f"Probability Distribution - {col}",
                                  f'outputs/probability_{col}.png')
```

---

## Step 10: Modeling – Classification (k-NN)

### Requirement
- Implement k-Nearest Neighbors (k-NN) on a numeric target variable or class
- Split data into training and testing sets, train the model, and evaluate accuracy

### Implementation Location
**File:** `modules/modeling.py`

### Key Functions

#### `prepare_data_for_classification(df, target_col, feature_cols)`
- **Purpose:** Prepares data for classification
- **Steps:**
  1. Selects feature columns (numeric only)
  2. Extracts target variable
  3. Encodes categorical target if needed (LabelEncoder)
- **Returns:** X (features), y (target), feature column names

#### `knn_classification(df, target_col, feature_cols, test_size, n_neighbors, random_state)`
- **Purpose:** Implements k-NN classification
- **Pipeline:**
  1. **Data Preparation:**
     - Extract features and target
     - Handle missing values (fill with 0)
  2. **Train-Test Split:**
     - Default: 80% train, 20% test
     - Stratified split (maintains class distribution)
  3. **Feature Scaling:**
     - StandardScaler (mean=0, std=1)
     - Critical for distance-based algorithms
  4. **Model Training:**
     - KNeighborsClassifier from scikit-learn
     - Default: k=5 neighbors
  5. **Prediction:**
     - Predicts on test set
  6. **Evaluation:**
     - Accuracy score
     - Classification report (precision, recall, F1-score)
     - Confusion matrix visualization
- **Returns:** Dictionary with model, predictions, accuracy, etc.

### Algorithm Explanation
- **k-NN (k-Nearest Neighbors):**
  - Non-parametric, instance-based learning
  - Classifies based on k nearest training examples
  - Distance metric: Euclidean distance
  - Voting: Majority class among k neighbors

### Usage in main.py
```python
# Create binary classification target
cleaned_df['HighValue'] = (cleaned_df['TotalValue'] > 
                          cleaned_df['TotalValue'].median()).astype(int)

# Train k-NN model
knn_results = knn_classification(cleaned_df, 'HighValue', 
                                 feature_cols=['Quantity', 'Price_x'],
                                 n_neighbors=5, test_size=0.2)
```

---

## Step 11: Modeling – Clustering (k-Means)

### Requirement
- Apply k-Means clustering to group numeric data points
- Visualize clusters using scatter plots or pair plots
- Interpret the clusters

### Implementation Location
**File:** `modules/modeling.py`

### Key Functions

#### `kmeans_clustering(df, n_clusters, feature_cols, random_state)`
- **Purpose:** Applies k-Means clustering
- **Pipeline:**
  1. **Data Preparation:**
     - Select numeric features
     - Handle missing values
  2. **Feature Scaling:**
     - StandardScaler (required for distance-based clustering)
  3. **Clustering:**
     - KMeans from scikit-learn
     - Default: k=3 clusters
     - n_init=10 (runs 10 times, keeps best)
  4. **Visualization:**
     - 2D scatter plot with cluster colors
     - Cluster centroids marked with red X
     - Pair plot with cluster coloring (if ≤4 features)
  5. **Interpretation:**
     - Cluster distribution (count and percentage)
     - Mean values for key features per cluster
- **Returns:** Dictionary with model, labels, clustered DataFrame

#### `find_optimal_k(df, max_k, feature_cols)`
- **Purpose:** Finds optimal number of clusters using Elbow Method
- **Method:**
  - Tests k from 1 to max_k
  - Calculates inertia (within-cluster sum of squares)
  - Plots inertia vs k
  - "Elbow" point = optimal k (where inertia decrease slows)
- **Output:** Elbow plot saved to outputs

### Algorithm Explanation
- **k-Means Clustering:**
  - Partitional clustering algorithm
  - Divides data into k clusters
  - Minimizes within-cluster variance
  - Iterative process:
    1. Initialize k centroids randomly
    2. Assign points to nearest centroid
    3. Update centroids to cluster means
    4. Repeat until convergence

### Usage in main.py
```python
# Find optimal k
find_optimal_k(cleaned_df, max_k=8, feature_cols=clustering_cols)

# Apply k-Means with k=3
kmeans_results = kmeans_clustering(cleaned_df, n_clusters=3,
                                   feature_cols=clustering_cols,
                                   random_state=42)
```

---

## Step 12: Summary & Insights

### Requirement
- Write a brief report summarizing findings, patterns, and insights from the data

### Implementation Location
**File:** `main.py` - `generate_summary_report()` function

### Function Details

#### `generate_summary_report(df)`
- **Purpose:** Generates comprehensive summary report
- **Insights Calculated:**
  1. **Revenue Metrics:**
     - Total revenue (sum of all transactions)
     - Average transaction value
     - Total number of transactions
  2. **Category Analysis:**
     - Top category by revenue
     - Revenue per category
  3. **Regional Analysis:**
     - Top region by revenue
     - Regional performance comparison
  4. **Temporal Analysis:**
     - Best performing year
     - Monthly trends (if applicable)
  5. **Product Analysis:**
     - Best selling product
     - Units sold per product
- **Output:**
  - Console printout with numbered insights
  - Text file saved to `outputs/summary_report.txt`
  - Includes timestamp

### Usage in main.py
```python
# Called at the end of main() function
generate_summary_report(cleaned_df)
```

---

## Main Workflow (`main.py`)

The `main.py` file orchestrates all 12 steps in sequence:

1. **Import Data** → Load and merge datasets
2. **Export Data** → Save merged dataset
3. **Data Cleaning** → Handle missing, duplicates, outliers
4. **Data Transformation** → Normalize features
5. **Descriptive Statistics** → Calculate and interpret stats
6. **Basic Visualization** → Line, bar, histogram plots
7. **Advanced Visualization** → Pair plots, heatmaps, violin plots
8. **Interactive Visualization** → Plotly dashboards
9. **Probability Analysis** → Distribution visualizations
10. **k-NN Classification** → Train and evaluate model
11. **k-Means Clustering** → Cluster analysis
12. **Summary & Insights** → Generate report

### Error Handling
- Try-except blocks in data loading
- Column existence checks before operations
- Graceful handling of missing columns

### Output Management
- All outputs saved to `outputs/` directory
- Organized file naming convention
- High-resolution plots (300 DPI)

---

## File Structure Summary

```
EDA_Project/
├── data/                          # Input data files
│   ├── Customers.csv
│   ├── Products.csv
│   └── Transactions.csv
│
├── modules/                       # Modular code implementation
│   ├── __init__.py               # Package initialization
│   ├── data_import.py            # Steps 1-2: Import/Export
│   ├── data_cleaning.py          # Step 3: Data Cleaning
│   ├── transformation.py         # Step 4: Transformation
│   ├── stats_analysis.py         # Step 5: Statistics
│   ├── visualization.py          # Steps 6-9: Visualizations
│   └── modeling.py               # Steps 10-11: ML Models
│
├── main.py                        # Main orchestrator (Step 12)
├── outputs/                       # Generated outputs
│   ├── *.png                     # Static visualizations
│   ├── *.html                    # Interactive dashboards
│   ├── *.csv                     # Exported data
│   └── summary_report.txt        # Insights report
│
├── README.md                      # Project overview
├── DOCUMENTATION.md               # This file
└── requirements.txt               # Dependencies
```

---

## Key Design Principles

1. **Modularity:** Each step in a separate module file
2. **Reusability:** Functions can be used independently
3. **Documentation:** Comprehensive docstrings and comments
4. **Error Handling:** Try-except blocks for robustness
5. **Visualization:** High-quality, customizable plots
6. **Scalability:** Can handle datasets of various sizes

---

## Dependencies

All required packages are listed in `requirements.txt`:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Static plotting
- `seaborn`: Statistical visualizations
- `plotly`: Interactive visualizations
- `scikit-learn`: Machine learning models
- `scipy`: Statistical functions
- `openpyxl`: Excel file support

---

## Conclusion

This project implements all 12 required steps from the project requirements in a well-organized, modular structure. Each module handles a specific aspect of the EDA workflow, making the code maintainable, testable, and reusable. The main script orchestrates the complete pipeline, ensuring all steps are executed in the correct order with proper error handling and output management.

