"""
Main Script for E-commerce Transaction Data Analysis and Insights
EDA Project - Complete Workflow
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import custom modules
from data_import import load_data_from_csv, display_data_info, export_data_to_csv, load_all_datasets
from data_cleaning import clean_dataset, handle_missing_values, remove_duplicates, handle_outliers
from transformation import normalize_data, standardize_data
from stats_analysis import calculate_descriptive_stats, interpret_statistics, calculate_correlation, calculate_covariance
from visualization import (
    create_line_plot, create_bar_chart, create_histogram, create_pair_plot,
    create_heatmap, create_violin_plot, create_probability_distribution,
    create_interactive_dashboard, create_interactive_scatter
)
from modeling import knn_classification, kmeans_clustering, find_optimal_k


def main():
    """
    Main function to run the complete EDA workflow
    """
    print("="*80)
    print("E-commerce Transaction Data Analysis and Insights")
    print("Complete EDA Workflow")
    print("="*80)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Import Data
    print("\n" + "="*80)
    print("STEP 1: IMPORT DATA")
    print("="*80)
    
    data_dir = 'data'
    datasets = load_all_datasets(data_dir)
    
    # Load individual datasets
    customers_df = datasets.get('Customers', None)
    products_df = datasets.get('Products', None)
    transactions_df = datasets.get('Transactions', None)
    
    if transactions_df is None:
        print("Error: Could not load Transactions dataset")
        return
    
    # Convert TransactionDate to datetime
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Merge datasets for comprehensive analysis
    print("\nMerging datasets...")
    merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
    merged_df = merged_df.merge(products_df, on='ProductID', how='left')
    
    print(f"Merged dataset shape: {merged_df.shape}")
    display_data_info(merged_df, "Merged Dataset")
    
    # Step 2: Export Data (after any preprocessing)
    print("\n" + "="*80)
    print("STEP 2: EXPORT DATA")
    print("="*80)
    export_data_to_csv(merged_df, 'outputs/merged_dataset.csv')
    
    # Step 3: Data Cleaning
    print("\n" + "="*80)
    print("STEP 3: DATA CLEANING")
    print("="*80)
    
    # Handle missing values
    cleaned_df = handle_missing_values(merged_df, strategy='mean')
    
    # Remove duplicates
    cleaned_df = remove_duplicates(cleaned_df)
    
    # Handle outliers for numeric columns
    numeric_cols = ['Quantity', 'TotalValue', 'Price_x', 'Price_y']
    numeric_cols = [col for col in numeric_cols if col in cleaned_df.columns]
    cleaned_df = handle_outliers(cleaned_df, columns=numeric_cols, method='cap', visualize=True)
    
    # Step 4: Data Transformation
    print("\n" + "="*80)
    print("STEP 4: DATA TRANSFORMATION")
    print("="*80)
    
    # Normalize numeric features (optional - for modeling)
    transform_cols = ['Quantity', 'TotalValue', 'Price_x']
    transform_cols = [col for col in transform_cols if col in cleaned_df.columns]
    if transform_cols:
        normalized_df, scaler = normalize_data(cleaned_df, columns=transform_cols, method='minmax')
    else:
        normalized_df = cleaned_df.copy()
    
    # Step 5: Descriptive Statistics
    print("\n" + "="*80)
    print("STEP 5: DESCRIPTIVE STATISTICS")
    print("="*80)
    
    stats_cols = ['Quantity', 'TotalValue', 'Price_x', 'Price_y']
    stats_cols = [col for col in stats_cols if col in cleaned_df.columns]
    if stats_cols:
        stats_df = calculate_descriptive_stats(cleaned_df, columns=stats_cols)
        interpret_statistics(stats_df)
        
        # Correlation and Covariance
        print("\nCorrelation Matrix:")
        corr_matrix = calculate_correlation(cleaned_df, columns=stats_cols)
        print(corr_matrix)
        
        print("\nCovariance Matrix:")
        cov_matrix = calculate_covariance(cleaned_df, columns=stats_cols)
        print(cov_matrix)
    
    # Step 6: Basic Visualization
    print("\n" + "="*80)
    print("STEP 6: BASIC VISUALIZATION")
    print("="*80)
    
    # Line plot - Sales over time
    if 'TransactionDate' in cleaned_df.columns:
        daily_sales = cleaned_df.groupby(cleaned_df['TransactionDate'].dt.date)['TotalValue'].sum().reset_index()
        daily_sales.columns = ['Date', 'TotalSales']
        create_line_plot(daily_sales.head(30), 'Date', 'TotalSales', 
                        "Daily Sales Over Time", 'outputs/line_plot_sales.png')
    
    # Bar chart - Sales by category
    if 'Category' in cleaned_df.columns:
        category_sales = cleaned_df.groupby('Category')['TotalValue'].sum().sort_values(ascending=False).reset_index()
        create_bar_chart(category_sales, 'Category', 'TotalValue',
                        "Total Sales by Category", 'outputs/bar_chart_category.png')
    
    # Histogram - Price distribution
    if 'Price_x' in cleaned_df.columns:
        create_histogram(cleaned_df, 'Price_x', bins=50,
                       title="Price Distribution", save_path='outputs/histogram_price.png')
    
    # Step 7: Advanced Visualization
    print("\n" + "="*80)
    print("STEP 7: ADVANCED VISUALIZATION")
    print("="*80)
    
    # Pair plot
    pair_cols = ['Quantity', 'TotalValue', 'Price_x']
    pair_cols = [col for col in pair_cols if col in cleaned_df.columns]
    if len(pair_cols) >= 2:
        create_pair_plot(cleaned_df[pair_cols], columns=pair_cols,
                      save_path='outputs/pair_plot.png')
    
    # Heatmap
    if stats_cols:
        create_heatmap(cleaned_df, columns=stats_cols,
                      title="Correlation Heatmap", save_path='outputs/heatmap.png')
    
    # Violin plot
    if 'Category' in cleaned_df.columns and 'TotalValue' in cleaned_df.columns:
        top_categories = cleaned_df['Category'].value_counts().head(5).index
        df_top_cat = cleaned_df[cleaned_df['Category'].isin(top_categories)]
        create_violin_plot(df_top_cat, 'Category', 'TotalValue',
                          title="Total Value Distribution by Category", 
                          save_path='outputs/violin_plot_category.png')
    
    # Step 8: Interactive Visualization
    print("\n" + "="*80)
    print("STEP 8: INTERACTIVE VISUALIZATION")
    print("="*80)
    
    # Interactive dashboard
    create_interactive_dashboard(cleaned_df.head(1000), 
                                'outputs/interactive_dashboard.html')
    
    # Interactive scatter plot
    if 'Quantity' in cleaned_df.columns and 'TotalValue' in cleaned_df.columns:
        scatter_df = cleaned_df.head(500)
        color_col = 'Category' if 'Category' in cleaned_df.columns else None
        create_interactive_scatter(scatter_df, 'Quantity', 'TotalValue', color_col,
                                  "Quantity vs Total Value", 
                                  'outputs/interactive_scatter.html')
    
    # Step 9: Probability Analysis
    print("\n" + "="*80)
    print("STEP 9: PROBABILITY ANALYSIS")
    print("="*80)
    
    prob_cols = ['TotalValue', 'Price_x', 'Quantity']
    for col in prob_cols:
        if col in cleaned_df.columns:
            create_probability_distribution(cleaned_df, col,
                                          f"Probability Distribution - {col}",
                                          f'outputs/probability_{col}.png')
    
    # Step 10: Modeling - Classification (k-NN)
    print("\n" + "="*80)
    print("STEP 10: MODELING - CLASSIFICATION (k-NN)")
    print("="*80)
    
    # Create a classification target (e.g., high value transaction)
    if 'TotalValue' in cleaned_df.columns:
        cleaned_df['HighValue'] = (cleaned_df['TotalValue'] > cleaned_df['TotalValue'].median()).astype(int)
        
        # Prepare features for classification
        feature_cols = ['Quantity', 'Price_x']
        feature_cols = [col for col in feature_cols if col in cleaned_df.columns]
        
        if len(feature_cols) >= 1:
            knn_results = knn_classification(cleaned_df, 'HighValue', 
                                            feature_cols=feature_cols,
                                            n_neighbors=5, test_size=0.2)
    
    # Step 11: Modeling - Clustering (k-Means)
    print("\n" + "="*80)
    print("STEP 11: MODELING - CLUSTERING (k-Means)")
    print("="*80)
    
    # Find optimal k
    clustering_cols = ['Quantity', 'TotalValue', 'Price_x']
    clustering_cols = [col for col in clustering_cols if col in cleaned_df.columns]
    
    if len(clustering_cols) >= 2:
        find_optimal_k(cleaned_df, max_k=8, feature_cols=clustering_cols)
        
        # Apply k-Means with k=3
        kmeans_results = kmeans_clustering(cleaned_df, n_clusters=3,
                                          feature_cols=clustering_cols,
                                          random_state=42)
    
    # Step 12: Summary & Insights
    print("\n" + "="*80)
    print("STEP 12: SUMMARY & INSIGHTS")
    print("="*80)
    
    generate_summary_report(cleaned_df)
    
    print("\n" + "="*80)
    print("EDA PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nAll outputs have been saved to the 'outputs' directory.")
    print("Check the following files:")
    print("  - Visualizations: outputs/*.png")
    print("  - Interactive dashboards: outputs/*.html")
    print("  - Exported data: outputs/merged_dataset.csv")


def generate_summary_report(df):
    """
    Generate a summary report with key insights
    """
    print("\nGenerating Summary Report...")
    
    insights = []
    
    # Basic statistics
    if 'TotalValue' in df.columns:
        total_revenue = df['TotalValue'].sum()
        avg_transaction = df['TotalValue'].mean()
        total_transactions = len(df)
        insights.append(f"Total Revenue: ${total_revenue:,.2f}")
        insights.append(f"Average Transaction Value: ${avg_transaction:.2f}")
        insights.append(f"Total Transactions: {total_transactions:,}")
    
    # Category insights
    if 'Category' in df.columns:
        top_category = df.groupby('Category')['TotalValue'].sum().idxmax()
        top_category_revenue = df.groupby('Category')['TotalValue'].sum().max()
        insights.append(f"Top Category by Revenue: {top_category} (${top_category_revenue:,.2f})")
    
    # Regional insights
    if 'Region' in df.columns:
        top_region = df.groupby('Region')['TotalValue'].sum().idxmax()
        top_region_revenue = df.groupby('Region')['TotalValue'].sum().max()
        insights.append(f"Top Region by Revenue: {top_region} (${top_region_revenue:,.2f})")
    
    # Time-based insights
    if 'TransactionDate' in df.columns:
        df['Year'] = df['TransactionDate'].dt.year
        df['Month'] = df['TransactionDate'].dt.month
        if 'Year' in df.columns:
            best_year = df.groupby('Year')['TotalValue'].sum().idxmax()
            insights.append(f"Best Year: {int(best_year)}")
    
    # Product insights
    if 'ProductName' in df.columns:
        top_product = df.groupby('ProductName')['Quantity'].sum().idxmax()
        top_product_qty = df.groupby('ProductName')['Quantity'].sum().max()
        insights.append(f"Best Selling Product: {top_product} ({int(top_product_qty)} units)")
    
    print("\nKey Insights:")
    print("-" * 60)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Save report to file
    with open('outputs/summary_report.txt', 'w') as f:
        f.write("E-commerce Transaction Data Analysis - Summary Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Key Insights:\n")
        f.write("-" * 60 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
    
    print("\nSummary report saved to outputs/summary_report.txt")


if __name__ == "__main__":
    main()

