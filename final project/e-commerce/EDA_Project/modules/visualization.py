"""
Visualization Module
Creates basic, advanced, and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def create_line_plot(df, x_col, y_col, title="Line Plot", save_path=None):
    """
    Create a line plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=4)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved line plot to {save_path}")
    plt.close()


def create_bar_chart(df, x_col, y_col, title="Bar Chart", save_path=None):
    """
    Create a bar chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_col], df[y_col], color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    plt.close()


def create_histogram(df, column, bins=30, title="Histogram", save_path=None):
    """
    Create a histogram
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to plot
    bins : int
        Number of bins
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    plt.close()


def create_pair_plot(df, columns=None, save_path=None):
    """
    Create a pair plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        Columns to include. If None, includes all numeric columns
    save_path : str
        Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to 5 columns
    
    sns.pairplot(df[columns], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pair Plot', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved pair plot to {save_path}")
    plt.close()


def create_heatmap(df, columns=None, title="Correlation Heatmap", save_path=None):
    """
    Create a correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        Columns to include. If None, includes all numeric columns
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    plt.close()


def create_violin_plot(df, x_col, y_col, title="Violin Plot", save_path=None):
    """
    Create a violin plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Categorical column for x-axis
    y_col : str
        Numeric column for y-axis
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x_col, y=y_col, palette='Set2')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved violin plot to {save_path}")
    plt.close()


def create_probability_distribution(df, column, title="Probability Distribution", save_path=None):
    """
    Visualize probability distribution for a numeric feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to visualize
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with density
    axes[0].hist(df[column].dropna(), bins=50, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='black')
    axes[0].set_title(f'Histogram - {column}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot - {column}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved probability distribution plot to {save_path}")
    plt.close()


def create_interactive_dashboard(df, save_path='outputs/interactive_dashboard.html'):
    """
    Create an interactive dashboard using Plotly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    save_path : str
        Path to save the HTML file
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales Over Time', 'Category Distribution', 
                       'Price Distribution', 'Top Products'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Get numeric columns for demonstration
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Plot 1: Line plot (if date column exists)
    if len(numeric_cols) >= 2:
        fig.add_trace(
            go.Scatter(x=df.index[:100], y=df[numeric_cols[0]][:100], 
                     mode='lines+markers', name=numeric_cols[0]),
            row=1, col=1
        )
    
    # Plot 2: Bar chart (if categorical column exists)
    if categorical_cols:
        value_counts = df[categorical_cols[0]].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, 
                  name=categorical_cols[0]),
            row=1, col=2
        )
    
    # Plot 3: Histogram
    if numeric_cols:
        fig.add_trace(
            go.Histogram(x=df[numeric_cols[0]], nbinsx=30, name=numeric_cols[0]),
            row=2, col=1
        )
    
    # Plot 4: Top values
    if len(numeric_cols) >= 2:
        top_values = df.nlargest(10, numeric_cols[1]) if len(df) > 10 else df
        fig.add_trace(
            go.Bar(x=top_values.index[:10], y=top_values[numeric_cols[1]][:10],
                  name=numeric_cols[1]),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Interactive E-commerce Dashboard",
        showlegend=True
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Saved interactive dashboard to {save_path}")


def create_interactive_scatter(df, x_col, y_col, color_col=None, 
                              title="Interactive Scatter Plot", 
                              save_path='outputs/interactive_scatter.html'):
    """
    Create an interactive scatter plot using Plotly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    color_col : str
        Column for color coding
    title : str
        Plot title
    save_path : str
        Path to save the HTML file
    """
    if color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=title, hover_data=df.columns.tolist())
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=title,
                        hover_data=df.columns.tolist())
    
    fig.update_layout(
        width=1000,
        height=600,
        title_font_size=16
    )
    
    fig.write_html(save_path)
    print(f"Saved interactive scatter plot to {save_path}")

