"""
Modeling Module
Implements k-NN classification and k-Means clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_classification(df, target_col, feature_cols=None):
    """
    Prepare data for classification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column for classification
    feature_cols : list
        Feature columns. If None, uses all numeric columns except target
    
    Returns:
    --------
    tuple
        X (features) and y (target) arrays
    """
    if feature_cols is None:
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Encode target if it's categorical
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y, feature_cols


def knn_classification(df, target_col, feature_cols=None, test_size=0.2, 
                      n_neighbors=5, random_state=42):
    """
    Implement k-Nearest Neighbors classification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column for classification
    feature_cols : list
        Feature columns. If None, uses all numeric columns except target
    test_size : float
        Proportion of test set
    n_neighbors : int
        Number of neighbors for k-NN
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and metrics
    """
    print(f"\n{'='*60}")
    print("k-Nearest Neighbors Classification")
    print(f"{'='*60}")
    
    # Prepare data
    X, y, feature_cols = prepare_data_for_classification(df, target_col, feature_cols)
    
    # Handle missing values
    X = pd.DataFrame(X, columns=feature_cols).fillna(0).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Number of neighbors: {n_neighbors}")
    print(f"  Training set size: {X_train.shape[0]}")
    print(f"  Test set size: {X_test.shape[0]}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - k-NN Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix to outputs/knn_confusion_matrix.png")
    
    return {
        'model': knn,
        'scaler': scaler,
        'y_pred': y_pred,
        'y_test': y_test,
        'accuracy': accuracy,
        'feature_cols': feature_cols
    }


def kmeans_clustering(df, n_clusters=3, feature_cols=None, random_state=42):
    """
    Apply k-Means clustering to numeric data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    n_clusters : int
        Number of clusters
    feature_cols : list
        Feature columns. If None, uses all numeric columns
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing model, labels, and visualizations
    """
    print(f"\n{'='*60}")
    print(f"k-Means Clustering (k={n_clusters})")
    print(f"{'='*60}")
    
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare data
    X = df[feature_cols].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply k-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Cluster distribution:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"    Cluster {cluster}: {count} points ({count/len(labels)*100:.1f}%)")
    
    # Visualize clusters
    if len(feature_cols) >= 2:
        # 2D scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
                            cmap='viridis', alpha=0.6, s=50, edgecolors='black')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.xlabel(feature_cols[0], fontsize=12)
        plt.ylabel(feature_cols[1], fontsize=12)
        plt.title(f'k-Means Clustering (k={n_clusters})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/kmeans_clusters_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved 2D cluster visualization to outputs/kmeans_clusters_2d.png")
        
        # Pair plot with clusters
        if len(feature_cols) <= 4:
            df_viz = pd.DataFrame(X_scaled, columns=feature_cols)
            df_viz['Cluster'] = labels
            sns.pairplot(df_viz, hue='Cluster', palette='viridis', diag_kind='kde')
            plt.suptitle(f'Pair Plot with k-Means Clusters (k={n_clusters})', 
                         fontsize=14, fontweight='bold', y=1.02)
            plt.savefig('outputs/kmeans_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved pair plot with clusters to outputs/kmeans_pairplot.png")
    
    # Interpret clusters
    print(f"\nCluster Interpretation:")
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        print(f"\n  Cluster {cluster_id}:")
        for col in feature_cols[:3]:  # Show first 3 features
            mean_val = cluster_data[col].mean()
            print(f"    {col}: {mean_val:.2f}")
    
    return {
        'model': kmeans,
        'scaler': scaler,
        'labels': labels,
        'df_clustered': df_clustered,
        'feature_cols': feature_cols,
        'inertia': kmeans.inertia_
    }


def find_optimal_k(df, max_k=10, feature_cols=None):
    """
    Find optimal number of clusters using elbow method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    max_k : int
        Maximum number of clusters to test
    feature_cols : list
        Feature columns
    
    Returns:
    --------
    None
        Creates elbow plot
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig('outputs/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved elbow method plot to outputs/elbow_method.png")

