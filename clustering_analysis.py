import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
from pathlib import Path

def load_optimal_combination(feature_results_path="features/combination_results.npy"):
    """
    Load the best feature combination from the results file.
    
    Args:
        feature_results_path: Path to combination_results.npy
        
    Returns:
        Tuple of (optimal feature indices, optimal feature names)
    """
    if not os.path.exists(feature_results_path):
        raise FileNotFoundError(f"Cannot find file {feature_results_path}")
    results = np.load(feature_results_path, allow_pickle=True)
    best_result = max(results, key=lambda x: x['silhouette_score'] if x['silhouette_score'] != -1 else -float('inf'))
    
    return best_result['feature_indices'], best_result['features']

def load_features(feature_dir="features", num_classes=3):
    """
    Load feature files from the specified directory.
    
    Args:
        feature_dir: Directory containing feature files
        num_classes: Number of classes excluding background
        
    Returns:
        Tuple of (list of feature arrays, list of video names)
    """
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.npy')]
    all_features, video_names = [], []
    expected_features = num_classes * 10 + (num_classes * (num_classes - 1)) // 2
    
    for file in feature_files:
        path = os.path.join(feature_dir, file)
        features = np.load(path)
        if features.shape[1] == 1 and features.shape[2] == expected_features:
            all_features.append(features)
            video_names.append(Path(file).stem.replace('_features', ''))
    
    return all_features, video_names

def compute_cluster_features(labels, n_clusters, num_frames=100):
    """
    Compute clustering features such as distribution and mean times.
    
    Args:
        labels: Cluster labels
        n_clusters: Number of clusters
        num_frames: Number of frames per video
        
    Returns:
        Tuple of (cluster distribution, cluster mean times)
    """
    time_indices = np.arange(num_frames)
    cluster_dist = np.bincount(labels, minlength=n_clusters) / len(labels)
    cluster_mean_times = [np.mean(time_indices[labels == i]) / num_frames if np.any(labels == i) else 0 for i in range(n_clusters)]
    return cluster_dist, cluster_mean_times

def cluster_optimal_features(features_list, optimal_feature_indices, n_clusters):
    """
    Perform clustering on optimal features.
    
    Args:
        features_list: List of feature arrays
        optimal_feature_indices: Indices of optimal features
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (results list, all labels, overall silhouette score)
    """
    results, all_labels = [], []
    
    for idx, features in enumerate(features_list):
        selected = features[:, :, optimal_feature_indices].reshape(-1, len(optimal_feature_indices))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(selected)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled)
        
        cluster_dist, cluster_mean_times = compute_cluster_features(labels, n_clusters)
        sil_score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else -1
        
        results.append({
            'cluster_dist': cluster_dist,
            'cluster_mean_times': cluster_mean_times,
            'labels': labels,
            'silhouette_score': sil_score
        })
        all_labels.extend(labels)
    
    all_combined = np.vstack([f[:, :, optimal_feature_indices].reshape(-1, len(optimal_feature_indices)) for f in features_list])
    overall_sil = silhouette_score(all_combined, all_labels) if len(set(all_labels)) > 1 else -1
    
    return results, np.array(all_labels), overall_sil

def save_clustering_results(results, video_names, output_path="features/clustering_results.npy"):
    """
    Save clustering results to a file.
    
    Args:
        results: List of clustering results
        video_names: List of video names
        output_path: Path to save results
    """
    formatted = [{
        'cluster_dist': r['cluster_dist'],
        'cluster_mean_times': r['cluster_mean_times'],
        'video_name': video_names[i]
    } for i, r in enumerate(results)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, formatted)