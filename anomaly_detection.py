import numpy as np
from sklearn.svm import OneClassSVM
import os

def load_clustering_results(results_path="features/clustering_results.npy"):
    """
    Load clustering results from the specified file.
    
    Args:
        results_path: Path to clustering_results.npy
        
    Returns:
        Tuple of (feature array, list of video names)
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Cannot find file: {results_path}")

    results = np.load(results_path, allow_pickle=True)
    features = []
    video_names = []

    for r in results:
        dist = np.array(r["cluster_dist"])
        mean_time = np.array(r["cluster_mean_times"])
        combined = np.concatenate([dist, mean_time])
        features.append(combined)
        video_names.append(r["video_name"])

    return np.array(features), video_names

def train_one_class_svm(features, nu=0.05, gamma=0.1):
    """
    Train a One-Class SVM model for anomaly detection.
    
    Args:
        features: Feature array for training
        nu: SVM nu parameter (proportion of outliers)
        gamma: SVM gamma parameter (kernel coefficient)
        
    Returns:
        Tuple of (trained model, anomaly scores, threshold)
    """
    model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    model.fit(features)
    scores = model.score_samples(features)
    threshold = np.mean(scores) - 1.5 * np.std(scores)
    return model, scores, threshold

def save_models(svm_model, svm_threshold, output_path="svm_model.npy"):
    """
    Save the trained SVM model and threshold.
    
    Args:
        svm_model: Trained One-Class SVM model
        svm_threshold: Anomaly detection threshold
        output_path: Path to save the model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, {
        "svm": svm_model,
        "svm_threshold": svm_threshold
    })

def summarize_scores(video_names, svm_scores):
    """
    Summarize anomaly scores for each video.
    
    Args:
        video_names: List of video names
        svm_scores: Anomaly scores from SVM
        
    Returns:
        Formatted summary string
    """
    summary = "\n📊 Anomaly Scores:\n"
    summary += "{:<25} {:>12}\n".format("Video", "SVM Score")
    for name, svm in zip(video_names, svm_scores):
        summary += "{:<25} {:>12.4f}\n".format(name, svm)
    return summary