import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import random
from math import comb

def evaluate_feature_combinations(self, features, n_clusters, min_features, max_features, max_combinations, tab='optimization'):
    """
    Evaluate feature combinations using KMeans clustering and save results to a file.

    Args:
        self: GUI instance for updating status.
        features: Numpy array containing features of videos.
        n_clusters: Number of clusters for KMeans.
        min_features: Minimum number of features in each combination.
        max_features: Maximum number of features in each combination.
        max_combinations: Maximum number of combinations to evaluate.
        tab: Tab name for status updates ('optimization').

    Returns:
        Best combination result based on silhouette score.
    """
    num_classes = len(self.config['feature_extraction']['class_names']) - 1  # Exclude background
    feature_names = []
    for cls_name in self.config['feature_extraction']['class_names']:
        if cls_name == 'background':
            continue
        feature_names.extend([
            f"{cls_name}_relative_area", f"{cls_name}_hu_moment_0", f"{cls_name}_hu_moment_1", f"{cls_name}_hu_moment_2",
            f"{cls_name}_perimeter_ratio", f"{cls_name}_convexity", f"{cls_name}_bounding_box_ratio",
            f"{cls_name}_aspect_ratio", f"{cls_name}_velocity_x", f"{cls_name}_velocity_y"
        ])
    # Add centroid distances
    for i, cls_name1 in enumerate(self.config['feature_extraction']['class_names']):
        if cls_name1 == 'background':
            continue
        for cls_name2 in self.config['feature_extraction']['class_names'][i+1:]:
            if cls_name2 == 'background':
                continue
            feature_names.append(f"centroid_distance_{cls_name1[0]}{cls_name2[0]}")

    total_combinations = sum(comb(len(feature_names), r) for r in range(min_features, min(max_features + 1, len(feature_names) + 1)))
    self.update_status(f"Total possible combinations: {total_combinations}", tab=tab)

    # Generate random combinations
    sampled_combinations = []
    n_features = len(feature_names)
    seen = set()

    while len(sampled_combinations) < min(max_combinations, total_combinations):
        r = random.randint(min_features, min(max_features, n_features))
        combo = tuple(sorted(random.sample(range(n_features), r)))
        if combo not in seen:
            seen.add(combo)
            sampled_combinations.append(combo)

    results = []
    for idx, combo in enumerate(sampled_combinations):
        selected_features = features[:, :, list(combo)]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features.reshape(-1, len(combo)))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)

        sil_score = -1
        if len(set(labels)) > 1:
            sil_score = silhouette_score(scaled_features, labels)

        results.append({
            'combo_idx': idx + 1,
            'features': [feature_names[i] for i in combo],
            'feature_indices': list(combo),  # Save the feature indices
            'silhouette_score': sil_score
        })
        self.update_status(f"Combination {idx + 1}/{len(sampled_combinations)}: {', '.join([feature_names[i] for i in combo])}", tab=tab)
        self.update_status(f"Silhouette Score: {sil_score:.4f}", tab=tab)
        self.optimization_progress['value'] = ((idx + 1) / len(sampled_combinations)) * 100
        self.root.update()

    best_result = max(results, key=lambda x: x['silhouette_score'] if x['silhouette_score'] != -1 else -float('inf'))
    self.update_status(f"\nBest combination (based on Silhouette Score):", tab=tab)
    self.update_status(f"Combination {best_result['combo_idx']}: {', '.join(best_result['features'])}", tab=tab)
    self.update_status(f"Silhouette Score: {best_result['silhouette_score']:.4f}", tab=tab)

    os.makedirs(self.output_dir.get(), exist_ok=True)
    np.save(os.path.join(self.output_dir.get(), 'combination_results.npy'), results)
    self.update_status(f"Saved combination results to {os.path.join(self.output_dir.get(), 'combination_results.npy')}", tab=tab)

    return best_result