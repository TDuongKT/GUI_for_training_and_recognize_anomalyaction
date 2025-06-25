import os
import json
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import cv2
import yaml
from model_utils import preprocess_frame, overlay_mask, load_model
from shape_features import calculate_shape_features
from clustering_analysis import load_optimal_combination, compute_cluster_features
from feature_extraction import interpolate_features

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_features_from_masks_custom(frame, model, frame_width, frame_height,
                                      max_areas, max_perimeters, class_names, colors,
                                      device, prev_centroids=None, frame_idx=0, fps=1, 
                                      optimal_feature_indices=None):
    """
    Extract features from predicted masks for a single frame.
    
    Args:
        frame: Input frame (numpy array)
        model: U2NETLite model
        frame_width, frame_height: Frame dimensions
        max_areas, max_perimeters: Lists of max areas and perimeters for normalization
        class_names: List of class names (excluding background)
        colors: List of RGB colors for masks
        device: Device for model inference (cuda/cpu)
        prev_centroids: List of previous centroids for velocity
        frame_idx: Current frame index
        fps: Frames per second
        optimal_feature_indices: Indices of selected features
        
    Returns:
        Tuple of (features, masks_to_rgb, current centroids)
    """
    features = []
    masks_to_rgb = []
    num_classes = len(class_names) + 1  # Including background
    prev_centroids = prev_centroids or [None] * len(class_names)
    current_centroids = [None] * len(class_names)

    frame_tensor, _ = preprocess_frame(frame=frame)
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        pred_masks, _ = model(frame_tensor)
    pred_masks = torch.argmax(pred_masks, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_masks = cv2.resize(pred_masks, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    class_data = [None] * len(class_names)
    for cls_idx, (cls_name, max_area, max_perimeter, prev_centroid, color) in enumerate(
        zip(class_names, max_areas, max_perimeters, prev_centroids, colors)
    ):
        mask = (pred_masks == (cls_idx + 1)).astype(np.uint8)  # Class indices start from 1 (0 is background)
        if np.any(mask):
            data = calculate_shape_features(mask, max_area, max_perimeter, prev_centroid, frame_idx, fps)
            class_data[cls_idx] = data
            masks_to_rgb.append((mask, tuple(color)))
            current_centroids[cls_idx] = data[5] if data else None

    def safe_data(data):
        if data:
            return list(data[:-1]) + data[-1]
        return [0] * 10

    class_features = [safe_data(data) for data in class_data]

    # Compute centroid distances between all pairs of classes
    centroid_distances = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if class_data[i] and class_data[j]:
                dist = np.linalg.norm(np.array(class_data[i][5]) - np.array(class_data[j][5])) / max(frame_width, frame_height)
            else:
                dist = 0
            centroid_distances.append(dist)

    feature_vector = sum(class_features, []) + centroid_distances

    if optimal_feature_indices is not None:
        feature_vector = [feature_vector[i] for i in optimal_feature_indices]

    features.append(feature_vector)
    return np.array(features), masks_to_rgb, current_centroids

def cluster_optimal_features_custom(features, optimal_feature_indices, n_clusters=3):
    """
    Cluster features using KMeans with optimal feature indices.
    
    Args:
        features: Feature array
        optimal_feature_indices: Indices of selected features
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with clustering results
    """
    selected_features = features.reshape(-1, features.shape[-1])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    cluster_dist, cluster_mean_times = compute_cluster_features(labels, n_clusters)
    sil_score = silhouette_score(scaled_features, labels) if len(set(labels)) > 1 else -1

    return {
        'cluster_dist': cluster_dist,
        'cluster_mean_times': cluster_mean_times,
        'labels': labels,
        'silhouette_score': sil_score
    }

def inference_video(self, video_path, model_path, svm_model_path, output_video_path, result_json_path, device='cuda'):
    """
    Perform inference on a video to detect anomalous actions.
    
    Args:
        self: GUI instance for status updates
        video_path: Path to input video
        model_path: Path to U2NETLite model
        svm_model_path: Path to SVM model
        output_video_path: Path to save segmented video
        result_json_path: Path to save JSON results
        device: Device for model inference
        
    Returns:
        Dictionary with inference results or None if failed
    """
    try:
        # Load configuration
        config = load_config()
        class_names = config.get('feature_extraction', {}).get('class_names', ['background', 'shell', 'tip', 'spring'])[1:]  # Exclude background
        colors = config.get('testing', {}).get('colors', [[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]])[1:]  # Exclude background color
        num_classes = len(class_names) + 1  # Including background

        self.update_status(f"Loading U2NETLite model from {model_path}...", tab='inference')
        model = load_model(model_path, num_classes=num_classes, device=device)
        self.update_status(f"Loading SVM model from {svm_model_path}...", tab='inference')
        anomaly_models = np.load(svm_model_path, allow_pickle=True).item()
        svm_model, svm_threshold = anomaly_models["svm"], anomaly_models["svm_threshold"]
        self.update_status(f"Loading optimal feature indices...", tab='inference')
        optimal_feature_indices, optimal_features = load_optimal_combination(feature_results_path=os.path.join(self.output_dir.get(), "combination_results.npy"))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError(f"Cannot open output video: {output_video_path}")

        # Compute max areas and perimeters
        max_areas = [0] * len(class_names)
        max_perimeters = [0] * len(class_names)
        cap_temp = cv2.VideoCapture(video_path)
        frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        self.update_status(f"Computing max areas and perimeters for {frame_count} frames...", tab='inference')
        while cap_temp.isOpened():
            ret, frame = cap_temp.read()
            if not ret:
                break
            frame_tensor, _ = preprocess_frame(frame=frame)
            frame_tensor = frame_tensor.to(device)
            with torch.no_grad():
                pred_masks, _ = model(frame_tensor)
            pred_masks = torch.argmax(pred_masks, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_masks = cv2.resize(pred_masks, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            for cls_idx in range(len(class_names)):
                mask = (pred_masks == (cls_idx + 1)).astype(np.uint8)
                if np.any(mask):
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        max_areas[cls_idx] = max(max_areas[cls_idx], area)
                        max_perimeters[cls_idx] = max(max_perimeters[cls_idx], perimeter)
        cap_temp.release()

        # Process video frames
        temporal_features = []
        frame_times = []
        frame_idx = 0
        centroids = [None] * len(class_names)
        processed_frames = 0

        self.update_status(f"Processing video frames...", tab='inference')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps if fps > 0 else 0

            features, masks_to_rgb, centroids = extract_features_from_masks_custom(
                frame, model, frame_width, frame_height,
                max_areas, max_perimeters, class_names, colors, device,
                centroids, frame_idx, fps, optimal_feature_indices
            )
            if features is not None and len(features) > 0:
                temporal_features.append(features)
                frame_times.append(current_time)

            # Overlay masks and write to output video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for mask, color in masks_to_rgb:
                frame_rgb = overlay_mask(frame_rgb, mask, color)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            frame_idx += 1
            processed_frames += 1
            self.inference_progress['value'] = (processed_frames / frame_count) * 100 if frame_count > 0 else 0
            self.root.update()

        cap.release()
        out.release()

        if temporal_features and len(temporal_features) >= 10:
            temporal_features = np.array(temporal_features)
            self.update_status(f"Interpolating features for {len(temporal_features)} frames...", tab='inference')
            interpolated_features = interpolate_features(temporal_features, frame_times)
            if interpolated_features is not None:
                self.update_status(f"Clustering features with {self.n_clusters.get()} clusters...", tab='inference')
                clustering_result = cluster_optimal_features_custom(
                    interpolated_features, optimal_feature_indices, n_clusters=int(self.n_clusters.get())
                )
                combined_features = np.concatenate([clustering_result["cluster_dist"], clustering_result["cluster_mean_times"]])
                svm_score = svm_model.score_samples([combined_features])[0]

                result = {
                    "video_path": str(video_path),
                    "datetime": datetime.now().isoformat(),
                    "silhouette_score": float(clustering_result['silhouette_score']),
                    "svm_score": float(svm_score),
                    "svm_anomaly": bool(svm_score < svm_threshold)
                }

                # Save results to JSON
                os.makedirs(os.path.dirname(result_json_path), exist_ok=True)
                if os.path.exists(result_json_path):
                    with open(result_json_path, 'r') as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(result)
                with open(result_json_path, 'w') as f:
                    json.dump(data, f, indent=2)

                self.update_status(f"Inference completed. Results saved to {result_json_path}", tab='inference')
                self.update_status(f"Silhouette Score: {result['silhouette_score']:.4f}", tab='inference')
                self.update_status(f"SVM Score: {result['svm_score']:.4f}, Anomaly: {result['svm_anomaly']}", tab='inference')
                self.update_status(f"Segmented video saved to {output_video_path}", tab='inference')
                return result
            else:
                self.update_status("Error: Feature interpolation failed", tab='inference')
        else:
            self.update_status(f"Error: Insufficient features extracted ({len(temporal_features)} frames)", tab='inference')
    except Exception as e:
        self.update_status(f"Error during inference: {str(e)}", tab='inference')
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
    return None