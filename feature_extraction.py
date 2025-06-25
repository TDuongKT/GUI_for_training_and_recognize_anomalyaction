import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d
import time
from model_utils import preprocess_frame, load_model
from shape_features import calculate_shape_features
import os

def extract_features_from_masks(frame, model, class_names, class_ids, frame_width, frame_height, 
                               max_areas, max_perimeters, device, prev_centroids=None, frame_idx=0, fps=1):
    features = []
    frame_tensor, frame_rgb = preprocess_frame(frame)
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        pred_masks, _ = model(frame_tensor)
    pred_masks = torch.argmax(pred_masks, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_masks = cv2.resize(pred_masks, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    feature_vector = []
    current_centroids = {}
    for cls_id, cls_name in class_ids.items():
        if cls_id == 0:  # Skip background
            continue
        mask = (pred_masks == cls_id).astype(np.uint8)
        prev_centroid = prev_centroids.get(cls_name, [0, 0]) if prev_centroids else [0, 0]
        max_area = max_areas.get(cls_name, 1)
        max_perimeter = max_perimeters.get(cls_name, 1)

        centroid = [0, 0]
        if np.any(mask):
            data = calculate_shape_features(mask, max_area, max_perimeter, prev_centroid, frame_idx, fps)
            relative_area, hu_moments, perimeter_ratio, convexity, bounding_box_ratio, centroid, aspect_ratio, velocity = data
            feature_vector.extend([relative_area, *hu_moments, perimeter_ratio, convexity, bounding_box_ratio, aspect_ratio, *velocity])
            current_centroids[cls_name] = centroid
        else:
            feature_vector.extend([0, *[0]*3, 0, 0, 0, 0, 0, 0])

    # Calculate centroid distances between all pairs of classes
    for i, cls_name1 in enumerate(class_ids.values()):
        if i == 0:  # Skip background
            continue
        for cls_name2 in list(class_ids.values())[i+1:]:
            if cls_name2 == 'background':
                continue
            centroid1 = current_centroids.get(cls_name1, [0, 0])
            centroid2 = current_centroids.get(cls_name2, [0, 0])
            distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2) / max(frame_width, frame_height)
            feature_vector.append(distance)

    features.append(feature_vector)
    return np.array(features) if features else None, current_centroids

def interpolate_features(features, frame_times, target_intervals=100):
    if len(features) < 2:
        return None
    original_times = np.array(frame_times) / max(frame_times) if max(frame_times) > 0 else np.array(frame_times)
    target_times = np.linspace(0, 1, target_intervals)
    interpolated = []
    for i in range(features.shape[2]):
        interp_func = interp1d(original_times, features[:, 0, i], kind='linear', fill_value="extrapolate")
        interpolated.append(interp_func(target_times))
    return np.array(interpolated).T.reshape(target_intervals, 1, -1)

def process_video(self, video_path, model_path, output_dir, device, num_classes, class_names):
    self.update_status(f"Processing video: {video_path}")
    model = load_model(model_path, num_classes=num_classes, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        self.update_status(f"Error: Cannot open video {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    class_ids = {i: name.strip() for i, name in enumerate(class_names)}
    max_areas = {name: 1 for name in class_ids.values() if name != 'background'}
    max_perimeters = {name: 1 for name in class_ids.values() if name != 'background'}

    cap_temp = cv2.VideoCapture(video_path)
    while cap_temp.isOpened():
        ret, frame = cap_temp.read()
        if not ret:
            break
        frame_tensor, _ = preprocess_frame(frame)
        frame_tensor = frame_tensor.to(device)
        with torch.no_grad():
            pred_masks, _ = model(frame_tensor)
        pred_masks = torch.argmax(pred_masks, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_masks = cv2.resize(pred_masks, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        for cls_id, cls_name in class_ids.items():
            if cls_id == 0:  # Skip background
                continue
            mask = (pred_masks == cls_id).astype(np.uint8)
            if np.any(mask):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    max_areas[cls_name] = max(max_areas[cls_name], area)
                    max_perimeters[cls_name] = max(max_perimeters[cls_name], perimeter)
    cap_temp.release()

    temporal_features = []
    frame_times = []
    frame_idx = 0
    prev_centroids = {}
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps if fps > 0 else 0
        frame_times.append(current_time)

        features, current_centroids = extract_features_from_masks(
            frame, model, class_names, class_ids, frame_width, frame_height,
            max_areas, max_perimeters, device, prev_centroids, frame_idx, fps
        )

        if features is not None:
            temporal_features.append(features)
        else:
            self.update_status(f"Skipped frame {frame_idx} in {video_path} due to missing features.")
        prev_centroids = current_centroids
        frame_idx += 1
        self.update_status(f"Processed frame {frame_idx}/{frame_count} for video {video_path}")

    cap.release()

    if temporal_features and len(temporal_features) >= 10:
        temporal_features = np.array(temporal_features)
        interpolated_features = interpolate_features(temporal_features, frame_times)
        if interpolated_features is not None:
            video_name = Path(video_path).stem
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{video_name}_features.npy")
            np.save(output_path, interpolated_features)
            self.update_status(f"Saved features for {video_name} to {output_path}")

            elapsed_time = time.time() - start_time
            fps_processed = frame_idx / elapsed_time if elapsed_time > 0 else 0
            self.update_status(f"Total time: {elapsed_time:.2f}s, FPS: {fps_processed:.2f}")

            return interpolated_features
        else:
            self.update_status(f"Video {video_path} is too short or has no features")
    return None