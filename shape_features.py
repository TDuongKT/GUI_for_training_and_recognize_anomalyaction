import cv2
import numpy as np

def calculate_shape_features(mask, max_area, max_perimeter, prev_centroid=None, frame_idx=0, fps=1):
    """
    Tính toán các đặc trưng hình dạng từ mask phân đoạn.
    
    Args:
        mask: Mask nhị phân của đối tượng.
        max_area (float): Diện tích tối đa để chuẩn hóa.
        max_perimeter (float): Chu vi tối đa để chuẩn hóa.
        prev_centroid (list): Tâm của đối tượng ở frame trước (nếu có).
        frame_idx (int): Chỉ số frame hiện tại.
        fps (float): Số frame trên giây của video.
    
    Returns:
        tuple: Các đặc trưng hình dạng (relative_area, hu_moments, perimeter_ratio, convexity,
               bounding_box_ratio, centroid, aspect_ratio, velocity).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, [0] * 3, 0, 0, 0, [0, 0], 0, [0, 0]

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    area = cv2.contourArea(contour)
    relative_area = area / max_area if max_area > 0 else 0

    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()[:3]
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    perimeter = cv2.arcLength(contour, True)
    perimeter_ratio = perimeter / max_perimeter if max_perimeter > 0 else 0

    convex_hull = cv2.convexHull(contour)
    convexity = area / cv2.contourArea(convex_hull) if cv2.contourArea(convex_hull) > 0 else 0

    bounding_box_ratio = area / (w * h) if w * h > 0 else 0

    aspect_ratio = w / h if h > 0 else 0

    centroid = [x + w / 2, y + h / 2]
    velocity = [0, 0]
    if prev_centroid is not None and frame_idx > 0:
        velocity = [(centroid[0] - prev_centroid[0]) * fps, (centroid[1] - prev_centroid[1]) * fps]

    return relative_area, hu_moments, perimeter_ratio, convexity, bounding_box_ratio, centroid, aspect_ratio, velocity