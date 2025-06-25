import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from u2net_lite import U2NETLite

# Transform để tiền xử lý frame
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_model(model_path, num_classes, device='cuda'):
    model = U2NETLite(in_ch=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=frame_rgb)
    frame_tensor = augmented['image'].unsqueeze(0)  # [1, 3, 256, 256]
    return frame_tensor, frame_rgb

def overlay_mask(frame, mask, colors):
    overlay = frame.copy()
    for cls in range(1, len(colors)):  # Bỏ background (cls=0)
        overlay[mask == cls] = colors[cls]
    return cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)