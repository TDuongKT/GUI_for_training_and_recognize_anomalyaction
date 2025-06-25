import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import time
from u2net_lite import U2NETLite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_video.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_transform(image_size):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def colorize_mask(mask, num_classes, colors):
    """
    Convert mask to colored image.
    
    Args:
        mask: Numpy array [H, W] with values [0, 1, ..., num_classes-1]
        num_classes: Number of classes (including background)
        colors: List of RGB colors for each class
    
    Returns:
        Colored mask as numpy array [H, W, 3]
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(num_classes):
        colored_mask[mask == cls] = colors[cls]
    return colored_mask

def test_video(config, input_video, output_video, model_path, progress_callback=None):
    num_classes = config['model']['num_classes']
    image_size = config['model']['image_size']
    colors = config['testing']['colors'][:num_classes]
    
    logger.info(f"Testing model on video: {input_video}")
    logger.info(f"Output video: {output_video}")
    logger.info(f"Model path: {model_path}, num_classes: {num_classes}, image_size: {image_size}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U2NETLite(in_ch=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded and set to evaluation mode on {device}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {input_video}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Get transform
    transform = get_transform(image_size)
    
    frame_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = transform(image=frame_rgb)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Run model
            d0, _ = model(input_tensor)
            pred = torch.argmax(d0, dim=1).squeeze(0).cpu().numpy()
            
            # Colorize mask
            colored_mask = colorize_mask(pred, num_classes, colors)
            
            # Resize mask to original frame size
            colored_mask = cv2.resize(colored_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            
            # Overlay mask on original frame
            output_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)
            
            # Write to output video
            out.write(output_frame)
            
            if progress_callback:
                progress_callback(frame_count, total_frames)
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    logger.info(f"Finished processing {frame_count} frames in {elapsed_time:.2f} seconds")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Output video saved at: {output_video}")

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    test_video(
        config=config,
        input_video=config['testing']['input_video_path'],
        output_video=config['testing']['output_video_path'],
        model_path=config['testing']['model_path']
    )