import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import logging

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from u2net_lite import U2NETLite
from coco_segmentation_dataset import create_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_metrics(pred, target, num_classes, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    iou_scores = []
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        iou_scores.append(iou.item())
        
        dice = (2 * intersection + eps) / (pred_cls.sum() + target_cls.sum() + eps)
        dice_scores.append(dice.item())
    
    return sum(iou_scores) / len(iou_scores), sum(dice_scores) / len(dice_scores)

def cutmix_data(images, masks, alpha=0.4):
    if random.random() > 0.5:
        return images, masks
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape, lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
    masks[:, bbx1:bbx2, bby1:bby2] = masks[index, bbx1:bbx2, bby1:bby2]
    
    return images, masks

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixstyle(images, alpha=0.1):
    if random.random() > 0.5:
        return images
    batch_size = images.size(0)
    mu = images.mean(dim=[2, 3], keepdim=True)
    sigma = images.std(dim=[2, 3], keepdim=True)
    mu_mix = mu[torch.randperm(batch_size)]
    sigma_mix = sigma[torch.randperm(batch_size)]
    lam = np.random.beta(alpha, alpha)
    return lam * (images - mu) / (sigma + 1e-6) * sigma_mix + mu_mix

class TextureAugment:
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, image, **kwargs):
        if random.random() > self.p:
            return image
        texture = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = np.clip(image + texture, 0, 255)
        return image

train_transform = A.Compose([
    A.Resize(256, 256),
    #A.RandomResizedCrop(height=256, width=256, scale=(0.08, 1.0), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.RandomFog(p=0.2),
    A.RandomRain(p=0.2),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
    A.RandomToneCurve(p=0.3),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=16, min_width=16, fill_value=0, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def train_model(config, progress_callback=None):
    num_classes = config['model']['num_classes']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U2NETLite(in_ch=3, num_classes=num_classes).to(device)
    criterion = smp.losses.DiceLoss(mode="multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = create_dataloader(
        image_dir=config['data']['train_image_dir'],
        annotation_file=config['data']['train_annotation'],
        transform=train_transform,
        batch_size=batch_size,
        shuffle=True,
        background_dir=config['data']['background_dir'],
        category_ids=config['data']['category_ids'],
        background_prob=config['data']['background_prob']
    )
    val_loader = create_dataloader(
        image_dir=config['data']['val_image_dir'],
        annotation_file=config['data']['val_annotation'],
        transform=val_transform,
        batch_size=batch_size,
        shuffle=False,
        background_dir=config['data']['background_dir'],
        category_ids=config['data']['category_ids'],
        background_prob=config['data']['background_prob']
    )

    best_val_iou = 0.0
    best_model_path = config['training']['best_model_path']
    last_model_path = config['training']['last_model_path']

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        train_batches = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            images, masks = cutmix_data(images, masks)
            images = mixstyle(images)
            
            optimizer.zero_grad()
            d0, d1 = model(images)
            loss = criterion(d0, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            iou, dice = calculate_metrics(d0, masks, num_classes)
            train_iou += iou * images.size(0)
            train_dice += dice * images.size(0)
            train_batches += images.size(0)
        
        train_loss /= train_batches
        train_iou /= train_batches
        train_dice /= train_batches

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                d0, d1 = model(images)
                loss = criterion(d0, masks)
                
                val_loss += loss.item() * images.size(0)
                iou, dice = calculate_metrics(d0, masks, num_classes)
                val_iou += iou * images.size(0)
                val_dice += dice * images.size(0)
                val_batches += images.size(0)
        
        val_loss /= val_batches
        val_iou /= val_batches
        val_dice /= val_batches
        
        log_message = (f"Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                       f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        logger.info(log_message)
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice)
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val IoU: {best_val_iou:.4f}")

    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Saved last model at epoch {epochs}")

if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_model(config)