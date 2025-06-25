import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, background_dir=None, category_ids=None, background_prob=0.7):
        """
        Dataset for COCO segmentation.

        Args:
            image_dir: Directory containing images (*.jpg).
            annotation_file: Path to JSON annotation file.
            transform: Transformations applied to image and mask.
            background_dir: Directory containing background images (optional).
            category_ids:tablet, optional): List of COCO category IDs to include (e.g., [1, 2]).
            background_prob: Probability of replacing background (default: 0.7).
        """
        self.image_dir = image_dir
        self.transform = transform
        self.background_dir = background_dir
        self.background_prob = background_prob
        self.backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) 
                           if f.endswith(('.jpg', '.png'))] if background_dir else []
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        
        self.category_ids = category_ids or self.coco.getCatIds()
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.category_ids)}
        print(f"Category IDs: {self.category_ids}")
        print(f"Category to Label mapping: {self.cat2label}")
        
        # Validate category_ids
        valid_coco_ids = self.coco.getCatIds()
        for cat_id in self.category_ids:
            if cat_id not in valid_coco_ids:
                raise ValueError(f"Invalid category ID: {cat_id}. Not found in COCO annotations.")

        # Filter images with at least one category
        self.valid_image_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            cat_ids_in_img = set(ann['category_id'] for ann in anns if 'segmentation' in ann and ann['segmentation'])
            if len(cat_ids_in_img.intersection(self.cat2label.keys())) >= 1:
                self.valid_image_ids.append(img_id)
        print(f"Total images: {len(self.image_ids)}, Valid images: {len(self.valid_image_ids)}")

    def __len__(self):
        return len(self.valid_image_ids)

    def replace_background(self, image, mask):
        """
        Replace background with a random image from background_dir.

        Args:
            image: Original image [H, W, 3]
            mask: Mask [H, W] with values [0, 1, 2, ...]

        Returns:
            Image with replaced background
        """
        if not self.backgrounds or random.random() > self.background_prob:
            return image
        bg_path = random.choice(self.backgrounds)
        background = np.array(Image.open(bg_path).convert('RGB').resize(image.shape[:2][::-1]))
        new_image = np.where(mask[..., None] > 0, image, background)
        return new_image

    def __getitem__(self, idx):
        img_id = self.valid_image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        image = np.array(Image.open(img_path).convert('RGB'))
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        
        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            cat_id = ann['category_id']
            if cat_id not in self.cat2label:
                continue
            label = self.cat2label[cat_id]
            ann_mask = self.coco.annToMask(ann)
            mask[ann_mask == 1] = label
        
        if self.background_dir:
            image = self.replace_background(image, mask)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask

def create_dataloader(image_dir, annotation_file, transform, batch_size=8, shuffle=True, 
                     background_dir=None, category_ids=None, background_prob=0.7):
    dataset = COCOSegmentationDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        background_dir=background_dir,
        category_ids=category_ids,
        background_prob=background_prob
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

if __name__ == "__main__":
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_loader = create_dataloader(
        image_dir="data/train/images",
        annotation_file="data/train/annotations/data.json",
        transform=test_transform,
        batch_size=8,
        shuffle=True,
        background_dir="data/backgrounds",
        category_ids=[1, 2],
        background_prob=0.7
    )
    
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Mask unique values: {torch.unique(masks)}")
        break