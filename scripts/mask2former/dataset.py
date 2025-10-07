from transformers import AutoImageProcessor
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import pytorch_lightning as pl
import albumentations as A
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os


class ImageSegmentationDataset(Dataset):    
    def __init__(self, images_dir: str, masks_dir: str, transform: Optional[callable] = None):
        self.images_dir     = images_dir
        self.masks_dir      = masks_dir
        self.transform      = transform
        self.filenames      = [
            os.path.splitext(f)[0] 
            for f in os.listdir(images_dir) 
            if not f.startswith('.') and f.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path    = os.path.join(self.images_dir, f"{self.filenames[idx]}.jpg")
        mask_path   = os.path.join(self.masks_dir, f"{self.filenames[idx]}.png")
        
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)
        
        np_image = np.array(image)                                          # H, W, C
        np_mask  = np.array(mask)                                           # H, W
        np_mask  = np.where(np_mask == 255, 1, np_mask).astype(np.uint8)    # 255->1 for FG
        
        if self.transform:
            aug = self.transform(image=np_image, mask=np_mask)
            np_image, np_mask = aug["image"], aug["mask"]
            
        # keep original return shape (C, H, W) to minimize downstream changes
        np_image = np_image.transpose(2, 0, 1)

        return np_image, np_mask
    
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int, num_workers: int, img_size: Tuple[int, int], local_processor: bool = False, keep_originals: bool = False):
        super().__init__()
        #model_id   = "facebook/mask2former-swin-base-ade-semantic"
        model_id   = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        proc_dir   = "artifacts/mask2former_image_processor"

        self.dataset_dir        = dataset_dir
        self.batch_size         = batch_size
        self.num_workers        = num_workers
        self.img_size           = img_size
        self.keep_originals     = keep_originals
        self.train_transform    = None

        # Load local copy if requested; otherwise try remote and persist for next runs.
        if local_processor:
            self.processor = AutoImageProcessor.from_pretrained(proc_dir, local_files_only=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            self.processor.save_pretrained(proc_dir)
            
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=(0.2, 1.3), p=1.0),

            # Crops the largest valid rectangle inside the rotated image
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, crop_border=True, p=0.3),

            # Safety pad + final crop
            A.PadIfNeeded(min_height=self.img_size[0], min_width=self.img_size[1],
                        border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(height=self.img_size[0], width=self.img_size[1], p=1.0),

            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        # Define dataset paths
        train_dir       = os.path.join(self.dataset_dir, 'images', 'train')
        val_dir         = os.path.join(self.dataset_dir, 'images', 'val')
        test_dir        = os.path.join(self.dataset_dir, 'images', 'test')
        train_mask_dir  = os.path.join(self.dataset_dir, 'labels', 'train')
        val_mask_dir    = os.path.join(self.dataset_dir, 'labels', 'val')
        test_mask_dir   = os.path.join(self.dataset_dir, 'labels', 'test')
        
        # Setup datasets based on stage
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageSegmentationDataset(
                images_dir=train_dir,
                masks_dir=train_mask_dir,
                transform=self.train_transform
            )
            self.val_dataset = ImageSegmentationDataset(
                images_dir=val_dir,
                masks_dir=val_mask_dir,
                transform=None
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ImageSegmentationDataset(
                images_dir=test_dir,
                masks_dir=test_mask_dir,
                transform=None
            )
    
    def train_dataloader(self) -> DataLoader:
        # DataLoader for training dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=False
        )
    
    def val_dataloader(self) -> DataLoader:
        # DataLoader for validation dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=False
        )
    
    def test_dataloader(self) -> DataLoader:
        # DataLoader for testing dataset
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
    def _collate_fn(self, batch):
        images, masks = zip(*batch)
        out = self.processor(
            images=images, segmentation_maps=masks,
            size=self.img_size, return_tensors="pt"
        )
        if self.keep_originals:
            out["original_images"] = images
            out["original_segmentation_maps"] = masks
        return out

