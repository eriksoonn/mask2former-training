from transformers import AutoImageProcessor
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import numpy as np
import os


class ImageSegmentationDataset(Dataset):    
    def __init__(self, images_dir: str, masks_dir: str, transform: Optional[callable] = None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform
        self.filenames  = [
            os.path.splitext(f)[0] 
            for f in os.listdir(images_dir) 
            if not f.startswith('.') and f.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path    = os.path.join(self.images_dir, f"{self.filenames[idx]}.jpg")
        mask_path   = os.path.join(self.masks_dir, f"{self.filenames[idx]}.png")
        
        # Load and preprocess image
        image       = Image.open(img_path).convert("RGB")
        np_image    = np.array(image).transpose(2, 0, 1)   # Convert to C, H, W
        
        # Load and preprocess mask
        mask    = Image.open(mask_path)
        np_mask = np.array(mask)
        np_mask = np.where(np_mask == 255, 1, np_mask)  # Convert 255 to 1
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return np_image, np_mask
    
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int, num_workers: int, img_size: Tuple[int, int], local_processor: bool = False):
        super().__init__()
        model_id   = "facebook/mask2former-swin-base-ade-semantic"
        proc_dir   = "artifacts/mask2former_image_processor"

        self.dataset_dir = dataset_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.img_size    = img_size

        # Load local copy if requested; otherwise try remote and persist for next runs.
        try:
            src = proc_dir if local_processor else model_id
            self.processor = AutoImageProcessor.from_pretrained(src, local_files_only=local_processor)
        except Exception:
            # Fallback: fetch from hub, then save locally for future local loads.
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            self.processor.save_pretrained(proc_dir)
    
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
                transform=None
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
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        # DataLoader for validation dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
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
        
    def _collate_fn(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        images, segmentation_maps = zip(*batch)
        processed_batch = self.processor(
            images=images,
            segmentation_maps=segmentation_maps,
            size=self.img_size,
            return_tensors="pt"
        )
        processed_batch["original_segmentation_maps"] = segmentation_maps
        processed_batch["original_images"] = images
        return processed_batch
