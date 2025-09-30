import pytorch_lightning as pl
import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
from torch import nn
import mask2former.config as config
import evaluate
import time
import json 
import numpy as np

class Mask2FormerFinetuner(pl.LightningModule):

    def __init__(self, id2label, lr ):
        super(Mask2FormerFinetuner, self).__init__()
        self.id2label = id2label
        self.lr = lr
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-ade-semantic",
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
        evaluate.load
        self.test_mean_iou = evaluate.load("mean_iou")
        self._per_image_metrics = []
        
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        # Your model's forward method
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['mask_labels'] = [label.to(device) for label in batch['mask_labels']]
        batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        return batch
        
    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch, 'training_time': total_time}
        with open('mask2former_hyperparameters.json', 'w') as f:
            json.dump(metrics, f)

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        loss = outputs.loss
        self.log("trainLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        self.log("valLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE, on_epoch=True,logger=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )

        # Post-process predictions to original sizes (list of [H,W])
        original_images = batch["original_images"]
        target_sizes = []
        for img in original_images:
            arr = img if isinstance(img, torch.Tensor) else torch.tensor(img)
            if arr.ndim == 3 and arr.shape[0] in (1,3):  # CHW
                H, W = arr.shape[1], arr.shape[2]
            else:                                        # HWC
                H, W = arr.shape[0], arr.shape[1]
            target_sizes.append((H, W))

        pred_maps = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        preds = [p.detach().cpu().numpy() for p in pred_maps]

        refs = batch["original_segmentation_maps"]
        refs = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r) for r in refs]

        # >>> Use add_batch over lists <<<
        self.test_mean_iou.add_batch(
            predictions=preds,
            references=refs,
        )

        # (Optional) return loss for logging
        return {"loss": outputs.loss.detach()}

    def on_test_epoch_end(self):
        # >>> Compute once for the entire test set <<<
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,         # <-- set this to your dataset's true ignore id
            reduce_labels=False,
        )

        # Handle per-category safely (NaNs when a class never appears)
        per_acc = np.array(metrics.pop("per_category_accuracy"))
        per_iou = np.array(metrics.pop("per_category_iou"))

        # If you prefer to log zeros instead of NaNs:
        per_acc_log = np.nan_to_num(per_acc, nan=0.0)
        per_iou_log = np.nan_to_num(per_iou, nan=0.0)

        # Log summary metrics
        self.log("mean_iou", float(metrics["mean_iou"]))
        self.log("mean_accuracy", float(metrics["mean_accuracy"]))

        # Log per-class metrics (replace NaN->0 just for logger readability)
        for i in range(self.num_classes):
            name = self.id2label[i]
            self.log(f"accuracy_{name}", float(per_acc_log[i]))
            self.log(f"iou_{name}", float(per_iou_log[i]))
        
    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
      
        # ReduceLROnPlateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.FACTOR, patience=config.PATIENCE),
            'reduce_on_plateau': True,  # Necessary for ReduceLROnPlateau
            'monitor': 'valLoss'  # Metric to monitor for reducing learning rate
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
