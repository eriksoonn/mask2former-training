from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, AutoImageProcessor
from typing import Dict, List, Optional, Tuple, Any
import mask2former.config as config
import pytorch_lightning as pl
import numpy as np
import evaluate
import torch
import json
import time


class Mask2FormerFinetuner(pl.LightningModule):
    def __init__(self, id2label: Dict[int, str], lr: float, use_pretrained: bool = True, compile: bool = False, local_processor: bool = False):
        super().__init__()
        model_id = "facebook/mask2former-swin-base-ade-semantic"
        proc_dir = "artifacts/mask2former_image_processor"
        cfg_dir  = "artifacts/mask2former_config"

        self.id2label    = id2label
        self.lr          = lr
        self.num_classes = len(id2label)
        self.label2id    = {v: k for k, v in id2label.items()}

        # Config
        if local_processor:
            config = Mask2FormerConfig.from_pretrained(cfg_dir, local_files_only=True)
        else:
            config = Mask2FormerConfig.from_pretrained(
                model_id,
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            config.save_pretrained(cfg_dir)

        # Model:
        if use_pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = Mask2FormerForUniversalSegmentation(config)

        if compile:
            self.model = torch.compile(self.model)

        # Processor
        if local_processor:
            self.processor = AutoImageProcessor.from_pretrained(proc_dir, local_files_only=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            self.processor.save_pretrained(proc_dir)

        self.test_mean_iou = evaluate.load("./metrics/mean_iou")
        self._per_image_metrics: List = []


    def forward(self, pixel_values: torch.Tensor, mask_labels: Optional[List[torch.Tensor]] = None, class_labels: Optional[List[torch.Tensor]] = None) -> Any:
        # Forward pass of the model
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
    
    def transfer_batch_to_device(self, batch: Dict, device: torch.device, dataloader_idx: int = 0) -> Dict:
        # Transfer batch data to the specified device.
        batch["pixel_values"]   = batch["pixel_values"].to(device)
        batch["mask_labels"]    = [label.to(device) for label in batch["mask_labels"]]
        batch["class_labels"]   = [label.to(device) for label in batch["class_labels"]]
        return batch

    def on_train_start(self) -> None:
        # Record the start time of training
        self.start_time = time.time()

    def on_train_end(self) -> None:
        # Save training metrics to a JSON file
        total_time  = time.time() - self.start_time
        metrics     = {"final_epoch": self.current_epoch, "training_time": total_time}

        with open("mask2former_hyperparameters.json", "w") as f:
            json.dump(metrics, f, indent=4)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Perform a training step
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("train_loss", loss, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        self.log("learning_rate", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Perform a validation step
        lr      = self.trainer.optimizers[0].param_groups[0]["lr"]
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )

        self.log("val_loss", outputs.loss, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        self.log("learning_rate", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)

        return outputs.loss

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        # Perform a test step
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )

        # Post-process predictions to original sizes
        original_images = batch["original_images"]
        target_sizes = []
        for img in original_images:
            arr = img if isinstance(img, torch.Tensor) else torch.tensor(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
                H, W = arr.shape[1], arr.shape[2]
            else:  # HWC
                H, W = arr.shape[0], arr.shape[1]
            target_sizes.append((H, W))

        pred_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        preds = [p.detach().cpu().numpy() for p in pred_maps]

        refs = batch["original_segmentation_maps"]
        refs = [
            r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
            for r in refs
        ]

        # Add batch to mean IoU evaluator
        self.test_mean_iou.add_batch(predictions=preds, references=refs)
        return {"loss": outputs.loss.detach()}

    def on_test_epoch_end(self) -> None:
        # Compute and log test metrics at the end of the test epoch
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        # Handle per-category metrics
        per_acc = np.array(metrics.pop("per_category_accuracy"))
        per_iou = np.array(metrics.pop("per_category_iou"))
        per_acc_log = np.nan_to_num(per_acc, nan=0.0)
        per_iou_log = np.nan_to_num(per_iou, nan=0.0)

        # Log summary metrics
        self.log("mean_iou", float(metrics["mean_iou"]), sync_dist=True)
        self.log("mean_accuracy", float(metrics["mean_accuracy"]), sync_dist=True)

        # Log per-class metrics
        for i in range(self.num_classes):
            name = self.id2label[i]
            self.log(f"accuracy_{name}", float(per_acc_log[i]), sync_dist=True)
            self.log(f"iou_{name}", float(per_iou_log[i]), sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Configure the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.LR_SCHEDULER.get('factor'),
                patience=config.LR_SCHEDULER.get('patience'),
            ),
            "reduce_on_plateau": True,
            "monitor": "valLoss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}