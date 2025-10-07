from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, AutoImageProcessor
from typing import Dict, List, Optional, Tuple, Any
from torchmetrics import JaccardIndex
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
        #model_id = "facebook/mask2former-swin-base-ade-semantic"
        model_id   = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        proc_dir = "artifacts/mask2former_image_processor"
        cfg_dir  = "artifacts/mask2former_config"

        self.id2label    = id2label
        self.lr          = lr
        self.num_classes = len(id2label)
        self.label2id    = {v: k for k, v in id2label.items()}

        # Config
        if local_processor:
            config_ = Mask2FormerConfig.from_pretrained(cfg_dir, local_files_only=True)
        else:
            config_ = Mask2FormerConfig.from_pretrained(
                model_id,
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            config_.save_pretrained(cfg_dir)

        # Model
        if use_pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = Mask2FormerForUniversalSegmentation(config_)

        if compile:
            self.model = torch.compile(self.model)

        # Processor
        if local_processor:
            self.processor = AutoImageProcessor.from_pretrained(proc_dir, local_files_only=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            self.processor.save_pretrained(proc_dir)

        # Metrics
        self.val_iou = JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=255)
        self.test_mean_iou = evaluate.load("./metrics/mean_iou")

        self._per_image_metrics: List = []

    def forward(self, pixel_values: torch.Tensor, mask_labels: Optional[List[torch.Tensor]] = None, class_labels: Optional[List[torch.Tensor]] = None) -> Any:
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def transfer_batch_to_device(self, batch: Dict, device: torch.device, dataloader_idx: int = 0) -> Dict:
        batch["pixel_values"] = batch["pixel_values"].to(device)
        batch["mask_labels"]  = [label.to(device) for label in batch["mask_labels"]]
        batch["class_labels"] = [label.to(device) for label in batch["class_labels"]]
        return batch

    # ----------------- TRAIN ----------------- #
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("loss/train", loss, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        self.log("lr/train", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)

        return loss
    
    def on_train_start(self) -> None:
        self.start_time = time.time()

    def on_train_end(self) -> None:
        total_time  = time.time() - self.start_time
        metrics     = {"final_epoch": self.current_epoch, "training_time": total_time}
        with open("mask2former_hyperparameters.json", "w") as f:
            json.dump(metrics, f, indent=4)
    # ----------------------------------------- #

    # -------------- VALIDATION --------------- #
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self(
                pixel_values=batch["pixel_values"],
                mask_labels=batch["mask_labels"],
                class_labels=batch["class_labels"],
            )

        self.log("loss/val", outputs.loss, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, prog_bar=True)
        self.log("lr/val", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, prog_bar=True)

        # Compute mIoU using mask_labels and class_labels
        with torch.inference_mode():
            # Post-process predictions to semantic segmentation maps
            pred_maps = self.processor.post_process_semantic_segmentation(outputs)

            # Convert instance masks to semantic masks
            refs = []
            for masks, classes in zip(batch["mask_labels"], batch["class_labels"]):
                # Initialize semantic mask with background (0)
                semantic_mask = torch.zeros(masks.shape[-2:], dtype=torch.int64, device=masks.device)
                for idx, (mask, cls) in enumerate(zip(masks, classes)):
                    # Only assign class if mask has non-zero elements to avoid overwriting
                    valid_mask = mask > 0
                    semantic_mask[valid_mask] = cls.item()  # Assign class ID to corresponding mask pixels
                refs.append(semantic_mask)

            # Stack predictions and references
            preds = torch.stack([torch.as_tensor(p, dtype=torch.int64) for p in pred_maps])
            refs = torch.stack(refs)

            # Ensure shapes match
            if preds.shape[-2:] != refs.shape[-2:]:
                refs = torch.nn.functional.interpolate(
                    refs.unsqueeze(0).float(), size=preds.shape[-2:], mode="nearest"
                ).long().squeeze(0)

            metric_device = getattr(self.val_iou.confmat, "device", self.device)
            preds = preds.to(metric_device, non_blocking=True)
            refs = refs.to(metric_device, non_blocking=True)

            self.val_iou.update(preds, refs)

        return outputs.loss

    # compute & log validation mIoU once per epoch
    def on_validation_epoch_end(self) -> None:
        try:
            miou = self.val_iou.compute().item()
        except Exception:
            return
        self.log("miou/val", miou, sync_dist=True, prog_bar=True)
        self.val_iou.reset()
    # ----------------------------------------- #

    # ----------------- TEST ----------------- #
    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )

        original_images = batch["original_images"]
        target_sizes = []
        for img in original_images:
            arr = img if isinstance(img, torch.Tensor) else torch.tensor(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
                H, W = arr.shape[1], arr.shape[2]
            else:  # HWC
                H, W = arr.shape[0], arr.shape[1]
            target_sizes.append((H, W))

        pred_maps = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        preds = [p.detach().cpu().numpy() for p in pred_maps]

        refs = batch["original_segmentation_maps"]
        refs = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r) for r in refs]

        self.test_mean_iou.add_batch(predictions=preds, references=refs)
        return {"loss": outputs.loss.detach()}

    def on_test_epoch_end(self) -> None:
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        per_acc = np.array(metrics.pop("per_category_accuracy"))
        per_iou = np.array(metrics.pop("per_category_iou"))
        per_acc_log = np.nan_to_num(per_acc, nan=0.0)
        per_iou_log = np.nan_to_num(per_iou, nan=0.0)

        self.log("miou/test", float(metrics["mean_iou"]), sync_dist=True)
        self.log("mean_accuracy/test", float(metrics["mean_accuracy"]), sync_dist=True)

        for i in range(self.num_classes):
            name = self.id2label[i]
            self.log(f"accuracy/test_{name}", float(per_acc_log[i]), sync_dist=True)
            self.log(f"iou/test_{name}", float(per_iou_log[i]), sync_dist=True)
    # ---------------------------------------- #

    # -------------- OPTIMIZER --------------- #
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.LR_SCHEDULER.get('factor'),
                patience=config.LR_SCHEDULER.get('patience'),
            ),
            "reduce_on_plateau": True,
            "monitor": "loss/val",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
