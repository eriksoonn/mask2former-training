#!/usr/bin/env python3
"""
Compute mIoU on the validation set for a given checkpoint.

Usage:
  python compute_miou.py --ckpt /path/to/model.ckpt

Optional:
  --device cuda|cpu
  --batch-size  (defaults to your module's BATCH_SIZE)
  --num-workers (defaults to your module's NUM_WORKERS)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor

# ---------------------------------------------------------------------
# Make repo root importable if this script lives under /tools or /scripts
# Adjust relative path if needed.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from mask2former import (  # noqa: E402
    Mask2FormerFinetuner,
    SegmentationDataModule,
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    ID2LABEL,
    LEARNING_RATE,
)

torch.set_float32_matmul_precision("medium")


# ------------------------------- Metrics --------------------------------------
@torch.inference_mode()
def compute_confusion_matrix(
    preds: torch.Tensor,  # (H, W) int64 in [0, C-1]
    target: torch.Tensor, # (H, W) int64 in [0, C-1]
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Returns confusion matrix of shape (C, C): rows=true class, cols=pred class.
    """
    assert preds.shape == target.shape, "Prediction/target shapes must match"

    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]

    # Flatten
    target = target.view(-1)
    preds = preds.view(-1)

    k = (target * num_classes + preds).to(torch.long)
    cm = torch.bincount(k, minlength=num_classes * num_classes)
    cm = cm.view(num_classes, num_classes)
    return cm


def miou_from_confmat(confmat: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given confusion matrix (C x C), return:
      per_class_iou: (C,)
      mean_iou: scalar
    IoU_c = TP / (TP + FP + FN)
    """
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    denom = tp + fp + fn + eps
    per_class_iou = tp / denom
    mean_iou = per_class_iou.mean()
    return per_class_iou, mean_iou


# ------------------------------- Helpers --------------------------------------
def build_semantic_from_instances(masks: torch.Tensor, classes: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Convert instance masks + class ids to a single semantic map.
      masks:   (N_inst, H, W)  binary/0-1 mask per instance
      classes: (N_inst,)       class id per instance (int64)
      out_hw:  (H, W)          desired output size (should match masks size)
    Returns:
      semantic_map: (H, W) int64
    """
    H, W = out_hw
    device = masks.device
    semantic = torch.zeros((H, W), dtype=torch.int64, device=device)  # background=0
    n_inst = masks.shape[0]

    for i in range(n_inst):
        m = masks[i] > 0
        if m.any():
            cls_id = int(classes[i].item())
            semantic[m] = cls_id
    return semantic


# ------------------------------ Main routine ----------------------------------
@torch.inference_mode()
def run_validation_miou(
    ckpt_path: str,
    dataset_dir: str,
    batch_size: int,
    num_workers: int,
    device: str,
    ignore_index: int | None = None,
) -> None:
    # Load processor & model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    model = Mask2FormerFinetuner.load_from_checkpoint(
        ckpt_path,
        id2label=ID2LABEL,
        lr=LEARNING_RATE,
        strict=False,
    )
    model.eval().to(device)

    # Data
    dm = SegmentationDataModule(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=getattr(dm, "img_size", None) if "dm" in locals() else (512, 512),  # fallback if needed
    )
    dm.setup(stage="fit")  # ensures val set is built
    val_loader = dm.val_dataloader()

    num_classes = len(ID2LABEL)

    # Accumulate confusion matrix on device (GPU for speed if available)
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for batch in val_loader:
        # Move inputs
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)  # (B, C, H, W)
        # Instance labels are lists; keep them on device when needed
        mask_labels: List[torch.Tensor] = [x.to(device, non_blocking=True) for x in batch["mask_labels"]]
        class_labels: List[torch.Tensor] = [x.to(device, non_blocking=True) for x in batch["class_labels"]]

        # Choose a common target size for post-process == network input size
        H, W = pixel_values.shape[-2:]
        target_sizes = [(H, W)] * pixel_values.shape[0]

        # Forward
        outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)

        # Semantic preds (list of HxW tensors with class ids)
        pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # Build semantic refs from instance masks/classes
        refs: List[torch.Tensor] = []
        for masks_i, classes_i in zip(mask_labels, class_labels):
            refs.append(build_semantic_from_instances(masks_i, classes_i, (H, W)))

        # Accumulate confusion matrix
        for pred, ref in zip(pred_maps, refs):
            pred_t = torch.as_tensor(pred, device=device, dtype=torch.int64)
            cm = compute_confusion_matrix(pred_t, ref, num_classes=num_classes, ignore_index=ignore_index)
            confmat += cm

    per_class_iou, mean_iou = miou_from_confmat(confmat)

    # Print results
    print("\n================  mIoU (validation)  ================\n")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Device     : {device}")
    print(f"Num classes: {num_classes}\n")

    # Per-class table
    id2label_sorted = [ID2LABEL[k] for k in sorted(ID2LABEL.keys(), key=lambda x: int(x))]
    for c, name in enumerate(id2label_sorted):
        print(f"Class {c:>2} ({name:<20}): IoU = {per_class_iou[c].item():.4f}")
    print("\n-----------------------------------------------------")
    print(f"Mean IoU: {mean_iou.item():.4f}")
    print("=====================================================\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute mIoU on validation set for a given Mask2Former checkpoint.")
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True,
                   help="Path to the .ckpt file produced by Lightning.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"], help="Device to run evaluation on.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Validation batch size.")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    p.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Dataset root directory.")
    p.add_argument("--ignore-index", type=int, default=None,
                   help="Optional class index to ignore in IoU (e.g., unlabeled).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_validation_miou(
        ckpt_path=args.ckpt,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        ignore_index=args.ignore_index,
    )


if __name__ == "__main__":
    main()
