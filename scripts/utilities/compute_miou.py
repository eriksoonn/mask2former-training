#!/usr/bin/env python3
"""
Compute mIoU on the validation split for a given Mask2Former Lightning checkpoint.

Examples
--------
python tools/compute_miou.py \
  --ckpt /path/to/model.ckpt \
  --img-size 512 512 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple

import torch
from transformers import AutoImageProcessor

# Make repo root importable if this file is under tools/ or scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Your package symbols
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


# ------------------------------- metrics --------------------------------------
@torch.inference_mode()
def confusion_matrix_2d(
    preds: torch.Tensor,    # (H, W), int64
    target: torch.Tensor,   # (H, W), int64
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Return (C x C) confusion matrix (rows = GT, cols = Pred)."""
    assert preds.shape == target.shape, "pred/target spatial shapes must match"
    if ignore_index is not None:
        keep = target != ignore_index
        preds = preds[keep]
        target = target[keep]
    k = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
    cm = torch.bincount(k, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def iou_from_confmat(confmat: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-class IoU and mean IoU from (C x C) confusion matrix."""
    tp = confmat.diag()
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    denom = tp + fp + fn + eps
    per_class_iou = tp / denom
    miou = per_class_iou.mean()
    return per_class_iou, miou


# --------------- instance -> semantic helper (keeps background=0) -------------
def instances_to_semantic(
    masks: torch.Tensor,      # (N_inst, H, W) binary
    classes: torch.Tensor,    # (N_inst,)
    out_hw: Tuple[int, int],
) -> torch.Tensor:
    H, W = out_hw
    device = masks.device
    sem = torch.zeros((H, W), dtype=torch.int64, device=device)
    for i in range(masks.shape[0]):
        m = masks[i] > 0
        if m.any():
            sem[m] = int(classes[i].item())
    return sem


# --------------------------------- core ---------------------------------------
@torch.inference_mode()
def evaluate_miou(
    ckpt_path: str,
    dataset_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: str,
    ignore_index: int | None,
) -> None:
    # Processor + model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    model = Mask2FormerFinetuner.load_from_checkpoint(
        ckpt_path, id2label=ID2LABEL, lr=LEARNING_RATE, strict=False
    ).to(device)
    model.eval()

    # Data (use validation split)
    dm = SegmentationDataModule(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        local_processor=False,
        keep_originals=False,
    )
    dm.setup(stage="fit")
    val_loader = dm.val_dataloader()

    num_classes = len(ID2LABEL)
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    # Iterate
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)          # (B,C,H,W)
        mask_labels  = [x.to(device, non_blocking=True) for x in batch["mask_labels"]]
        class_labels = [x.to(device, non_blocking=True) for x in batch["class_labels"]]

        # Target sizes for post-process == chosen img_size
        H, W = img_size
        target_sizes = [(H, W)] * pixel_values.shape[0]

        outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # Build refs and accumulate CM
        for pred, masks_i, classes_i in zip(pred_maps, mask_labels, class_labels):
            pred_t = torch.as_tensor(pred, dtype=torch.int64, device=device)  # (H,W)
            ref_t  = instances_to_semantic(masks_i, classes_i, (H, W))        # (H,W), int64
            confmat += confusion_matrix_2d(pred_t, ref_t, num_classes=num_classes, ignore_index=ignore_index)

    per_class_iou, miou = iou_from_confmat(confmat)

    # Pretty print
    print("\n================  mIoU (validation)  ================\n")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Image size : {img_size[0]}x{img_size[1]}")
    print(f"Device     : {device}")
    print(f"Classes    : {num_classes}\n")

    # Sort ID2LABEL by numeric key if keys are strings
    def _key_int(k): 
        try: return int(k)
        except: return k

    ordered = [ID2LABEL[k] for k in sorted(ID2LABEL.keys(), key=_key_int)]
    for cid, name in enumerate(ordered):
        print(f"Class {cid:>2} ({name:<20}): IoU = {per_class_iou[cid].item():.4f}")

    print("\n-----------------------------------------------------")
    print(f"Mean IoU: {miou.item():.4f}")
    print("=====================================================\n")


# --------------------------------- CLI ----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute mIoU on validation set for a given checkpoint.")
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True, type=str,
                   help="Path to Lightning .ckpt file.")
    p.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
                   help="Dataset root (default from package).")
    p.add_argument("--img-size", nargs=2, type=int, metavar=("H", "W"), required=True,
                   help="Evaluation image size H W (e.g., 512 512).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for validation.")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    p.add_argument("--device", choices=["cuda", "cpu"],
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device.")
    p.add_argument("--ignore-index", type=int, default=None,
                   help="Optional class id to ignore in IoU (e.g., unlabeled).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_size = (args.img_size[0], args.img_size[1])
    evaluate_miou(
        ckpt_path=args.ckpt,
        dataset_dir=args.dataset_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        ignore_index=args.ignore_index,
    )


if __name__ == "__main__":
    main()
