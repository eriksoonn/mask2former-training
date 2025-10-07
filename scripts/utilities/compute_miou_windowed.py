#!/usr/bin/env python3
"""
Compute mIoU on the validation split for a given Mask2Former Lightning checkpoint.

Examples
--------
# Single-pass (original behavior)
python tools/compute_miou.py \
  --ckpt /path/to/model.ckpt \
  --img-size 960 960 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda

# Windowed inference at original canvas 1920x1088, feeding 480x480
python tools/compute_miou.py \
  --ckpt /path/to/model.ckpt \
  --img-size 1920 1088 \
  --window-size 480 480 \
  --window-overlap 0.5 \
  --window-halo 64 \
  --batch-size 1 \
  --num-workers 4 \
  --device cuda
"""

from __future__ import annotations
import argparse
import os
import sys
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
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


# -------------------------- window utils & inference --------------------------
def _cosine_weight(h: int, w: int, device: torch.device) -> torch.Tensor:
    """2D cosine (Hann) window for smooth blending."""
    wy = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, h, device=device))
    wx = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, w, device=device))
    return torch.outer(wy, wx)  # (h,w)


@torch.inference_mode()
def predict_semantic_windowed(
    model,
    processor,
    img_chw: torch.Tensor,               # (C,H,W) float tensor on device
    num_classes: int,
    window_size: Optional[Tuple[int, int]] = None,   # (h, w) or None (no windowing)
    window_overlap: float = 0.25,
    halo: int = 64,
) -> torch.Tensor:
    """
    Returns predicted label map (H,W) int64.
    If window_size is None -> single full-frame pass.
    Otherwise -> sliding-window with halo context + cosine blending of hard labels.
    """
    C, H, W = img_chw.shape

    # No windowing: original behavior
    if window_size is None:
        outputs = model(pixel_values=img_chw.unsqueeze(0))
        pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])
        pred = torch.as_tensor(pred_maps[0], dtype=torch.int64, device=img_chw.device)
        return pred

    # Windowed path
    wh, ww = window_size
    wh = min(wh, H)
    ww = min(ww, W)
    assert 0.0 <= window_overlap < 1.0, "window_overlap must be in [0,1)"
    sy = max(int(wh * (1.0 - window_overlap)), 1)
    sx = max(int(ww * (1.0 - window_overlap)), 1)

    # Accumulators for weighted hard voting
    votes = torch.zeros((num_classes, H, W), dtype=torch.float32, device=img_chw.device)
    weights = torch.zeros((H, W), dtype=torch.float32, device=img_chw.device)

    # Precompute center-window weights
    w_center = _cosine_weight(wh, ww, img_chw.device)  # (wh, ww)

    # Tile starts (ensure full coverage of borders)
    y_starts = list(range(0, max(H - wh + 1, 1), sy))
    x_starts = list(range(0, max(W - ww + 1, 1), sx))
    if len(y_starts) == 0:
        y_starts = [0]
    if len(x_starts) == 0:
        x_starts = [0]
    if y_starts[-1] != H - wh:
        y_starts.append(max(H - wh, 0))
    if x_starts[-1] != W - ww:
        x_starts.append(max(W - ww, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            y1, x1 = y0 + wh, x0 + ww

            # Expand with halo (clamped to image)
            yh0 = max(y0 - halo, 0)
            xh0 = max(x0 - halo, 0)
            yh1 = min(y1 + halo, H)
            xh1 = min(x1 + halo, W)

            big = img_chw[:, yh0:yh1, xh0:xh1].unsqueeze(0)  # (1,C,wh+2h, ww+2h)

            # Predict on the halo-extended tile, then center-crop back to (wh,ww)
            outputs = model(pixel_values=big)
            pred_big = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(yh1 - yh0, xh1 - xh0)]
            )[0]
            pred_big = torch.as_tensor(pred_big, dtype=torch.int64, device=img_chw.device)

            # Coordinates of center region within the big tile
            cy0 = y0 - yh0
            cx0 = x0 - xh0
            pred_tile = pred_big[cy0:cy0+wh, cx0:cx0+ww]  # (wh, ww)

            # Weighted hard voting into the full canvas
            one_hot = F.one_hot(pred_tile, num_classes=num_classes).permute(2, 0, 1).float()  # (C,wh,ww)
            votes[:, y0:y1, x0:x1] += one_hot * w_center.unsqueeze(0)
            weights[y0:y1, x0:x1] += w_center

    # Normalize & argmax
    weights = weights.clamp_min(1e-6)
    probs = votes / weights.unsqueeze(0)  # (C,H,W)
    pred = probs.argmax(dim=0).to(torch.int64)
    return pred


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
    window_size: Optional[Tuple[int, int]],
    window_overlap: float,
    window_halo: int,
) -> None:
    # Processor + model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    model = Mask2FormerFinetuner.load_from_checkpoint(
        ckpt_path, id2label=ID2LABEL, lr=LEARNING_RATE, strict=False
    ).to(device)
    model.eval()

    # Data (use validation split) — img_size is the "inference canvas"
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

        H, W = img_size

        for b in range(pixel_values.shape[0]):
            img_chw = pixel_values[b]  # (C,H,W)

            pred_t = predict_semantic_windowed(
                model=model,
                processor=processor,
                img_chw=img_chw,
                num_classes=num_classes,
                window_size=window_size,          # None -> single-pass
                window_overlap=window_overlap,
                halo=window_halo,
            )  # (H,W) int64

            ref_t = instances_to_semantic(mask_labels[b], class_labels[b], (H, W))
            confmat += confusion_matrix_2d(
                pred_t, ref_t, num_classes=num_classes, ignore_index=ignore_index
            )

    per_class_iou, miou = iou_from_confmat(confmat)

    # Pretty print
    print("\n================  mIoU (validation)  ================\n")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Image size : {img_size[0]}x{img_size[1]}  (inference canvas)")
    if window_size is None:
        print(f"Windowing  : disabled (single pass)")
    else:
        print(f"Windowing  : {window_size[0]}x{window_size[1]} tiles, overlap={window_overlap:.2f}, halo={window_halo}")
    print(f"Device     : {device}")
    print(f"Classes    : {num_classes}\n")

    # Sort ID2LABEL by numeric key if keys are strings
    def _key_int(k):
        try:
            return int(k)
        except Exception:
            return k

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
                   help="Inference canvas size H W (e.g., 1920 1088).")
    p.add_argument("--window-size", nargs=2, type=int, metavar=("H", "W"), default=None,
                   help="Window/crop size fed to the network (e.g., 480 480). If omitted, windowing is disabled.")
    p.add_argument("--window-overlap", type=float, default=0.25,
                   help="Fractional overlap between windows (0.0-0.99).")
    p.add_argument("--window-halo", type=int, default=64,
                   help="Context halo (pixels) added around each window before center-cropping back.")
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

    window_size = None
    if args.window_size is not None:
        # argparse gives None or a list; interpret list -> tuple
        if isinstance(args.window_size, list):
            window_size = (args.window_size[0], args.window_size[1])
        else:
            window_size = args.window_size  # already tuple

    evaluate_miou(
        ckpt_path=args.ckpt,
        dataset_dir=args.dataset_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        ignore_index=args.ignore_index,
        window_size=window_size,
        window_overlap=float(args.window_overlap),
        window_halo=int(args.window_halo),
    )


if __name__ == "__main__":
    main()
