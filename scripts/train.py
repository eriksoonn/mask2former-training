import pytorch_lightning as pl
from mask2former import (
    Mask2FormerFinetuner,
    SegmentationDataModule,
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    ID2LABEL,
    LEARNING_RATE,
    LOGGER,
    DEVICES,
    CHECKPOINT_CALLBACK,
    EPOCHS,
)
import torch

def main():
    # Set random seed and precision for reproducibility
    torch.manual_seed(1)
    torch.set_float32_matmul_precision("medium")

    # Initialize data module
    data_module = SegmentationDataModule(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )

    # Initialize model
    model = Mask2FormerFinetuner(
        id2label=ID2LABEL,
        learning_rate=LEARNING_RATE,
    )

    # Set up trainer
    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator="cuda",
        devices=DEVICES,
        strategy="ddp",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS,
    )

    # Start training
    print("Training starts!!")
    trainer.fit(model, data_module)

    # Save final model checkpoint
    print("Saving model!")
    trainer.save_checkpoint("mask2former.ckpt")

if __name__ == "__main__":
    main()