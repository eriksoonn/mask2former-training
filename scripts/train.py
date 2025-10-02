from transformers.utils import logging as hf_logging
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
    COMPILE,
    PRETRAINED,
    PRECISION,
    ACCUMULATE_GRAD_BATCHES,
    TENSORBOARD_LOGGER,
)
import warnings
import torch

def main():
    # Suppress specific Hugging Face warnings
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=r".*not valid for `Mask2FormerImageProcessor.__init__`.*")
    warnings.filterwarnings("ignore", message=r".*Using a slow image processor as `use_fast` is unset.*")
    warnings.filterwarnings("ignore", message=r".*Precision 16-mixed is not supported by the model summary.*")
    warnings.filterwarnings("ignore", message=r".*You have overridden `transfer_batch_to_device`.*")

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
        lr=LEARNING_RATE,
        use_pretrained=PRETRAINED,
        compile=COMPILE
    )

    # Set up trainer
    trainer = pl.Trainer(
        logger=[LOGGER, TENSORBOARD_LOGGER],
        accelerator="cuda",
        devices=DEVICES,
        strategy="ddp",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS,
        precision=PRECISION,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=10
    )

    # Start training
    print("Training starts!!")
    trainer.fit(model, data_module)

    # Save final model checkpoint
    print("Saving model!")
    trainer.save_checkpoint("mask2former.ckpt")

if __name__ == "__main__":
    main()