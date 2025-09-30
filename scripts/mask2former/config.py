from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from typing import Dict, List, Tuple
import os


# Training Hyperparameters
LEARNING_RATE: float = 5e-5
EPOCHS: int = 180
PRECISION: str = "16-mixed"
DEVICES: List[int] = [0, 1]
IMG_SIZE: Tuple[int, int] = (640, 640)

NUM_WORKERS: int = 8
BATCH_SIZE: int = 4

# Checkpoint Callback Configuration
CHECKPOINT_CALLBACK: ModelCheckpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="valLoss",
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    filename="{epoch}-{valLoss:.4f}",
    save_weights_only=False,
)

# Logger Configuration
LOGGER: CSVLogger = CSVLogger(
    save_dir="outputs",
    name="lightning_logs_csv",
)

# Learning Rate Scheduler Configuration
LR_SCHEDULER: Dict[str, float] = {
    "factor": 0.5,
    "patience": 8,
}

# Dataset Configuration
DATASET_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "rs19")
)

ID2LABEL: Dict[int, str] = {
    0: "Background",
    1: "road",
    2: "sidewalk",
    3: "construction",
    4: "tram-track",
    5: "fence",
    6: "pole",
    7: "traffic-light",
    8: "traffic-sign",
    9: "vegetation",
    10: "terrain",
    11: "sky",
    12: "human",
    13: "rail-track",
    14: "car",
    15: "truck",
    16: "trackbed",
    17: "on-rails",
    18: "rail-raised",
    19: "rail-embedded",
}
