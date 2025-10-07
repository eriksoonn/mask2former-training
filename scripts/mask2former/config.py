from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Dict, List, Tuple
import os


# Training Hyperparameters
#LEARNING_RATE: float = 5e-5
LEARNING_RATE: float = 1e-5
EPOCHS: int = 180
PRECISION: str = "bf16-mixed"
DEVICES: List[int] = [0]
IMG_SIZE: Tuple[int, int] = (1080, 1920)    # (H, W)

PRETRAINED:bool = True
LOCAL_PROCESSOR: bool = False

COMPILE: bool = False
NUM_WORKERS: int = 8
BATCH_SIZE: int = 6
ACCUMULATE_GRAD_BATCHES: int = 4

# Learning Rate Scheduler Configuration
LR_SCHEDULER: Dict[str, float] = {
    "factor": 0.5,
    "patience": 8,
}

# Checkpoint Callback Configuration
CHECKPOINT_CALLBACK: ModelCheckpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="loss/val",
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    filename="{epoch}-{loss/val:.4f}",
    save_weights_only=False,
)

# Logger Configuration
LOGGER: CSVLogger = CSVLogger(
    save_dir="outputs",
    name="lightning_logs_csv",
)

TENSORBOARD_LOGGER = TensorBoardLogger(
    save_dir="outputs",
    name="lightning_logs_tb",
)

# Dataset Configuration
# DATASET_DIR: str = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "data", "rs19")
# )

DATASET_DIR: str = '/mnt/nvme1/datasets/rs19'


ID2LABEL: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "construction",
    3: "tram-track",
    4: "fence",
    5: "pole",
    6: "traffic-light",
    7: "traffic-sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "human",
    12: "rail-track",
    13: "car",
    14: "truck",
    15: "trackbed",
    16: "on-rails",
    17: "rail-raised",
    18: "rail-embedded",
}
