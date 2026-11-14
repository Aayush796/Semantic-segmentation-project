import torch


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

SEED = 42


# --------------------------------------------------
# Dataset
# --------------------------------------------------

DATA_ROOT = "./data"

IMAGE_SIZE = 256

NUM_CLASSES = 3

CLASS_NAMES = [
    "background",
    "pet",
    "boundary"
]


# --------------------------------------------------
# Training
# --------------------------------------------------

BATCH_SIZE = 1

NUM_EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.0


# --------------------------------------------------
# Validation
# --------------------------------------------------

VAL_RATIO = 0.20


# --------------------------------------------------
# Hardware
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Results
# --------------------------------------------------

RESULTS_DIR = "./results"

CHECKPOINT_DIR = "./results/checkpoints"

METRICS_DIR = "./results/metrics"

QUALITATIVE_DIR = "./results/qualitative"