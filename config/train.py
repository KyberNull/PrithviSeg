"""Train-specific configuration."""

from .pretrain import NUM_EPOCHS_PRETRAIN

NUM_CLASSES_TRAIN = 4
NUM_EPOCHS_TRAIN = NUM_EPOCHS_PRETRAIN + 50
NUM_VAL_SAMPLES_TRAIN = 280
TRAIN_IMG_DIR = "data/train/training_dataset/processed_datasets"
TRAIN_MASK_DIR = "data/train/training_dataset/processed_masks"
VAL_IMG_DIR = "data/train/validation_dataset/processed_datasets"
VAL_MASK_DIR = "data/train/validation_dataset/processed_masks"