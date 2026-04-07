"""Phase 3 (downstream training) specific configuration."""

from .pretrain import NUM_EPOCHS_PRETRAIN

NUM_CLASSES_PHASE_3 = 4
NUM_EPOCHS_PHASE_3 = NUM_EPOCHS_PRETRAIN + 50
NUM_VAL_SAMPLES_PHASE_3 = 280
PHASE3_TRAIN_IMG_DIR = "data/phase-3/TrainningDataset/processed_datasets"
PHASE3_TRAIN_MASK_DIR = "data/phase-3/TrainningDataset/processed_masks"
PHASE3_VAL_IMG_DIR = "data/phase-3/ValidationDataset/processed_datasets"
PHASE3_VAL_MASK_DIR = "data/phase-3/ValidationDataset/processed_masks"
