"""Large-image inference configuration."""

PATCH_SIZE = 1024
STRIDE = PATCH_SIZE
NUM_CLASSES_INFERENCE = 4
USE_TORCH_COMPILE = True
TEMP_DATASET_DIR = "processed_datasets"
TEMP_MASK_DIR = "processed_masks"
