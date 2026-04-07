"""Shared configuration package for training, evaluation, and inference."""

from .shared import (
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_WORKERS,
    USE_GRADIENT_CHECKPOINTING,
    VAL_INTERVAL,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)

__all__ = [
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "LEARNING_RATE",
    "MODEL_PATH",
    "NUM_WORKERS",
    "USE_GRADIENT_CHECKPOINTING",
    "VAL_INTERVAL",
    "WARMUP_EPOCHS",
    "WEIGHT_DECAY",
]
