"""Environment-based configuration helpers."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def _load_environment() -> None:
    root_dir = Path(__file__).resolve().parent
    env_example_path = root_dir / ".env.example"
    env_path = root_dir / ".env"

    if env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path)

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

_load_environment()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    restart_cycle_epochs: int
    restart_cycle_mult: int
    model_path: str
    batch_size: int
    num_classes: int
    num_epochs: int
    num_workers: int
    val_interval: int
    num_val_samples: int

    @property
    def num_batches(self) -> int:
        return self.batch_size


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    num_workers: int
    batch_size: int
    num_classes: int
    max_examples: int
    ignore_label: int

    @property
    def num_batches(self) -> int:
        return self.batch_size


def get_train_config() -> TrainConfig:
    return TrainConfig(
        learning_rate=_get_float("TRAIN_LEARNING_RATE", 0.003),
        weight_decay=_get_float("TRAIN_WEIGHT_DECAY", 1e-4),
        warmup_epochs=_get_int("TRAIN_WARMUP_EPOCHS", 5),
        restart_cycle_epochs=_get_int("TRAIN_RESTART_CYCLE_EPOCHS", 20),
        restart_cycle_mult=_get_int("TRAIN_RESTART_CYCLE_MULT", 2),
        model_path=_get_str("MODEL_PATH", "model.pt"),
        batch_size=_get_int("TRAIN_BATCH_SIZE", 32),
        num_classes=_get_int("NUM_CLASSES", 21),
        num_epochs=_get_int("TRAIN_EPOCHS", 300),
        num_workers=_get_int("NUM_WORKERS", min(4, os.cpu_count() or 1)),
        val_interval=_get_int("VAL_INTERVAL", 5),
        num_val_samples=_get_int("NUM_VAL_SAMPLES", 140),
    )


def get_eval_config() -> EvalConfig:
    return EvalConfig(
        model_path=_get_str("MODEL_PATH", "model.pt"),
        num_workers=_get_int("NUM_WORKERS", min(4, os.cpu_count() or 1)),
        batch_size=_get_int("EVAL_BATCH_SIZE", 16),
        num_classes=_get_int("NUM_CLASSES", 21),
        max_examples=_get_int("EVAL_MAX_EXAMPLES", 10),
        ignore_label=_get_int("IGNORE_LABEL", 255),
    )