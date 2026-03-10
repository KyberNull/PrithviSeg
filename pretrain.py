"""Pretraining loop for segmentation on the SBD dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

from config import get_train_config
import logging
from losses import dice_loss, compute_means
from model import UNet
from rich.logging import RichHandler
import gc
import os
import shutil
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transforms import TrainTransforms, EvalTransforms


config = get_train_config()
LEARNING_RATE = config.learning_rate
WEIGHT_DECAY = config.weight_decay
WARMUP_EPOCHS = config.warmup_epochs
PRETRAIN_MODEL_PATH = "model.pt"
NUM_BATCHES = config.num_batches
NUM_CLASSES = config.num_classes
NUM_EPOCHS = 20
NUM_WORKERS = config.num_workers
VAL_INTERVAL = config.val_interval
NUM_VAL_SAMPLES = config.num_val_samples
CHECKPOINT_INTERVAL = 1
CHECKPOINT_RAM_HEADROOM_GB = 0.1

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)


def freeze_encoder(model) -> None:
	"""Freeze encoder weights and keep BatchNorm stats fixed during pretraining."""
	encoder = getattr(model, "encoder", None)
	if encoder is None:
		orig_mod = getattr(model, "_orig_mod", None)
		encoder = getattr(orig_mod, "encoder", None)
	if encoder is None:
		return

	for p in encoder.parameters():
		p.requires_grad = False
	encoder.eval()


def _bytes_in_obj(obj) -> int:
	if torch.is_tensor(obj):
		return obj.numel() * obj.element_size()
	if isinstance(obj, dict):
		return sum(_bytes_in_obj(v) for v in obj.values())
	if isinstance(obj, (list, tuple)):
		return sum(_bytes_in_obj(v) for v in obj)
	return 0


def _available_ram_bytes() -> int:
	"""Return MemAvailable from /proc/meminfo (Linux)."""
	try:
		with open("/proc/meminfo", "r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("MemAvailable:"):
					parts = line.split()
					# kB -> bytes
					return int(parts[1]) * 1024
	except (FileNotFoundError, ValueError, OSError):
		return -1
	return -1


def _estimate_checkpoint_bytes(model, optimizer) -> int:
	# Model tensors are cheap to estimate from params + buffers directly.
	model_bytes = 0
	for param in model.parameters():
		model_bytes += param.numel() * param.element_size()
	for buf in model.buffers():
		model_bytes += buf.numel() * buf.element_size()

	# Optimizer state can be large; count tensor payloads recursively.
	optim_state = optimizer.state_dict()
	optim_bytes = _bytes_in_obj(optim_state.get("state", {}))

	# Serialization buffers/metadata can add overhead; keep a conservative cushion.
	return int((model_bytes + optim_bytes) * 1.25)


def build_sbd_dataset(image_set: str, transforms):
	"""Load SBD from local files first; only download if missing."""
	try:
		return datasets.SBDataset(
			"./data",
			image_set=image_set,
			mode="segmentation",
			transforms=transforms,
			download=False,
		)
	except RuntimeError:
		logger.warning(
			"SBD dataset not found locally for split '%s'. Attempting download.",
			image_set,
		)

	try:
		return datasets.SBDataset(
			"./data",
			image_set=image_set,
			mode="segmentation",
			transforms=transforms,
			download=True,
		)
	except shutil.Error as err:
		logger.warning(
			"SBD download encountered existing files (%s). Reusing local copy.",
			err,
		)
		return datasets.SBDataset(
			"./data",
			image_set=image_set,
			mode="segmentation",
			transforms=transforms,
			download=False,
		)


def get_adamw_param_groups(model: nn.Module):
	decay_params = []
	no_decay_params = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if param.ndim <= 1 or name.endswith(".bias"):
			no_decay_params.append(param)
		else:
			decay_params.append(param)

	return [
		{"params": decay_params, "weight_decay": WEIGHT_DECAY},
		{"params": no_decay_params, "weight_decay": 0.0},
	]


def handle_shutdown(sig, frame):
	del frame
	global shutdown_requested
	logger.warning(f"Shutdown requested! Signal: {sig}")
	shutdown_requested = True


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path, force: bool = False) -> bool:
	headroom_bytes = int(CHECKPOINT_RAM_HEADROOM_GB * 1024 * 1024 * 1024)
	estimated_bytes = _estimate_checkpoint_bytes(model, optimizer)
	available_bytes = _available_ram_bytes()

	if available_bytes > 0 and available_bytes < estimated_bytes + headroom_bytes:
		msg = (
			"Skipping checkpoint at epoch %d: available RAM %.2f GB < required %.2f GB "
			"(estimated checkpoint %.2f GB + headroom %.2f GB)."
		)
		if not force:
			logger.warning(
				msg,
				epoch,
				available_bytes / (1024 ** 3),
				(estimated_bytes + headroom_bytes) / (1024 ** 3),
				estimated_bytes / (1024 ** 3),
				headroom_bytes / (1024 ** 3),
			)
			return False
		logger.warning(
			"Low RAM during forced checkpoint at epoch %d; attempting save anyway.",
			epoch,
		)

	state = {
		"epoch": epoch,
		"checkpoint_mode": "full",
		"model_state": model.state_dict(),
		"optim_state": optimizer.state_dict(),
		"scheduler_state": scheduler.state_dict(),
		"scaler_state": scaler.state_dict(),
	}

	tmp_path = f"{path}.tmp"
	gc.collect()
	torch.save(state, tmp_path)
	os.replace(tmp_path, path)
	return True


def validate(model, validation_loader, device, criterion):
	model.eval()
	running_val_loss = 0.0
	total_iou = 0.0
	val_iterator = iter(validation_loader)

	with torch.no_grad():
		for _ in range(NUM_VAL_SAMPLES):
			try:
				val_input, val_output = next(val_iterator)
			except StopIteration:
				val_iterator = iter(validation_loader)
				val_input, val_output = next(val_iterator)

			val_input = val_input.to(device, non_blocking=True)
			val_output = val_output.squeeze(1).to(device, non_blocking=True).long()

			with autocast(device_type=device.type, dtype=amp_dtype):
				val_prediction = model(val_input)
				val_loss = criterion(val_prediction, val_output)

			running_val_loss += val_loss.item()

			_, iou = compute_means(val_prediction, val_output, NUM_CLASSES)
			total_iou += iou.item()

		total_iou /= NUM_VAL_SAMPLES
		running_val_loss /= NUM_VAL_SAMPLES

		logger.info(f"mCEL: {running_val_loss:.4f}")
		logger.info(f"mIoU: {total_iou:.4f}")

	model.train()
	freeze_encoder(model)


def main(device, model_path):
	logger.info(
		"Using full-state checkpoints every %d epoch(s) with %.2f GB RAM headroom requirement.",
		CHECKPOINT_INTERVAL,
		CHECKPOINT_RAM_HEADROOM_GB,
	)

	# GradScaler is only useful on CUDA where float16 gradients can underflow.
	scaler = torch.GradScaler(enabled=(device.type == "cuda"))

	train_dataset = build_sbd_dataset("train_noval", TrainTransforms())
	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=NUM_BATCHES,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
		persistent_workers=NUM_WORKERS > 0,
	)

	validation_dataset = build_sbd_dataset("val", EvalTransforms())
	validation_loader = DataLoader(
		dataset=validation_dataset,
		batch_size=NUM_BATCHES,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
	)

	model = UNet(num_classes=NUM_CLASSES).to(device=device, non_blocking=True)
	freeze_encoder(model)
	optimizer = optim.AdamW(get_adamw_param_groups(model), lr=LEARNING_RATE)
	model = torch.compile(model)

	warmup_steps = min(
		max(0, WARMUP_EPOCHS * len(train_loader)),
		max(0, NUM_EPOCHS * len(train_loader) - 1),
	)

	scheduler = CosineAnnealingLR(
		optimizer,
		T_max=NUM_EPOCHS * len(train_loader) - warmup_steps,
		eta_min=LEARNING_RATE * 0.1,
	)
	if warmup_steps > 0:
		warmup_scheduler = LinearLR(
			optimizer,
			start_factor=0.1,
			end_factor=1.0,
			total_iters=warmup_steps,
		)
		scheduler = SequentialLR(
			optimizer,
			schedulers=[warmup_scheduler, scheduler],
			milestones=[warmup_steps],
		)

	criterion = nn.CrossEntropyLoss(ignore_index=255)

	start_epoch = 0
	try:
		ckpt = torch.load(model_path, map_location=device)
		model.load_state_dict(ckpt["model_state"])

		if all(k in ckpt for k in ("optim_state", "scheduler_state", "scaler_state")):
			optimizer.load_state_dict(ckpt["optim_state"])
			scheduler.load_state_dict(ckpt["scheduler_state"])
			scaler.load_state_dict(ckpt["scaler_state"])
		else:
			logger.info(
				"Loaded weights-only checkpoint. Optimizer/scheduler/scaler reinitialized."
			)

		start_epoch = ckpt.get("epoch", -1) + 1
		logger.info(f"Resuming pretraining from epoch {start_epoch}")
	except FileNotFoundError:
		logger.warning("Pretrain checkpoint not found. Starting from scratch.")
	except (RuntimeError, KeyError) as err:
		logger.error(f"Checkpoint incompatible with current model architecture: {err}")
		logger.warning("Starting pretraining from scratch.")

	model.train()
	freeze_encoder(model)
	for epoch in range(start_epoch, NUM_EPOCHS):
		epoch_bar = tqdm(
			train_loader,
			desc=f"Pretrain Epoch {epoch + 1}/{NUM_EPOCHS}",
			leave=True,
			disable=not sys.stdout.isatty(),
			position=0,
		)
		running_loss = 0.0

		for batch, (input_tensor, output_tensor) in enumerate(epoch_bar):
			if shutdown_requested:
				saved = save_checkpoint(
					model,
					optimizer,
					scheduler,
					scaler,
					epoch,
					PRETRAIN_MODEL_PATH,
					force=True,
				)
				if saved:
					logger.info("Checkpoint saved. Gracefully exiting...")
				else:
					logger.warning("Checkpoint skipped due to RAM budget. Exiting gracefully.")
				return

			input_tensor = input_tensor.to(device, non_blocking=True)
			output_tensor = output_tensor.squeeze(1).to(device, non_blocking=True).long()

			optimizer.zero_grad(set_to_none=True)

			with autocast(device_type=device.type, dtype=amp_dtype):
				prediction = model(input_tensor)
				loss = criterion(prediction, output_tensor)
				loss += dice_loss(prediction, output_tensor, NUM_CLASSES)

			scaler.scale(loss).backward()
			scale_before_step = scaler.get_scale()
			scaler.step(optimizer)
			scaler.update()
			scale_after_step = scaler.get_scale()

			if scale_after_step >= scale_before_step:
				scheduler.step()

			running_loss += loss.item()
			avg_loss = running_loss / (batch + 1)
			epoch_bar.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr()[0])

		if (epoch + 1) % VAL_INTERVAL == 0:
			validate(model, validation_loader, device, criterion)

		should_save = (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS
		if should_save:
			saved = save_checkpoint(
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				scaler=scaler,
				epoch=epoch,
				path=PRETRAIN_MODEL_PATH,
			)
			if saved:
				logger.info("Checkpoint stored at epoch %d", epoch)

	logger.info("Pretraining complete. Checkpoint saved.")


if __name__ == "__main__":
	signal.signal(signal.SIGINT, handle_shutdown)
	signal.signal(signal.SIGTERM, handle_shutdown)

	device = torch.device("cpu")

	if torch.cuda.is_available():
		device = torch.device("cuda")
		pin_memory = True
		amp_dtype = torch.float16
		torch.backends.cudnn.benchmark = True
	elif torch.mps.is_available():
		device = torch.device("mps")
		amp_dtype = torch.float16

	logging.basicConfig(
		level=logging.INFO,
		format="%(message)s",
		handlers=[RichHandler()],
		force=True,
	)

	main(device, PRETRAIN_MODEL_PATH)
