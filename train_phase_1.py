"""
Pretraining loop for segmentation on the SBD dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

import logging
from losses import CBCE
from model import UNet
import os
import signal
import sys
import torch
from torch import optim, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transforms import TrainTransforms, EvalTransforms
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging, freeze_encoder

###-------CONSTANTS-------###
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 5
MODEL_PATH = "model.pt"
NUM_BATCHES = 4
NUM_CLASSES = 20
NUM_EPOCHS_PHASE_1 = 20
NUM_EPOCHS = NUM_EPOCHS_PHASE_1
NUM_WORKERS = min(4, os.cpu_count() or 1)
VAL_INTERVAL = 1
NUM_VAL_SAMPLES = 280
CHECKPOINT_RAM_HEADROOM_GB = 0.1
###-----------------------###

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def handle_shutdown(sig, frame):
	del frame
	global shutdown_requested
	logger.warning(f"Shutdown requested! Signal: {sig}")
	shutdown_requested = True

def train_batch(model, epoch, train_loader, optimizer, scheduler, scaler):
	epoch_bar = tqdm(
		train_loader,
		desc=f"Phase 1 Epoch {epoch + 1}/{NUM_EPOCHS}",
		leave=True,
		disable=not sys.stdout.isatty(),
		position=0,
	)
	running_loss = 0.0

	for batch, (input_tensor, output_tensor) in enumerate(epoch_bar):
		if shutdown_requested:
			save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)
			return

		input_tensor = input_tensor.to(device, non_blocking=True)
		output_tensor = output_tensor.squeeze(1).to(device, non_blocking=True).float()

		optimizer.zero_grad(set_to_none=True)

		with autocast(device_type=device.type, dtype=amp_dtype):
			prediction = model(input_tensor)
			loss = CBCE(prediction, output_tensor, NUM_CLASSES)

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

def validate(model, validation_loader, device):
	model.eval()
	running_val_loss = 0.0
	val_iterator = iter(validation_loader)

	with torch.no_grad():
		for _ in range(NUM_VAL_SAMPLES):
			try:
				val_input, val_output = next(val_iterator)
			except StopIteration:
				val_iterator = iter(validation_loader)
				val_input, val_output = next(val_iterator)

			val_input = val_input.to(device, non_blocking=True)
			val_output = val_output.squeeze(1).to(device, non_blocking=True).float()

			val_prediction = model(val_input)

			val_loss = CBCE(val_prediction, val_output, NUM_CLASSES)

			running_val_loss += val_loss.item()

		running_val_loss /= NUM_VAL_SAMPLES

		logger.info(f"mCBCE: {running_val_loss:.4f}")

	model.train()
	freeze_encoder(model)

def load_checkpoint(model_path, model):
	start_epoch = 0
	train_loader, validation_loader = get_dataloaders()
	optimizer = optim.AdamW(get_adamw_param_groups(model, learning_rate=LEARNING_RATE, backbone_lr=0, weight_decay=WEIGHT_DECAY), lr=LEARNING_RATE)
	scheduler = setup_scheduler(train_loader, optimizer)
	scaler = torch.GradScaler(enabled=(device.type == "cuda")) # GradScaler is only useful on CUDA where float16 gradients can underflow.
	try:
		ckpt = torch.load(model_path, map_location=device)
		model.load_state_dict(ckpt["model_state"])
		start_epoch = ckpt.get("epoch", -1) + 1
		scheduler = setup_scheduler(train_loader, optimizer)
		
		optimizer.load_state_dict(ckpt["optim_state"])
		scheduler.load_state_dict(ckpt["scheduler_state"])
		scaler.load_state_dict(ckpt["scaler_state"])
	except FileNotFoundError:
		logger.warning("Pretrain checkpoint not found. Starting from scratch.")
	except (RuntimeError, KeyError) as err:
		logger.error(f"Checkpoint incompatible with current model architecture: {err}")
		logger.warning("Starting from scratch.")

	logger.info(f"Resuming pretraining from epoch {start_epoch+1}")

	return model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader

def get_dataloaders():
	train_dataset = datasets.SBDataset(
		root='./data/phase-1',
		image_set='train_noval',
		mode='boundaries',
		download=True,
		transforms=TrainTransforms()
	)
	val_dataset = datasets.SBDataset(
		root='./data/phase-1',
		image_set='train',
		mode='boundaries',
		download=True,
		transforms=EvalTransforms()
	)
	train_dataloader = DataLoader(
		dataset=train_dataset,
		batch_size=NUM_BATCHES,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
		persistent_workers=NUM_WORKERS > 0
		)
	val_dataloader = DataLoader(
		dataset=val_dataset,
		batch_size=NUM_BATCHES,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
		persistent_workers=NUM_WORKERS > 0
		)
	return train_dataloader, val_dataloader

def setup_scheduler(train_loader, optimizer):
	warmup_steps = min(
		max(0, WARMUP_EPOCHS * len(train_loader)),
		max(0, NUM_EPOCHS * len(train_loader) - 1),
	)

	scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader) - warmup_steps, eta_min=LEARNING_RATE * 0.1)

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
	return scheduler

def main(device, model_path):
	model = UNet(num_classes=NUM_CLASSES).to(device=device, non_blocking=True)

	freeze_encoder(model)

	model = torch.compile(model)

	model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader = load_checkpoint(model_path, model)

	model.train()
	freeze_encoder(model)

	for epoch in range(start_epoch, NUM_EPOCHS):
		train_batch(model, epoch, train_loader, optimizer, scheduler, scaler)

		if (epoch + 1) % VAL_INTERVAL == 0:
			validate(model, validation_loader, device)

		save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)

	logger.info("Pretraining complete. Checkpoint saved.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main(device, MODEL_PATH)


