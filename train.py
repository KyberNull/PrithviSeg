"""
Training loop for the GeoSpatial dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

from datasets import geospatial_dataset
import logging
from losses import dice_loss, iou
from model import UNet
import os
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_phase_1 import NUM_EPOCHS_PHASE_1
from train_phase_2 import NUM_EPOCHS_PHASE_2
from transforms import TrainTransforms, EvalTransforms
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging, freeze_encoder

###-------CONSTANTS-------###
LEARNING_RATE = 1e-4
BACKBONE_FACTOR = 10
BACKBONE_LEARNING_RATE = LEARNING_RATE / BACKBONE_FACTOR
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 5
MODEL_PATH = "model.pt"
NUM_BATCHES = 16
NUM_CLASSES = 4
NUM_EPOCHS_PHASE_3 = 100
NUM_WORKERS = min(4, os.cpu_count() or 1)
VAL_INTERVAL = 1
NUM_VAL_SAMPLES = 280
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

def train_batch(model, epoch, train_loader, optimizer, scheduler, scaler, criterion):
	epoch_bar = tqdm(
		train_loader,
		desc=f"[Train] Epoch {epoch + 1}/{NUM_EPOCHS_PHASE_3}",
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
		output_tensor = output_tensor.squeeze(1).to(device, non_blocking=True).long()

		optimizer.zero_grad(set_to_none=True)

		with autocast(device_type=device.type, dtype=amp_dtype):
			prediction = model(input_tensor)
			loss = criterion(prediction, output_tensor)
			loss += dice_loss(prediction, output_tensor, NUM_CLASSES)

		if not torch.isfinite(loss):
			logger.warning(f"Non-finite loss at epoch {epoch+1}, batch {batch}; skipping step.")
			optimizer.zero_grad(set_to_none=True)
			continue

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

def validate(model, validation_loader, device, criterion, epoch):
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

			if not torch.isfinite(val_loss):
				continue

			running_val_loss += val_loss.item()

			_, iou = iou(val_prediction, val_output, NUM_CLASSES)
			total_iou += iou.item()

		total_iou /= NUM_VAL_SAMPLES
		running_val_loss /= NUM_VAL_SAMPLES

		logger.info(f"mCEL: {running_val_loss:.4f}")
		logger.info(f"mIoU: {total_iou:.4f}")

	model.train()
	#freeze_encoder(model)

def load_checkpoint(model_path, model):
	start_epoch = NUM_EPOCHS_PHASE_1 + NUM_EPOCHS_PHASE_2
	train_loader, validation_loader = get_dataloaders()
	optimizer = optim.AdamW(get_adamw_param_groups(model, LEARNING_RATE, BACKBONE_LEARNING_RATE, WEIGHT_DECAY), lr=LEARNING_RATE)
	scheduler = setup_scheduler(train_loader, optimizer)
	scaler = torch.GradScaler(enabled=(device.type == "cuda")) # GradScaler is only useful on CUDA where float16 gradients can underflow.
	try:
		ckpt = torch.load(model_path, map_location=device)
		state_dict = ckpt["model_state"]
		ckpt_epoch = ckpt.get("epoch", -1) + 1
			
		# Remove segmentation head params from checkpoint when number of classes differ.
		keys_to_remove = []
		for k, v in list(state_dict.items()):
			if k.endswith('head.weight'):
				if v.shape[0] != NUM_CLASSES:
					keys_to_remove.append(k)
					# Replacing 'weight' with 'bias' to remove the corresponding biases.
					bias_k = k[:-len('weight')] + 'bias'
					if bias_k in state_dict:
						keys_to_remove.append(bias_k)

		if keys_to_remove:
			# report original classes if possible
			try:
				# Finding the first item in the list which matches the needs.
				sample_key = next(key for key in keys_to_remove if key.endswith('head.weight'))
				orig_classes = state_dict[sample_key].shape[0]
			except StopIteration:
				orig_classes = 'unknown'

			logger.warning(f"Pre-trained head has {orig_classes} classes, but current task has {NUM_CLASSES}. Excluding head parameters: {keys_to_remove}")
			for k in keys_to_remove:
				state_dict.pop(k, None)
		
		model.load_state_dict(state_dict, strict=False)

		phase_3_start = NUM_EPOCHS_PHASE_1 + NUM_EPOCHS_PHASE_2
		is_phase_3_resume = phase_3_start < ckpt_epoch < NUM_EPOCHS_PHASE_3
		has_train_state = all(k in ckpt for k in ("optim_state", "scheduler_state", "scaler_state"))

		if keys_to_remove:
			logger.warning("Head class mismatch detected earlier — resetting optimizer/scheduler/scaler for clean phase transition.")
			start_epoch = phase_3_start
		elif is_phase_3_resume and has_train_state:
			optimizer.load_state_dict(ckpt["optim_state"])
			scheduler.load_state_dict(ckpt["scheduler_state"])
			scaler.load_state_dict(ckpt["scaler_state"])
			start_epoch = ckpt_epoch
			logger.info("Resuming phase-3 optimizer/scheduler state.")
		else:
			start_epoch = phase_3_start
			if ckpt_epoch >= NUM_EPOCHS_PHASE_3:
				logger.info("Phase-3 checkpoint already complete; starting from phase-3 boundary with fresh optimizer/scheduler.")
			else:
				logger.info("Using checkpoint model weights with fresh optimizer/scheduler/scaler.")
	except FileNotFoundError:
		logger.warning("Pretrain checkpoint not found. Starting from scratch.")
	except (RuntimeError, KeyError) as err:
		logger.error(f"Checkpoint incompatible with current model architecture: {err}")
		logger.warning("Starting from scratch.")

	logger.info(f"Resuming pretraining from epoch {start_epoch+1}")

	return model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader

def get_dataloaders():
	#Importing the trainning and validation villages
	train_img_dir = "data/phase-3/TrainningDataset/processed_datasets"
	train_mask_dir = "data/phase-3/TrainningDataset/processed_masks"
	val_img_dir = "data/phase-3/ValidationDataset/processed_datasets"
	val_mask_dir = "data/phase-3/ValidationDataset/processed_masks"

	train_dataset = geospatial_dataset(
		img_dir=train_img_dir,
		img_mask=train_mask_dir,
		transform=TrainTransforms()
		)
	val_dataset = geospatial_dataset(
		img_dir=val_img_dir,
		img_mask=val_mask_dir,
		transform=EvalTransforms()
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
		pin_memory=pin_memory
		)
	return train_dataloader, val_dataloader

def setup_scheduler(train_loader, optimizer):
	warmup_steps = min(
		max(0, WARMUP_EPOCHS * len(train_loader)),
		max(0, NUM_EPOCHS_PHASE_3 * len(train_loader) - 1),
	)

	scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PHASE_3 * len(train_loader) - warmup_steps, eta_min=LEARNING_RATE * 0.1)

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

	#freeze_encoder(model)

	model = torch.compile(model)

	model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader = load_checkpoint(model_path, model)

	criterion = nn.CrossEntropyLoss()

	model.train()
	#freeze_encoder(model)

	for epoch in range(start_epoch, NUM_EPOCHS_PHASE_3):
		train_batch(model, epoch, train_loader, optimizer, scheduler, scaler, criterion)

		if (epoch + 1) % VAL_INTERVAL == 0:
			validate(model, validation_loader, device, criterion, epoch)

		save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)

	logger.info("Pretraining complete. Checkpoint saved.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main(device, MODEL_PATH)
