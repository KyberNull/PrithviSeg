"""
Training loop for the GeoSpatial dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

from datasets import geospatial_dataset
import logging
from losses import dice_loss, iou_metric, iou_metric_processed_fast, focal_loss
from model import SegFormer
import signal
import sys
import torch
from torch import optim, autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pretrain import NUM_EPOCHS_PRETRAIN
from transforms import TrainTransforms, EvalTransforms, PostProcessing
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging, handle_shutdown, shutdown_requested

###-------CONSTANTS-------###
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
MODEL_PATH = "model.pt"
BATCH_SIZE = 8
NUM_CLASSES = 4
NUM_EPOCHS_PHASE_3 = NUM_EPOCHS_PRETRAIN + 50
NUM_WORKERS = 2
VAL_INTERVAL = 1
NUM_VAL_SAMPLES = 280
###-----------------------###

pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def train_batch(model, epoch, train_loader, optimizer, scheduler, scaler, criterion):
	"""Trains one batch of images"""
	epoch_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}/{NUM_EPOCHS_PHASE_3}", leave=True, disable=not sys.stdout.isatty(), position=0 )
	running_loss = 0.0

	for batch, (input_tensor, output_tensor) in enumerate(epoch_bar):
		if shutdown_requested:
			save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)
			return

		input_tensor = input_tensor.to(device, non_blocking=True)
		output_tensor = output_tensor.squeeze(1).to(device, non_blocking=True).long()

		optimizer.zero_grad(set_to_none=True)

		with autocast(device_type=device.type, dtype=amp_dtype):
			backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
			with sdpa_kernel(backends=backends, set_priority=True):
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
	total_iou_processed = 0.0
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
				processed_mask = PostProcessing(NUM_CLASSES)(val_prediction)

			if not torch.isfinite(val_loss):
				continue

			running_val_loss += val_loss.item()

			iou = iou_metric(val_prediction, val_output, NUM_CLASSES)
			iou_processed = iou_metric_processed_fast(processed_mask, val_output, NUM_CLASSES)
			total_iou += float(iou)
			total_iou_processed += float(iou_processed)

		total_iou /= NUM_VAL_SAMPLES
		total_iou_processed /= NUM_VAL_SAMPLES
		running_val_loss /= NUM_VAL_SAMPLES

		logger.info(f"mCEL: {running_val_loss:.4f}")
		logger.info(f"mIoU: {total_iou:.4f}")
		logger.info(f"mIoU (Processed): {total_iou_processed:.4f}")


	model.train()

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
	start_epoch = NUM_EPOCHS_PRETRAIN  # Default to starting at phase 3 
	new_segmentation_head = False

	try:
		ckpt = torch.load(path, map_location="cpu")
		state_dict = ckpt["model_state"]

		# ---- Handle segmentation head mismatch (safe version) ----
		for k in list(state_dict.keys()):
			if "segmentation_head.0.weight" in k:
				# Check the shape in the checkpoint vs current model
				# model.state_dict() will have the '_orig_mod' prefix if compiled
				model_state = model.state_dict()
				
				if k in model_state:
					old_num_classes = state_dict[k].shape[0]
					new_num_classes = model_state[k].shape[0]

					if old_num_classes != new_num_classes:
						logger.warning(f"Shape mismatch at {k}: {old_num_classes} -> {new_num_classes}. Dropping head.")
						new_segmentation_head = True
						del state_dict[k]
						# Also remove the bias
						bias_key = k.replace("weight", "bias")
						if bias_key in state_dict:
							del state_dict[bias_key]
				break

		# ---- Load model weights ----
		missing, unexpected = model.load_state_dict(state_dict, strict=False)
		if missing:
			logger.warning(f"Missing keys: {missing}")
		if unexpected:
			logger.warning(f"Unexpected keys: {unexpected}")

		# ---- Resume training state if available ----
		has_train_state = all(
			k in ckpt for k in ("optim_state", "scheduler_state", "scaler_state")
		)

		if has_train_state and not new_segmentation_head and optimizer and scheduler and scaler:
			optimizer.load_state_dict(ckpt["optim_state"])
			scheduler.load_state_dict(ckpt["scheduler_state"])
			scaler.load_state_dict(ckpt["scaler_state"])
			start_epoch = ckpt.get("epoch", 0) + 1
			logger.info(f"Resuming training from epoch {start_epoch}")
		else:
			logger.info("Loaded model weights only (fresh optimizer/scheduler).")

	except FileNotFoundError:
		logger.warning("Checkpoint not found. Starting from scratch.")

	except (RuntimeError, KeyError) as err:
		logger.error(f"Incompatible checkpoint: {err}")
		logger.warning("Starting from scratch.")

	return start_epoch

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
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
		persistent_workers=NUM_WORKERS > 0,
		prefetch_factor=1,
	)
	val_dataloader = DataLoader(
		dataset=val_dataset,
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

	scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PHASE_3 * len(train_loader) - warmup_steps - NUM_EPOCHS_PRETRAIN, eta_min=LEARNING_RATE * 0.1)

	if warmup_steps > 0:
		warmup_scheduler = LinearLR(
			optimizer,
			start_factor=0.01,
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
	model = SegFormer(num_classes=NUM_CLASSES)
	model = torch.compile(model)
	model = model.to(device=device, non_blocking=True)

	train_loader, validation_loader = get_dataloaders()

	optimizer = optim.AdamW(get_adamw_param_groups(model, WEIGHT_DECAY), lr=LEARNING_RATE)
	scheduler = setup_scheduler(train_loader, optimizer)
	scaler = torch.GradScaler(enabled=(device.type == "cuda")) # GradScaler is only useful on CUDA where float16 gradients can underflow.

	start_epoch = load_checkpoint(model_path, model, optimizer, scheduler, scaler)

	criterion = focal_loss

	model.train()

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
