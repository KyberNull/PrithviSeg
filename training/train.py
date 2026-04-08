"""
Training loop for the GeoSpatial dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

from config.pretrain import NUM_EPOCHS_PRETRAIN
from config.train import (
	NUM_CLASSES_TRAIN,
	NUM_EPOCHS_TRAIN,
	NUM_VAL_SAMPLES_TRAIN,
	TRAIN_IMG_DIR,
	TRAIN_MASK_DIR,
	VAL_IMG_DIR,
	VAL_MASK_DIR,
)
from config.shared import (
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
import logging
from losses import dice_loss, focal_loss
from model import SegFormer
from .primitives import setup_scheduler, train_batch, validate
from .phase_io import get_train_dataloaders, load_checkpoint_train
import signal
import torch
from torch import optim
from processing import EvalTransforms, PostProcessing, TrainTransforms, GeospatialDataset
import utils
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging, handle_shutdown

###-------CONSTANTS-------###
NUM_CLASSES = NUM_CLASSES_TRAIN
NUM_VAL_SAMPLES = NUM_VAL_SAMPLES_TRAIN
###-----------------------###

pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def main(device, model_path):
	if GRAD_ACCUM_STEPS < 1:
		raise ValueError(f"GRAD_ACCUM_STEPS must be >= 1, got {GRAD_ACCUM_STEPS}")

	model = SegFormer(num_classes=NUM_CLASSES, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
	model = torch.compile(model)
	model = model.to(device=device, non_blocking=True)

	train_loader, validation_loader = get_train_dataloaders(
		geospatial_dataset_cls=GeospatialDataset,
		train_img_dir=TRAIN_IMG_DIR,
		train_mask_dir=TRAIN_MASK_DIR,
		val_img_dir=VAL_IMG_DIR,
		val_mask_dir=VAL_MASK_DIR,
		train_transform=TrainTransforms(),
		eval_transform=EvalTransforms(),
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
	)

	optimizer = optim.AdamW(get_adamw_param_groups(model, WEIGHT_DECAY), lr=LEARNING_RATE)
	scheduler = setup_scheduler(
		train_loader=train_loader,
		optimizer=optimizer,
		grad_accum_steps=GRAD_ACCUM_STEPS,
		total_epochs=NUM_EPOCHS_TRAIN,
		warmup_epochs=WARMUP_EPOCHS,
		learning_rate=LEARNING_RATE,
		warmup_start_factor=0.01,
		pretrain_epoch_offset=NUM_EPOCHS_PRETRAIN,
	)
	scaler = torch.GradScaler(enabled=(device.type == "cuda")) # GradScaler is only useful on CUDA where float16 gradients can underflow.

	start_epoch = load_checkpoint_train(
		path=model_path,
		model=model,
		start_epoch_default=NUM_EPOCHS_PRETRAIN,
		logger=logger,
		optimizer=optimizer,
		scheduler=scheduler,
		scaler=scaler,
	)

	criterion = focal_loss

	model.train()

	for epoch in range(start_epoch, NUM_EPOCHS_TRAIN):
		train_batch(
			model=model,
			epoch=epoch,
			total_epochs=NUM_EPOCHS_TRAIN,
			train_loader=train_loader,
			optimizer=optimizer,
			scheduler=scheduler,
			scaler=scaler,
			criterion=criterion,
			dice_loss_fn=dice_loss,
			num_classes=NUM_CLASSES,
			grad_accum_steps=GRAD_ACCUM_STEPS,
			phase_label="Train",
			model_path=MODEL_PATH,
			device=device,
			amp_dtype=amp_dtype,
			logger=logger,
			save_checkpoint_fn=save_checkpoint,
			should_stop=lambda: utils.shutdown_requested,
		)

		if (epoch + 1) % VAL_INTERVAL == 0:
			validate(
				model=model,
				validation_loader=validation_loader,
				device=device,
				criterion=criterion,
				num_classes=NUM_CLASSES,
				num_val_samples=NUM_VAL_SAMPLES,
				amp_dtype=amp_dtype,
				logger=logger,
				compute_processed=True,
				post_processor=PostProcessing(NUM_CLASSES),
			)

		save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)

	logger.info("Training complete. Checkpoint saved.")