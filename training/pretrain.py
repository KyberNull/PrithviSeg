"""
Pretraining loop for segmentation on the GeoSpatial dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

from config.pretrain import (
	NUM_CLASSES_PRETRAIN,
	NUM_EPOCHS_PRETRAIN,
	NUM_VAL_SAMPLES_PRETRAIN,
	PRETRAIN_DATA_ROOT,
	PRETRAIN_SCENES,
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
from losses import dice_loss, dou_loss, focal_loss
from model import SegFormer
from .primitives import setup_scheduler, train_batch, validate
from .phase_io import get_pretrain_dataloaders, load_checkpoint_pretrain
import signal
import torch
from torchgeo.datasets import LoveDA
from processing import EvalTransforms, TrainTransforms
import utils
from utils import get_adamw_param_groups, save_checkpoint, device_setup, setup_logging, handle_shutdown

###-------CONSTANTS-------###
NUM_CLASSES = NUM_CLASSES_PRETRAIN
NUM_EPOCHS = NUM_EPOCHS_PRETRAIN
NUM_VAL_SAMPLES = NUM_VAL_SAMPLES_PRETRAIN
###-----------------------###

pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def main(device, model_path):
	if GRAD_ACCUM_STEPS < 1:
		raise ValueError(f"GRAD_ACCUM_STEPS must be >= 1, got {GRAD_ACCUM_STEPS}")

	model = SegFormer(num_classes=NUM_CLASSES, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING).to(device=device, non_blocking=True)
	
	train_loader, validation_loader = get_pretrain_dataloaders(
		loveda_cls=LoveDA,
		root=PRETRAIN_DATA_ROOT,
		scenes=PRETRAIN_SCENES,
		train_transform=TrainTransforms(),
		eval_transform=EvalTransforms(),
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
	)

	model = torch.compile(model)
	model, optimizer, scheduler, scaler, start_epoch, train_loader = load_checkpoint_pretrain(
		model_path=model_path,
		model=model,
		train_loader=train_loader,
		setup_scheduler_fn=setup_scheduler,
		get_adamw_param_groups_fn=get_adamw_param_groups,
		learning_rate=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY,
		grad_accum_steps=GRAD_ACCUM_STEPS,
		total_epochs=NUM_EPOCHS,
		warmup_epochs=WARMUP_EPOCHS,
		device=device,
		num_classes=NUM_CLASSES,
		logger=logger,
	)
	criterion = focal_loss

	model.train()
	for epoch in range(start_epoch, NUM_EPOCHS):
		train_batch(
			model=model,
			epoch=epoch,
			total_epochs=NUM_EPOCHS,
			train_loader=train_loader,
			optimizer=optimizer,
			scheduler=scheduler,
			scaler=scaler,
			criterion=criterion,
			dice_loss_fn=dice_loss,
			dou_loss_fn=dou_loss,
			num_classes=NUM_CLASSES,
			grad_accum_steps=GRAD_ACCUM_STEPS,
			phase_label="Pretrain",
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
				cast_prediction_float=True,
			)

		save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)

	logger.info("Pretraining complete. Checkpoint saved.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main(device, MODEL_PATH)


