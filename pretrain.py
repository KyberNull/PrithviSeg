"""
Pretraining loop for segmentation on the SBD dataset.

This script is intentionally similar to train.py, but uses SBD for source-task
pretraining so train.py can be reserved for downstream/domain training.
"""

import logging
from losses import dice_loss, compute_means
from model import UNet
import os
import signal
import sys
import torch
from torch import nn, optim, autocast
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
PRETRAIN_MODEL_PATH = "model.pt"
NUM_BATCHES = 32
NUM_CLASSES = 21
NUM_EPOCHS_PHASE_1 = 20
NUM_EPOCHS_PHASE_2 = 20
NUM_EPOCHS_PHASE_3 = 40
NUM_EPOCHS = NUM_EPOCHS_PHASE_1 + NUM_EPOCHS_PHASE_2 + NUM_EPOCHS_PHASE_3
NUM_WORKERS = min(4, os.cpu_count() or 1)
VAL_INTERVAL = 10
NUM_VAL_SAMPLES = 280
CHECKPOINT_INTERVAL = 1
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

def train_batch(model, epoch, train_loader, optimizer, scheduler, scaler, criterion):
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
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, PRETRAIN_MODEL_PATH)
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
    freeze_encoder(model, 0)

def load_checkpoint(model_path, model, optimizer, scaler):
    start_epoch = 0
    train_loader, validation_loader = get_dataloaders(start_epoch)
    scheduler = setup_scheduler(train_loader, optimizer, start_epoch)
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", -1) + 1

        train_loader, validation_loader = get_dataloaders(start_epoch)
        scheduler = setup_scheduler(train_loader, optimizer, start_epoch)
        
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
    except FileNotFoundError:
        logger.warning("Pretrain checkpoint not found. Starting from scratch.")
    except (RuntimeError, KeyError) as err:
        logger.error(f"Checkpoint incompatible with current model architecture: {err}")
        logger.warning("Starting from scratch.")


    train_loader, validation_loader = get_dataloaders(start_epoch)
    scheduler = setup_scheduler(train_loader, optimizer, start_epoch)

    logger.info(f"Resuming pretraining from epoch {start_epoch+1}")

    return model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader

def get_dataloaders(epoch: int):
    if epoch < NUM_EPOCHS_PHASE_1:
        train_dataset = datasets.SBDataset(
            root='./data',
            image_set='train_noval',
            mode='boundaries',
            transforms=TrainTransforms()
        )
        val_dataset = datasets.SBDataset(
            root='./data',
            image_set='train',
            mode='boundaries',
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
    elif epoch < NUM_EPOCHS_PHASE_1 + NUM_EPOCHS_PHASE_2:
        #TODO: Road seg dataset
        pass
    return train_dataloader, val_dataloader

def setup_scheduler(train_loader, optimizer, epoch):
    if epoch < NUM_EPOCHS_PHASE_1:
        num_epochs = NUM_EPOCHS_PHASE_1
    elif epoch < NUM_EPOCHS_PHASE_1 + NUM_EPOCHS_PHASE_2:
        num_epochs = NUM_EPOCHS_PHASE_2
    else:
        num_epochs = NUM_EPOCHS_PHASE_3

    warmup_steps = min(
        max(0, WARMUP_EPOCHS * len(train_loader)),
        max(0, num_epochs * len(train_loader) - 1),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader) - warmup_steps,
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
    return scheduler

def main(device, model_path):
    
    # GradScaler is only useful on CUDA where float16 gradients can underflow.
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    model = UNet(num_classes=NUM_CLASSES).to(device=device, non_blocking=True)
    freeze_encoder(model, 0)
    #TODO based on the phase, encoder lr should be sent
    optimizer = optim.AdamW(get_adamw_param_groups(model, LEARNING_RATE, LEARNING_RATE/10, WEIGHT_DECAY), lr=LEARNING_RATE)
    model = torch.compile(model)

    model, optimizer, scheduler, scaler, start_epoch, train_loader, validation_loader = load_checkpoint(model_path, model, optimizer, scaler)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    freeze_encoder(model, 0)

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_batch(model, epoch, train_loader, optimizer, scheduler, scaler, criterion)

        if (epoch + 1) % VAL_INTERVAL == 0:
            validate(model, validation_loader, device, criterion)

        should_save = (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS
        if should_save:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                path=PRETRAIN_MODEL_PATH,
            )
            logger.info("Checkpoint stored at epoch %d", epoch)

        if epoch == NUM_EPOCHS_PHASE_1:
            train_loader, validation_loader = get_dataloaders(epoch)
            scheduler = setup_scheduler(train_loader, optimizer, epoch)

        elif epoch == NUM_EPOCHS_PHASE_2:
            train_loader, validation_loader = get_dataloaders(epoch)
            scheduler = setup_scheduler(train_loader, optimizer, epoch)

    logger.info("Pretraining complete. Checkpoint saved.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main(device, PRETRAIN_MODEL_PATH)
