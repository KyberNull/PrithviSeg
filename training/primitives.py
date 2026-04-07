"""Shared training/validation primitives used by train and pretrain entrypoints."""

import math
import sys

import torch
from torch import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from losses import iou_metric, iou_metric_processed_fast


def train_batch(
    *,
    model,
    epoch,
    total_epochs,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    dice_loss_fn,
    num_classes,
    grad_accum_steps,
    phase_label,
    model_path,
    device,
    amp_dtype,
    logger,
    save_checkpoint_fn,
    should_stop,
):
    """Run one training epoch with gradient accumulation and scheduler stepping."""
    epoch_bar = tqdm(train_loader, desc=f"[{phase_label}] Epoch {epoch + 1}/{total_epochs}", leave=True, disable=not sys.stdout.isatty(), position=0)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch, (input_tensor, output_tensor) in enumerate(epoch_bar):
        if should_stop():
            save_checkpoint_fn(model, optimizer, scheduler, scaler, epoch, model_path)
            return

        input_tensor = input_tensor.to(device, non_blocking=True)
        output_tensor = output_tensor.squeeze(1).to(device, non_blocking=True).long()

        with autocast(device_type=device.type, dtype=amp_dtype):
            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            with sdpa_kernel(backends=backends, set_priority=True):
                prediction = model(input_tensor)
            loss = criterion(prediction, output_tensor)
            loss += dice_loss_fn(prediction, output_tensor, num_classes)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at epoch {epoch+1}, batch {batch}; skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / grad_accum_steps).backward()
        should_step = ((batch + 1) % grad_accum_steps == 0) or ((batch + 1) == len(train_loader))
        if should_step:
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            if scale_after_step >= scale_before_step:
                scheduler.step()

        running_loss += loss.item()
        epoch_bar.set_postfix(loss=running_loss / (batch + 1), lr=scheduler.get_last_lr()[0])


def validate(
    *,
    model,
    validation_loader,
    device,
    criterion,
    num_classes,
    num_val_samples,
    amp_dtype,
    logger,
    compute_processed=False,
    post_processor=None,
    cast_prediction_float=False,
):
    """Run validation loop and report mCEL/mIoU (+ optional processed mIoU)."""
    model.eval()
    running_val_loss = 0.0
    total_iou = 0.0
    total_iou_processed = 0.0
    val_iterator = iter(validation_loader)

    with torch.no_grad():
        for _ in range(num_val_samples):
            try:
                val_input, val_output = next(val_iterator)
            except StopIteration:
                val_iterator = iter(validation_loader)
                val_input, val_output = next(val_iterator)

            val_input = val_input.to(device, non_blocking=True)
            val_output = val_output.squeeze(1).to(device, non_blocking=True).long()

            with autocast(device_type=device.type, dtype=amp_dtype):
                val_prediction = model(val_input)
                prediction_for_loss = val_prediction.float() if cast_prediction_float else val_prediction
                val_loss = criterion(prediction_for_loss, val_output)
                processed_mask = post_processor(val_prediction) if compute_processed and post_processor else None

            if not torch.isfinite(val_loss):
                continue

            running_val_loss += val_loss.item()
            total_iou += float(iou_metric(val_prediction, val_output, num_classes))
            if compute_processed and processed_mask is not None:
                total_iou_processed += float(iou_metric_processed_fast(processed_mask, val_output, num_classes))

        total_iou /= num_val_samples
        running_val_loss /= num_val_samples
        logger.info(f"mCEL: {running_val_loss:.4f}")
        logger.info(f"mIoU: {total_iou:.4f}")
        if compute_processed:
            total_iou_processed /= num_val_samples
            logger.info(f"mIoU (Processed): {total_iou_processed:.4f}")

    model.train()


def setup_scheduler(
    *,
    train_loader,
    optimizer,
    grad_accum_steps,
    total_epochs,
    warmup_epochs,
    learning_rate,
    warmup_start_factor,
    pretrain_epoch_offset=0,
):
    """Create warmup + cosine scheduler with optional pretrain offset in cosine span."""
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = min(max(0, warmup_epochs * steps_per_epoch), max(0, total_steps - 1))
    pretrain_steps = max(0, pretrain_epoch_offset * steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps - pretrain_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=learning_rate * 0.1)

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps])
    return scheduler
