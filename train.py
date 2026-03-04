"""Training loop for the segmentation model."""

from config import get_train_config
from itertools import cycle
import logging
from losses import dice_loss, compute_means
from model import UNet
from rich.logging import RichHandler
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transforms import VOCTrainTransforms, VOCEvalTransforms

config = get_train_config()
LEARNING_RATE = config.learning_rate
WEIGHT_DECAY = config.weight_decay
WARMUP_EPOCHS = config.warmup_epochs
RESTART_CYCLE_EPOCHS = config.restart_cycle_epochs
RESTART_CYCLE_MULT = config.restart_cycle_mult
MODEL_PATH = config.model_path
NUM_BATCHES = config.num_batches
NUM_CLASSES = config.num_classes
NUM_EPOCHS = config.num_epochs
NUM_WORKERS = config.num_workers
VAL_INTERVAL = config.val_interval
NUM_VAL_SAMPLES = config.num_val_samples

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)


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
    global shutdown_requested
    logger.warning(f'Shutdown requested! Signal: {sig}')
    shutdown_requested = True

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }, path)

def validate(model, val_iterator, device, criterion):
    model.eval()
    running_val_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for i in range(NUM_VAL_SAMPLES):
            val_input, val_output = next(val_iterator)

            val_input, val_output = val_input.to(device, non_blocking=True), val_output.squeeze(1).to(device, non_blocking=True).long()

            val_prediction = model(val_input)
            val_loss = criterion(val_prediction, val_output)

            running_val_loss += val_loss.item()

            _, iou = compute_means(val_prediction, val_output, NUM_CLASSES)
            total_iou += iou

        total_iou /= NUM_VAL_SAMPLES

        running_val_loss /= NUM_VAL_SAMPLES
            
        logging.info(f"mCEL: {running_val_loss:.4f}")
        logging.info(f"mIoU: {total_iou:.4f}")

        model.train()

def main(device, model_path):
    # GradScaler is only useful on CUDA where float16 gradients can underflow.
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    trainDataset = datasets.VOCSegmentation('./data', year = '2012', image_set = 'train', transforms = VOCTrainTransforms())
    trainLoader = DataLoader(dataset=trainDataset, batch_size=NUM_BATCHES, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)
    vailidationDataset = datasets.VOCSegmentation('./data', year = '2012', image_set = 'val', transforms = VOCEvalTransforms())
    validationLoader = DataLoader(dataset=vailidationDataset, batch_size=NUM_BATCHES, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = UNet(NUM_CLASSES).to(device)
    optimizer = optim.AdamW(get_adamw_param_groups(model), lr=LEARNING_RATE)
    model = torch.compile(model)
    warmup_steps = min(max(0, WARMUP_EPOCHS * len(trainLoader)), max(0, NUM_EPOCHS * len(trainLoader) - 1))
    restart_cycle_steps = max(1, RESTART_CYCLE_EPOCHS * len(trainLoader))
    restart_cycle_mult = max(1, RESTART_CYCLE_MULT)
    
    scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_cycle_steps,
            T_mult=restart_cycle_mult,
            eta_min=LEARNING_RATE * 0.1,
        )
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps],
        )

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0
    val_iterator = cycle(validationLoader) # Cycle loops around the iterable and keeps memory of the last position.

    try:
        # Resume model, optimizer, scheduler, and scaler states to continue training seamlessly.
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Training from scratch.")
    except (RuntimeError, KeyError) as err:
        logger.error(f"Checkpoint incompatible with current model architecture: {err}")
        logger.warning("Training from scratch.")

    model.train()
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_bar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True, disable=not sys.stdout.isatty(), position=0)
        running_loss = 0.0
        for batch, (input, output) in enumerate(epoch_bar):
            if shutdown_requested:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, MODEL_PATH)
                logger.info("Checkpoint saved. Gracefully exiting...")
                return

            # VOC masks come as [N, 1, H, W]; CrossEntropyLoss expects [N, H, W] class ids.
            input, output = input.to(device, non_blocking=True), output.squeeze(1).to(device, non_blocking=True).long()

            # set_to_none avoids unnecessary memory writes compared to zeroing tensors.
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=amp_dtype):
                prediction = model(input)
                loss = criterion(prediction, output)
                
                # Combine pixel-wise CE with overlap-aware Dice for better segmentation quality.
                loss += dice_loss(prediction, output, NUM_CLASSES)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            running_loss += loss.item()

            avg_loss = running_loss / (batch + 1)
            epoch_bar.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr()[0]) # Shows AVG Loss NOT Batch Loss
        
        # Validation loop
        if (epoch + 1) % VAL_INTERVAL == 0:
            validate(model, val_iterator, device, criterion)
        
        #TODO: Maybe save best model and current model
        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, path=MODEL_PATH)
    logger.info("Training complete. Checkpoint saved.")

if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        amp_dtype = torch.float16
        torch.backends.cudnn.benchmark = True

    elif torch.mps.is_available():
        device = torch.device('mps')
        amp_dtype = torch.float16

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()],
        force=True,
    )

    main(device, MODEL_PATH)