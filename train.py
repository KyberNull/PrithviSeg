"""Training loop for the segmentation model."""

from config import get_train_config
import logging
from losses import dice_loss
from model import UNet
from rich.logging import RichHandler
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transforms import VOCTrainTransforms

config = get_train_config()
LEARNING_RATE = config.learning_rate
MODEL_PATH = config.model_path
NUM_BATCHES = config.num_batches
NUM_CLASSES = config.num_classes
NUM_EPOCHS = config.num_epochs
NUM_WORKERS = config.num_workers

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

def handle_shutdown(sig, frame):
    global shutdown_requested
    logger.warning(f'Shutdown requested! Signal: {sig}')
    shutdown_requested = True

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)

def main(device, model_path):
    # GradScaler is only useful on CUDA where float16 gradients can underflow.
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    trainData = datasets.VOCSegmentation(
        './data',
        '2012',
        image_set='train',
        transforms=VOCTrainTransforms(),
    )
    trainLoader = DataLoader(dataset=trainData, batch_size=NUM_BATCHES, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)

    model = UNet(NUM_CLASSES).to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0

    try:
        # Resume both model and optimizer states to continue training seamlessly.
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        logger.warning("Checkpoint not found. Training from scratch.")
    except RuntimeError as err:
        logger.error(f"Checkpoint incompatible with current model architecture: {err}")
        logger.warning("Training from scratch.")

    model.train() #TODO: Add Validation 
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_bar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True, disable=not sys.stdout.isatty(), position=0)
        running_loss = 0.0
        for batch, (input, output) in enumerate(epoch_bar):
            if shutdown_requested:
                save_checkpoint(model, optimizer, epoch, MODEL_PATH)
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
            scaler.update()

            running_loss += loss.item()

            avg_loss = running_loss / (batch + 1)
            epoch_bar.set_postfix(loss=avg_loss) # Shows AVG Loss NOT Batch Loss
        
        #TODO: Maybe save best model and current model
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=MODEL_PATH)
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