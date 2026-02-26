import logging
from losses import dice_loss
from model import UNet
import os
import signal
import sys
import torch
from torch import nn, optim, autocast
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transforms import IMAGE_TRANSFORM, MASK_TRANSFORM

# TODO: Make parameters depend on .env/config.yaml/something similar for better flexibility.
#===----CONSTANTS----===#
LEARNING_RATE = 0.0005 #TODO: Add learning rate scheduler and make this more dynamic.
MODEL_PATH = 'model.pt'
NUM_BATCHES = 24
NUM_CLASSES = 21
NUM_EPOCHS = 300
NUM_WORKERS = min(4, os.cpu_count() or 1)
#===-----------------===#

shutdown_requested = False
pin_memory = False
amp_dtype = torch.bfloat16

def handle_shutdown(sig, frame):
    global shutdown_requested
    logging.info(f'Signal {sig} received. Will save checkpoint after batch...')
    shutdown_requested = True

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)
    logging.info("Checkpoint saved.")

def main(device, model_path):
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    trainData = datasets.VOCSegmentation('./data', '2012', image_set='train', transform=IMAGE_TRANSFORM, target_transform=MASK_TRANSFORM)
    trainLoader = DataLoader(dataset=trainData, batch_size=NUM_BATCHES, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)

    model = UNet(NUM_CLASSES).to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0

    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        logging.warning("Checkpoint not found. Training from scratch.")

    model.train() #TODO: Add Validation 
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_bar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True, disable=not sys.stdout.isatty(), position=0)
        running_loss = 0.0
        for batch, (input, output) in enumerate(epoch_bar):
            if shutdown_requested:
                save_checkpoint(model, optimizer, epoch, MODEL_PATH)
                logging.info("Gracefully exiting ...")
                return

            input, output = input.to(device, non_blocking=True), output.squeeze(1).to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=amp_dtype):
                prediction = model(input)
                loss = criterion(prediction, output)
                
            loss += dice_loss(prediction, output, NUM_CLASSES)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            avg_loss = running_loss / (batch + 1)
            epoch_bar.set_postfix(loss=avg_loss) # Shows AVG Loss NOT Batch Loss
        
        #TODO: Maybe save best model and current model
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=MODEL_PATH)

if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        amp_dtype = torch.float16
    elif torch.mps.is_available():
        device = torch.device('mps')
        amp_dtype = torch.float16

    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main(device, MODEL_PATH)