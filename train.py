import logging
from losses import dice_loss
from model import UNet
import os
import signal
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transforms import IMAGE_TRANSFORM, MASK_TRANSFORM

#TODO: Make paramaters depend on environment variables for better flexibility.
#===----CONSTANTS----===#
LEARNING_RATE = 0.0001
MODEL_PATH = 'model.pt'
NUM_BATCHES = 16
NUM_CLASSES = 21
NUM_EPOCHS = 60
NUM_WORKERS = min(4, os.cpu_count() or 1)
#===-----------------===#

shutdown_requested = False
pin_memory = False

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

    trainData = datasets.VOCSegmentation('./data', '2012', image_set='train', transform=IMAGE_TRANSFORM, target_transform=MASK_TRANSFORM)
    trainLoader = DataLoader(dataset=trainData, batch_size=NUM_BATCHES, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=True)

    model = UNet(NUM_CLASSES).to(device=device, dtype=torch.bfloat16)
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    for epoch in tqdm(range(start_epoch, NUM_EPOCHS), desc="Training", position=0):
        epoch_bar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True, disable=not sys.stdout.isatty(), position=1)
        running_loss = 0.0
        for batch, (input, output) in enumerate(epoch_bar):
            if shutdown_requested:
                save_checkpoint(model, optimizer, epoch, MODEL_PATH)
                logging.info("Gracefully exiting ...")
                return

            input, output = input.to(device, non_blocking=True, dtype=torch.bfloat16), output.to(device, non_blocking=True, dtype=torch.bfloat16)

            output = torch.squeeze(output, 1).long()
            optimizer.zero_grad(set_to_none=True)

            prediction = model(input)
            
            #TODO: Add IoU for evaluation
            loss = dice_loss(prediction, output, num_classes=NUM_CLASSES) + criterion(prediction, output)
            loss.backward()
            optimizer.step()

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
    elif torch.mps.is_available(): device = torch.device('mps')

    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main(device, MODEL_PATH)