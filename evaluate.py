"""Evaluation and qualitative visualization utilities for model predictions."""

from datasets import geospatial_dataset
import logging
import os
from losses import iou_metric, iou_metric_processed_fast
import matplotlib.pyplot as plt
from model import SegFormer
import numpy
import signal
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import EvalTransforms, PostProcessing, IMAGENET_MEAN, IMAGENET_STD
from utils import device_setup, setup_logging

MODEL_PATH = "model.pt"
NUM_WORKERS = min(4, os.cpu_count() or 1)
NUM_BATCHES = 16
NUM_CLASSES = 4
MAX_EXAMPLES = 10
IGNORE_LABEL = 255
IMG_DIR = "data/phase-3/TestingDataset/processed_datasets"
MASK_DIR = "data/phase-3/TestingDataset/processed_masks"

pin_memory = False
results_to_view = []
shutdown_requested = False
logger = logging.getLogger(__name__)

def handle_shutdown(sig, frame):
	"""Handles the shutdown and saving of the model"""
	del frame
	global shutdown_requested
	logger.warning(f"Shutdown requested! Signal: {sig}")
	shutdown_requested = True

def test_model():

    test_dataset = geospatial_dataset(img_dir=IMG_DIR, img_mask=MASK_DIR, transform=EvalTransforms())
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=NUM_BATCHES, shuffle=True, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    model = SegFormer(NUM_CLASSES).to(device=device, non_blocking=True)
    model = torch.compile(model=model)


    try:
        ckpt = torch.load(MODEL_PATH)
        model.load_state_dict(ckpt["model_state"])
    except FileNotFoundError:
        logger.error("Saved model cannot be found. Train a model first")
        return
    except RuntimeError as err:
        logger.error(f"Saved checkpoint is incompatible with current model architecture: {err}")
        return

    model.eval()
    count = 0
    total_CEL = 0
    total_iou = 0
    total_iou_processed = 0

    testing_bar = tqdm(test_dataloader, desc = "Evaluating Model", leave=True)

    with torch.no_grad():
        for (test_input, target) in testing_bar:
            if shutdown_requested:
                sys.exit(0)
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Convert masks from [N, 1, H, W] to [N, H, W] class ids.
            target = target.squeeze(1).long()
            preds = model(test_input)

            if (len(results_to_view) < MAX_EXAMPLES):
                # Keep one qualitative sample per batch for quick visual inspection.
                test_input_img = test_input[0].cpu().numpy().transpose(1,2,0)
                pred_mask = torch.argmax(preds[0], dim = 0).cpu().numpy()
                true_mask = target[0].cpu().numpy()
                processed_mask = PostProcessing(NUM_CLASSES)(preds[0:1])[0].cpu().numpy()

                results_to_view.append({"image":test_input_img, 
                                        "pred_mask": pred_mask, 
                                        "true_mask": true_mask,
                                        "processed_mask": processed_mask})
            val_loss = criterion(preds, target)

            pred_mask_batch = PostProcessing(NUM_CLASSES)(preds)
            iou = iou_metric(preds, target, NUM_CLASSES)
            iou_processed = iou_metric_processed_fast(pred_mask_batch, target, NUM_CLASSES)
            total_CEL += val_loss.item()
            total_iou += iou.item()
            total_iou_processed += iou_processed.item()
            count += 1

    total_CEL /= count
    total_iou /= count
    total_iou_processed /= count

    logger.info(f"mCEL: {total_CEL:.4f}")
    logger.info(f"mIoU: {total_iou:.4f}")
    logger.info(f"mIoU (Processed): {total_iou_processed:.4f}")


def view_results():
    
    for i,data in enumerate(results_to_view):
        plt.figure(figsize=(15,5))
        image = data["image"]
        image = (image * numpy.array(IMAGENET_STD)) + numpy.array(IMAGENET_MEAN)

        plt.subplot(1,4,1)
        plt.imshow(numpy.clip(image, 0, 1))
        plt.title(f"Example {i+1}: Input")
        plt.axis('off')

        plt.subplot(1,4,2)
        true_mask = numpy.ma.masked_equal(data["true_mask"], IGNORE_LABEL)
        plt.imshow(true_mask, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.imshow(data["pred_mask"], cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.imshow(data["processed_mask"], cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        plt.title("Processed Predicted Mask")
        plt.axis("off")


        plt.tight_layout()
        plt.show()

def main():
    test_model()
    view_results()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main()