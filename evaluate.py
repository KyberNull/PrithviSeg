"""Evaluation and qualitative visualization utilities for model predictions."""

from datasets import geospatial_dataset
import logging
import os
from losses import compute_means
import matplotlib.pyplot as plt
from model import UNet
import numpy
from rich.logging import RichHandler
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import EvalTransforms, IMAGENET_MEAN, IMAGENET_STD

MODEL_PATH = "model.pt"
NUM_WORKERS = min(4, os.cpu_count() or 1)
NUM_BATCHES = 16
NUM_CLASSES = 4
MAX_EXAMPLES = 10
IGNORE_LABEL = 255

pin_memory = False
results_to_view = []
logger = logging.getLogger(__name__)

def test_model():
        
    test_img_dir = "data/phase-3/TestingDataset/processed_datasets"
    test_mask_dir = "data/phase-3/TestingDataset/processed_masks"

    test_dataset = geospatial_dataset(img_dir=test_img_dir, img_mask=test_mask_dir, transform=EvalTransforms())
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=NUM_BATCHES, shuffle=True, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    model = UNet(NUM_CLASSES).to(device=device, non_blocking=True)
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

    testing_bar = tqdm(test_dataloader, desc = "Evaluating Model", leave=True)

    with torch.no_grad():
        for (test_input, target) in testing_bar:
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

                results_to_view.append({"image":test_input_img, 
                                        "pred_mask": pred_mask, 
                                        "true_mask": true_mask})
            val_loss = criterion(preds, target)
            _, iou = compute_means(preds, target, NUM_CLASSES)
            total_CEL += val_loss.item()
            total_iou += iou.item()
            count += 1

    total_CEL /= count
    total_iou /= count

    logger.info(f"mCEL: {total_CEL:.4f}")
    logger.info(f"mIoU: {total_iou:.4f}")


def view_results():
    
    for i,data in enumerate(results_to_view):
        plt.figure(figsize=(15,5))
        image = data["image"]
        image = (image * numpy.array(IMAGENET_STD)) + numpy.array(IMAGENET_MEAN)

        plt.subplot(1,3,1)
        plt.imshow(numpy.clip(image, 0, 1))
        plt.title(f"Example {i+1}: Input")
        plt.axis('off')

        plt.subplot(1,3,2)
        true_mask = numpy.ma.masked_equal(data["true_mask"], IGNORE_LABEL)
        plt.imshow(true_mask, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(data["pred_mask"], cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def main():
    test_model()
    view_results()


if __name__ == "__main__":

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
        torch.backends.cudnn.benchmark = True

    elif torch.mps.is_available(): device = torch.device("mps")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()],
        force=True,
    )
    main()