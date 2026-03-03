"""Evaluation and qualitative visualization utilities for model predictions."""

from config import get_eval_config
import logging
from losses import compute_means
import matplotlib.pyplot as plt
from model import UNet
import numpy
from rich.logging import RichHandler
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import VOCEvalTransforms

config = get_eval_config()
MODEL_PATH = config.model_path
NUM_WORKERS = config.num_workers
NUM_BATCHES = config.num_batches
NUM_CLASSES = config.num_classes
MAX_EXAMPLES = config.max_examples
IGNORE_LABEL = config.ignore_label

pin_memory = False
results_to_view = []
logger = logging.getLogger(__name__)

def test_model():
        
    testData = datasets.VOCSegmentation('./data', year = '2012', image_set = 'val', transforms = VOCEvalTransforms())
    testLoader = DataLoader(dataset=testData, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, batch_size=NUM_BATCHES, persistent_workers=NUM_WORKERS > 0)


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
    total_DC = 0
    total_iou = 0

    testing_bar = tqdm(testLoader, desc = "Evaluating Model", leave=True)

    with torch.no_grad():
        for (test_input, target) in testing_bar:
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Convert masks from [N, 1, H, W] to [N, H, W] class ids.
            target = target.squeeze(1)
            preds = model(test_input)

            if (len(results_to_view) < MAX_EXAMPLES):
                # Keep one qualitative sample per batch for quick visual inspection.
                test_input_img = test_input[0].cpu().numpy().transpose(1,2,0)
                pred_mask = torch.argmax(preds[0], dim = 0).cpu().numpy()
                true_mask = target[0].cpu().numpy()

                results_to_view.append({"image":test_input_img, 
                                        "pred_mask": pred_mask, 
                                        "true_mask": true_mask})

            dice_coefficient, iou = compute_means(preds, target, NUM_CLASSES)
            total_DC += dice_coefficient
            total_iou += iou
            count += 1

    total_DC /= count
    total_iou /= count

    logger.info(f"mDC: {total_DC:.4f}")
    logger.info(f"mIoU: {total_iou:.4f}")


def view_results():
    
    for i,data in enumerate(results_to_view):
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.imshow(numpy.clip(data["image"], 0, 1))
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