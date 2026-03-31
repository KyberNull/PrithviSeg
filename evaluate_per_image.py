"""Per-image evaluation and visualization with metrics shown in matplotlib."""

from datasets import geospatial_dataset
import logging
from losses import iou_metric, iou_metric_processed_fast, pixel_accuracy_metric
import matplotlib.pyplot as plt
from model import SegFormer
import os
import numpy
import signal
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import EvalTransforms, PostProcessing, IMAGENET_MEAN, IMAGENET_STD
from utils import device_setup, setup_logging, handle_shutdown, shutdown_requested

MODEL_PATH = "model.pt"
NUM_WORKERS = min(4, os.cpu_count() or 1)
NUM_BATCHES = 8
NUM_CLASSES = 4
MAX_EXAMPLES = 5
IGNORE_LABEL = 255

INPUT_DIR = "data/input_demo"
MASK_DIR = "data/masks_demo"

pin_memory = False
results_to_view = []
logger = logging.getLogger(__name__)


def test_model():
    test_dataset = geospatial_dataset(
        img_dir=INPUT_DIR,
        img_mask=MASK_DIR,
        transform=EvalTransforms(),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=NUM_BATCHES,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=NUM_WORKERS,
    )

    model = SegFormer(NUM_CLASSES).to(device=device, non_blocking=True)
    model = torch.compile(model=model)
    postprocess = PostProcessing(NUM_CLASSES)

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

    total_iou = 0.0
    total_pix_acc = 0.0
    total_iou_processed = 0.0
    total_pix_acc_processed = 0.0
    count = 0

    testing_bar = tqdm(test_dataloader, desc="Evaluating Per-Image Metrics", leave=True)

    with torch.no_grad():
        for test_input, target in testing_bar:
            if shutdown_requested:
                sys.exit(0)
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target.squeeze(1).long()

            preds = model(test_input)
            pred_mask_batch = postprocess(preds)

            for idx in range(test_input.size(0)):
                single_pred = preds[idx : idx + 1]
                single_target = target[idx : idx + 1]
                single_processed = pred_mask_batch[idx : idx + 1]
                single_target_cpu = single_target.cpu()

                sample_iou = iou_metric(single_pred, single_target, NUM_CLASSES).item()
                sample_pix_acc = pixel_accuracy_metric(single_pred, single_target, IGNORE_LABEL).item()
                sample_iou_processed = iou_metric_processed_fast(single_processed, single_target_cpu, NUM_CLASSES).item()
                sample_pix_acc_processed = pixel_accuracy_metric(single_processed, single_target_cpu, IGNORE_LABEL).item()

                total_iou += sample_iou
                total_pix_acc += sample_pix_acc
                total_iou_processed += sample_iou_processed
                total_pix_acc_processed += sample_pix_acc_processed
                count += 1

                if len(results_to_view) < MAX_EXAMPLES:
                    test_input_img = test_input[idx].cpu().numpy().transpose(1, 2, 0)
                    pred_mask = torch.argmax(single_pred[0], dim=0).cpu().numpy()
                    true_mask = single_target[0].cpu().numpy()
                    processed_mask = single_processed[0].cpu().numpy()

                    results_to_view.append(
                        {
                            "image": test_input_img,
                            "pred_mask": pred_mask,
                            "true_mask": true_mask,
                            "processed_mask": processed_mask,
                            "iou": sample_iou,
                            "pixel_acc": sample_pix_acc,
                            "iou_processed": sample_iou_processed,
                            "pixel_acc_processed": sample_pix_acc_processed,
                        }
                    )

            if len(results_to_view) >= MAX_EXAMPLES:
                break

    if count == 0:
        logger.warning("No samples were evaluated.")
        return

    logger.info(f"Mean Pixel Accuracy: {total_pix_acc / count:.4f}")
    logger.info(f"Mean IoU: {total_iou / count:.4f}")
    logger.info(f"Mean Pixel Accuracy (Processed): {total_pix_acc_processed / count:.4f}")
    logger.info(f"Mean IoU (Processed): {total_iou_processed / count:.4f}")


def view_results():
    for i, data in enumerate(results_to_view):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        image = data["image"]
        image = (image * numpy.array(IMAGENET_STD)) + numpy.array(IMAGENET_MEAN)

        axes[0].imshow(numpy.clip(image, 0, 1))
        axes[0].set_title("Input")
        axes[0].axis("off")

        true_mask = numpy.ma.masked_equal(data["true_mask"], IGNORE_LABEL)
        axes[1].imshow(true_mask, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(data["pred_mask"], cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[2].set_title("Predicted")
        axes[2].axis("off")

        axes[3].imshow(data["processed_mask"], cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[3].set_title("Processed Predicted")
        axes[3].axis("off")

        metrics_text = (
            f"Example {i + 1}  |  Pixel Acc: {data['pixel_acc']:.4f}  |  IoU: {data['iou']:.4f}"
            f"\nProcessed Pixel Acc: {data['pixel_acc_processed']:.4f}  |  Processed IoU: {data['iou_processed']:.4f}"
        )
        fig.text(
            0.5,
            0.01,
            metrics_text,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
        )

        fig.suptitle("Per-Image Segmentation Metrics", fontsize=12)
        plt.tight_layout(rect=(0, 0.07, 1, 0.95))
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