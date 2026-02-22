import logging
from losses import compute_means
import matplotlib.pyplot as plt
from model import UNet
import numpy
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transforms import IMAGE_TRANSFORM, MASK_TRANSFORM

#------CONSTANTS-------#
MODEL_PATH = 'model.pt'
NUM_WORKERS = min(4, os.cpu_count() or 1)
NUM_BATCHES = 6
NUM_CLASSES = 21
MAX_EXAMPLES = 10

pin_memory = False
results_to_view = []

def test_model():
        
    testData = datasets.VOCSegmentation('./data', year = '2012', image_set = 'val', transform = IMAGE_TRANSFORM, target_transform = MASK_TRANSFORM)
    testLoader = DataLoader(dataset=testData, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, batch_size=16, persistent_workers=True)


    model = UNet(NUM_CLASSES).to(device=device, non_blocking=True)
    model = torch.compile(model=model)


    try:
        ckpt = torch.load("model.pt")
        model.load_state_dict(ckpt["model_state"])
    except FileNotFoundError:
        logging.warning("Saved model cannot be found. Train a model first")

    model.eval()
    count = 0
    total_DC = 0
    total_iou = 0

    testing_bar = tqdm(testLoader, desc = "Evaluating Model", leave=True)

    with torch.no_grad():
        for (test_input, target) in testing_bar:
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target.squeeze(1)
            preds = model(test_input)

            if (len(results_to_view) < MAX_EXAMPLES):
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

    logging.info(f"DC: {total_DC}")
    logging.info(f"IoU: {total_iou}")


def view_results():
    
    for i,data in enumerate(results_to_view):
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        img = (data["image"] * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        plt.imshow(numpy.clip(img, 0, 1))
        plt.title(f"Example {i+1}: Input")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(data["true_mask"])
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(data["pred_mask"])
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
    elif torch.mps.is_available(): device = torch.device("mps")
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()