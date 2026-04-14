"""Data augmentation and preprocessing transforms for segmentation."""

import numpy as np
import torch
from processing.nsegment import NoisySegmentPlus
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class TrainTransforms:
    """Data augmentation transforms used during training."""

    def __init__(self, noisy_mask_prob=0.25, noisy_area_thresh=1000, ignore_index=255):
        self.flips = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        self.rotate90 = v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270)),
        ])
        self.noisy_segment = NoisySegmentPlus(
            prob=noisy_mask_prob,
            area_thresh=noisy_area_thresh,
            ignore_label=ignore_index,
        )

    def __call__(self, image, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample["image"], sample["mask"]
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)
        image, mask = self.flips(image, mask)
        image, mask = self.rotate90(image, mask)
        mask_np = np.asarray(mask, dtype=np.int64)
        mask = tv_tensors.Mask(torch.from_numpy(self.noisy_segment(mask_np)))
        image = F.to_image(image)

        mask = mask.to(torch.int64)
        return image, mask


class EvalTransforms:
    """Resize and normalize transforms used for eval/inference."""

    def __call__(self, image, mask=None):
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample["image"], sample["mask"]
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.to_image(image)
        mask = mask.to(torch.int64)
        return image, mask